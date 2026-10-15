//! The host side of [`HostFfi`]: what a plugin's `Host` handle talks to.
//!
//! Every call is validated, runs on the caller's thread and returns without
//! waiting for the UI thread. Registries (contributions, layers, surface ids,
//! settings) change immediately, so a plugin always reads back what it just
//! wrote; visible effects are queued as [`Command`]s.

use super::shared::{Command, HostShared, PluginContext, SurfaceCommand, ViewCommand};
use crate::overlay::store::StoredRaster;
use crate::overlay::{Owner, raster::premultiply};
use abi_stable::{
    sabi_trait::TD_Opaque,
    std_types::{RErr, ROk, RStr, RString},
};
use eov_plugin_api::abi::{
    FfiError, FfiErrorKind, FfiResult, HostBox, HostFfi, HostFfi_TO, MAX_RASTER_PIXELS,
    MAX_REGION_PIXELS, Pixels, RasterUpload, RegionRequest,
};
use eov_plugin_api::protocol::{methods, *};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

/// Largest text a plugin may put on the clipboard.
const MAX_CLIPBOARD_BYTES: usize = 16 * 1024 * 1024;

pub struct HostApi {
    ctx: Arc<PluginContext>,
    shared: Arc<HostShared>,
    owner: Owner,
}

impl HostApi {
    pub fn new(ctx: Arc<PluginContext>, shared: Arc<HostShared>) -> Self {
        let owner = Owner::Plugin(Arc::clone(&ctx.id));
        Self { ctx, shared, owner }
    }

    pub fn into_ffi(self) -> HostBox {
        HostFfi_TO::from_value(self, TD_Opaque)
    }

    fn push(&self, command: Command) {
        self.shared.push(&self.ctx.id, command);
    }

    fn check_access(&self) -> Result<(), FfiError> {
        if self.ctx.is_revoked() {
            return Err(FfiError::new(
                FfiErrorKind::Revoked,
                format!("plugin '{}' is disabled", self.ctx.id),
            ));
        }
        Ok(())
    }

    fn open_file_exists(&self, file_id: FileId) -> Result<(), FfiError> {
        if self.shared.published().files.contains_key(&file_id) {
            Ok(())
        } else {
            Err(FfiError::failed(format!("no open file with id {file_id}")))
        }
    }

    fn handle(&self, method: &str, args: &str) -> Result<Value, FfiError> {
        let null = Value::Null;
        Ok(match method {
            methods::LOG => {
                let args: LogArgs = parse(args)?;
                let id = &*self.ctx.id;
                match args.level {
                    LogLevel::Trace => tracing::trace!(target: "plugin", "[{id}] {}", args.message),
                    LogLevel::Debug => tracing::debug!(target: "plugin", "[{id}] {}", args.message),
                    LogLevel::Info => tracing::info!(target: "plugin", "[{id}] {}", args.message),
                    LogLevel::Warn => tracing::warn!(target: "plugin", "[{id}] {}", args.message),
                    LogLevel::Error => tracing::error!(target: "plugin", "[{id}] {}", args.message),
                }
                // Problems reach the user, not only the log file (PLG-15).
                if matches!(args.level, LogLevel::Warn | LogLevel::Error) {
                    self.push(Command::Notify(NotifyArgs {
                        level: if args.level == LogLevel::Error {
                            NotifyLevel::Error
                        } else {
                            NotifyLevel::Warning
                        },
                        message: args.message,
                        detail: None,
                    }));
                }
                null
            }
            methods::NOTIFY => {
                let mut args: NotifyArgs = parse(args)?;
                truncate(&mut args.message, 1000);
                if let Some(detail) = &mut args.detail {
                    truncate(detail, 20_000);
                }
                self.push(Command::Notify(args));
                null
            }
            methods::STATUS => {
                let mut args: StatusArgs = parse(args)?;
                if let Some(text) = &mut args.text {
                    truncate(text, 200);
                }
                self.push(Command::Status(args.text));
                null
            }
            methods::SNAPSHOT => to_value(&self.shared.published().snapshot)?,
            methods::FILE_INFO => {
                let args: FileArgs = parse(args)?;
                let published = self.shared.published();
                let file = published.files.get(&args.file_id).ok_or_else(|| {
                    FfiError::failed(format!("no open file with id {}", args.file_id))
                })?;
                to_value(&file.info)?
            }
            methods::FILE_PROPERTIES => {
                let args: FileArgs = parse(args)?;
                let published = self.shared.published();
                let file = published.files.get(&args.file_id).ok_or_else(|| {
                    FfiError::failed(format!("no open file with id {}", args.file_id))
                })?;
                to_value(&*file.properties)?
            }
            methods::FILE_OPEN => {
                let args: OpenFileArgs = parse(args)?;
                let path = PathBuf::from(&args.path);
                if !path.exists() {
                    return Err(FfiError::failed(format!("{} does not exist", args.path)));
                }
                self.push(Command::OpenFile(path));
                null
            }

            methods::TOOLBAR_ADD => {
                let spec: ToolbarButtonSpec = parse(args)?;
                let id = spec.button_id.clone();
                let warning = self
                    .shared
                    .contributions
                    .lock()
                    .add_toolbar(&self.ctx.id, spec)
                    .map_err(invalid)?;
                self.refused(&id, warning);
                self.push(Command::ContributionsChanged);
                null
            }
            methods::TOOLBAR_UPDATE => {
                let update: ButtonUpdate = parse(args)?;
                self.shared
                    .contributions
                    .lock()
                    .update_toolbar(&self.ctx.id, &update)
                    .map_err(FfiError::failed)?;
                self.push(Command::ContributionsChanged);
                null
            }
            methods::TOOLBAR_REMOVE => {
                let args: IdArgs = parse(args)?;
                self.shared
                    .contributions
                    .lock()
                    .remove_toolbar(&self.ctx.id, &args.id);
                self.push(Command::ContributionsChanged);
                null
            }
            methods::HUD_ADD => {
                let spec: HudButtonSpec = parse(args)?;
                self.shared
                    .contributions
                    .lock()
                    .add_hud(&self.ctx.id, spec)
                    .map_err(invalid)?;
                self.push(Command::ContributionsChanged);
                null
            }
            methods::HUD_UPDATE => {
                let update: ButtonUpdate = parse(args)?;
                self.shared
                    .contributions
                    .lock()
                    .update_hud(&self.ctx.id, &update)
                    .map_err(FfiError::failed)?;
                self.push(Command::ContributionsChanged);
                null
            }
            methods::HUD_REMOVE => {
                let args: IdArgs = parse(args)?;
                self.shared
                    .contributions
                    .lock()
                    .remove_hud(&self.ctx.id, &args.id);
                self.push(Command::ContributionsChanged);
                null
            }
            methods::HOTKEY_REGISTER => {
                let spec: HotkeySpec = parse(args)?;
                let id = spec.hotkey_id.clone();
                let warning = self
                    .shared
                    .contributions
                    .lock()
                    .register_hotkey(&self.ctx.id, spec)
                    .map_err(invalid)?;
                self.refused(&id, warning);
                null
            }
            methods::HOTKEY_UNREGISTER => {
                let args: IdArgs = parse(args)?;
                self.shared
                    .contributions
                    .lock()
                    .unregister_hotkey(&self.ctx.id, &args.id);
                null
            }
            methods::TOOL_REGISTER => {
                let spec: ToolSpec = parse(args)?;
                if let Some(style) = &spec.preview_style {
                    check_style(style)?;
                }
                let id = spec.tool_id.clone();
                let warning = self
                    .shared
                    .contributions
                    .lock()
                    .register_tool(&self.ctx.id, spec)
                    .map_err(invalid)?;
                self.refused(&id, warning);
                self.push(Command::ContributionsChanged);
                null
            }
            methods::TOOL_UNREGISTER => {
                let args: IdArgs = parse(args)?;
                self.shared
                    .contributions
                    .lock()
                    .unregister_tool(&self.ctx.id, &args.id);
                self.push(Command::ContributionsChanged);
                null
            }
            methods::TOOL_ACTIVATE => {
                let args: IdArgs = parse(args)?;
                if self
                    .shared
                    .contributions
                    .lock()
                    .tool(&self.ctx.id, &args.id)
                    .is_none()
                {
                    return Err(FfiError::failed(format!("unknown tool '{}'", args.id)));
                }
                self.push(Command::ActivateTool(args.id));
                null
            }
            methods::TOOL_DEACTIVATE => {
                self.push(Command::DeactivateTools);
                null
            }

            methods::LAYER_CREATE => {
                let spec: LayerSpec = parse(args)?;
                self.open_file_exists(spec.file_id)?;
                self.shared
                    .overlays
                    .create_layer(self.owner.clone(), spec)
                    .map_err(invalid)?;
                self.push(Command::OverlaysChanged);
                null
            }
            methods::LAYER_UPDATE => {
                let update: LayerUpdate = parse(args)?;
                self.shared
                    .overlays
                    .update_layer(&self.owner, update)
                    .map_err(invalid)?;
                self.push(Command::OverlaysChanged);
                null
            }
            methods::LAYER_REMOVE => {
                let args: IdArgs = parse(args)?;
                self.shared.overlays.remove_layer(&self.owner, &args.id);
                self.push(Command::OverlaysChanged);
                null
            }
            methods::LAYER_SELECT => {
                let args: SelectArgs = parse(args)?;
                self.shared
                    .overlays
                    .select(&self.owner, &args.layer_id, &args.object_ids)
                    .map_err(FfiError::failed)?;
                self.push(Command::OverlaysChanged);
                null
            }
            methods::LAYER_UPSERT => {
                let args: UpsertObjectsArgs = parse(args)?;
                self.shared
                    .overlays
                    .upsert(&self.owner, &args.layer_id, args.objects, args.replace_all)
                    .map_err(invalid)?;
                self.push(Command::OverlaysChanged);
                null
            }
            methods::LAYER_REMOVE_OBJECTS => {
                let args: RemoveObjectsArgs = parse(args)?;
                self.shared
                    .overlays
                    .remove_objects(&self.owner, &args.layer_id, &args.object_ids)
                    .map_err(FfiError::failed)?;
                self.push(Command::OverlaysChanged);
                null
            }
            methods::BADGE_SET => {
                let args: BadgeArgs = parse(args)?;
                validate_badge_id(&args.badge_id)?;
                self.shared
                    .set_badge(&self.ctx.id, &args.badge_id, args.file_id, args.text)
                    .map_err(invalid)?;
                self.push(Command::Render);
                null
            }
            methods::LAYER_HIGHLIGHT => {
                let args: SelectArgs = parse(args)?;
                self.shared
                    .overlays
                    .set_highlight(&self.owner, &args.layer_id, &args.object_ids)
                    .map_err(invalid)?;
                self.push(Command::OverlaysChanged);
                null
            }
            methods::LAYER_SET_MIN_ZOOM => {
                let args: MinZoomArgs = parse(args)?;
                self.shared
                    .overlays
                    .set_min_zoom(&self.owner, &args.layer_id, args.min_zoom)
                    .map_err(invalid)?;
                self.push(Command::OverlaysChanged);
                null
            }
            methods::LAYER_SET_PANES => {
                let args: PaneFilterArgs = parse(args)?;
                self.shared
                    .overlays
                    .set_panes(&self.owner, &args.layer_id, args.panes)
                    .map_err(invalid)?;
                self.push(Command::OverlaysChanged);
                null
            }
            methods::LAYER_SET_PANE_VISIBILITY => {
                let args: PaneVisibilityArgs = parse(args)?;
                self.shared
                    .overlays
                    .set_pane_visibility(&self.owner, &args.layer_id, args.pane, args.visible)
                    .map_err(invalid)?;
                self.push(Command::OverlaysChanged);
                null
            }
            methods::LAYER_CLEAR => {
                let args: IdArgs = parse(args)?;
                self.shared
                    .overlays
                    .clear_layer(&self.owner, &args.id)
                    .map_err(FfiError::failed)?;
                self.push(Command::OverlaysChanged);
                null
            }

            methods::SURFACE_OPEN => {
                let mut spec: SurfaceSpec = parse(args)?;
                let path = resolve_plugin_path(&self.ctx.root, &spec.ui_path).map_err(invalid)?;
                if !is_identifier(&spec.component) {
                    return Err(invalid(format!(
                        "'{}' is not a valid Slint component name",
                        spec.component
                    )));
                }
                for size in [spec.width, spec.height, spec.min_width, spec.min_height]
                    .into_iter()
                    .flatten()
                {
                    if !size.is_finite() || !(0.0..=16_384.0).contains(&size) {
                        return Err(invalid("surface sizes must be finite and within 0–16384"));
                    }
                }
                spec.ui_path = path.to_string_lossy().into_owned();
                let surface_id = self.shared.fresh_id(&format!("{}-surface", self.ctx.id));
                self.shared.register_surface(&surface_id, &self.ctx.id);
                self.push(Command::Surface(SurfaceCommand::Open {
                    surface_id: surface_id.clone(),
                    spec,
                }));
                to_value(&SurfaceOpened { surface_id })?
            }
            methods::SURFACE_SET => {
                let args: SurfaceProperties = parse(args)?;
                self.own_surface(&args.surface_id)?;
                self.push(Command::Surface(SurfaceCommand::Set {
                    surface_id: args.surface_id,
                    properties: args.properties,
                }));
                null
            }
            methods::SURFACE_CLOSE => {
                let args: IdArgs = parse(args)?;
                if self.own_surface(&args.id).is_ok() {
                    self.shared.unregister_surface(&args.id);
                    self.push(Command::Surface(SurfaceCommand::Close {
                        surface_id: args.id,
                    }));
                }
                null
            }
            methods::SURFACE_IS_OPEN => {
                let args: IdArgs = parse(args)?;
                json!(self.own_surface(&args.id).is_ok())
            }

            methods::DIALOG_CONFIRM => {
                let mut args: ConfirmArgs = parse(args)?;
                truncate(&mut args.title, 200);
                truncate(&mut args.message, 5000);
                let request_id = self.shared.fresh_id("confirm");
                self.push(Command::Confirm {
                    request_id: request_id.clone(),
                    args,
                });
                to_value(&RequestId { request_id })?
            }
            methods::DIALOG_OPEN_FILE | methods::DIALOG_SAVE_FILE => {
                let args: FileDialogArgs = parse(args)?;
                let request_id = self.shared.fresh_id("dialog");
                self.push(Command::FileDialog {
                    request_id: request_id.clone(),
                    args,
                    save: method == methods::DIALOG_SAVE_FILE,
                });
                to_value(&RequestId { request_id })?
            }

            methods::VIEW_SET => {
                let args: ViewSetArgs = parse(args)?;
                finite(&[args.center_x, args.center_y, args.zoom])?;
                if args.zoom <= 0.0 {
                    return Err(invalid("zoom must be positive"));
                }
                self.push(Command::View(ViewCommand::Set(args)));
                null
            }
            methods::VIEW_ORIENTATION => {
                let args: ViewOrientationArgs = parse(args)?;
                if eov_common::viewport::Orientation::from_degrees(args.rotation_deg, args.flipped)
                    .is_none()
                {
                    return Err(invalid("rotation_deg must be a multiple of 90"));
                }
                self.push(Command::View(ViewCommand::Orientation(args)));
                null
            }
            methods::VIEW_PAN => {
                let args: ViewPanArgs = parse(args)?;
                finite(&[args.dx, args.dy])?;
                self.push(Command::View(ViewCommand::Pan(args)));
                null
            }
            methods::VIEW_ZOOM => {
                let args: ViewZoomArgs = parse(args)?;
                finite(&[args.factor])?;
                finite(
                    &args
                        .anchor_x
                        .into_iter()
                        .chain(args.anchor_y)
                        .collect::<Vec<_>>(),
                )?;
                if args.factor <= 0.0 {
                    return Err(invalid("zoom factor must be positive"));
                }
                self.push(Command::View(ViewCommand::Zoom(args)));
                null
            }
            methods::VIEW_DISPLAY => {
                let args: PaneArgs = parse(args)?;
                let published = self.shared.published();
                let pane = args.pane.or(published.snapshot.focused_pane);
                let display = pane
                    .and_then(|pane| published.displays.get(&pane).cloned())
                    .unwrap_or_default();
                to_value(&display)?
            }
            methods::VIEW_FIT => {
                let args: PaneArgs = parse(args)?;
                self.push(Command::View(ViewCommand::Fit(args.pane)));
                null
            }
            methods::VIEW_FRAME => {
                let args: ViewFrameArgs = parse(args)?;
                let r = args.rect;
                finite(&[r.x, r.y, r.width, r.height])?;
                if r.width <= 0.0 || r.height <= 0.0 {
                    return Err(invalid("the frame rectangle must have a positive size"));
                }
                self.push(Command::View(ViewCommand::Frame(args)));
                null
            }
            methods::APP_CYCLE_PANE => {
                let args: OffsetArgs = parse(args)?;
                self.push(Command::CyclePane(args.offset));
                null
            }
            methods::APP_CYCLE_TAB => {
                let args: OffsetArgs = parse(args)?;
                self.push(Command::CycleTab(args.offset));
                null
            }
            methods::APP_SERIES => {
                let args: SeriesArgs = parse(args)?;
                self.push(Command::Series(args.action));
                null
            }
            methods::RENDER_REQUEST => {
                self.push(Command::Render);
                null
            }

            methods::UNDO_PUSH => {
                let mut args: UndoPushArgs = parse(args)?;
                if args.entry_id.is_empty() || args.entry_id.len() > 256 {
                    return Err(invalid("undo entry ids must be 1–256 bytes"));
                }
                truncate(&mut args.label, 100);
                self.push(Command::UndoPush(args));
                null
            }
            methods::UNDO_CLEAR => {
                let args: UndoClearArgs = parse(args)?;
                self.push(Command::UndoClear(args));
                null
            }

            methods::FILTER_REGISTER => {
                let spec: FilterSpec = parse(args)?;
                self.shared
                    .contributions
                    .lock()
                    .register_filter(&self.ctx.id, spec)
                    .map_err(invalid)?;
                self.push(Command::ContributionsChanged);
                self.push(Command::Render);
                null
            }
            methods::FILTER_SET_ENABLED => {
                let args: FilterEnableArgs = parse(args)?;
                let changed = self
                    .shared
                    .contributions
                    .lock()
                    .set_filter_enabled(&self.ctx.id, &args.filter_id, args.enabled)
                    .map_err(FfiError::failed)?;
                if changed {
                    self.push(Command::ContributionsChanged);
                    self.push(Command::Render);
                }
                null
            }
            methods::FILTER_UNREGISTER => {
                let args: IdArgs = parse(args)?;
                if self
                    .shared
                    .contributions
                    .lock()
                    .unregister_filter(&self.ctx.id, &args.id)
                {
                    self.push(Command::ContributionsChanged);
                    self.push(Command::Render);
                }
                null
            }

            methods::SETTINGS_GET => {
                let args: KeyArgs = parse(args)?;
                self.shared.settings.get(&self.ctx.id, &args.key)
            }
            methods::SETTINGS_SET => {
                let args: SettingsSetArgs = parse(args)?;
                self.shared
                    .settings
                    .set(&self.ctx.id, &args.key, args.value)
                    .map_err(FfiError::failed)?;
                null
            }

            methods::JOB_START => {
                let mut args: JobStartArgs = parse(args)?;
                truncate(&mut args.title, 200);
                let job_id = self.shared.fresh_id(&format!("{}-job", self.ctx.id));
                self.push(Command::JobStart {
                    job_id: job_id.clone(),
                    args,
                });
                to_value(&JobStarted { job_id })?
            }
            methods::JOB_UPDATE => {
                let mut args: JobUpdateArgs = parse(args)?;
                if let Some(fraction) = args.fraction {
                    if !fraction.is_finite() {
                        return Err(invalid("job progress must be finite"));
                    }
                    args.fraction = Some(fraction.clamp(0.0, 1.0));
                }
                if let Some(message) = &mut args.message {
                    truncate(message, 200);
                }
                self.push(Command::JobUpdate(args));
                null
            }
            methods::JOB_FINISH => {
                let args: JobFinishArgs = parse(args)?;
                self.push(Command::JobFinish(args));
                null
            }

            methods::CLIPBOARD_SET_TEXT => {
                let args: TextArgs = parse(args)?;
                if args.text.len() > MAX_CLIPBOARD_BYTES {
                    return Err(invalid("clipboard text is too large"));
                }
                self.push(Command::Clipboard(args.text));
                null
            }
            methods::ROI_GET => {
                let args: FileArgs = parse(args)?;
                let published = self.shared.published();
                json!(published.files.get(&args.file_id).and_then(|f| f.roi))
            }
            methods::MEASUREMENTS_LIST => {
                let args: FileArgs = parse(args)?;
                let published = self.shared.published();
                to_value(
                    &published
                        .files
                        .get(&args.file_id)
                        .map(|f| f.measurements.clone())
                        .unwrap_or_default(),
                )?
            }
            methods::POST => {
                let args: PostArgs = parse(args)?;
                if args.token.len() > 4096 {
                    return Err(invalid("post tokens are limited to 4096 bytes"));
                }
                self.push(Command::Post(args.token));
                null
            }
            methods::EVENTS_SUBSCRIBE => {
                let args: SubscribeArgs = parse(args)?;
                let events: HashSet<String> = args.events.into_iter().collect();
                self.shared.set_subscriptions(&self.ctx.id, Some(events));
                null
            }
            other => {
                return Err(FfiError::new(
                    FfiErrorKind::Unsupported,
                    format!("unknown method '{other}'"),
                ));
            }
        })
    }

    fn own_surface(&self, surface_id: &str) -> Result<(), FfiError> {
        match self.shared.surface_owner(surface_id) {
            Some(owner) if owner == self.ctx.id => Ok(()),
            Some(_) => Err(FfiError::failed("the surface belongs to another plugin")),
            None => Err(FfiError::failed(format!(
                "surface '{surface_id}' is not open"
            ))),
        }
    }

    /// Reports a refused hotkey to the user and to the plugin.
    fn refused(&self, id: &str, warning: Option<String>) {
        if let Some(message) = warning {
            tracing::warn!("{message}");
            self.push(Command::Notify(NotifyArgs {
                level: NotifyLevel::Warning,
                message: message.clone(),
                detail: None,
            }));
            self.push(Command::Refused {
                what: "hotkey".to_string(),
                id: id.to_string(),
                reason: message,
            });
        }
    }

    fn read_region_inner(&self, request: RegionRequest) -> Result<Pixels, FfiError> {
        let pixels = u64::from(request.width) * u64::from(request.height);
        if pixels == 0 {
            return Err(invalid("regions must be at least 1×1"));
        }
        if pixels > MAX_REGION_PIXELS {
            return Err(invalid(format!(
                "region of {}×{} exceeds the {MAX_REGION_PIXELS}-pixel limit; read it in tiles",
                request.width, request.height
            )));
        }
        let published = self.shared.published();
        let file = published
            .files
            .get(&request.file_id)
            .ok_or_else(|| FfiError::failed(format!("no open file with id {}", request.file_id)))?;
        if file.info.levels.len() as u64 <= u64::from(request.level) {
            return Err(invalid(format!(
                "level {} does not exist (the slide has {})",
                request.level,
                file.info.levels.len()
            )));
        }
        drop(published);
        let reader = self
            .shared
            .reader(request.file_id)
            .ok_or_else(|| FfiError::failed("the slide is not readable"))?;
        let data = reader.read(&request).map_err(FfiError::failed)?;
        Ok(Pixels {
            width: request.width,
            height: request.height,
            data: data.into(),
        })
    }

    fn put_raster_inner(&self, raster: RasterUpload) -> Result<(), FfiError> {
        if !raster.pixels.is_well_formed() {
            return Err(invalid("raster size does not match its data"));
        }
        let count = u64::from(raster.pixels.width) * u64::from(raster.pixels.height);
        if count == 0 || count > MAX_RASTER_PIXELS {
            return Err(invalid(format!(
                "rasters must have 1–{MAX_RASTER_PIXELS} pixels"
            )));
        }
        let mut data = raster.pixels.data.into_vec();
        premultiply(&mut data);
        self.shared
            .overlays
            .put_raster(
                &self.owner,
                raster.layer_id.as_str(),
                StoredRaster {
                    object_id: raster.object_id.into_string(),
                    rect: Rect {
                        x: raster.x,
                        y: raster.y,
                        width: raster.width,
                        height: raster.height,
                    },
                    width: raster.pixels.width,
                    height: raster.pixels.height,
                    premultiplied: Arc::new(data),
                    smooth: raster.smooth,
                    seq: 0,
                },
            )
            .map_err(invalid)?;
        self.push(Command::OverlaysChanged);
        Ok(())
    }
}

/// Runs a host call, turning host-side panics into errors instead of
/// aborting the process (PLG-7).
fn guarded<T>(ctx: &PluginContext, f: impl FnOnce() -> Result<T, FfiError>) -> FfiResult<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => ROk(value),
        Ok(Err(error)) => RErr(error),
        Err(_) => {
            tracing::error!("host panicked while serving plugin '{}'", ctx.id);
            RErr(FfiError::failed("internal host error"))
        }
    }
}

impl HostFfi for HostApi {
    fn call(&self, method: RStr<'_>, args_json: RStr<'_>) -> FfiResult<RString> {
        guarded(&self.ctx, || {
            self.check_access()?;
            self.handle(method.as_str(), args_json.as_str())
                .map(|value| RString::from(value.to_string()))
        })
    }

    fn read_region(&self, request: RegionRequest) -> FfiResult<Pixels> {
        guarded(&self.ctx, || {
            self.check_access()?;
            self.read_region_inner(request)
        })
    }

    fn read_associated_image(&self, file_id: i64, name: RStr<'_>) -> FfiResult<Pixels> {
        guarded(&self.ctx, || {
            self.check_access()?;
            self.open_file_exists(file_id)?;
            let reader = self
                .shared
                .reader(file_id)
                .ok_or_else(|| FfiError::failed("the slide is not readable"))?;
            let (width, height, data) = reader
                .associated_image(name.as_str())
                .map_err(FfiError::failed)?;
            Ok(Pixels {
                width,
                height,
                data: data.into(),
            })
        })
    }

    fn put_raster(&self, raster: RasterUpload) -> FfiResult<()> {
        guarded(&self.ctx, || {
            self.check_access()?;
            self.put_raster_inner(raster)
        })
    }
}

fn validate_badge_id(id: &str) -> Result<(), FfiError> {
    if id.is_empty() || id.len() > 64 {
        return Err(invalid("badge ids must be 1–64 bytes"));
    }
    Ok(())
}

fn parse<T: DeserializeOwned>(args: &str) -> Result<T, FfiError> {
    serde_json::from_str(args).map_err(|e| invalid(format!("invalid arguments: {e}")))
}

fn to_value<T: serde::Serialize + ?Sized>(value: &T) -> Result<Value, FfiError> {
    serde_json::to_value(value).map_err(|e| FfiError::failed(e.to_string()))
}

fn invalid(message: impl Into<String>) -> FfiError {
    FfiError::new(FfiErrorKind::InvalidArguments, message)
}

fn finite(values: &[f64]) -> Result<(), FfiError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(invalid("coordinates and factors must be finite"))
    }
}

fn check_style(style: &Style) -> Result<(), FfiError> {
    let sizes = [style.stroke_width, style.point_radius, style.font_size];
    if sizes
        .iter()
        .all(|v| v.is_finite() && (0.0..=10_000.0).contains(v))
    {
        Ok(())
    } else {
        Err(invalid("style sizes must be finite and within 0–10000"))
    }
}

fn truncate(text: &mut String, max_chars: usize) {
    if let Some((index, _)) = text.char_indices().nth(max_chars) {
        text.truncate(index);
        text.push('…');
    }
}

/// Whether `name` can be a Slint component name.
pub fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        && name.len() <= 128
}

/// Resolves a path a plugin gave relative to its root, refusing anything
/// that escapes the root (absolute paths, `..`, drive prefixes, symlinks
/// pointing outside) — PLG-13.
pub fn resolve_plugin_path(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let rel = Path::new(relative);
    if relative.is_empty()
        || rel.is_absolute()
        || rel
            .components()
            .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
    {
        return Err(format!(
            "'{relative}' must be a relative path inside the plugin package"
        ));
    }
    let root = dunce::canonicalize(root).map_err(|e| format!("{}: {e}", root.display()))?;
    let path = dunce::canonicalize(root.join(rel)).map_err(|e| format!("{relative}: {e}"))?;
    if !path.starts_with(&root) {
        return Err(format!("'{relative}' resolves outside the plugin package"));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlay::OverlayStore;
    use crate::plugin_runtime::settings::SettingsStore;
    use crate::plugin_runtime::shared::{Published, PublishedFile, SlideReader};
    use eov_plugin_api::sdk::Host;

    struct Grey;

    impl SlideReader for Grey {
        fn read(&self, request: &RegionRequest) -> Result<Vec<u8>, String> {
            Ok([100, 100, 100, 255].repeat((request.width * request.height) as usize))
        }

        fn associated_image(&self, name: &str) -> Result<(u32, u32, Vec<u8>), String> {
            if name == "label" {
                Ok((1, 1, vec![1, 2, 3, 255]))
            } else {
                Err("missing".into())
            }
        }
    }

    fn setup() -> (tempfile::TempDir, Arc<HostShared>, Arc<PluginContext>, Host) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("ui")).unwrap();
        std::fs::write(
            dir.path().join("ui/panel.slint"),
            "export component Panel {}",
        )
        .unwrap();
        let settings = SettingsStore::new();
        settings.load("p", &dir.path().join("data"));
        let shared = Arc::new(HostShared::new(
            Arc::new(OverlayStore::new()),
            settings,
            "1.0.0",
        ));
        let mut published = Published::default();
        published.files.insert(
            1,
            PublishedFile {
                info: FileInfo {
                    file_id: 1,
                    levels: vec![LevelInfo {
                        level: 0,
                        width: 1000,
                        height: 1000,
                        downsample: 1.0,
                    }],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        shared.publish(published);
        shared.set_reader(1, Arc::new(Grey));
        let ctx = Arc::new(PluginContext::new("p", dir.path().to_path_buf()));
        let host = Host::from_ffi(HostApi::new(Arc::clone(&ctx), Arc::clone(&shared)).into_ffi());
        (dir, shared, ctx, host)
    }

    #[test]
    fn mutations_queue_commands_and_registries_update_immediately() {
        let (_dir, shared, _ctx, host) = setup();
        host.add_toolbar_button(ToolbarButtonSpec {
            button_id: "go".into(),
            tooltip: "Go".into(),
            icon_svg: "<svg/>".into(),
            ..Default::default()
        })
        .unwrap();
        assert!(
            shared
                .contributions
                .lock()
                .toolbar_button("p", "go")
                .is_some()
        );
        let layer = host.create_layer(LayerSpec::new("cells", 1)).unwrap();
        layer
            .upsert(vec![OverlayObject::new(
                "a",
                Shape::Point { x: 1.0, y: 1.0 },
            )])
            .unwrap();
        assert_eq!(
            shared
                .overlays
                .layer(&Owner::plugin("p"), "cells")
                .unwrap()
                .len(),
            1
        );
        let commands: Vec<_> = shared.take_commands().into_iter().map(|(_, c)| c).collect();
        assert_eq!(commands[0], Command::ContributionsChanged);
        assert!(commands.contains(&Command::OverlaysChanged));
        // Layers need an open file.
        assert!(host.create_layer(LayerSpec::new("x", 99)).is_err());
    }

    #[test]
    fn surfaces_are_owned_and_paths_contained() {
        let (dir, shared, _ctx, host) = setup();
        let surface = host
            .open_surface(SurfaceSpec::new(
                SurfaceKind::Sidebar,
                "ui/panel.slint",
                "Panel",
            ))
            .unwrap();
        assert!(surface.is_open());
        assert!(
            host.open_surface(SurfaceSpec::new(
                SurfaceKind::Sidebar,
                "../escape.slint",
                "Panel"
            ))
            .is_err()
        );
        assert!(
            host.open_surface(SurfaceSpec::new(
                SurfaceKind::Sidebar,
                dir.path()
                    .join("ui/panel.slint")
                    .to_string_lossy()
                    .into_owned(),
                "Panel"
            ))
            .is_err()
        );
        assert!(
            host.open_surface(SurfaceSpec::new(
                SurfaceKind::Sidebar,
                "ui/panel.slint",
                "Bad Name"
            ))
            .is_err()
        );
        // Another plugin cannot touch it.
        let other = Arc::new(PluginContext::new("q", dir.path().into()));
        let other_host = Host::from_ffi(HostApi::new(other, Arc::clone(&shared)).into_ffi());
        assert!(other_host.surface(surface.id()).set("x", 1).is_err());
        surface.close().unwrap();
        assert!(!surface.is_open());
    }

    #[test]
    fn invalid_numbers_are_rejected() {
        let (_dir, _shared, _ctx, host) = setup();
        assert!(host.pan(None, f64::NAN, 0.0).is_err());
        assert!(host.zoom(None, 0.0).is_err());
        assert!(
            host.frame(
                None,
                Rect {
                    x: 0.0,
                    y: 0.0,
                    width: -1.0,
                    height: 1.0
                }
            )
            .is_err()
        );
    }

    #[test]
    fn region_reads_are_bounded_and_validated() {
        let (_dir, _shared, _ctx, host) = setup();
        let image = host
            .read_region(RegionRequest {
                file_id: 1,
                level: 0,
                x: -10,
                y: -10,
                width: 4,
                height: 2,
                background: 0,
            })
            .unwrap();
        assert_eq!((image.width, image.height, image.data.len()), (4, 2, 32));
        let too_big = host.read_region(RegionRequest {
            file_id: 1,
            level: 0,
            x: 0,
            y: 0,
            width: 100_000,
            height: 100_000,
            background: 0,
        });
        assert_eq!(too_big.unwrap_err().kind, FfiErrorKind::InvalidArguments);
        let bad_level = host.read_region(RegionRequest {
            file_id: 1,
            level: 5,
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            background: 0,
        });
        assert!(bad_level.is_err());
        assert_eq!(
            host.read_associated_image(1, "label").unwrap().data,
            vec![1, 2, 3, 255]
        );
    }

    #[test]
    fn revoked_plugins_lose_access() {
        let (_dir, _shared, ctx, host) = setup();
        ctx.revoke();
        let error = host.snapshot().unwrap_err();
        assert_eq!(error.kind, FfiErrorKind::Revoked);
    }

    #[test]
    fn unknown_methods_are_unsupported_and_settings_persist() {
        let (_dir, _shared, _ctx, host) = setup();
        assert!(
            host.call::<_, Value>("nope", &json!({}))
                .unwrap_err()
                .is_unsupported()
        );
        host.set_setting("k", &"v").unwrap();
        assert_eq!(host.setting::<String>("k").unwrap().as_deref(), Some("v"));
    }

    #[test]
    fn rasters_are_premultiplied_and_stored() {
        let (_dir, shared, _ctx, host) = setup();
        let layer = host.create_layer(LayerSpec::new("heat", 1)).unwrap();
        layer
            .put_raster(
                "h",
                Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
                eov_plugin_api::sdk::Image {
                    width: 1,
                    height: 1,
                    data: vec![200, 100, 50, 128],
                },
                true,
            )
            .unwrap();
        let stored = shared.overlays.layer(&Owner::plugin("p"), "heat").unwrap();
        let raster = stored.rasters().next().unwrap();
        assert_eq!(&raster.premultiplied[..], &[100, 50, 25, 128]);
        let bad = layer.put_raster(
            "bad",
            Rect {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
            },
            eov_plugin_api::sdk::Image {
                width: 2,
                height: 2,
                data: vec![0; 3],
            },
            true,
        );
        assert!(bad.is_err());
    }
}

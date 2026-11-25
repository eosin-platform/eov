//! The safe Rust SDK for writing eov plugins.
//!
//! Implement [`Plugin`] and export it with [`crate::export_plugin!`]:
//!
//! ```no_run
//! use eov_plugin_api::prelude::*;
//!
//! struct Hello {
//!     host: Host,
//! }
//!
//! impl Plugin for Hello {
//!     fn create(host: Host, _init: PluginInit) -> Result<Self, String> {
//!         host.add_toolbar_button(ToolbarButtonSpec {
//!             button_id: "hello".into(),
//!             tooltip: "Say hello".into(),
//!             icon_svg: "<svg xmlns='http://www.w3.org/2000/svg'/>".into(),
//!             ..Default::default()
//!         })
//!         .map_err(|e| e.to_string())?;
//!         Ok(Self { host })
//!     }
//!
//!     fn on_event(&self, event: HostEvent) -> Result<(), String> {
//!         if let HostEvent::ToolbarButtonClicked { .. } = event {
//!             self.host.notify(NotifyLevel::Info, "Hello from a plugin");
//!         }
//!         Ok(())
//!     }
//! }
//!
//! eov_plugin_api::export_plugin!(Hello);
//! ```
//!
//! Every call into the plugin is wrapped in `catch_unwind`: a panic becomes an
//! error the host reports (and the host disables the plugin) instead of
//! aborting eov.
//!
//! # Slide formats
//!
//! A plugin that reads slides eov cannot read itself declares the formats
//! in its manifest (`[[slide_formats]]`), implements
//! [`Plugin::open_slide`] and returns a [`SlideReader`] per slide;
//! [`crate::export_plugin!`] exports the rest:
//!
//! ```no_run
//! use eov_plugin_api::prelude::*;
//!
//! struct Flat;
//!
//! impl SlideReader for Flat {
//!     fn info(&self) -> SlideInfo {
//!         SlideInfo {
//!             levels: vec![SlideLevel { width: 1024, height: 1024, ..Default::default() }],
//!             ..Default::default()
//!         }
//!     }
//!
//!     fn read_region(&self, _region: &SlideRegion, out: &mut [u8]) -> Result<(), String> {
//!         for px in out.chunks_exact_mut(4) {
//!             px.copy_from_slice(&[230, 200, 220, 255]);
//!         }
//!         Ok(())
//!     }
//! }
//!
//! struct FlatFormat;
//!
//! impl Plugin for FlatFormat {
//!     fn create(_host: Host, _init: PluginInit) -> Result<Self, String> {
//!         Ok(Self)
//!     }
//!
//!     fn open_slide(&self, request: &SlideOpenRequest) -> Result<Box<dyn SlideReader>, String> {
//!         std::fs::metadata(&request.location).map_err(|e| e.to_string())?;
//!         Ok(Box::new(Flat))
//!     }
//! }
//!
//! eov_plugin_api::export_plugin!(FlatFormat);
//! ```

use crate::abi::{
    self, FfiError, FfiErrorKind, FfiResult, FilterFrame, FilterView, HostBox, Pixels, PluginBox,
    PluginFfi, PluginFfi_TO, PluginModule, PluginModuleRef, RasterUpload, RegionRequest, SlideBox,
    SlideFfi, SlideFfi_TO, SlideRegion, SlideSourceBox, SlideSourceFfi, SlideSourceFfi_TO,
};
use crate::i18n::Catalog;
use crate::protocol::{methods, *};
use abi_stable::{
    prefix_type::PrefixTypeTrait,
    sabi_trait::{TD_CanDowncast, TD_Opaque},
    std_types::{RErr, ROk, RSliceMut, RStr, RString},
};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

/// Name of the static `abi_stable` looks up in a plugin library. Spelled out
/// so that [`crate::export_plugin!`] needs no `abi_stable` dependency in the
/// plugin crate; a unit test keeps it in sync.
pub const ROOT_MODULE_LOADER_SYMBOL: &str = "_1as_0lib_1header_0root_bmodule_bloader";

// ---------------------------------------------------------------------------
// The plugin trait
// ---------------------------------------------------------------------------

/// An eov plugin.
///
/// The host calls [`on_event`](Self::on_event), [`context_menu`](Self::context_menu)
/// and [`ui_callback`](Self::ui_callback) on its UI thread, one at a time, with
/// no host locks held; keep them short and move slow work to your own threads
/// (every [`Host`] method may be called from any thread).
/// [`apply_filter`](Self::apply_filter) runs on render worker threads,
/// possibly concurrently for different panes.
pub trait Plugin: Send + Sync + Sized + 'static {
    /// Creates the plugin. Keep `host` to call back into eov later.
    fn create(host: Host, init: PluginInit) -> Result<Self, String>;

    /// Handles an event. A returned error is shown to the user as a
    /// warning, as it is (translate it with `tr!`).
    fn on_event(&self, event: HostEvent) -> Result<(), String> {
        let _ = event;
        Ok(())
    }

    /// Contributes items to a viewport context menu.
    fn context_menu(&self, context: &MenuContext) -> Vec<MenuItem> {
        let _ = context;
        Vec::new()
    }

    /// Handles a callback declared in one of the plugin's `.slint` surfaces.
    /// The returned value becomes the callback's return value (use `null` for
    /// callbacks without one).
    fn ui_callback(&self, surface_id: &str, name: &str, args: &[Value]) -> Result<Value, String> {
        let _ = (surface_id, name, args);
        Ok(Value::Null)
    }

    /// Applies the CPU filter `filter_id` (registered with
    /// [`Host::register_filter`]) to a rendered frame in place.
    fn apply_filter(&self, filter_id: &str, frame: &mut Frame<'_>) -> Result<(), String> {
        let _ = (filter_id, frame);
        Ok(())
    }

    /// Opens a slide of one of the formats the manifest declares
    /// (`[[slide_formats]]`): a file whose extension, or a
    /// URI whose scheme, matched the format `request.format_id`.
    ///
    /// Called on worker threads, possibly several at once, with no host
    /// locks held; it may take a while (a network round trip), but it must
    /// return. A panic stops the plugin. An error says why the slide did not
    /// open, in English, as eov's own readers say it: eov writes it to its
    /// log, returns it to scripts, and shows it to the user inside its own,
    /// translated message (do not translate it, or [`crate::i18n::isolate`]
    /// values in it).
    fn open_slide(&self, request: &SlideOpenRequest) -> Result<Box<dyn SlideReader>, String> {
        Err(format!(
            "this plugin reads no slide formats (asked for '{}')",
            request.format_id
        ))
    }

    /// Called once before the plugin is unloaded. Stop your threads here.
    /// Slides the plugin opened may still be read (and dropped) afterwards,
    /// until eov has let go of them.
    fn shutdown(&self) {}
}

/// One slide opened by [`Plugin::open_slide`].
///
/// eov reads regions from its render workers, several threads at once on
/// the same reader, so it is `Send + Sync`: guard non-thread-safe decoders
/// with a lock of your own. Every call runs inside `catch_unwind`; a panic
/// fails the call and stops the plugin, an `Err` fails only that read (the
/// tile shows as missing and is tried again later).
pub trait SlideReader: Send + Sync + 'static {
    /// The slide's metadata, asked for once right after opening. eov checks
    /// it ([`SlideInfo::validate`]) and refuses slides it cannot use.
    fn info(&self) -> SlideInfo;

    /// Fills `out` with the pixels of `region`, which lies inside its level
    /// (in pixels of that level, see [`SlideRegion`]): `width·height·4`
    /// bytes of RGBA with straight alpha, rows top to bottom. `out` arrives
    /// transparent; leave pixels without image data as they are.
    fn read_region(&self, region: &SlideRegion, out: &mut [u8]) -> Result<(), String>;

    /// Decodes an associated image listed in [`SlideInfo::associated_images`]
    /// at the size listed there, as RGBA with straight alpha.
    fn read_associated_image(&self, name: &str) -> Result<Image, String> {
        Err(format!("no associated image '{name}'"))
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// An error returned by a host call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostError {
    pub kind: FfiErrorKind,
    pub message: String,
}

impl HostError {
    fn new(kind: FfiErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// The host does not know this method (it is older than the plugin).
    pub fn is_unsupported(&self) -> bool {
        self.kind == FfiErrorKind::Unsupported
    }
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for HostError {}

impl From<FfiError> for HostError {
    fn from(error: FfiError) -> Self {
        Self::new(error.kind(), error.message.into_string())
    }
}

impl From<HostError> for String {
    fn from(error: HostError) -> Self {
        error.message
    }
}

pub type HostResult<T> = Result<T, HostError>;

fn from_ffi<T>(result: FfiResult<T>) -> HostResult<T> {
    result.into_result().map_err(HostError::from)
}

// ---------------------------------------------------------------------------
// Images and frames
// ---------------------------------------------------------------------------

/// An RGBA8 image with straight alpha, row-major, without padding.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Image {
    /// The RGBA value at `(x, y)`.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = (y as usize * self.width as usize + x as usize) * 4;
        self.data.get(i..i + 4).map(|p| [p[0], p[1], p[2], p[3]])
    }

    fn into_pixels(self) -> abi::Pixels {
        abi::Pixels {
            width: self.width,
            height: self.height,
            data: self.data.into(),
        }
    }
}

impl From<abi::Pixels> for Image {
    fn from(pixels: abi::Pixels) -> Self {
        Self {
            width: pixels.width,
            height: pixels.height,
            data: pixels.data.into_vec(),
        }
    }
}

/// A rendered frame handed to [`Plugin::apply_filter`].
pub struct Frame<'a> {
    pixels: &'a mut [u8],
    width: u32,
    height: u32,
    stride: usize,
    view: FilterView,
}

impl<'a> Frame<'a> {
    /// Wraps a pixel buffer. `stride` is in bytes and at least `width·4`.
    /// Returns `None` when the buffer is too small for the dimensions.
    pub fn new(
        pixels: &'a mut [u8],
        width: u32,
        height: u32,
        stride: usize,
        view: FilterView,
    ) -> Option<Self> {
        let row = (width as usize).checked_mul(4)?;
        if stride < row {
            return None;
        }
        let needed = if height == 0 {
            0
        } else {
            stride.checked_mul(height as usize - 1)?.checked_add(row)?
        };
        (pixels.len() >= needed).then_some(Self {
            pixels,
            width,
            height,
            stride,
            view,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Bytes between the starts of consecutive rows.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The pane being rendered.
    pub fn view(&self) -> &FilterView {
        &self.view
    }

    /// Row `y` as `width·4` RGBA bytes.
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(y < self.height, "row {y} out of range");
        let start = y as usize * self.stride;
        &mut self.pixels[start..start + self.width as usize * 4]
    }

    /// Every row, top to bottom, each `width·4` RGBA bytes.
    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [u8]> + '_ {
        let row = self.width as usize * 4;
        let height = self.height as usize;
        self.pixels
            .chunks_mut(self.stride.max(1))
            .take(height)
            .map(move |chunk| &mut chunk[..row])
    }

    /// Calls `f` for every pixel with its coordinates.
    pub fn for_each_pixel(&mut self, mut f: impl FnMut(u32, u32, &mut [u8])) {
        let width = self.width;
        for (y, row) in self.rows_mut().enumerate() {
            let (pixels, _) = row.as_chunks_mut::<4>();
            for (x, px) in pixels.iter_mut().enumerate().take(width as usize) {
                f(x as u32, y as u32, px);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host handle
// ---------------------------------------------------------------------------

/// A cheap, cloneable, thread-safe handle to the host.
#[derive(Clone)]
pub struct Host {
    ffi: Arc<HostBox>,
}

impl Host {
    /// Wraps the host object passed to the plugin. Used by
    /// [`crate::export_plugin!`] and [`crate::testing`].
    pub fn from_ffi(ffi: HostBox) -> Self {
        Self { ffi: Arc::new(ffi) }
    }

    /// Invokes a host method with pre-encoded JSON.
    pub fn call_raw(&self, method: &str, args_json: &str) -> HostResult<String> {
        from_ffi(
            self.ffi
                .call(RStr::from_str(method), RStr::from_str(args_json)),
        )
        .map(RString::into_string)
    }

    /// Invokes a host method with typed arguments and result.
    pub fn call<A, R>(&self, method: &str, args: &A) -> HostResult<R>
    where
        A: Serialize + ?Sized,
        R: DeserializeOwned,
    {
        let args = serde_json::to_string(args)
            .map_err(|e| HostError::new(FfiErrorKind::InvalidArguments, e.to_string()))?;
        let reply = self.call_raw(method, &args)?;
        serde_json::from_str(&reply).map_err(|e| {
            HostError::new(
                FfiErrorKind::Failed,
                format!("unexpected reply to '{method}': {e}"),
            )
        })
    }

    fn command<A: Serialize + ?Sized>(&self, method: &str, args: &A) -> HostResult<()> {
        self.call::<A, Value>(method, args).map(|_| ())
    }

    // -- feedback ----------------------------------------------------------

    /// Writes `message` to eov's log as it is. Warnings and errors also reach
    /// the user: eov shows the same text, unchanged, as a notification.
    ///
    /// eov's log stays in English (it goes into bug reports), so write log
    /// messages in English, without [`isolate`](crate::i18n::isolate). To
    /// tell the user about a problem in their language, show the translated
    /// text with [`notify`](Self::notify) or
    /// [`notify_error`](Self::notify_error) and log the English one with
    /// [`info`](Self::info) or [`debug`](Self::debug), both written by the
    /// same code with
    /// [`i18n::shown_and_english`](crate::i18n::shown_and_english):
    /// `let (shown, english) = i18n::shown_and_english(|| tr!("{} failed",
    /// i18n::isolate(&name)));` ([`i18n::in_english`](crate::i18n::in_english)
    /// runs any code that makes texts in English). A plugin that hands
    /// translated text to [`warn`](Self::warn) or [`error`](Self::error) has
    /// it shown in the user's language, and accepts that eov's log gets that
    /// language too.
    pub fn log(&self, level: LogLevel, message: impl Into<String>) {
        let _ = self.command(
            methods::LOG,
            &LogArgs {
                level,
                message: message.into(),
            },
        );
    }

    pub fn debug(&self, message: impl Into<String>) {
        self.log(LogLevel::Debug, message);
    }

    pub fn info(&self, message: impl Into<String>) {
        self.log(LogLevel::Info, message);
    }

    /// Logs a warning and shows the same text to the user as a warning
    /// notification. eov's log is in English: for a text in the user's
    /// language, see [`log`](Self::log).
    pub fn warn(&self, message: impl Into<String>) {
        self.log(LogLevel::Warn, message);
    }

    /// Logs an error and shows the same text to the user as an error
    /// notification. eov's log is in English: for a text in the user's
    /// language, see [`log`](Self::log).
    pub fn error(&self, message: impl Into<String>) {
        self.log(LogLevel::Error, message);
    }

    /// Shows a toast notification.
    pub fn notify(&self, level: NotifyLevel, message: impl Into<String>) {
        let _ = self.command(
            methods::NOTIFY,
            &NotifyArgs {
                level,
                message: message.into(),
                detail: None,
            },
        );
    }

    /// Shows an error toast with an expandable detail text.
    pub fn notify_error(&self, message: impl Into<String>, detail: impl Into<String>) {
        let _ = self.command(
            methods::NOTIFY,
            &NotifyArgs {
                level: NotifyLevel::Error,
                message: message.into(),
                detail: Some(detail.into()),
            },
        );
    }

    /// Sets (or with `None` clears) the plugin's status-bar text.
    pub fn set_status(&self, text: Option<&str>) {
        let _ = self.command(
            methods::STATUS,
            &StatusArgs {
                text: text.map(str::to_string),
            },
        );
    }

    // -- state ---------------------------------------------------------------

    /// A recent, consistent copy of the host state.
    pub fn snapshot(&self) -> HostResult<HostSnapshot> {
        self.call(methods::SNAPSHOT, &Map::new())
    }

    pub fn file_info(&self, file_id: FileId) -> HostResult<FileInfo> {
        self.call(methods::FILE_INFO, &FileArgs { file_id })
    }

    /// Every vendor property of a slide (OpenSlide property map).
    pub fn file_properties(&self, file_id: FileId) -> HostResult<PropertyMap> {
        self.call(methods::FILE_PROPERTIES, &FileArgs { file_id })
    }

    /// Opens a slide in a new tab.
    pub fn open_file(&self, path: impl Into<String>) -> HostResult<()> {
        self.command(methods::FILE_OPEN, &OpenFileArgs { path: path.into() })
    }

    /// Reads slide pixels on the calling thread. For RGB slides they are
    /// the slide's own colours, as scanned: eov's colour adjustments and
    /// its colour management (ICC profile to sRGB, see
    /// [`DisplayAdjustments::color_managed`]) apply to the display only.
    ///
    /// Multichannel slides (see [`FileInfo::channel_names`]) come back as
    /// RGBA too: their visible channels composited with the user's current
    /// channel display, as the viewer shows them.
    pub fn read_region(&self, request: RegionRequest) -> HostResult<Image> {
        from_ffi(self.ffi.read_region(request)).map(Image::from)
    }

    /// Reads an associated image (`"label"`, `"macro"`, `"thumbnail"`, …).
    pub fn read_associated_image(&self, file_id: FileId, name: &str) -> HostResult<Image> {
        from_ffi(
            self.ffi
                .read_associated_image(file_id, RStr::from_str(name)),
        )
        .map(Image::from)
    }

    /// The host region of interest of a slide.
    pub fn roi(&self, file_id: FileId) -> HostResult<Option<Rect>> {
        self.call(methods::ROI_GET, &FileArgs { file_id })
    }

    /// The host measurements of a slide.
    pub fn measurements(&self, file_id: FileId) -> HostResult<Vec<MeasurementInfo>> {
        self.call(methods::MEASUREMENTS_LIST, &FileArgs { file_id })
    }

    // -- toolbar, HUD and tools ----------------------------------------------

    pub fn add_toolbar_button(&self, spec: ToolbarButtonSpec) -> HostResult<()> {
        self.command(methods::TOOLBAR_ADD, &spec)
    }

    pub fn update_toolbar_button(&self, update: ButtonUpdate) -> HostResult<()> {
        self.command(methods::TOOLBAR_UPDATE, &update)
    }

    /// Shorthand for toggling a toolbar button's pressed state.
    pub fn set_toolbar_button_active(&self, button_id: &str, active: bool) -> HostResult<()> {
        self.update_toolbar_button(ButtonUpdate {
            button_id: button_id.to_string(),
            active: Some(active),
            ..Default::default()
        })
    }

    pub fn remove_toolbar_button(&self, button_id: &str) -> HostResult<()> {
        self.command(
            methods::TOOLBAR_REMOVE,
            &IdArgs {
                id: button_id.into(),
            },
        )
    }

    pub fn add_hud_button(&self, spec: HudButtonSpec) -> HostResult<()> {
        self.command(methods::HUD_ADD, &spec)
    }

    pub fn update_hud_button(&self, update: ButtonUpdate) -> HostResult<()> {
        self.command(methods::HUD_UPDATE, &update)
    }

    pub fn remove_hud_button(&self, button_id: &str) -> HostResult<()> {
        self.command(
            methods::HUD_REMOVE,
            &IdArgs {
                id: button_id.into(),
            },
        )
    }

    /// Shows a short notice fixed to the corner of the panes showing
    /// `file_id` (every pane with a slide for `None`); it does not move with
    /// the view. `text: None` removes it.
    pub fn set_badge(
        &self,
        badge_id: &str,
        file_id: Option<FileId>,
        text: Option<&str>,
    ) -> HostResult<()> {
        self.command(
            methods::BADGE_SET,
            &BadgeArgs {
                badge_id: badge_id.to_string(),
                file_id,
                text: text.map(str::to_string),
            },
        )
    }

    /// Offers a command for headless batch runs (`eov batch run
    /// <plugin>:<command> SLIDES --out DIR`); slides then arrive as
    /// [`HostEvent::BatchRun`]. Call it in [`Plugin::create`].
    pub fn register_batch_command(&self, spec: BatchCommandSpec) -> HostResult<()> {
        self.command(methods::BATCH_REGISTER, &spec)
    }

    /// Reports how far the slide of batch run `run_id` is.
    pub fn batch_progress(
        &self,
        run_id: &str,
        fraction: Option<f32>,
        message: Option<&str>,
    ) -> HostResult<()> {
        self.command(
            methods::BATCH_PROGRESS,
            &BatchProgressArgs {
                run_id: run_id.to_string(),
                fraction,
                message: message.map(str::to_string),
            },
        )
    }

    /// Ends the slide of a batch run: the next slide comes once this is
    /// sent.
    pub fn finish_batch(&self, result: BatchFinishArgs) -> HostResult<()> {
        self.command(methods::BATCH_FINISH, &result)
    }

    /// Binds a key of its own (not tied to a button or tool); presses
    /// arrive as [`HostEvent::HotkeyPressed`]. A key eov uses (its shortcut
    /// table, which users can change, and Escape, Enter, Delete, Backspace,
    /// Tab and plain arrows) or another plugin's key is refused with
    /// [`HostEvent::RegistrationRefused`]; tool and button hotkeys are
    /// checked the same way.
    pub fn register_hotkey(&self, spec: HotkeySpec) -> HostResult<()> {
        self.command(methods::HOTKEY_REGISTER, &spec)
    }

    pub fn unregister_hotkey(&self, hotkey_id: &str) -> HostResult<()> {
        self.command(
            methods::HOTKEY_UNREGISTER,
            &IdArgs {
                id: hotkey_id.to_string(),
            },
        )
    }

    pub fn register_tool(&self, spec: ToolSpec) -> HostResult<()> {
        self.command(methods::TOOL_REGISTER, &spec)
    }

    pub fn unregister_tool(&self, tool_id: &str) -> HostResult<()> {
        self.command(methods::TOOL_UNREGISTER, &IdArgs { id: tool_id.into() })
    }

    pub fn activate_tool(&self, tool_id: &str) -> HostResult<()> {
        self.command(methods::TOOL_ACTIVATE, &IdArgs { id: tool_id.into() })
    }

    /// Returns to the Navigate tool if one of this plugin's tools is active.
    pub fn deactivate_tools(&self) -> HostResult<()> {
        self.command(methods::TOOL_DEACTIVATE, &Map::new())
    }

    // -- overlays --------------------------------------------------------------

    /// Creates (or replaces) an overlay layer.
    pub fn create_layer(&self, spec: LayerSpec) -> HostResult<Layer> {
        let layer_id = spec.layer_id.clone();
        self.command(methods::LAYER_CREATE, &spec)?;
        Ok(self.layer(layer_id))
    }

    /// A handle to an existing layer.
    pub fn layer(&self, layer_id: impl Into<String>) -> Layer {
        Layer {
            host: self.clone(),
            layer_id: layer_id.into(),
        }
    }

    // -- surfaces --------------------------------------------------------------

    /// Opens a Slint surface (sidebar, window or modal).
    pub fn open_surface(&self, spec: SurfaceSpec) -> HostResult<Surface> {
        let opened: SurfaceOpened = self.call(methods::SURFACE_OPEN, &spec)?;
        Ok(self.surface(opened.surface_id))
    }

    /// A handle to a surface opened earlier.
    pub fn surface(&self, surface_id: impl Into<String>) -> Surface {
        Surface {
            host: self.clone(),
            surface_id: surface_id.into(),
        }
    }

    // -- dialogs ---------------------------------------------------------------

    /// Asks a yes/no question; the answer arrives as
    /// [`HostEvent::ConfirmResult`] with the returned request id.
    pub fn confirm(&self, args: ConfirmArgs) -> HostResult<String> {
        self.call::<_, RequestId>(methods::DIALOG_CONFIRM, &args)
            .map(|r| r.request_id)
    }

    /// Shows an open-file dialog; the result arrives as
    /// [`HostEvent::DialogResult`] with the returned request id.
    pub fn open_file_dialog(&self, args: FileDialogArgs) -> HostResult<String> {
        self.call::<_, RequestId>(methods::DIALOG_OPEN_FILE, &args)
            .map(|r| r.request_id)
    }

    /// Shows a save-file dialog; the result arrives as
    /// [`HostEvent::DialogResult`] with the returned request id.
    pub fn save_file_dialog(&self, args: FileDialogArgs) -> HostResult<String> {
        self.call::<_, RequestId>(methods::DIALOG_SAVE_FILE, &args)
            .map(|r| r.request_id)
    }

    // -- models ----------------------------------------------------------------

    /// Makes a model available in eov's model store: answered at once when
    /// it is stored already, otherwise downloaded once the user agrees and
    /// verified against its SHA-256. The answer arrives as
    /// [`HostEvent::ModelFetched`] with the returned request id. Hosts
    /// without a model store answer [`HostError::is_unsupported`].
    pub fn fetch_model(&self, args: ModelFetchArgs) -> HostResult<String> {
        self.call::<_, RequestId>(methods::MODEL_FETCH, &args)
            .map(|r| r.request_id)
    }

    /// The stored model whose main file has this SHA-256, if any.
    pub fn resolve_model(&self, sha256: &str) -> HostResult<Option<ModelRecord>> {
        self.call(
            methods::MODEL_RESOLVE,
            &ModelResolveArgs {
                sha256: sha256.to_ascii_lowercase(),
            },
        )
    }

    /// Every model in eov's model store.
    pub fn models(&self) -> HostResult<Vec<ModelRecord>> {
        self.call(methods::MODEL_LIST, &Map::new())
    }

    // -- navigation --------------------------------------------------------------

    pub fn set_view(&self, args: ViewSetArgs) -> HostResult<()> {
        self.command(methods::VIEW_SET, &args)
    }

    /// Pans a pane (default: the focused one) by logical screen pixels.
    pub fn pan(&self, pane: Option<PaneId>, dx: f64, dy: f64) -> HostResult<()> {
        self.command(methods::VIEW_PAN, &ViewPanArgs { pane, dx, dy })
    }

    /// Zooms a pane by `factor` (> 1 zooms in) around its centre.
    pub fn zoom(&self, pane: Option<PaneId>, factor: f64) -> HostResult<()> {
        self.command(
            methods::VIEW_ZOOM,
            &ViewZoomArgs {
                pane,
                factor,
                anchor_x: None,
                anchor_y: None,
                animate: false,
            },
        )
    }

    pub fn fit(&self, pane: Option<PaneId>) -> HostResult<()> {
        self.command(methods::VIEW_FIT, &PaneArgs { pane })
    }

    /// Turns a pane's view (default: the focused one) about its centre:
    /// `rotation_deg` clockwise (any angle), then mirrored left to right if
    /// `flipped`. Level-0 coordinates are unaffected.
    pub fn set_view_orientation(
        &self,
        pane: Option<PaneId>,
        rotation_deg: f64,
        flipped: bool,
    ) -> HostResult<()> {
        self.command(
            methods::VIEW_ORIENTATION,
            &ViewOrientationArgs {
                pane,
                rotation_deg,
                flipped,
            },
        )
    }

    /// How a pane displays its slide (tone adjustments, normalisation), so
    /// overlays made from slide pixels can match it.
    pub fn display_adjustments(&self, pane: Option<PaneId>) -> HostResult<DisplayAdjustments> {
        self.call(methods::VIEW_DISPLAY, &PaneArgs { pane })
    }

    /// Frames a level-0 rectangle in a pane.
    pub fn frame(&self, pane: Option<PaneId>, rect: Rect) -> HostResult<()> {
        self.command(
            methods::VIEW_FRAME,
            &ViewFrameArgs {
                pane,
                rect,
                padding: None,
            },
        )
    }

    pub fn cycle_pane(&self, offset: i32) -> HostResult<()> {
        self.command(methods::APP_CYCLE_PANE, &OffsetArgs { offset })
    }

    pub fn cycle_tab(&self, offset: i32) -> HostResult<()> {
        self.command(methods::APP_CYCLE_TAB, &OffsetArgs { offset })
    }

    pub fn series(&self, action: SeriesAction) -> HostResult<()> {
        self.command(methods::APP_SERIES, &SeriesArgs { action })
    }

    /// Asks the host to redraw every pane (e.g. after a filter's output
    /// changed).
    pub fn request_render(&self) -> HostResult<()> {
        self.command(methods::RENDER_REQUEST, &Map::new())
    }

    // -- undo --------------------------------------------------------------------

    /// Records an undoable change. Ctrl/⌘+Z later sends
    /// [`HostEvent::Undo`] with `entry_id`.
    pub fn push_undo(
        &self,
        entry_id: impl Into<String>,
        file_id: Option<FileId>,
        label: impl Into<String>,
    ) -> HostResult<()> {
        self.command(
            methods::UNDO_PUSH,
            &UndoPushArgs {
                entry_id: entry_id.into(),
                file_id,
                label: label.into(),
            },
        )
    }

    /// Forgets the plugin's history entries (of one file, or all).
    pub fn clear_undo(&self, file_id: Option<FileId>) -> HostResult<()> {
        self.command(methods::UNDO_CLEAR, &UndoClearArgs { file_id })
    }

    // -- filters -----------------------------------------------------------------

    pub fn register_filter(&self, spec: FilterSpec) -> HostResult<()> {
        self.command(methods::FILTER_REGISTER, &spec)
    }

    pub fn set_filter_enabled(&self, filter_id: &str, enabled: bool) -> HostResult<()> {
        self.command(
            methods::FILTER_SET_ENABLED,
            &FilterEnableArgs {
                filter_id: filter_id.into(),
                enabled,
            },
        )
    }

    pub fn unregister_filter(&self, filter_id: &str) -> HostResult<()> {
        self.command(
            methods::FILTER_UNREGISTER,
            &IdArgs {
                id: filter_id.into(),
            },
        )
    }

    // -- settings ------------------------------------------------------------------

    /// Reads a persisted setting.
    pub fn setting<T: DeserializeOwned>(&self, key: &str) -> HostResult<Option<T>> {
        let value: Value = self.call(methods::SETTINGS_GET, &KeyArgs { key: key.into() })?;
        if value.is_null() {
            return Ok(None);
        }
        serde_json::from_value(value)
            .map(Some)
            .map_err(|e| HostError::new(FfiErrorKind::Failed, format!("setting '{key}': {e}")))
    }

    /// Persists a setting.
    pub fn set_setting<T: Serialize + ?Sized>(&self, key: &str, value: &T) -> HostResult<()> {
        let value = serde_json::to_value(value)
            .map_err(|e| HostError::new(FfiErrorKind::InvalidArguments, e.to_string()))?;
        self.command(
            methods::SETTINGS_SET,
            &SettingsSetArgs {
                key: key.into(),
                value,
            },
        )
    }

    pub fn remove_setting(&self, key: &str) -> HostResult<()> {
        self.command(
            methods::SETTINGS_SET,
            &SettingsSetArgs {
                key: key.into(),
                value: Value::Null,
            },
        )
    }

    // -- jobs ------------------------------------------------------------------------

    /// Shows a background job with progress in the host UI.
    pub fn start_job(&self, title: impl Into<String>, cancellable: bool) -> HostResult<Job> {
        let started: JobStarted = self.call(
            methods::JOB_START,
            &JobStartArgs {
                title: title.into(),
                cancellable,
            },
        )?;
        Ok(Job {
            host: self.clone(),
            job_id: started.job_id,
            finished: false,
        })
    }

    // -- miscellaneous -----------------------------------------------------------------

    pub fn copy_text(&self, text: impl Into<String>) -> HostResult<()> {
        self.command(methods::CLIPBOARD_SET_TEXT, &TextArgs { text: text.into() })
    }

    /// Asks for [`HostEvent::Posted`] with `token` to be delivered on the UI
    /// thread.
    pub fn post(&self, token: impl Into<String>) -> HostResult<()> {
        self.command(
            methods::POST,
            &PostArgs {
                token: token.into(),
            },
        )
    }

    /// Restricts the optional events delivered to this plugin (see
    /// [`crate::protocol::event_names`]).
    pub fn subscribe(&self, events: &[&str]) -> HostResult<()> {
        self.command(
            methods::EVENTS_SUBSCRIBE,
            &SubscribeArgs {
                events: events.iter().map(|e| e.to_string()).collect(),
            },
        )
    }
}

/// A handle to an overlay layer.
#[derive(Clone)]
pub struct Layer {
    host: Host,
    layer_id: String,
}

impl Layer {
    pub fn id(&self) -> &str {
        &self.layer_id
    }

    /// Adds or replaces objects.
    pub fn upsert(&self, objects: Vec<OverlayObject>) -> HostResult<()> {
        self.host.command(
            methods::LAYER_UPSERT,
            &UpsertObjectsArgs {
                layer_id: self.layer_id.clone(),
                objects,
                replace_all: false,
            },
        )
    }

    /// Atomically replaces every object of the layer.
    pub fn replace_all(&self, objects: Vec<OverlayObject>) -> HostResult<()> {
        self.host.command(
            methods::LAYER_UPSERT,
            &UpsertObjectsArgs {
                layer_id: self.layer_id.clone(),
                objects,
                replace_all: true,
            },
        )
    }

    pub fn remove(&self, object_ids: Vec<String>) -> HostResult<()> {
        self.host.command(
            methods::LAYER_REMOVE_OBJECTS,
            &RemoveObjectsArgs {
                layer_id: self.layer_id.clone(),
                object_ids,
            },
        )
    }

    pub fn clear(&self) -> HostResult<()> {
        self.host.command(
            methods::LAYER_CLEAR,
            &IdArgs {
                id: self.layer_id.clone(),
            },
        )
    }

    /// Changes layer properties; set `layer_id` is filled in.
    pub fn update(&self, mut update: LayerUpdate) -> HostResult<()> {
        update.layer_id = self.layer_id.clone();
        self.host.command(methods::LAYER_UPDATE, &update)
    }

    pub fn set_visible(&self, visible: bool) -> HostResult<()> {
        self.update(LayerUpdate {
            visible: Some(visible),
            ..Default::default()
        })
    }

    /// Replaces the host-side selection within this layer.
    pub fn select(&self, object_ids: Vec<String>) -> HostResult<()> {
        self.host.command(
            methods::LAYER_SELECT,
            &SelectArgs {
                layer_id: self.layer_id.clone(),
                object_ids,
            },
        )
    }

    /// Adds or replaces a raster object (heat map) stretched over `rect`.
    pub fn put_raster(
        &self,
        object_id: impl Into<String>,
        rect: Rect,
        image: Image,
        smooth: bool,
    ) -> HostResult<()> {
        from_ffi(self.host.ffi.put_raster(RasterUpload {
            layer_id: RString::from(self.layer_id.as_str()),
            object_id: RString::from(object_id.into()),
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            pixels: image.into_pixels(),
            smooth,
        }))
    }

    /// Draws these objects with the layer's hover style (an empty list
    /// clears it), e.g. while a row of the plugin's own list is hovered.
    pub fn highlight(&self, object_ids: Vec<String>) -> HostResult<()> {
        self.host.command(
            methods::LAYER_HIGHLIGHT,
            &SelectArgs {
                layer_id: self.layer_id.clone(),
                object_ids,
            },
        )
    }

    /// Hides the layer below `min_zoom` screen pixels per level-0 pixel
    /// (`None`: every zoom) without recreating it.
    pub fn set_min_zoom(&self, min_zoom: Option<f64>) -> HostResult<()> {
        self.host.command(
            methods::LAYER_SET_MIN_ZOOM,
            &MinZoomArgs {
                layer_id: self.layer_id.clone(),
                min_zoom,
            },
        )
    }

    /// Shows the layer only in `panes` (`None`: in every pane). Unlike
    /// [`set_visible_in_pane`](Self::set_visible_in_pane) this also covers
    /// panes opened later; call it right after creating the layer.
    pub fn set_panes(&self, panes: Option<Vec<PaneId>>) -> HostResult<()> {
        self.host.command(
            methods::LAYER_SET_PANES,
            &PaneFilterArgs {
                layer_id: self.layer_id.clone(),
                panes,
            },
        )
    }

    /// Shows or hides the layer in one pane only; the layer's own
    /// visibility still applies.
    pub fn set_visible_in_pane(&self, pane: PaneId, visible: bool) -> HostResult<()> {
        self.host.command(
            methods::LAYER_SET_PANE_VISIBILITY,
            &PaneVisibilityArgs {
                layer_id: self.layer_id.clone(),
                pane,
                visible,
            },
        )
    }

    /// Deletes the layer and its objects.
    pub fn remove_layer(self) -> HostResult<()> {
        self.host.command(
            methods::LAYER_REMOVE,
            &IdArgs {
                id: self.layer_id.clone(),
            },
        )
    }
}

/// A handle to an open surface.
#[derive(Clone)]
pub struct Surface {
    host: Host,
    surface_id: String,
}

impl Surface {
    pub fn id(&self) -> &str {
        &self.surface_id
    }

    /// Sets one property (by its Slint name).
    pub fn set(&self, name: &str, value: impl Serialize) -> HostResult<()> {
        let value = serde_json::to_value(value)
            .map_err(|e| HostError::new(FfiErrorKind::InvalidArguments, e.to_string()))?;
        let mut properties = Map::new();
        properties.insert(name.to_string(), value);
        self.set_many(properties)
    }

    /// Sets several properties. A value the surface already shows is left
    /// alone (so its bindings stay); setting a field back to a value the
    /// user edited away from does apply.
    pub fn set_many(&self, properties: Map<String, Value>) -> HostResult<()> {
        self.host.command(
            methods::SURFACE_SET,
            &SurfaceProperties {
                surface_id: self.surface_id.clone(),
                properties,
            },
        )
    }

    pub fn is_open(&self) -> bool {
        self.host
            .call::<_, bool>(
                methods::SURFACE_IS_OPEN,
                &IdArgs {
                    id: self.surface_id.clone(),
                },
            )
            .unwrap_or(false)
    }

    pub fn close(&self) -> HostResult<()> {
        self.host.command(
            methods::SURFACE_CLOSE,
            &IdArgs {
                id: self.surface_id.clone(),
            },
        )
    }
}

/// A background job shown in the host UI. Dropping an unfinished job marks
/// it finished.
pub struct Job {
    host: Host,
    job_id: String,
    finished: bool,
}

impl Job {
    pub fn id(&self) -> &str {
        &self.job_id
    }

    /// Reports progress (`None` = indeterminate).
    pub fn update(&self, fraction: Option<f32>, message: Option<&str>) {
        let _ = self.host.command(
            methods::JOB_UPDATE,
            &JobUpdateArgs {
                job_id: self.job_id.clone(),
                fraction: fraction.map(|f| f.clamp(0.0, 1.0)),
                message: message.map(str::to_string),
            },
        );
    }

    /// Marks the job done, or failed with `error`.
    pub fn finish(mut self, error: Option<String>) {
        self.finish_inner(error);
    }

    fn finish_inner(&mut self, error: Option<String>) {
        if self.finished {
            return;
        }
        self.finished = true;
        let _ = self.host.command(
            methods::JOB_FINISH,
            &JobFinishArgs {
                job_id: self.job_id.clone(),
                error,
            },
        );
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        self.finish_inner(None);
    }
}

// ---------------------------------------------------------------------------
// FFI adapter (used by `export_plugin!`)
// ---------------------------------------------------------------------------

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Runs `f`, turning panics into [`FfiErrorKind::Panicked`] errors.
fn guarded<T>(f: impl FnOnce() -> Result<T, FfiError>) -> FfiResult<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => ROk(value),
        Ok(Err(error)) => RErr(error),
        Err(payload) => RErr(FfiError::new(
            FfiErrorKind::Panicked,
            format!("plugin panicked: {}", panic_message(payload.as_ref())),
        )),
    }
}

#[doc(hidden)]
pub struct Adapter<P: Plugin> {
    /// Shared with the slide source ([`slide_source`]).
    plugin: Arc<P>,
    /// The folder and domain of the plugin's catalogs (`[i18n]`).
    catalogs: Option<(std::path::PathBuf, String)>,
    /// For warnings about catalogs that cannot be read.
    host: Option<Host>,
}

/// The folder and domain of the catalogs the manifest in `plugin_root`
/// declares, if any.
fn plugin_catalogs(plugin_root: &str) -> Result<Option<(std::path::PathBuf, String)>, String> {
    let root = std::path::Path::new(plugin_root);
    let path = root.join(crate::manifest::MANIFEST_FILENAME);
    if !path.is_file() {
        return Ok(None);
    }
    let manifest = crate::PluginManifest::from_file(&path).map_err(|e| e.to_string())?;
    Ok(manifest.catalogs(root))
}

/// Makes `tag` the language of [`crate::i18n`] in this library and loads
/// the plugin's catalog for it. English (no catalog) when the language is
/// unknown or the plugin has no catalog for it; the error says why a
/// catalog could not be read.
fn switch_language(
    catalogs: Option<&(std::path::PathBuf, String)>,
    tag: Option<&str>,
) -> Result<(), String> {
    use crate::i18n;
    let language = tag.and_then(i18n::by_tag).unwrap_or(i18n::source());
    let catalog = match catalogs {
        Some((dir, domain)) => Catalog::load(dir, domain, language).map_err(|e| e.message),
        None => Ok(None),
    };
    match catalog {
        Ok(catalog) => {
            i18n::set_catalog(language, catalog);
            match tag {
                Some(tag) if i18n::by_tag(tag).is_none() => Err(format!(
                    "eov's language '{tag}' is unknown to this plugin's SDK; texts are in English"
                )),
                _ => Ok(()),
            }
        }
        Err(error) => {
            i18n::set_catalog(language, None);
            Err(format!("translations not loaded: {error}"))
        }
    }
}

impl<P: Plugin> Adapter<P> {
    fn dispatch(&self, event_json: &str) -> Result<String, FfiError> {
        let event: HostEvent = serde_json::from_str(event_json).map_err(|e| {
            FfiError::new(FfiErrorKind::InvalidArguments, format!("bad event: {e}"))
        })?;
        if let HostEvent::LanguageChanged { language } = &event
            && let Err(error) = switch_language(self.catalogs.as_ref(), Some(language))
            && let Some(host) = &self.host
        {
            // The plugin still gets the event, and shows English texts. Logged
            // only: eov shows warnings to users, and a bad catalog is for
            // the plugin's developers to fix.
            host.info(error);
        }
        match event {
            HostEvent::ContextMenuRequested(context) => {
                let items = self.plugin.context_menu(&context);
                to_json(&ContextMenuResponse { items })
            }
            HostEvent::UiCallback {
                surface_id,
                name,
                args,
            } => {
                let value = self
                    .plugin
                    .ui_callback(&surface_id, &name, &args)
                    .map_err(FfiError::failed)?;
                to_json(&value)
            }
            HostEvent::Unknown => Err(FfiError::new(
                FfiErrorKind::Unsupported,
                "event not supported by this plugin version",
            )),
            other => self
                .plugin
                .on_event(other)
                .map(|()| "null".to_string())
                .map_err(FfiError::failed),
        }
    }
}

fn to_json<T: Serialize>(value: &T) -> Result<String, FfiError> {
    serde_json::to_string(value).map_err(|e| FfiError::failed(e.to_string()))
}

impl<P: Plugin> PluginFfi for Adapter<P> {
    fn handle_event(&self, event_json: RStr<'_>) -> FfiResult<RString> {
        guarded(|| self.dispatch(event_json.as_str()).map(RString::from))
    }

    fn apply_filter(&self, filter_id: RStr<'_>, frame: FilterFrame<'_>) -> FfiResult<()> {
        guarded(|| {
            let FilterFrame {
                pixels,
                width,
                height,
                stride,
                view,
            } = frame;
            let mut frame = Frame::new(
                pixels.into_mut_slice(),
                width,
                height,
                stride as usize,
                view,
            )
            .ok_or_else(|| {
                FfiError::new(
                    FfiErrorKind::InvalidArguments,
                    format!("frame buffer too small for {width}x{height} (stride {stride})"),
                )
            })?;
            self.plugin
                .apply_filter(filter_id.as_str(), &mut frame)
                .map_err(FfiError::failed)
        })
    }

    fn shutdown(&self) {
        let _ = catch_unwind(AssertUnwindSafe(|| self.plugin.shutdown()));
    }
}

/// Wraps a plugin value in the FFI instance type. It can be downcast, so
/// [`slide_source`] finds the plugin again (only this library does that:
/// the host never sees the type).
#[doc(hidden)]
pub fn into_plugin_box<P: Plugin>(plugin: P) -> PluginBox {
    into_plugin_box_with_catalogs(plugin, None, None)
}

fn into_plugin_box_with_catalogs<P: Plugin>(
    plugin: P,
    catalogs: Option<(std::path::PathBuf, String)>,
    host: Option<Host>,
) -> PluginBox {
    PluginFfi_TO::from_value(
        Adapter {
            plugin: Arc::new(plugin),
            catalogs,
            host,
        },
        TD_CanDowncast,
    )
}

/// The slide source of an instance: its
/// [`Plugin::open_slide`].
extern "C" fn slide_source<P: Plugin>(instance: &PluginBox) -> FfiResult<SlideSourceBox> {
    slide_source_of::<P>(instance)
}

/// [`slide_source`] for the test harness.
#[doc(hidden)]
pub fn slide_source_of<P: Plugin>(instance: &PluginBox) -> FfiResult<SlideSourceBox> {
    guarded(|| {
        let adapter = instance
            .obj
            .downcast_as::<Adapter<P>>()
            .map_err(|_| FfiError::failed("the instance was not created by this plugin library"))?;
        Ok(SlideSourceFfi_TO::from_value(
            SourceAdapter {
                plugin: Arc::clone(&adapter.plugin),
            },
            TD_Opaque,
        ))
    })
}

struct SourceAdapter<P: Plugin> {
    plugin: Arc<P>,
}

impl<P: Plugin> SlideSourceFfi for SourceAdapter<P> {
    fn open(&self, request_json: RStr<'_>) -> FfiResult<SlideBox> {
        guarded(|| {
            let request: SlideOpenRequest =
                serde_json::from_str(request_json.as_str()).map_err(|e| {
                    FfiError::new(FfiErrorKind::InvalidArguments, format!("bad request: {e}"))
                })?;
            let reader = self.plugin.open_slide(&request).map_err(FfiError::failed)?;
            Ok(SlideFfi_TO::from_value(SlideAdapter { reader }, TD_Opaque))
        })
    }
}

struct SlideAdapter {
    reader: Box<dyn SlideReader>,
}

impl SlideFfi for SlideAdapter {
    fn info(&self) -> FfiResult<RString> {
        guarded(|| to_json(&self.reader.info()).map(RString::from))
    }

    fn read_region(&self, region: SlideRegion, out: RSliceMut<'_, u8>) -> FfiResult<()> {
        guarded(|| {
            let out = out.into_mut_slice();
            let expected = region.width as usize * region.height as usize * 4;
            if out.len() != expected {
                return Err(FfiError::new(
                    FfiErrorKind::InvalidArguments,
                    format!(
                        "the buffer holds {} bytes; {}×{} RGBA needs {expected}",
                        out.len(),
                        region.width,
                        region.height
                    ),
                ));
            }
            self.reader
                .read_region(&region, out)
                .map_err(FfiError::failed)
        })
    }

    fn read_associated_image(&self, name: RStr<'_>) -> FfiResult<Pixels> {
        guarded(|| {
            let image = self
                .reader
                .read_associated_image(name.as_str())
                .map_err(FfiError::failed)?;
            Ok(Pixels {
                width: image.width,
                height: image.height,
                data: image.data.into(),
            })
        })
    }

    fn call(&self, method: RStr<'_>, _args_json: RStr<'_>) -> FfiResult<RString> {
        RErr(FfiError::new(
            FfiErrorKind::Unsupported,
            format!("the slide has no method '{}'", method.as_str()),
        ))
    }
}

/// Creates a plugin through the same code path the host uses.
#[doc(hidden)]
pub fn create_plugin<P: Plugin>(host: HostBox, init_json: &str) -> Result<PluginBox, FfiError> {
    guarded(|| {
        let init: PluginInit = serde_json::from_str(init_json)
            .map_err(|e| FfiError::new(FfiErrorKind::InvalidArguments, format!("bad init: {e}")))?;
        let host = Host::from_ffi(host);
        // The plugin's texts are in the interface's language from the start.
        // Problems with the plugin's catalogs are logged, not shown: the
        // plugin then runs in English, and its developers fix the catalog.
        let catalogs = plugin_catalogs(&init.plugin_root).unwrap_or_else(|error| {
            host.info(format!("translations not loaded: {error}"));
            None
        });
        if let Err(error) = switch_language(catalogs.as_ref(), init.language.as_deref()) {
            host.info(error);
        }
        let plugin = P::create(host.clone(), init).map_err(FfiError::failed)?;
        Ok(into_plugin_box_with_catalogs(plugin, catalogs, Some(host)))
    })
    .into_result()
}

extern "C" fn create_instance<P: Plugin>(
    host: HostBox,
    init_json: RStr<'_>,
) -> FfiResult<PluginBox> {
    create_plugin::<P>(host, init_json.as_str()).into()
}

/// The root module of a plugin library (used by [`crate::export_plugin!`]).
#[doc(hidden)]
pub fn root_module<P: Plugin>() -> PluginModuleRef {
    PluginModule {
        abi_version: abi::ABI_VERSION,
        api_version: RStr::from_str(abi::API_VERSION),
        create: create_instance::<P>,
        slide_source: slide_source::<P>,
    }
    .leak_into_prefix()
}

/// Exports a [`Plugin`] implementation as an eov plugin library.
///
/// Use it once, at the crate root of a `cdylib` crate.
#[macro_export]
macro_rules! export_plugin {
    ($plugin:ty) => {
        #[doc(hidden)]
        #[unsafe(no_mangle)]
        #[allow(non_upper_case_globals)]
        pub static _1as_0lib_1header_0root_bmodule_bloader: $crate::__private::LibHeader = {
            extern "C" fn __eov_erased_root_module() -> $crate::__private::RootModuleResult {
                fn __eov_root_module() -> $crate::abi::PluginModuleRef {
                    $crate::sdk::root_module::<$plugin>()
                }
                $crate::__private::call_root_module_loader(__eov_root_module)
            }
            unsafe {
                $crate::__private::LibHeader::from_constructor::<$crate::abi::PluginModuleRef>(
                    __eov_erased_root_module,
                    $crate::__private::CheckTypeLayout::Yes,
                )
            }
        };
    };
}

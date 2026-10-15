//! Plugin UI surfaces (sidebar, modal, window) and confirmation dialogs.
//!
//! - Compiled definitions are cached by path, component and modification
//!   time, so reopening a surface does not recompile it.
//! - Each surface has an id and an owner; only one sidebar and one modal are
//!   shown at a time (others replace the sidebar, modals queue).
//! - Properties are pushed by the plugin. A value the surface already shows
//!   is skipped (compared with the live value, so a field the user edited
//!   can be reset), and array properties update their model row by row so
//!   Slint keeps the existing rows (PLG-15).
//! - Callbacks carry their surface id and can return values.

use super::UiHost;
use crate::plugin_runtime::EventTarget;
use crate::plugin_runtime::shared::SurfaceCommand;
use eov_plugin_api::protocol::{ConfirmArgs, HostEvent, NotifyLevel, SurfaceKind, SurfaceSpec};
use i_slint_compiler::langtype::Type;
use serde_json::{Map, Value as Json};
use slint::{ComponentFactory, ComponentHandle, Model, VecModel};
use slint_interpreter::{ComponentDefinition, ComponentInstance, Value};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::SystemTime;

/// The Slint version plugin surfaces are compiled with.
pub(crate) const SLINT_VERSION: &str = "1.17.1";

const DEFAULT_SIDEBAR_WIDTH: f32 = 300.0;

/// What a surface's factory and callbacks share with the manager.
struct SurfaceData {
    surface_id: String,
    plugin: Arc<str>,
    properties: Map<String, Json>,
    types: HashMap<String, Type>,
    returns: HashMap<String, Type>,
    instance: Option<slint::Weak<ComponentInstance>>,
    /// The last value of each array property, to skip unchanged arrays
    /// without converting every row.
    applied_arrays: HashMap<String, Json>,
    models: HashMap<String, Rc<VecModel<Value>>>,
}

impl SurfaceData {
    fn new(surface_id: &str, plugin: &Arc<str>, definition: &ComponentDefinition) -> Self {
        let mut types = HashMap::new();
        let mut returns = HashMap::new();
        for (name, (ty, _visibility)) in definition.properties_and_callbacks() {
            match &ty {
                Type::Callback(function) => {
                    returns.insert(name, function.return_type.clone());
                }
                ty if ty.is_property_type() => {
                    types.insert(name, ty.clone());
                }
                _ => {}
            }
        }
        Self {
            surface_id: surface_id.to_string(),
            plugin: Arc::clone(plugin),
            properties: Map::new(),
            types,
            returns,
            instance: None,
            applied_arrays: HashMap::new(),
            models: HashMap::new(),
        }
    }

    fn live_instance(&self) -> Option<ComponentInstance> {
        self.instance.as_ref().and_then(|weak| weak.upgrade())
    }

    /// Applies one property; problems are reported and skipped.
    fn apply_one(
        &mut self,
        instance: &ComponentInstance,
        name: &str,
        json: &Json,
    ) -> Result<(), String> {
        let Some(ty) = self.types.get(name).cloned() else {
            return Err(format!("the component has no public property '{name}'"));
        };
        if let (Type::Array(inner), Json::Array(rows)) = (&ty, json) {
            if self.models.contains_key(name) && self.applied_arrays.get(name) == Some(json) {
                return Ok(());
            }
            let values = rows
                .iter()
                .map(|row| slint_interpreter::json::value_from_json(inner, row))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("property '{name}': {e}"))?;
            match self.models.get(name) {
                Some(model) => {
                    // Update rows in place so Slint keeps row elements.
                    let common = model.row_count().min(values.len());
                    for (i, value) in values.iter().enumerate().take(common) {
                        if model.row_data(i).as_ref() != Some(value) {
                            model.set_row_data(i, value.clone());
                        }
                    }
                    while model.row_count() > values.len() {
                        model.remove(model.row_count() - 1);
                    }
                    for value in values.into_iter().skip(common) {
                        model.push(value);
                    }
                }
                None => {
                    let model = Rc::new(VecModel::from(values));
                    instance
                        .set_property(name, Value::Model(slint::ModelRc::from(Rc::clone(&model))))
                        .map_err(|e| format!("property '{name}': {e}"))?;
                    self.models.insert(name.to_string(), model);
                }
            }
            self.applied_arrays.insert(name.to_string(), json.clone());
        } else {
            let value = slint_interpreter::json::value_from_json(&ty, json)
                .map_err(|e| format!("property '{name}': {e}"))?;
            // Skip what the surface shows already: setting a property breaks
            // its bindings, and the user may have edited it since.
            if instance.get_property(name).ok().as_ref() != Some(&value) {
                instance
                    .set_property(name, value)
                    .map_err(|e| format!("property '{name}': {e}"))?;
            }
        }
        Ok(())
    }

    /// Applies `properties` to the live instance; returns the problems.
    fn apply(&mut self, properties: &Map<String, Json>) -> Vec<String> {
        let Some(instance) = self.live_instance() else {
            return Vec::new();
        };
        properties
            .iter()
            .filter_map(|(name, value)| self.apply_one(&instance, name, value).err())
            .collect()
    }

    /// Takes over a freshly created instance.
    fn attach(
        data: &Rc<RefCell<Self>>,
        instance: &ComponentInstance,
        definition: &ComponentDefinition,
    ) {
        {
            let mut d = data.borrow_mut();
            d.instance = Some(instance.as_weak());
            d.applied_arrays.clear();
            d.models.clear();
            let properties = d.properties.clone();
            let problems = d.apply(&properties);
            for problem in problems {
                tracing::warn!("Plugin '{}' surface {}: {problem}", d.plugin, d.surface_id);
            }
        }
        for name in definition.callbacks() {
            let data = Rc::clone(data);
            let callback_name = name.clone();
            let result = instance.set_callback(&name, move |args| {
                let (surface_id, plugin, return_type) = {
                    let d = data.borrow();
                    (
                        d.surface_id.clone(),
                        Arc::clone(&d.plugin),
                        d.returns.get(&callback_name).cloned(),
                    )
                };
                let args: Vec<Json> = args
                    .iter()
                    .map(|value| {
                        slint_interpreter::json::value_to_json(value).unwrap_or(Json::Null)
                    })
                    .collect();
                let reply = super::with_host(|host| {
                    host.request(
                        &plugin,
                        &HostEvent::UiCallback {
                            surface_id,
                            name: callback_name.clone(),
                            args,
                        },
                    )
                })
                .flatten();
                match (return_type, reply) {
                    (Some(ty), Some(reply)) if ty != Type::Void => {
                        slint_interpreter::json::value_from_json(&ty, &reply).unwrap_or(Value::Void)
                    }
                    _ => Value::Void,
                }
            });
            if let Err(error) = result {
                tracing::debug!("Cannot wire callback '{name}': {error}");
            }
        }
    }
}

struct OpenSurface {
    plugin: Arc<str>,
    spec: SurfaceSpec,
    definition: ComponentDefinition,
    data: Rc<RefCell<SurfaceData>>,
    /// Separate windows are owned here; embedded surfaces are owned by their
    /// container.
    window: Option<ComponentInstance>,
}

/// Who asked a confirmation, and so who gets the answer.
enum ConfirmOwner {
    Plugin {
        plugin: Arc<str>,
        request_id: String,
    },
    /// eov itself; called with the answer once the dialog has closed.
    Host(Box<dyn FnOnce(bool)>),
}

struct PendingConfirm {
    owner: ConfirmOwner,
    args: ConfirmArgs,
}

#[derive(Default)]
pub(super) struct Surfaces {
    compiled: HashMap<(PathBuf, String), (Option<SystemTime>, ComponentDefinition)>,
    open: HashMap<String, OpenSurface>,
    sidebar: Option<String>,
    modal: Option<String>,
    modal_queue: VecDeque<String>,
    confirm: Option<PendingConfirm>,
    confirm_queue: VecDeque<PendingConfirm>,
}

fn compile(
    path: &Path,
    component: &str,
    plugin_root: Option<&Path>,
) -> Result<ComponentDefinition, String> {
    let mut compiler = slint_interpreter::Compiler::default();
    if let Some(root) = plugin_root {
        compiler.set_include_paths(vec![root.to_path_buf()]);
    }
    let result = spin_on(compiler.build_from_path(path));
    let errors: Vec<String> = result
        .diagnostics()
        .filter(|d| d.level() == slint_interpreter::DiagnosticLevel::Error)
        .map(|d| d.to_string())
        .collect();
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    result.component(component).ok_or_else(|| {
        format!(
            "component '{component}' is not exported by {}",
            path.display()
        )
    })
}

fn spin_on<T>(future: impl std::future::Future<Output = T>) -> T {
    use std::task::{Context, Poll, Waker};
    let mut cx = Context::from_waker(Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

impl Surfaces {
    /// Whether one of the plugin windows has keyboard focus.
    pub(super) fn window_focused(&self) -> bool {
        use slint::winit_030::WinitWindowAccessor;
        self.open
            .values()
            .filter_map(|surface| surface.window.as_ref())
            .any(|window| {
                window
                    .window()
                    .with_winit_window(|window| window.has_focus())
                    .unwrap_or(false)
            })
    }

    fn definition(
        &mut self,
        path: &Path,
        component: &str,
        root: Option<&Path>,
    ) -> Result<ComponentDefinition, String> {
        let mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        let key = (path.to_path_buf(), component.to_string());
        if let Some((cached_mtime, definition)) = self.compiled.get(&key)
            && *cached_mtime == mtime
        {
            return Ok(definition.clone());
        }
        let definition = compile(path, component, root)?;
        self.compiled.insert(key, (mtime, definition.clone()));
        Ok(definition)
    }

    pub(super) fn sidebar_owner(&self) -> Option<String> {
        let id = self.sidebar.as_ref()?;
        self.open.get(id).map(|s| s.plugin.to_string())
    }

    pub(super) fn apply(&mut self, host: &UiHost, plugin: &Arc<str>, command: SurfaceCommand) {
        match command {
            SurfaceCommand::Open { surface_id, spec } => {
                self.open_surface(host, plugin, surface_id, spec)
            }
            SurfaceCommand::Set {
                surface_id,
                properties,
            } => {
                let Some(surface) = self.open.get(&surface_id).filter(|s| s.plugin == *plugin)
                else {
                    return;
                };
                let problems = {
                    let mut data = surface.data.borrow_mut();
                    let problems = data.apply(&properties);
                    // Kept for instances created later (a sidebar shown
                    // again).
                    data.properties.extend(properties);
                    problems
                };
                for problem in problems.into_iter().take(3) {
                    host.notify(
                        NotifyLevel::Warning,
                        &problem,
                        None,
                        Some(&host.plugin_name(plugin)),
                    );
                }
            }
            SurfaceCommand::Close { surface_id } => {
                if self
                    .open
                    .get(&surface_id)
                    .is_some_and(|s| s.plugin == *plugin)
                {
                    self.close(host, &surface_id, false);
                }
            }
        }
    }

    fn open_surface(
        &mut self,
        host: &UiHost,
        plugin: &Arc<str>,
        surface_id: String,
        spec: SurfaceSpec,
    ) {
        let path = PathBuf::from(&spec.ui_path);
        let root = host
            .runtime
            .borrow()
            .slot(plugin)
            .map(|slot| slot.info.root.clone());
        let definition = match self.definition(&path, &spec.component, root.as_deref()) {
            Ok(definition) => definition,
            Err(error) => {
                host.shared.unregister_surface(&surface_id);
                host.notify(
                    NotifyLevel::Error,
                    &format!(
                        "Cannot show {}",
                        path.file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default()
                    ),
                    Some(&error),
                    Some(&host.plugin_name(plugin)),
                );
                host.post(
                    EventTarget::Plugin(Arc::clone(plugin)),
                    HostEvent::SurfaceClosed { surface_id },
                );
                return;
            }
        };
        let data = Rc::new(RefCell::new(SurfaceData::new(
            &surface_id,
            plugin,
            &definition,
        )));
        data.borrow_mut().properties = spec.properties.clone();
        let surface = OpenSurface {
            plugin: Arc::clone(plugin),
            spec: spec.clone(),
            definition: definition.clone(),
            data: Rc::clone(&data),
            window: None,
        };
        match spec.kind {
            SurfaceKind::Sidebar => {
                if let Some(previous) = self.sidebar.take() {
                    self.close(host, &previous, true);
                }
                self.open.insert(surface_id.clone(), surface);
                self.sidebar = Some(surface_id.clone());
                if let Some(ui) = host.ui() {
                    let width = spec
                        .width
                        .unwrap_or(DEFAULT_SIDEBAR_WIDTH)
                        .clamp(180.0, 640.0);
                    ui.set_plugin_sidebar_width(width);
                    ui.set_plugin_sidebar_captures_keyboard(spec.capture_keyboard);
                    ui.set_plugin_sidebar_factory(embedded_factory(definition, data));
                    ui.set_show_plugin_sidebar(true);
                }
            }
            SurfaceKind::Modal => {
                self.open.insert(surface_id.clone(), surface);
                if self.modal.is_some() {
                    self.modal_queue.push_back(surface_id);
                } else {
                    self.show_modal(host, &surface_id);
                }
            }
            SurfaceKind::Window => {
                let instance = match definition.create() {
                    Ok(instance) => instance,
                    Err(error) => {
                        host.shared.unregister_surface(&surface_id);
                        host.notify(
                            NotifyLevel::Error,
                            "Cannot open a plugin window",
                            Some(&error.to_string()),
                            Some(&host.plugin_name(plugin)),
                        );
                        return;
                    }
                };
                SurfaceData::attach(&data, &instance, &definition);
                if let Some(title) = &spec.title {
                    let _ = instance.set_property("title", Value::String(title.clone().into()));
                }
                if let (Some(w), Some(h)) = (spec.width, spec.height) {
                    // Never larger than the main window (PLG-8's forced
                    // width ran off small screens).
                    let (max_w, max_h) = host
                        .ui()
                        .map(|ui| {
                            let size = ui.window().size().to_logical(ui.window().scale_factor());
                            (size.width.max(640.0), size.height.max(480.0))
                        })
                        .unwrap_or((1280.0, 800.0));
                    instance.window().set_size(slint::LogicalSize::new(
                        w.clamp(200.0, max_w),
                        h.clamp(150.0, max_h),
                    ));
                }
                let close_id = surface_id.clone();
                instance.window().on_close_requested(move || {
                    let id = close_id.clone();
                    // Close outside the window's own event handler.
                    slint::Timer::single_shot(std::time::Duration::ZERO, move || {
                        super::with_host(|host| {
                            let mut surfaces = host.surfaces.borrow_mut();
                            surfaces.close(host, &id, true);
                        });
                    });
                    slint::CloseRequestResponse::HideWindow
                });
                {
                    use slint::winit_030::{EventResult, WinitWindowAccessor, winit};
                    instance.window().on_winit_window_event(|_, event| {
                        if let winit::event::WindowEvent::Focused(_) = event {
                            super::window_focus_changed();
                        }
                        EventResult::Propagate
                    });
                }
                if let Err(error) = instance.show() {
                    tracing::error!("Cannot show plugin window: {error}");
                }
                let mut surface = surface;
                surface.window = Some(instance);
                self.open.insert(surface_id, surface);
            }
        }
    }

    fn show_modal(&mut self, host: &UiHost, surface_id: &str) {
        let Some(surface) = self.open.get(surface_id) else {
            return;
        };
        self.modal = Some(surface_id.to_string());
        if let Some(ui) = host.ui() {
            ui.set_plugin_modal_width(surface.spec.width.unwrap_or(480.0).clamp(200.0, 1600.0));
            ui.set_plugin_modal_height(surface.spec.height.unwrap_or(320.0).clamp(120.0, 1200.0));
            ui.set_plugin_modal_captures_keyboard(surface.spec.capture_keyboard);
            ui.set_plugin_modal_factory(embedded_factory(
                surface.definition.clone(),
                Rc::clone(&surface.data),
            ));
            ui.set_show_plugin_modal(true);
        }
    }

    /// Closes a surface; `notify` sends `SurfaceClosed` to its owner (when
    /// the user or another plugin closed it).
    pub(super) fn close(&mut self, host: &UiHost, surface_id: &str, notify: bool) {
        let Some(surface) = self.open.remove(surface_id) else {
            return;
        };
        host.shared.unregister_surface(surface_id);
        if self.sidebar.as_deref() == Some(surface_id) {
            self.sidebar = None;
            if let Some(ui) = host.ui() {
                ui.set_show_plugin_sidebar(false);
                ui.set_plugin_sidebar_factory(ComponentFactory::default());
            }
        }
        if self.modal.as_deref() == Some(surface_id) {
            self.modal = None;
            if let Some(ui) = host.ui() {
                ui.set_show_plugin_modal(false);
                ui.set_plugin_modal_factory(ComponentFactory::default());
                ui.invoke_focus_keyboard();
            }
            if let Some(next) = self.modal_queue.pop_front() {
                self.show_modal(host, &next);
            }
        }
        self.modal_queue.retain(|id| id != surface_id);
        if let Some(window) = surface.window {
            let _ = window.hide();
        }
        if notify {
            host.post(
                EventTarget::Plugin(Arc::clone(&surface.plugin)),
                HostEvent::SurfaceClosed {
                    surface_id: surface_id.to_string(),
                },
            );
        }
    }

    /// Escape on a modal, or its backdrop.
    pub(super) fn dismiss_modal(&mut self, host: &UiHost) {
        if let Some(id) = self.modal.clone() {
            self.close(host, &id, true);
        }
    }

    pub(super) fn close_plugin(&mut self, host: &UiHost, plugin_id: &str) {
        let ids: Vec<String> = self
            .open
            .iter()
            .filter(|(_, s)| &*s.plugin == plugin_id)
            .map(|(id, _)| id.clone())
            .collect();
        for id in ids {
            self.close(host, &id, false);
        }
        self.confirm_queue.retain(
            |c| !matches!(&c.owner, ConfirmOwner::Plugin { plugin, .. } if &**plugin == plugin_id),
        );
        if self.confirm.as_ref().is_some_and(
            |c| matches!(&c.owner, ConfirmOwner::Plugin { plugin, .. } if &**plugin == plugin_id),
        ) {
            self.confirm = None;
            self.show_next_confirm(host);
        }
    }

    pub(super) fn close_all(&mut self, host: &UiHost) {
        let ids: Vec<String> = self.open.keys().cloned().collect();
        for id in ids {
            self.close(host, &id, false);
        }
    }

    // -- confirmations -------------------------------------------------------------

    pub(super) fn queue_confirm(
        &mut self,
        host: &UiHost,
        plugin: &Arc<str>,
        request_id: String,
        args: ConfirmArgs,
    ) {
        self.confirm_queue.push_back(PendingConfirm {
            owner: ConfirmOwner::Plugin {
                plugin: Arc::clone(plugin),
                request_id,
            },
            args,
        });
        if self.confirm.is_none() {
            self.show_next_confirm(host);
        }
    }

    pub(super) fn queue_host_confirm(
        &mut self,
        host: &UiHost,
        args: ConfirmArgs,
        answer: Box<dyn FnOnce(bool)>,
    ) {
        self.confirm_queue.push_back(PendingConfirm {
            owner: ConfirmOwner::Host(answer),
            args,
        });
        if self.confirm.is_none() {
            self.show_next_confirm(host);
        }
    }

    fn show_next_confirm(&mut self, host: &UiHost) {
        let Some(next) = self.confirm_queue.pop_front() else {
            if let Some(ui) = host.ui() {
                ui.set_plugin_confirm_visible(false);
            }
            return;
        };
        if let Some(ui) = host.ui() {
            ui.set_plugin_confirm_title(next.args.title.clone().into());
            ui.set_plugin_confirm_message(next.args.message.clone().into());
            ui.set_plugin_confirm_confirm_label(next.args.confirm_label.clone().into());
            ui.set_plugin_confirm_cancel_label(next.args.cancel_label.clone().into());
            ui.set_plugin_confirm_destructive(next.args.destructive);
            ui.set_plugin_confirm_visible(true);
        }
        self.confirm = Some(next);
    }

    /// The user answered the confirmation dialog. A question of eov's own
    /// is returned, to be answered once the surfaces are no longer borrowed.
    pub(super) fn answer_confirm(
        &mut self,
        host: &UiHost,
        confirmed: bool,
    ) -> Option<Box<dyn FnOnce(bool)>> {
        let mut host_answer = None;
        if let Some(pending) = self.confirm.take() {
            match pending.owner {
                ConfirmOwner::Plugin { plugin, request_id } => host.post(
                    EventTarget::Plugin(plugin),
                    HostEvent::ConfirmResult {
                        request_id,
                        confirmed,
                    },
                ),
                ConfirmOwner::Host(answer) => host_answer = Some(answer),
            }
        }
        self.show_next_confirm(host);
        host_answer
    }
}

fn embedded_factory(
    definition: ComponentDefinition,
    data: Rc<RefCell<SurfaceData>>,
) -> ComponentFactory {
    ComponentFactory::new(move |ctx| {
        let instance = match definition.create_embedded(ctx) {
            Ok(instance) => instance,
            Err(error) => {
                tracing::error!("Cannot create plugin surface: {error}");
                return None;
            }
        };
        SurfaceData::attach(&data, &instance, &definition);
        Some(instance)
    })
}

impl UiHost {
    pub(crate) fn dismiss_plugin_modal(&self) {
        self.surfaces.borrow_mut().dismiss_modal(self);
    }

    pub(crate) fn answer_plugin_confirm(&self, confirmed: bool) {
        let host_answer = self.surfaces.borrow_mut().answer_confirm(self, confirmed);
        if let Some(answer) = host_answer {
            answer(confirmed);
        }
    }

    /// Asks the user to confirm something eov is about to do, in the same
    /// dialog plugins use; `answer` gets the reply.
    pub(crate) fn confirm(
        &self,
        title: &str,
        message: &str,
        confirm_label: &str,
        destructive: bool,
        answer: impl FnOnce(bool) + 'static,
    ) {
        let args = ConfirmArgs {
            title: title.to_string(),
            message: message.to_string(),
            confirm_label: confirm_label.to_string(),
            cancel_label: "Cancel".to_string(),
            destructive,
        };
        self.surfaces
            .borrow_mut()
            .queue_host_confirm(self, args, Box::new(answer));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `.slint` file of the in-tree plugins compiles with this eov's
    /// Slint, the way surfaces are compiled (so a Slint upgrade cannot break
    /// plugin panels unnoticed). Plugins whose submodule is not checked out
    /// are skipped.
    #[test]
    fn in_tree_plugin_surfaces_compile() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let roots = std::fs::read_dir(repo.join("plugins"))
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .chain(std::iter::once(repo.join("plugin_api/example-plugin")));
        let mut compiled = 0;
        for root in roots {
            let Ok(files) = std::fs::read_dir(root.join("ui")) else {
                continue;
            };
            for path in files.filter_map(Result::ok).map(|entry| entry.path()) {
                if path.extension().and_then(|e| e.to_str()) != Some("slint") {
                    continue;
                }
                let mut compiler = slint_interpreter::Compiler::default();
                compiler.set_include_paths(vec![root.clone()]);
                let result = spin_on(compiler.build_from_path(&path));
                // Files of globals and widgets (no component of their own)
                // are compiled through the files that import them.
                let errors: Vec<String> = result
                    .diagnostics()
                    .filter(|d| d.level() == slint_interpreter::DiagnosticLevel::Error)
                    .map(|d| d.to_string())
                    .filter(|message| !message.contains("No component found"))
                    .collect();
                assert!(
                    errors.is_empty(),
                    "{}:\n{}",
                    path.display(),
                    errors.join("\n")
                );
                compiled += 1;
            }
        }
        assert!(compiled > 0, "the example plugin's sidebar at least");
    }

    /// A plugin can reset a field the user edited (the edited value is not
    /// mistaken for the one the plugin sent before), and arrays update in
    /// place.
    #[test]
    fn properties_are_compared_with_what_the_surface_shows() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let source = r#"
                export component Panel {
                    in-out property <string> name;
                    in property <[string]> rows;
                }
            "#;
            let result = spin_on(
                slint_interpreter::Compiler::default()
                    .build_from_source(source.into(), "panel.slint".into()),
            );
            let definition = result.component("Panel").expect("compiles");
            let instance = definition.create().unwrap();
            let plugin: Arc<str> = Arc::from("p");
            let data = Rc::new(RefCell::new(SurfaceData::new("s", &plugin, &definition)));
            SurfaceData::attach(&data, &instance, &definition);
            let set = |json: Json| {
                let Json::Object(map) = json else {
                    unreachable!()
                };
                data.borrow_mut().apply(&map)
            };
            assert!(set(serde_json::json!({"name": "Original", "rows": ["a", "b"]})).is_empty());
            // The user types into the field (a two-way binding writes the
            // property), then the plugin resets it.
            instance
                .set_property("name", Value::String("Edited".into()))
                .unwrap();
            assert!(set(serde_json::json!({"name": "Original"})).is_empty());
            assert_eq!(
                instance.get_property("name").unwrap(),
                Value::String("Original".into())
            );
            let model = Rc::clone(data.borrow().models.get("rows").unwrap());
            assert!(set(serde_json::json!({"rows": ["a", "c", "d"]})).is_empty());
            assert_eq!(model.row_count(), 3);
            assert_eq!(model.row_data(1), Some(Value::String("c".into())));
            assert!(!set(serde_json::json!({"missing": 1})).is_empty());
        });
    }

    #[test]
    fn slint_version_matches_the_lockfile() {
        let lock = include_str!("../../../Cargo.lock");
        let version = lock
            .split("[[package]]")
            .find(|block| block.contains("\nname = \"slint\"\n"))
            .and_then(|block| block.lines().find(|l| l.starts_with("version = ")))
            .map(|line| line.trim_start_matches("version = ").trim_matches('"'))
            .unwrap();
        assert_eq!(version, SLINT_VERSION);
    }
}

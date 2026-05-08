//! Plugin system for EOV.
//!
//! This module contains the host-side plugin infrastructure:
//!
//! - **discovery** – scanning a directory for plugin subdirectories
//! - **manifest** – re-exports from `plugin_api`
//! - **manager** – orchestrates discovery, validation, activation
//! - **host_context** – the concrete `HostContext` the host passes to plugins
//!
//! Plugins contribute toolbar buttons and `.slint` UI panels loaded at runtime
//! via the Slint interpreter.

pub mod discovery;
pub mod host_context;
pub mod manager;
pub(crate) mod registry;
pub mod toolbar;

pub use manager::{ActionOutcome, PluginManager};

use abi_stable::library::RawLibrary;
use abi_stable::std_types::RString;
use host_context::WindowOpenRequest;
use plugin_api::ffi::{self, PluginVTable, UiPropertyFFI};
use slint::{CloseRequestResponse, ComponentHandle, Timer, TimerMode};
use slint::winit_030::WinitWindowAccessor;
use slint_interpreter::json::{value_from_json_str, value_to_json};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;
use std::rc::Rc;
use std::time::Duration;
use tracing::{error, info};

fn apply_plugin_window_geometry(
    plugin_id: &str,
    instance: &slint_interpreter::ComponentInstance,
) {
    let Some(parent) = crate::plugin_host::main_window_geometry() else {
        return;
    };

    let base_width = ((parent.width as f32) * 0.76).round();
    let target_width = if plugin_id == "gamepad" {
        (base_width * 2.0).clamp(1040.0, 1760.0)
    } else {
        base_width.clamp(520.0, 1080.0)
    };
    let target_height = ((parent.height as f32) * 0.7).round().clamp(420.0, 880.0);
    instance
        .window()
        .set_size(slint::LogicalSize::new(target_width, target_height));

    instance.window().with_winit_window(|window: &slint::winit_030::winit::window::Window| {
        use slint::winit_030::winit::dpi::LogicalPosition;

        let offset_x = (parent.width as f32 * 0.08).round() as i32 + 36;
        let offset_y = (parent.height as f32 * 0.1).round() as i32 + 42;
        window.set_outer_position(LogicalPosition::new(
            parent.x + offset_x,
            parent.y + offset_y,
        ));
    });
}

struct PluginWindowEntry {
    plugin_id: String,
    toolbar_button_id: Option<String>,
    instance: slint_interpreter::ComponentInstance,
    refresh_timer: Rc<Timer>,
}

// Thread-local storage for plugin window handles so they are not dropped.
thread_local! {
    static PLUGIN_WINDOWS: RefCell<Vec<PluginWindowEntry>> = const { RefCell::new(Vec::new()) };
}

fn remove_plugin_window(plugin_id: &str, hide_window: bool) -> Option<PluginWindowEntry> {
    PLUGIN_WINDOWS.with(|windows| {
        let mut windows = windows.borrow_mut();
        let index = windows
            .iter()
            .position(|entry| entry.plugin_id == plugin_id)?;
        let entry = windows.remove(index);
        entry.refresh_timer.stop();
        if hide_window {
            let _ = entry.instance.hide();
        }
        Some(entry)
    })
}

fn close_plugin_window(plugin_id: &str) -> bool {
    let Some(entry) = remove_plugin_window(plugin_id, true) else {
        return false;
    };
    if let Some(button_id) = entry.toolbar_button_id.as_deref() {
        let _ = crate::plugin_host::set_local_toolbar_button_active(plugin_id, button_id, false);
    }
    true
}

fn plugin_window_is_open(plugin_id: &str) -> bool {
    PLUGIN_WINDOWS.with(|windows| {
        windows
            .borrow()
            .iter()
            .any(|entry| entry.plugin_id == plugin_id)
    })
}

fn apply_plugin_window_properties(
    plugin_id: &str,
    vtable: PluginVTable,
    instance: &slint_interpreter::ComponentInstance,
    skip_mapping_rows: bool,
) -> anyhow::Result<()> {
    let mut applied_values = BTreeMap::new();
    let property_types: Vec<_> = instance
        .definition()
        .properties_and_callbacks()
        .filter_map(|(name, (ty, _visibility))| ty.is_property_type().then_some((name, ty)))
        .collect();

    for UiPropertyFFI { name, json_value } in (vtable.get_sidebar_properties)().into_iter() {
        let property_name = name.to_string();
        if skip_mapping_rows && plugin_id == "gamepad" && property_name == "mapping-rows" {
            continue;
        }
        let Some((_, property_type)) = property_types
            .iter()
            .find(|(candidate, _)| candidate == &property_name)
        else {
            continue;
        };

        let value = value_from_json_str(property_type, json_value.as_str()).map_err(|err| {
            anyhow::anyhow!(
                "failed to decode plugin window property '{}:{}' from JSON: {err}",
                plugin_id,
                property_name
            )
        })?;
        let target_json =
            serde_json::from_str::<serde_json::Value>(json_value.as_str()).map_err(|err| {
                anyhow::anyhow!(
                    "failed to parse plugin window property JSON '{}:{}': {err}",
                    plugin_id,
                    property_name
                )
            })?;
        let current_matches = instance
            .get_property(&property_name)
            .ok()
            .and_then(|current| value_to_json(&current).ok())
            .is_some_and(|current_json| current_json == target_json);
        applied_values.insert(property_name.clone(), target_json);
        if current_matches {
            continue;
        }
        instance
            .set_property(&property_name, value)
            .map_err(|err| {
                anyhow::anyhow!(
                    "failed to set plugin window property '{}:{}': {err}",
                    plugin_id,
                    property_name
                )
            })?;
    }

    Ok(())
}

/// Open a plugin window using the Slint runtime interpreter.
fn open_plugin_window(
    req: &WindowOpenRequest,
    vtable: &PluginVTable,
    toolbar_button_id: Option<&str>,
) -> anyhow::Result<()> {
    info!(
        "Opening plugin window for '{}': {} (component: {})",
        req.plugin_id,
        req.ui_path.display(),
        req.component
    );

    let source = std::fs::read_to_string(&req.ui_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to read plugin UI file {}: {e}",
            req.ui_path.display()
        )
    })?;

    let compiler = slint_interpreter::Compiler::default();
    let result = spin_on(compiler.build_from_source(source, req.ui_path.clone()));

    let has_errors = result
        .diagnostics()
        .any(|d| d.level() == slint_interpreter::DiagnosticLevel::Error);
    for diag in result.diagnostics() {
        if diag.level() == slint_interpreter::DiagnosticLevel::Error {
            error!("Slint compile error: {diag}");
        }
    }
    if has_errors {
        return Err(anyhow::anyhow!(
            "Failed to compile plugin UI '{}' for plugin '{}'",
            req.ui_path.display(),
            req.plugin_id
        ));
    }

    let definition = result.component(&req.component).ok_or_else(|| {
        anyhow::anyhow!(
            "Component '{}' not found in plugin UI '{}' for plugin '{}'. Available: {:?}",
            req.component,
            req.ui_path.display(),
            req.plugin_id,
            result.component_names().collect::<Vec<_>>()
        )
    })?;

    let instance = definition
        .create()
        .map_err(|e| anyhow::anyhow!("Failed to create plugin component: {e}"))?;

    let plugin_id = req.plugin_id.clone();
    let toolbar_button_id_owned = toolbar_button_id.map(ToOwned::to_owned);
    let toolbar_button_id_for_close = toolbar_button_id_owned.clone();
    instance.window().on_close_requested(move || {
        let _ = remove_plugin_window(&plugin_id, false);
        if let Some(button_id) = toolbar_button_id_for_close.as_deref() {
            let _ =
                crate::plugin_host::set_local_toolbar_button_active(&plugin_id, button_id, false);
        }
        CloseRequestResponse::HideWindow
    });

    apply_plugin_window_properties(&req.plugin_id, *vtable, &instance, false)?;

    // Wire all user-defined callbacks in the .slint to call through the
    // plugin's on_ui_callback vtable entry.
    let on_ui_cb = vtable.on_ui_callback;
    let property_vtable = *vtable;
    for name in definition.callbacks() {
        let cb_name = name.clone();
        let instance_weak = instance.as_weak();
        let plugin_id = req.plugin_id.clone();
        if instance
            .set_callback(&name, move |args| {
                let args_json = serde_json::Value::Array(
                    args.iter()
                        .map(value_to_json)
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap_or_default(),
                )
                .to_string();
                (on_ui_cb)(RString::from(cb_name.as_str()), RString::from(args_json));
                if let Some(instance) = instance_weak.upgrade() {
                    let _ = apply_plugin_window_properties(&plugin_id, property_vtable, &instance, false);
                }
                slint_interpreter::Value::Void
            })
            .is_err()
        {
            info!(
                "Could not wire callback '{name}' for plugin '{}'",
                req.plugin_id
            );
        }
    }

    instance
        .show()
        .map_err(|e| anyhow::anyhow!("Failed to show plugin window: {e}"))?;

    apply_plugin_window_geometry(&req.plugin_id, &instance);

    let refresh_timer = Rc::new(Timer::default());
    let refresh_timer_for_callback = Rc::clone(&refresh_timer);
    let instance_weak = instance.as_weak();
    let plugin_id = req.plugin_id.clone();
    let property_vtable = *vtable;
    refresh_timer.start(TimerMode::Repeated, Duration::from_millis(200), move || {
        let Some(instance) = instance_weak.upgrade() else {
            refresh_timer_for_callback.stop();
            return;
        };
        let _ = apply_plugin_window_properties(&plugin_id, property_vtable, &instance, true);
    });

    PLUGIN_WINDOWS.with(|windows| {
        windows.borrow_mut().push(PluginWindowEntry {
            plugin_id: req.plugin_id.clone(),
            toolbar_button_id: toolbar_button_id_owned,
            instance,
            refresh_timer,
        });
    });

    info!("Plugin window opened for '{}'", req.plugin_id);
    Ok(())
}

pub fn toggle_rust_plugin_window(
    plugin_root: &Path,
    toolbar_button_id: &str,
) -> anyhow::Result<bool> {
    let manifest = plugin_api::PluginManifest::from_file(
        &plugin_root.join(plugin_api::manifest::MANIFEST_FILENAME),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load plugin manifest: {e}"))?;

    if plugin_window_is_open(&manifest.id) {
        close_plugin_window(&manifest.id);
        return Ok(false);
    }

    let lib_name = ffi::plugin_library_filename(&manifest.id);
    let lib_path = plugin_root.join(&lib_name);
    if !lib_path.exists() {
        return Err(anyhow::anyhow!(
            "Plugin shared library not found: {}",
            lib_path.display()
        ));
    }

    let raw = RawLibrary::load_at(&lib_path)
        .map_err(|e| anyhow::anyhow!("Failed to load plugin library: {e}"))?;
    let raw: &'static RawLibrary = Box::leak(Box::new(raw));
    let vtable: PluginVTable = unsafe {
        let sym = raw
            .get::<ffi::GetPluginVTableFn>(ffi::PLUGIN_VTABLE_SYMBOL)
            .map_err(|e| anyhow::anyhow!("Failed to find plugin vtable symbol: {e}"))?;
        (*sym)()
    };

    let ui_path = manifest
        .resolve_entry_ui(plugin_root)
        .ok_or_else(|| anyhow::anyhow!("Plugin '{}' has no entry_ui", manifest.id))?;
    let req = WindowOpenRequest {
        plugin_id: manifest.id.clone(),
        ui_path,
        component: manifest
            .entry_component
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Plugin '{}' has no entry_component", manifest.id))?,
    };

    open_plugin_window(&req, &vtable, Some(toolbar_button_id))?;
    Ok(true)
}

/// Entry point for `eov plugin-window <plugin_root>`.
///
/// Runs in a child process with its own Slint event loop and GPU context.
/// Loads the plugin manifest, shared library, and .slint UI; wires callbacks
/// through the vtable; shows the window; and blocks until it is closed.
pub fn run_plugin_window_standalone(plugin_root: &Path) -> anyhow::Result<()> {
    let manifest = plugin_api::PluginManifest::from_file(
        &plugin_root.join(plugin_api::manifest::MANIFEST_FILENAME),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load plugin manifest: {e}"))?;

    let lib_name = ffi::plugin_library_filename(&manifest.id);
    let lib_path = plugin_root.join(&lib_name);
    if !lib_path.exists() {
        return Err(anyhow::anyhow!(
            "Plugin shared library not found: {}",
            lib_path.display()
        ));
    }

    // Load the shared library and get the vtable.
    let raw = RawLibrary::load_at(&lib_path)
        .map_err(|e| anyhow::anyhow!("Failed to load plugin library: {e}"))?;
    let raw: &'static RawLibrary = Box::leak(Box::new(raw));
    let vtable: PluginVTable = unsafe {
        let sym = raw
            .get::<ffi::GetPluginVTableFn>(ffi::PLUGIN_VTABLE_SYMBOL)
            .map_err(|e| anyhow::anyhow!("Failed to find plugin vtable symbol: {e}"))?;
        (*sym)()
    };

    let ui_path = manifest
        .resolve_entry_ui(plugin_root)
        .ok_or_else(|| anyhow::anyhow!("Plugin '{}' has no entry_ui", manifest.id))?;
    let req = WindowOpenRequest {
        plugin_id: manifest.id.clone(),
        ui_path,
        component: manifest
            .entry_component
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Plugin '{}' has no entry_component", manifest.id))?,
    };

    // Open the window directly (no deferral needed — we own the event loop).
    open_plugin_window(&req, &vtable, None)?;

    // Run until the window is closed.
    slint::run_event_loop()?;
    Ok(())
}

/// Minimal synchronous executor for futures that are not truly async.
fn spin_on<T>(future: impl std::future::Future<Output = T>) -> T {
    use std::task::{Context, Poll, Wake, Waker};
    struct NoopWaker;
    impl Wake for NoopWaker {
        fn wake(self: std::sync::Arc<Self>) {}
    }
    let waker = Waker::from(std::sync::Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spin_on_for_test<T>(future: impl std::future::Future<Output = T>) -> T {
        spin_on(future)
    }

    #[test]
    fn annotations_sidebar_runtime_compiles_and_creates() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let ui_path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../plugins/annotations/ui/annotations-sidebar.slint");
            let source = std::fs::read_to_string(&ui_path).unwrap();

            let compiler = slint_interpreter::Compiler::default();
            let result = spin_on_for_test(compiler.build_from_source(source, ui_path.clone()));

            let diagnostics = result.diagnostics().collect::<Vec<_>>();
            assert!(
                diagnostics.iter().all(|diagnostic| {
                    diagnostic.level() != slint_interpreter::DiagnosticLevel::Error
                }),
                "runtime compile diagnostics: {diagnostics:?}"
            );

            let definition = result.component("AnnotationsSidebar").unwrap();
            let _instance = definition.create().unwrap();
        });
    }

    #[test]
    fn eovae_sidebar_runtime_compiles_and_creates() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let ui_path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../plugins/eovae/ui/eovae-sidebar.slint");
            let source = std::fs::read_to_string(&ui_path).unwrap();

            let compiler = slint_interpreter::Compiler::default();
            let result = spin_on_for_test(compiler.build_from_source(source, ui_path.clone()));

            let diagnostics = result.diagnostics().collect::<Vec<_>>();
            assert!(
                diagnostics.iter().all(|diagnostic| {
                    diagnostic.level() != slint_interpreter::DiagnosticLevel::Error
                }),
                "runtime compile diagnostics: {diagnostics:?}"
            );

            let definition = result.component("EovaeSidebar").unwrap();
            let _instance = definition.create().unwrap();
        });
    }

    #[test]
    fn gamepad_sidebar_runtime_compiles_and_creates() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let ui_path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../plugins/gamepad/ui/gamepad-window.slint");
            let source = std::fs::read_to_string(&ui_path).unwrap();

            let compiler = slint_interpreter::Compiler::default();
            let result = spin_on_for_test(compiler.build_from_source(source, ui_path.clone()));

            let diagnostics = result.diagnostics().collect::<Vec<_>>();
            assert!(
                diagnostics.iter().all(|diagnostic| {
                    diagnostic.level() != slint_interpreter::DiagnosticLevel::Error
                }),
                "runtime compile diagnostics: {diagnostics:?}"
            );

            let definition = result.component("GamepadWindow").unwrap();
            let _instance = definition.create().unwrap();
        });
    }

    #[test]
    fn eovae_viewport_grid_runtime_compiles_and_creates() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let ui_path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../plugins/eovae/ui/eovae-viewport-grid.slint");
            let source = std::fs::read_to_string(&ui_path).unwrap();

            let compiler = slint_interpreter::Compiler::default();
            let result = spin_on_for_test(compiler.build_from_source(source, ui_path.clone()));

            let diagnostics = result.diagnostics().collect::<Vec<_>>();
            assert!(
                diagnostics.iter().all(|diagnostic| {
                    diagnostic.level() != slint_interpreter::DiagnosticLevel::Error
                }),
                "runtime compile diagnostics: {diagnostics:?}"
            );

            let definition = result.component("EovaeViewportGrid").unwrap();
            let _instance = definition.create().unwrap();
        });
    }
}

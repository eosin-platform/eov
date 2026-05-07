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
use slint::{ComponentHandle, Timer, TimerMode};
use slint_interpreter::json::{value_from_json_str, value_to_json};
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::time::Duration;
use tracing::{error, info, warn};

// Thread-local storage for plugin window handles so they are not dropped.
thread_local! {
    static PLUGIN_WINDOWS: RefCell<Vec<slint_interpreter::ComponentInstance>> = const { RefCell::new(Vec::new()) };
    static PLUGIN_WINDOW_TIMERS: RefCell<Vec<Rc<Timer>>> = const { RefCell::new(Vec::new()) };
}

fn apply_plugin_window_properties(
    plugin_id: &str,
    vtable: PluginVTable,
    instance: &slint_interpreter::ComponentInstance,
) -> anyhow::Result<()> {
    let property_types: Vec<_> = instance
        .definition()
        .properties_and_callbacks()
        .filter_map(|(name, (ty, _visibility))| ty.is_property_type().then_some((name, ty)))
        .collect();

    for UiPropertyFFI { name, json_value } in (vtable.get_sidebar_properties)().into_iter() {
        let property_name = name.to_string();
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
        instance.set_property(&property_name, value).map_err(|err| {
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
fn open_plugin_window(req: &WindowOpenRequest, vtable: &PluginVTable) -> anyhow::Result<()> {
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

    apply_plugin_window_properties(&req.plugin_id, *vtable, &instance)?;

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
                    let _ = apply_plugin_window_properties(&plugin_id, property_vtable, &instance);
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
        let _ = apply_plugin_window_properties(&plugin_id, property_vtable, &instance);
    });

    PLUGIN_WINDOWS.with(|windows| {
        windows.borrow_mut().push(instance);
    });
    PLUGIN_WINDOW_TIMERS.with(|timers| {
        timers.borrow_mut().push(refresh_timer);
    });

    info!("Plugin window opened for '{}'", req.plugin_id);
    Ok(())
}

/// Spawn a Python plugin's entry script as a subprocess.
///
/// The script is expected to use `slint`-python to create its own window
/// and handle callbacks. The process runs independently of the host.
///
/// The plugin is expected to have a `.venv/` directory inside its root
/// with the slint package installed. The host uses the venv's python3.
pub fn spawn_python_plugin(
    script_path: &Path,
    plugin_root: &Path,
    initial_action_id: Option<&str>,
) {
    info!(
        "Spawning Python plugin: {} (cwd: {})",
        script_path.display(),
        plugin_root.display()
    );

    // Use the venv python inside the plugin directory.
    let venv_python = plugin_root.join(".venv").join("bin").join("python3");
    let python = if venv_python.exists() {
        info!("Using plugin venv: {}", venv_python.display());
        venv_python.into_os_string()
    } else {
        warn!(
            "No .venv found in {}, falling back to system python3",
            plugin_root.display()
        );
        std::ffi::OsString::from("python3")
    };

    let mut cmd = std::process::Command::new(&python);
    cmd.arg(script_path)
        .current_dir(plugin_root)
        .stdin(std::process::Stdio::null());

    // Pass the extension host address if available.
    if let Ok(host_addr) = std::env::var("EOV_EXTENSION_HOST") {
        cmd.env("EOV_EXTENSION_HOST", &host_addr);
    }
    if let Some(action_id) = initial_action_id {
        cmd.env("EOV_INITIAL_PLUGIN_ACTION", action_id);
    }

    match cmd.spawn() {
        Ok(child) => {
            info!("Python plugin process started (pid {})", child.id());
        }
        Err(e) => {
            error!(
                "Failed to spawn Python plugin {}: {e}",
                script_path.display()
            );
        }
    }
}

/// Spawn the Rust plugin window in a child process.
///
/// Instead of opening a second Slint window in the same process (which
/// crashes the wgpu renderer), we re-exec `eov plugin-window <dir>`.
/// The child process gets its own GPU context and Slint event loop.
pub fn spawn_rust_plugin_window(plugin_root: &Path) {
    let eov_exe = std::env::current_exe().unwrap_or_else(|_| "eov".into());
    info!(
        "Spawning Rust plugin window subprocess: {} plugin-window {}",
        eov_exe.display(),
        plugin_root.display()
    );

    let mut cmd = std::process::Command::new(&eov_exe);
    cmd.arg("plugin-window")
        .arg(plugin_root)
        .stdin(std::process::Stdio::null());

    // Pass the extension host address if available.
    if let Ok(host_addr) = std::env::var("EOV_EXTENSION_HOST") {
        cmd.env("EOV_EXTENSION_HOST", &host_addr);
    }

    match cmd.spawn() {
        Ok(child) => {
            info!("Rust plugin window process started (pid {})", child.id());
        }
        Err(e) => {
            error!(
                "Failed to spawn plugin window process for {}: {e}",
                plugin_root.display()
            );
        }
    }
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
    open_plugin_window(&req, &vtable)?;

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
                .join("../plugins/gamepad/ui/gamepad-sidebar.slint");
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

            let definition = result.component("GamepadSidebar").unwrap();
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

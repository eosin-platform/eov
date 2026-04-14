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

pub use host_context::AppHostContext;
pub use manager::PluginManager;
pub use toolbar::ToolbarManager;

use abi_stable::std_types::RString;
use host_context::WindowOpenRequest;
use plugin_api::ffi::PluginVTable;
use slint::{ComponentHandle, Timer};
use std::cell::RefCell;
use std::time::Duration;
use tracing::{error, info};

// Thread-local storage for plugin window handles so they are not dropped.
thread_local! {
    static PLUGIN_WINDOWS: RefCell<Vec<slint_interpreter::ComponentInstance>> = RefCell::new(Vec::new());
}

/// Schedule a plugin window to open on the next event loop tick.
///
/// Deferring avoids a wgpu surface conflict that occurs when creating a new
/// window inside a Slint callback (the main window's swap chain gets
/// invalidated before the current frame finishes).
pub fn schedule_open_plugin_window(req: WindowOpenRequest, vtable: PluginVTable) {
    Timer::single_shot(Duration::ZERO, move || {
        if let Err(e) = open_plugin_window(&req, &vtable) {
            error!("Failed to open plugin window: {e}");
        }
    });
}

/// Open a plugin window using the Slint runtime interpreter.
fn open_plugin_window(
    req: &WindowOpenRequest,
    vtable: &PluginVTable,
) -> anyhow::Result<()> {
    info!(
        "Opening plugin window for '{}': {} (component: {})",
        req.plugin_id,
        req.ui_path.display(),
        req.component
    );

    let source = std::fs::read_to_string(&req.ui_path)
        .map_err(|e| anyhow::anyhow!("Failed to read plugin UI file {}: {e}", req.ui_path.display()))?;

    let compiler = slint_interpreter::Compiler::default();
    let result = spin_on(compiler.build_from_source(source, req.ui_path.clone()));

    let has_errors = result.diagnostics().any(|d| d.level() == slint_interpreter::DiagnosticLevel::Error);
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

    let instance = definition.create()
        .map_err(|e| anyhow::anyhow!("Failed to create plugin component: {e}"))?;

    // Wire all user-defined callbacks in the .slint to call through the
    // plugin's on_ui_callback vtable entry.
    let on_ui_cb = vtable.on_ui_callback;
    for name in definition.callbacks() {
        let cb_name = name.clone();
        if instance.set_callback(&name, move |_args| {
            (on_ui_cb)(RString::from(cb_name.as_str()));
            slint_interpreter::Value::Void
        }).is_err() {
            info!("Could not wire callback '{name}' for plugin '{}'", req.plugin_id);
        }
    }

    instance.show()
        .map_err(|e| anyhow::anyhow!("Failed to show plugin window: {e}"))?;

    PLUGIN_WINDOWS.with(|windows| {
        windows.borrow_mut().push(instance);
    });

    info!("Plugin window opened for '{}'", req.plugin_id);
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

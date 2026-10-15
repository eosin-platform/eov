//! Chooses what draws the window (RND-4).
//!
//! The window is drawn with femtovg on wgpu when a GPU adapter is
//! available (Vulkan, DirectX 12 or Metal), which also lets slides be
//! composited on the GPU. Without one — virtual desktops, remote sessions,
//! VMs, machines without a GPU driver — Slint's software renderer draws the
//! window and slides are composited on the CPU; the user is told why.
//! `EOV_RENDERER=software` forces the software renderer; `WGPU_BACKEND`
//! picks the wgpu backends (OpenGL only on request: a window drawn through
//! wgpu's OpenGL backend did not appear in testing).

use anyhow::Result;
use slint::BackendSelector;
use slint::wgpu_29::wgpu;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WindowGeometry {
    pub(crate) width: Option<u32>,
    pub(crate) height: Option<u32>,
    pub(crate) x: Option<i32>,
    pub(crate) y: Option<i32>,
}

/// What draws the window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum UiRenderer {
    /// femtovg on wgpu: slides can be drawn on the GPU too.
    Gpu,
    /// Slint's software renderer, for the given reason: slides are drawn on
    /// the CPU.
    Software(String),
}

pub(crate) fn select_backend(window_geometry: WindowGeometry) -> Result<UiRenderer> {
    let mut settings = slint::wgpu_29::WGPUSettings::default();
    settings.backends = preferred_backends();

    let reason = if software_requested() {
        "the software renderer was requested (EOV_RENDERER=software)".to_string()
    } else {
        match gpu_adapter(&settings) {
            Ok(adapter) => {
                tracing::info!("Drawing with the GPU: {adapter}");
                match winit_backend_selector(window_geometry)
                    .renderer_name("femtovg-wgpu".to_string())
                    .require_wgpu_29(slint::wgpu_29::WGPUConfiguration::Automatic(settings))
                    .select()
                {
                    Ok(()) => return Ok(UiRenderer::Gpu),
                    Err(err) => format!("the GPU renderer could not start ({err})"),
                }
            }
            Err(reason) => reason,
        }
    };

    tracing::warn!("Drawing with the software renderer: {reason}");
    winit_backend_selector(window_geometry)
        .renderer_name("software".to_string())
        .select()?;
    Ok(UiRenderer::Software(reason))
}

fn software_requested() -> bool {
    std::env::var("EOV_RENDERER").is_ok_and(|value| value.eq_ignore_ascii_case("software"))
}

/// `WGPU_BACKEND`, else Vulkan and DirectX 12 (Metal on macOS).
fn preferred_backends() -> wgpu::Backends {
    if let Some(backends) = wgpu::Backends::from_env() {
        return backends;
    }
    if cfg!(target_os = "macos") {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN | wgpu::Backends::DX12
    }
}

/// The adapter femtovg-wgpu would draw with, or why there is none. Applies
/// the same test as Slint, which refuses to start femtovg-wgpu otherwise:
/// a GPU adapter, or a software one with `SLINT_WGPU_CPU` set.
fn gpu_adapter(settings: &slint::wgpu_29::WGPUSettings) -> Result<String, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: settings.backends,
        flags: settings.instance_flags,
        backend_options: settings.backend_options.clone(),
        memory_budget_thresholds: settings.instance_memory_budget_thresholds,
        display: None,
    });
    let adapters: Vec<wgpu::AdapterInfo> =
        poll_once(instance.enumerate_adapters(settings.backends))
            .unwrap_or_default()
            .iter()
            .map(wgpu::Adapter::get_info)
            .collect();
    usable_adapter(&adapters, std::env::var_os("SLINT_WGPU_CPU").is_some())
}

fn usable_adapter(adapters: &[wgpu::AdapterInfo], allow_cpu: bool) -> Result<String, String> {
    if let Some(info) = adapters
        .iter()
        .find(|info| allow_cpu || info.device_type != wgpu::DeviceType::Cpu)
    {
        return Ok(format!("{} ({:?})", info.name, info.backend));
    }
    Err(if adapters.is_empty() {
        "no Vulkan, DirectX 12 or Metal graphics adapter was found".to_string()
    } else {
        let names: Vec<&str> = adapters.iter().map(|info| info.name.as_str()).collect();
        format!(
            "only software graphics adapters were found ({})",
            names.join(", ")
        )
    })
}

/// The output of a future that is ready at once (wgpu on native platforms).
fn poll_once<F: std::future::Future>(future: F) -> Option<F::Output> {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match future.as_mut().poll(&mut context) {
        std::task::Poll::Ready(output) => Some(output),
        std::task::Poll::Pending => None,
    }
}

/// Tells the user in a dialog why the window could not open. (The Windows
/// GUI build shows fatal errors in a message box itself, SH-2.)
pub(crate) fn report_startup_error(error: &anyhow::Error) {
    #[cfg(not(target_os = "windows"))]
    {
        let _ = rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Error)
            .set_title("eov could not start")
            .set_description(format!("{error:#}"))
            .set_buttons(rfd::MessageButtons::Ok)
            .show();
    }
    #[cfg(target_os = "windows")]
    let _ = error;
}

fn winit_backend_selector(window_geometry: WindowGeometry) -> BackendSelector {
    use slint::winit_030::winit;

    BackendSelector::new()
        .backend_name("winit".to_string())
        .with_winit_window_attributes_hook(move |attributes| {
            let mut attributes = attributes;

            if let (Some(width), Some(height)) = (window_geometry.width, window_geometry.height) {
                attributes =
                    attributes.with_inner_size(winit::dpi::LogicalSize::new(width, height));
            }

            if let (Some(x), Some(y)) = (window_geometry.x, window_geometry.y) {
                attributes = attributes.with_position(winit::dpi::LogicalPosition::new(x, y));
            }

            #[cfg(target_os = "macos")]
            {
                use slint::winit_030::winit::platform::macos::WindowAttributesExtMacOS;

                attributes = attributes
                    .with_titlebar_transparent(true)
                    .with_title_hidden(true)
                    .with_fullsize_content_view(true);
            }

            attributes
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(name: &str, device_type: wgpu::DeviceType) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.to_string(),
            vendor: 0,
            device: 0,
            device_type,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
            subgroup_min_size: 0,
            subgroup_max_size: 0,
            transient_saves_memory: false,
        }
    }

    #[test]
    fn software_adapters_alone_do_not_count_as_a_gpu() {
        assert!(
            usable_adapter(&[], false)
                .unwrap_err()
                .contains("no Vulkan")
        );
        let cpu = [adapter("llvmpipe", wgpu::DeviceType::Cpu)];
        let reason = usable_adapter(&cpu, false).unwrap_err();
        assert!(reason.contains("only software") && reason.contains("llvmpipe"));
        assert!(
            usable_adapter(&cpu, true).is_ok(),
            "unless SLINT_WGPU_CPU is set"
        );
        let both = [
            adapter("llvmpipe", wgpu::DeviceType::Cpu),
            adapter("Radeon", wgpu::DeviceType::DiscreteGpu),
        ];
        assert!(usable_adapter(&both, false).unwrap().starts_with("Radeon"));
    }
}

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

pub(crate) fn select_backend(window_geometry: WindowGeometry) -> Result<()> {
    let mut settings = slint::wgpu_29::WGPUSettings::default();
    settings.backends = preferred_backends();

    winit_backend_selector(window_geometry)
        .renderer_name("femtovg-wgpu".to_string())
        .require_wgpu_29(slint::wgpu_29::WGPUConfiguration::Automatic(settings))
        .select()?;
    Ok(())
}

fn preferred_backends() -> wgpu::Backends {
    #[cfg(target_os = "macos")]
    {
        wgpu::Backends::METAL
    }

    #[cfg(not(target_os = "macos"))]
    {
        wgpu::Backends::VULKAN
    }
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

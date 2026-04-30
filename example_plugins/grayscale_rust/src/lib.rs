//! Grayscale viewport filter plugin (Rust FFI).
//!
//! Demonstrates implementing a viewport filter via the abi_stable FFI surface.
//! Registers a toolbar button (smiley face icon) that toggles a grayscale
//! post-processing filter on the viewport. Supports both CPU and GPU rendering.
//!
//! The GPU path uses an ash-based compute pipeline with a GLSL compute shader
//! compiled to SPIR-V. The shader operates in-place on the filter image via
//! `imageLoad`/`imageStore`.

use abi_stable::std_types::{ROption, RString, RVec};
use ash::vk;
use ash::vk::Handle;
use plugin_api::ffi::{
    ActionResponseFFI, GpuFilterContextFFI, HostApiVTable, HostLogLevelFFI, HudToolbarButtonFFI,
    PluginVTable, ToolbarButtonFFI, UiPropertyFFI, ViewportContextMenuItemFFI, ViewportFilterFFI,
    ViewportOverlayComponentRequestFFI, ViewportOverlayPointFFI, ViewportOverlayPolygonFFI,
    ViewportOverlayVertexFFI, ViewportSnapshotFFI,
};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Smiley face SVG icon for the toolbar button.
const SMILEY_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>"#;

const FILTER_ID: &str = "grayscale";
const BUTTON_ID: &str = "toggle_grayscale";

static ENABLED: AtomicBool = AtomicBool::new(false);
static HOST_API: Mutex<Option<HostApiVTable>> = Mutex::new(None);

extern "C" fn set_host_api_ffi(host_api: HostApiVTable) {
    *HOST_API.lock().unwrap() = Some(host_api);
}

// ---------------------------------------------------------------------------
// FFI exports
// ---------------------------------------------------------------------------

extern "C" fn get_toolbar_buttons_ffi() -> RVec<ToolbarButtonFFI> {
    RVec::from(vec![ToolbarButtonFFI {
        button_id: RString::from(BUTTON_ID),
        tooltip: RString::from("Toggle Grayscale"),
        icon_svg: RString::from(SMILEY_SVG),
        action_id: RString::from(BUTTON_ID),
        tool_mode: abi_stable::std_types::ROption::RNone,
        hotkey: abi_stable::std_types::ROption::RNone,
    }])
}

extern "C" fn get_hud_toolbar_buttons_ffi() -> RVec<HudToolbarButtonFFI> {
    RVec::new()
}

extern "C" fn on_action_ffi(action_id: RString) -> ActionResponseFFI {
    if action_id.as_str() == BUTTON_ID {
        let prev = ENABLED.fetch_xor(true, Ordering::Relaxed);
        if let Some(host_api) = *HOST_API.lock().unwrap() {
            let enabled = !prev;
            let _ = (host_api.set_toolbar_button_active)(
                host_api.context,
                RString::from(BUTTON_ID),
                enabled,
            );
            let message = if !prev {
                "Grayscale filter enabled"
            } else {
                "Grayscale filter disabled"
            };
            (host_api.log_message)(
                host_api.context,
                HostLogLevelFFI::Info,
                RString::from(message),
            );
        }
        println!(
            "[grayscale_plugin] Grayscale toggled {}",
            if !prev { "ON" } else { "OFF" }
        );
    }
    ActionResponseFFI { open_window: false }
}

extern "C" fn on_hud_action_ffi(
    _action_id: RString,
    _viewport: ViewportSnapshotFFI,
) -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}

extern "C" fn on_ui_callback_ffi(_callback_name: RString, _args_json: RString) {}

extern "C" fn get_sidebar_properties_ffi() -> RVec<UiPropertyFFI> {
    RVec::new()
}

extern "C" fn get_viewport_context_menu_items_ffi(
    _viewport: ViewportSnapshotFFI,
) -> RVec<ViewportContextMenuItemFFI> {
    RVec::new()
}

extern "C" fn on_viewport_context_menu_action_ffi(
    _item_id: RString,
    _viewport: ViewportSnapshotFFI,
) -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}

extern "C" fn get_viewport_overlay_points_ffi(
    _viewport: ViewportSnapshotFFI,
) -> RVec<ViewportOverlayPointFFI> {
    RVec::new()
}

extern "C" fn get_viewport_overlay_polygons_ffi(
    _viewport: ViewportSnapshotFFI,
) -> RVec<ViewportOverlayPolygonFFI> {
    RVec::new()
}

extern "C" fn get_viewport_overlay_component_ffi() -> ROption<ViewportOverlayComponentRequestFFI> {
    ROption::RNone
}

extern "C" fn get_viewport_overlay_properties_ffi(
    _viewport: ViewportSnapshotFFI,
) -> RVec<UiPropertyFFI> {
    RVec::new()
}

extern "C" fn on_point_annotation_placed_ffi(
    _viewport: ViewportSnapshotFFI,
    _x_level0: f64,
    _y_level0: f64,
) {
}

extern "C" fn on_polygon_annotation_placed_ffi(
    _viewport: ViewportSnapshotFFI,
    _vertices: RVec<ViewportOverlayVertexFFI>,
) {
}

extern "C" fn get_viewport_filters_ffi() -> RVec<ViewportFilterFFI> {
    RVec::from(vec![ViewportFilterFFI {
        filter_id: RString::from(FILTER_ID),
        name: RString::from("Grayscale"),
        supports_cpu: true,
        supports_gpu: true,
        enabled: ENABLED.load(Ordering::Relaxed),
    }])
}

extern "C" fn apply_filter_cpu_ffi(
    _filter_id: RString,
    rgba_data: *mut u8,
    len: u32,
    _width: u32,
    _height: u32,
    _viewport: *const plugin_api::ffi::ViewportSnapshotFFI,
) -> bool {
    if !ENABLED.load(Ordering::Relaxed) || rgba_data.is_null() {
        return false;
    }
    let data = unsafe { std::slice::from_raw_parts_mut(rgba_data, len as usize) };
    for pixel in data.chunks_exact_mut(4) {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        let lum = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
        pixel[0] = lum;
        pixel[1] = lum;
        pixel[2] = lum;
        // alpha unchanged
    }
    true
}

extern "C" fn apply_filter_gpu_ffi(_filter_id: RString, ctx: *const GpuFilterContextFFI) -> bool {
    if !ENABLED.load(Ordering::Relaxed) || ctx.is_null() {
        return false;
    }
    let ctx = unsafe { &*ctx };
    match gpu_grayscale::apply(ctx) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("[grayscale_plugin] GPU filter error: {e:?}");
            false
        }
    }
}

/// GPU compute pipeline for grayscale (lazily initialized, cached).
mod gpu_grayscale {
    use super::*;

    /// Pre-compiled SPIR-V for the grayscale compute shader.
    const SHADER_SPV: &[u8] = include_bytes!("../shaders/grayscale.comp.spv");

    /// Cached pipeline state — created once per VkDevice, reused across frames.
    struct PipelineState {
        device: ash::Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
        pipeline_layout: vk::PipelineLayout,
        pipeline: vk::Pipeline,
        descriptor_pool: vk::DescriptorPool,
        command_pool: vk::CommandPool,
        /// The VkDevice handle this state was created for.
        device_handle: u64,
    }

    // SAFETY: The ash::Device dispatch table is thread-safe. The Vulkan
    // objects are only used from the plugin FFI call which is serialized
    // by the host.
    unsafe impl Send for PipelineState {}

    static PIPELINE: Mutex<Option<PipelineState>> = Mutex::new(None);

    pub fn apply(ctx: &GpuFilterContextFFI) -> Result<(), vk::Result> {
        let mut guard = PIPELINE.lock().unwrap();

        // Recreate if the device changed (shouldn't happen normally).
        if guard
            .as_ref()
            .is_some_and(|s| s.device_handle != ctx.vk_device)
        {
            // Drop old state.
            if let Some(old) = guard.take() {
                unsafe { cleanup(&old) };
            }
        }

        if guard.is_none() {
            *guard = Some(unsafe { create_pipeline(ctx)? });
        }

        let state = guard.as_ref().unwrap();
        unsafe { dispatch(state, ctx) }
    }

    unsafe fn create_pipeline(ctx: &GpuFilterContextFFI) -> Result<PipelineState, vk::Result> {
        // Reconstruct ash::Device from the raw handle.
        // SAFETY: The host guarantees the VkDevice is valid for the duration
        // of this call and that the function pointers are loaded.
        let vk_device = vk::Device::from_raw(ctx.vk_device);

        // Load Vulkan entry point (dynamically links libvulkan.so).
        let entry = unsafe { ash::Entry::load().expect("Failed to load Vulkan entry") };

        // Use vkGetInstanceProcAddr(NULL, "vkGetDeviceProcAddr") to obtain
        // a device-level function pointer resolver, then load all device
        // functions from the existing VkDevice.
        let device = unsafe {
            ash::Device::load_with(
                |name| {
                    // vkGetDeviceProcAddr via the global entry point.
                    // NULL instance is allowed per Vulkan spec for
                    // vkGetInstanceProcAddr to resolve global functions.
                    let gdpa_name = c"vkGetDeviceProcAddr";
                    let gdpa: ash::vk::PFN_vkGetDeviceProcAddr = std::mem::transmute(
                        entry.get_instance_proc_addr(vk::Instance::null(), gdpa_name.as_ptr()),
                    );
                    std::mem::transmute(gdpa(vk_device, name.as_ptr()))
                },
                vk_device,
            )
        };

        // Create shader module from SPIR-V
        let spv_words: Vec<u32> = SHADER_SPV
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_words);
        let shader_module = unsafe { device.create_shader_module(&shader_info, None)? };

        // Descriptor set layout: 1 storage image binding
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let ds_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&binding));
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&ds_layout_info, None)? };

        // Pipeline layout
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // Compute pipeline
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
        }
        .map_err(|(_, e)| e)?;

        // Clean up shader module (no longer needed after pipeline creation).
        unsafe { device.destroy_shader_module(shader_module, None) };

        // Descriptor pool
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(std::slice::from_ref(&pool_size));
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        // Command pool
        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&cmd_pool_info, None)? };

        Ok(PipelineState {
            device,
            descriptor_set_layout,
            pipeline_layout,
            pipeline: pipelines[0],
            descriptor_pool,
            command_pool,
            device_handle: ctx.vk_device,
        })
    }

    unsafe fn dispatch(state: &PipelineState, ctx: &GpuFilterContextFFI) -> Result<(), vk::Result> {
        let device = &state.device;
        let image = vk::Image::from_raw(ctx.vk_image);
        let queue = vk::Queue::from_raw(ctx.vk_queue);
        let done_fence = vk::Fence::from_raw(ctx.vk_done_fence);

        // Allocate a descriptor set
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(state.descriptor_pool)
            .set_layouts(std::slice::from_ref(&state.descriptor_set_layout));
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
        let descriptor_set = descriptor_sets[0];

        // Create an image view for the storage image binding
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::from_raw(ctx.vk_format as i32))
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let image_view = unsafe { device.create_image_view(&view_info, None)? };

        // Update descriptor set
        let image_write = vk::DescriptorImageInfo::default()
            .image_view(image_view)
            .image_layout(vk::ImageLayout::GENERAL);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&image_write));

        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };

        // Allocate command buffer
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(state.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = unsafe { device.allocate_command_buffers(&cmd_alloc)? };
        let cmd = cmd_bufs[0];

        // Record
        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, state.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                state.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            // Dispatch: ceil(width/16) x ceil(height/16)
            let gx = ctx.width.div_ceil(16);
            let gy = ctx.height.div_ceil(16);
            device.cmd_dispatch(cmd, gx, gy, 1);

            device.end_command_buffer(cmd)?;
        }

        // Submit and signal the done fence
        unsafe {
            let cmd_bufs_arr = [cmd];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs_arr);
            device.queue_submit(queue, &[submit_info], done_fence)?;
        }

        // Wait for completion so we can clean up transient resources.
        unsafe {
            device.wait_for_fences(&[done_fence], true, u64::MAX)?;
        }

        // Clean up transient resources
        unsafe {
            device.free_command_buffers(state.command_pool, &[cmd]);
            device.destroy_image_view(image_view, None);
            device.reset_descriptor_pool(
                state.descriptor_pool,
                vk::DescriptorPoolResetFlags::empty(),
            )?;
        }

        Ok(())
    }

    unsafe fn cleanup(state: &PipelineState) {
        unsafe {
            let d = &state.device;
            d.destroy_command_pool(state.command_pool, None);
            d.destroy_descriptor_pool(state.descriptor_pool, None);
            d.destroy_pipeline(state.pipeline, None);
            d.destroy_pipeline_layout(state.pipeline_layout, None);
            d.destroy_descriptor_set_layout(state.descriptor_set_layout, None);
        }
    }
}

extern "C" fn set_filter_enabled_ffi(_filter_id: RString, enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

extern "C" fn on_point_annotation_moved_ffi(
    _viewport: ViewportSnapshotFFI,
    _annotation_id: RString,
    _x_level0: f64,
    _y_level0: f64,
) {
}

extern "C" fn on_polygon_annotation_moved_ffi(
    _viewport: ViewportSnapshotFFI,
    _annotation_id: RString,
    _vertices: RVec<ViewportOverlayVertexFFI>,
) {
}

extern "C" fn on_undo_ffi() -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}

extern "C" fn on_redo_ffi() -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}

extern "C" fn on_viewport_annotation_selected_ffi(
    _viewport: ViewportSnapshotFFI,
    _annotation_id: RString,
) {
}

#[unsafe(no_mangle)]
pub extern "C" fn eov_get_plugin_vtable() -> PluginVTable {
    PluginVTable {
        set_host_api: set_host_api_ffi,
        get_toolbar_buttons: get_toolbar_buttons_ffi,
        get_hud_toolbar_buttons: get_hud_toolbar_buttons_ffi,
        on_action: on_action_ffi,
        on_hud_action: on_hud_action_ffi,
        on_ui_callback: on_ui_callback_ffi,
        get_sidebar_properties: get_sidebar_properties_ffi,
        get_viewport_context_menu_items: get_viewport_context_menu_items_ffi,
        on_viewport_context_menu_action: on_viewport_context_menu_action_ffi,
        get_viewport_overlay_points: get_viewport_overlay_points_ffi,
        get_viewport_overlay_polygons: get_viewport_overlay_polygons_ffi,
        get_viewport_overlay_component: get_viewport_overlay_component_ffi,
        get_viewport_overlay_properties: get_viewport_overlay_properties_ffi,
        on_point_annotation_placed: on_point_annotation_placed_ffi,
        on_polygon_annotation_placed: on_polygon_annotation_placed_ffi,
        on_undo: on_undo_ffi,
        on_redo: on_redo_ffi,
        on_point_annotation_moved: on_point_annotation_moved_ffi,
        on_polygon_annotation_moved: on_polygon_annotation_moved_ffi,
        on_viewport_annotation_selected: on_viewport_annotation_selected_ffi,
        get_viewport_filters: get_viewport_filters_ffi,
        apply_filter_cpu: apply_filter_cpu_ffi,
        apply_filter_gpu: apply_filter_gpu_ffi,
        set_filter_enabled: set_filter_enabled_ffi,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grayscale_conversion() {
        // RGBA pixel: red
        let mut buf = vec![255u8, 0, 0, 255];
        apply_filter_cpu_ffi(
            RString::from(FILTER_ID),
            buf.as_mut_ptr(),
            4,
            1,
            1,
            std::ptr::null(),
        );
        // Grayscale not applied because ENABLED is false by default
        assert_eq!(buf, [255, 0, 0, 255]);

        ENABLED.store(true, Ordering::Relaxed);
        apply_filter_cpu_ffi(
            RString::from(FILTER_ID),
            buf.as_mut_ptr(),
            4,
            1,
            1,
            std::ptr::null(),
        );
        // lum = 0.299 * 255 ≈ 76
        let lum = (0.299 * 255.0) as u8;
        assert_eq!(buf[0], lum);
        assert_eq!(buf[1], lum);
        assert_eq!(buf[2], lum);
        assert_eq!(buf[3], 255);
    }
}

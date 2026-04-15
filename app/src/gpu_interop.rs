//! Vulkan interop for DMA-BUF export and raw handle extraction.
//!
//! This module provides utilities for:
//! - Extracting raw Vulkan handles from wgpu objects via wgpu-hal
//! - Creating VkImages with external memory support (DMA-BUF exportable)
//! - Exporting VkDeviceMemory as DMA-BUF file descriptors
//! - Creating and exporting sync fences as fd
//!
//! These are used by the GPU viewport filter pipeline to share textures
//! with in-process FFI plugins (raw VkDevice/VkImage) and out-of-process
//! gRPC plugins (DMA-BUF fd over Unix socket).

use ash::vk;
use slint::wgpu_28::wgpu;

#[cfg(unix)]
use std::os::fd::RawFd;

#[cfg(not(unix))]
type RawFd = std::os::raw::c_int;

struct ImageLayoutTransition {
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
}

/// Raw Vulkan handles extracted from the wgpu runtime.
#[allow(dead_code)]
pub(crate) struct VulkanHandles {
    /// `ash::Instance` dispatch table (not owned — lifetime tied to wgpu).
    pub instance_fn: ash::Instance,
    /// `ash::Device` dispatch table (not owned — lifetime tied to wgpu).
    pub device_fn: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    /// KHR_external_memory_fd extension loader.
    pub external_memory_fd: ash::khr::external_memory_fd::Device,
    /// KHR_external_fence_fd extension loader.
    pub external_fence_fd: ash::khr::external_fence_fd::Device,
    /// Command pool for filter copy operations.
    pub command_pool: vk::CommandPool,
    /// Re-usable command buffer for copy operations.
    pub command_buffer: vk::CommandBuffer,
    /// Fence for synchronizing copy operations.
    pub copy_fence: vk::Fence,
}

/// A VkImage backed by dedicated, DMA-BUF-exportable memory.
#[allow(dead_code)]
pub(crate) struct ExportableImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub memory_size: u64,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    /// Stride in bytes (may be padded by the driver).
    pub stride: u32,
}

impl ExportableImage {
    /// Destroy the Vulkan resources. Must be called before the device is dropped.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Extract raw Vulkan handles from wgpu Device.
///
/// # Safety
/// The returned handles borrow the wgpu internals. The caller must ensure
/// the wgpu Device outlives the returned `VulkanHandles`.
pub(crate) unsafe fn extract_vulkan_handles(wgpu_device: &wgpu::Device) -> Option<VulkanHandles> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = wgpu_device;
        return None;
    }

    #[cfg(target_os = "linux")]
    {
    // Access the wgpu-hal Vulkan device.
    let hal_device_guard = unsafe { wgpu_device.as_hal::<wgpu::hal::api::Vulkan>()? };
    let raw_device: ash::Device = hal_device_guard.raw_device().clone();
    let physical_device = hal_device_guard.raw_physical_device();

    // The shared instance gives us the ash::Instance for extension loading.
    let shared_instance = hal_device_guard.shared_instance();
    let ash_instance = shared_instance.raw_instance();

    // Queue family index and raw VkQueue are both on the HAL device.
    let queue_family_index = hal_device_guard.queue_family_index();
    let raw_queue = hal_device_guard.raw_queue();

    let ash_instance_clone = ash_instance.clone();

    // Extension loaders for external memory/fence operations.
    let external_memory_fd = ash::khr::external_memory_fd::Device::new(ash_instance, &raw_device);
    let external_fence_fd = ash::khr::external_fence_fd::Device::new(ash_instance, &raw_device);

    // Create a command pool + buffer for filter copy operations.
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool = unsafe { raw_device.create_command_pool(&pool_info, None).ok()? };

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffers = unsafe { raw_device.allocate_command_buffers(&alloc_info).ok()? };
    let command_buffer = command_buffers[0];

    let fence_info = vk::FenceCreateInfo::default();
    let copy_fence = unsafe { raw_device.create_fence(&fence_info, None).ok()? };

    Some(VulkanHandles {
        instance_fn: ash_instance_clone,
        device_fn: raw_device,
        physical_device,
        queue: raw_queue,
        queue_family_index,
        external_memory_fd,
        external_fence_fd,
        command_pool,
        command_buffer,
        copy_fence,
    })
    }
}

/// Create a VkImage with dedicated, DMA-BUF-exportable memory.
///
/// The image is created with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`
/// so its backing memory can be exported as a DMA-BUF file descriptor.
#[allow(dead_code)]
pub(crate) unsafe fn create_exportable_image(
    handles: &VulkanHandles,
    width: u32,
    height: u32,
) -> Result<ExportableImage, vk::Result> {
    let format = vk::Format::R8G8B8A8_UNORM;

    // External memory image create info — declares the handle type.
    let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::LINEAR) // LINEAR for DMA-BUF mmap compatibility
        .usage(
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external_info);

    let image = unsafe { handles.device_fn.create_image(&image_info, None)? };

    // Query memory requirements.
    let mem_reqs = unsafe { handles.device_fn.get_image_memory_requirements(image) };

    // Query physical device memory properties to find a suitable type.
    let mem_props = unsafe {
        handles
            .instance_fn
            .get_physical_device_memory_properties(handles.physical_device)
    };

    // Use dedicated allocation with export info.
    let mut export_info = vk::ExportMemoryAllocateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(find_memory_type_index(
            &mem_props,
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ))
        .push_next(&mut export_info)
        .push_next(&mut dedicated_info);

    let memory = unsafe { handles.device_fn.allocate_memory(&alloc_info, None)? };

    unsafe {
        handles.device_fn.bind_image_memory(image, memory, 0)?;
    }

    // Query the stride (row pitch) for LINEAR tiling.
    let subresource = vk::ImageSubresource {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        array_layer: 0,
    };
    let layout = unsafe {
        handles
            .device_fn
            .get_image_subresource_layout(image, subresource)
    };

    Ok(ExportableImage {
        image,
        memory,
        memory_size: mem_reqs.size,
        width,
        height,
        format,
        stride: layout.row_pitch as u32,
    })
}

/// Export a VkDeviceMemory as a DMA-BUF file descriptor.
#[allow(dead_code)]
pub(crate) unsafe fn export_dma_buf_fd(
    handles: &VulkanHandles,
    memory: vk::DeviceMemory,
) -> Result<RawFd, vk::Result> {
    let get_fd_info = vk::MemoryGetFdInfoKHR::default()
        .memory(memory)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    unsafe { handles.external_memory_fd.get_memory_fd(&get_fd_info) }
}

/// Create a VkFence and export it as a sync file descriptor.
#[allow(dead_code)]
pub(crate) unsafe fn create_and_export_fence(
    handles: &VulkanHandles,
) -> Result<(vk::Fence, RawFd), vk::Result> {
    let mut export_info = vk::ExportFenceCreateInfo::default()
        .handle_types(vk::ExternalFenceHandleTypeFlags::SYNC_FD);

    let fence_info = vk::FenceCreateInfo::default().push_next(&mut export_info);

    let fence = unsafe { handles.device_fn.create_fence(&fence_info, None)? };

    // The sync fd can only be exported after the fence is signaled.
    // For now return the fence; the fd will be exported after signaling.
    Ok((fence, -1))
}

/// Export a signaled fence as a sync fd.
#[allow(dead_code)]
pub(crate) unsafe fn export_fence_fd(
    handles: &VulkanHandles,
    fence: vk::Fence,
) -> Result<RawFd, vk::Result> {
    let get_fd_info = vk::FenceGetFdInfoKHR::default()
        .fence(fence)
        .handle_type(vk::ExternalFenceHandleTypeFlags::SYNC_FD);

    unsafe { handles.external_fence_fd.get_fence_fd(&get_fd_info) }
}

/// Import a sync fd as a VkFence (for waiting on plugin completion).
#[allow(dead_code)]
pub(crate) unsafe fn import_fence_from_fd(
    handles: &VulkanHandles,
    fd: RawFd,
) -> Result<vk::Fence, vk::Result> {
    let fence_info = vk::FenceCreateInfo::default();
    let fence = unsafe { handles.device_fn.create_fence(&fence_info, None)? };

    let import_info = vk::ImportFenceFdInfoKHR::default()
        .fence(fence)
        .handle_type(vk::ExternalFenceHandleTypeFlags::SYNC_FD)
        .flags(vk::FenceImportFlags::TEMPORARY)
        .fd(fd);

    unsafe {
        handles.external_fence_fd.import_fence_fd(&import_info)?;
    }

    Ok(fence)
}

/// Transition image layout using a pipeline barrier.
unsafe fn transition_image_layout(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    transition: ImageLayoutTransition,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(transition.old_layout)
        .new_layout(transition.new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(transition.image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(transition.src_access)
        .dst_access_mask(transition.dst_access);

    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            transition.src_stage,
            transition.dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

/// Find a suitable memory type index for the given type bits and desired flags.
/// Prefers the first type matching both bits and flags; falls back to first matching type.
fn find_memory_type_index(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    desired: vk::MemoryPropertyFlags,
) -> u32 {
    // First pass: find a type that satisfies both bits and desired flags.
    for i in 0..mem_props.memory_type_count {
        if (type_bits >> i) & 1 == 1
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(desired)
        {
            return i;
        }
    }
    // Fallback: first type that satisfies the bits.
    for i in 0..mem_props.memory_type_count {
        if (type_bits >> i) & 1 == 1 {
            return i;
        }
    }
    0
}

/// Copy a wgpu surface texture to the filter image (pre-filter) using raw
/// Vulkan commands. The wgpu queue must be idle before calling this.
///
/// After this call, `filter_image` is in `GENERAL` layout, ready for plugin access.
///
/// # Safety
/// - `surface_vk_image` must be the raw VkImage backing the wgpu surface texture.
/// - The wgpu device/queue must be idle (call `device.poll(Wait)` first).
/// - `handles.command_buffer` must not be in a recording state.
#[allow(dead_code)]
pub(crate) unsafe fn copy_surface_to_filter(
    handles: &VulkanHandles,
    surface_vk_image: vk::Image,
    filter_image: &ExportableImage,
) -> Result<(), vk::Result> {
    let device = &handles.device_fn;
    let cmd = handles.command_buffer;

    unsafe {
        device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        // Transition surface image: SHADER_READ_ONLY → TRANSFER_SRC
        transition_image_layout(
            device,
            cmd,
            ImageLayoutTransition {
                image: surface_vk_image,
                old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_access: vk::AccessFlags::SHADER_READ,
                dst_access: vk::AccessFlags::TRANSFER_READ,
                src_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                dst_stage: vk::PipelineStageFlags::TRANSFER,
            },
        );

        // Transition filter image: UNDEFINED → TRANSFER_DST
        transition_image_layout(
            device,
            cmd,
            ImageLayoutTransition {
                image: filter_image.image,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_access: vk::AccessFlags::empty(),
                dst_access: vk::AccessFlags::TRANSFER_WRITE,
                src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                dst_stage: vk::PipelineStageFlags::TRANSFER,
            },
        );

        // Copy surface → filter
        let region = vk::ImageCopy {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            extent: vk::Extent3D {
                width: filter_image.width,
                height: filter_image.height,
                depth: 1,
            },
        };
        device.cmd_copy_image(
            cmd,
            surface_vk_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            filter_image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        // Transition filter image: TRANSFER_DST → GENERAL (for plugin access)
        transition_image_layout(
            device,
            cmd,
            ImageLayoutTransition {
                image: filter_image.image,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_access: vk::AccessFlags::TRANSFER_WRITE,
                dst_access: vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                src_stage: vk::PipelineStageFlags::TRANSFER,
                dst_stage: vk::PipelineStageFlags::ALL_COMMANDS,
            },
        );

        // Transition surface back: TRANSFER_SRC → SHADER_READ_ONLY
        transition_image_layout(
            device,
            cmd,
            ImageLayoutTransition {
                image: surface_vk_image,
                old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_access: vk::AccessFlags::TRANSFER_READ,
                dst_access: vk::AccessFlags::SHADER_READ,
                src_stage: vk::PipelineStageFlags::TRANSFER,
                dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
            },
        );

        device.end_command_buffer(cmd)?;

        // Submit and wait
        device.reset_fences(&[handles.copy_fence])?;
        let cmd_bufs = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
        device.queue_submit(handles.queue, &[submit_info], handles.copy_fence)?;
        device.wait_for_fences(&[handles.copy_fence], true, u64::MAX)?;
    }

    Ok(())
}

/// Copy the filter image back to the wgpu surface texture (post-filter).
/// The wgpu queue must be idle before calling this.
///
/// # Safety
/// - The filter image must be in `GENERAL` layout.
/// - The wgpu device/queue must be idle.
#[allow(dead_code)]
pub(crate) unsafe fn copy_filter_to_surface(
    handles: &VulkanHandles,
    filter_image: &ExportableImage,
    surface_vk_image: vk::Image,
) -> Result<(), vk::Result> {
    let device = &handles.device_fn;
    let cmd = handles.command_buffer;

    unsafe {
        device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        // Transition filter image: GENERAL → TRANSFER_SRC
        transition_image_layout(
            device,
            cmd,
            ImageLayoutTransition {
                image: filter_image.image,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_access: vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                dst_access: vk::AccessFlags::TRANSFER_READ,
                src_stage: vk::PipelineStageFlags::ALL_COMMANDS,
                dst_stage: vk::PipelineStageFlags::TRANSFER,
            },
        );

        // Transition surface: SHADER_READ_ONLY → TRANSFER_DST
        transition_image_layout(
            device,
            cmd,
            ImageLayoutTransition {
                image: surface_vk_image,
                old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_access: vk::AccessFlags::SHADER_READ,
                dst_access: vk::AccessFlags::TRANSFER_WRITE,
                src_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                dst_stage: vk::PipelineStageFlags::TRANSFER,
            },
        );

        // Copy filter → surface
        let region = vk::ImageCopy {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            extent: vk::Extent3D {
                width: filter_image.width,
                height: filter_image.height,
                depth: 1,
            },
        };
        device.cmd_copy_image(
            cmd,
            filter_image.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            surface_vk_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        // Transition surface back: TRANSFER_DST → SHADER_READ_ONLY
        transition_image_layout(
            device,
            cmd,
            ImageLayoutTransition {
                image: surface_vk_image,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_access: vk::AccessFlags::TRANSFER_WRITE,
                dst_access: vk::AccessFlags::SHADER_READ,
                src_stage: vk::PipelineStageFlags::TRANSFER,
                dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
            },
        );

        device.end_command_buffer(cmd)?;

        // Submit and wait
        device.reset_fences(&[handles.copy_fence])?;
        let cmd_bufs = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
        device.queue_submit(handles.queue, &[submit_info], handles.copy_fence)?;
        device.wait_for_fences(&[handles.copy_fence], true, u64::MAX)?;
    }

    Ok(())
}

/// Get the raw VkImage handle from a wgpu Texture.
///
/// # Safety
/// The wgpu Texture must be backed by a Vulkan image.
#[allow(dead_code)]
pub(crate) unsafe fn get_surface_vk_image(texture: &wgpu::Texture) -> Option<vk::Image> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = texture;
        return None;
    }

    #[cfg(target_os = "linux")]
    {
    let hal_texture = unsafe { texture.as_hal::<wgpu::hal::api::Vulkan>()? };
    Some(unsafe { hal_texture.raw_handle() })
    }
}

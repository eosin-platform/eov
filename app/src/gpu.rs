use crate::AppWindow;
use crate::state::FilteringMode;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use common::{TileCoord, TileData};
use slint::ComponentHandle;
use slint::wgpu_28::wgpu;
use slint::{GraphicsAPI, Image, RenderingState};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::rc::Rc;
use std::sync::Arc;

const MAX_GPU_TILE_TEXTURES: usize = 4096;

const SHADER_SOURCE: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) fine_uv: vec2<f32>,
    @location(2) coarse_uv: vec2<f32>,
    @location(3) mip_blend: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) fine_uv: vec2<f32>,
    @location(1) coarse_uv: vec2<f32>,
    @location(2) mip_blend: f32,
};

struct Adjustments {
    gamma: f32,
    brightness: f32,
    contrast: f32,
    stain_norm_enabled: f32,
    inv_stain_row0: vec4<f32>,
    inv_stain_row1: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.fine_uv = input.fine_uv;
    output.coarse_uv = input.coarse_uv;
    output.mip_blend = input.mip_blend;
    return output;
}

@group(0) @binding(0)
var fine_texture: texture_2d<f32>;

@group(0) @binding(1)
var coarse_texture: texture_2d<f32>;

@group(0) @binding(2)
var tile_sampler: sampler;

@group(0) @binding(3)
var<uniform> adjustments: Adjustments;

fn apply_stain_norm(color: vec4<f32>) -> vec4<f32> {
    if adjustments.stain_norm_enabled < 0.5 {
        return color;
    }
    let rgb = max(color.rgb, vec3<f32>(1.0 / 255.0));
    let od = -log(rgb);
    let od_sum = od.r + od.g + od.b;
    if od_sum <= 0.15 || od_sum >= 6.0 {
        return color;
    }
    let c0 = max(dot(adjustments.inv_stain_row0.xyz, od), 0.0);
    let c1 = max(dot(adjustments.inv_stain_row1.xyz, od), 0.0);
    let nc0 = c0 * adjustments.inv_stain_row0.w;
    let nc1 = c1 * adjustments.inv_stain_row1.w;
    let ref_h = vec3<f32>(0.6442, 0.7170, 0.2668);
    let ref_e = vec3<f32>(0.0927, 0.9545, 0.2832);
    let new_od = ref_h * nc0 + ref_e * nc1;
    let new_rgb = exp(-new_od);
    return vec4<f32>(clamp(new_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), color.a);
}

fn apply_adj(color: vec4<f32>) -> vec4<f32> {
    let inv_gamma = 1.0 / max(adjustments.gamma, 0.001);
    let g = vec3<f32>(pow(color.r, inv_gamma), pow(color.g, inv_gamma), pow(color.b, inv_gamma));
    let b = g + vec3<f32>(adjustments.brightness);
    let c = (b - vec3<f32>(0.5)) * adjustments.contrast + vec3<f32>(0.5);
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), color.a);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let fine = textureSample(fine_texture, tile_sampler, input.fine_uv);
    let coarse = textureSample(coarse_texture, tile_sampler, input.coarse_uv);
    return apply_adj(apply_stain_norm(mix(fine, coarse, input.mip_blend)));
}
"#;

const MIPGEN_SHADER_SOURCE: &str = r#"
struct MipgenVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_mipgen(@builtin(vertex_index) vi: u32) -> MipgenVertexOutput {
    var out: MipgenVertexOutput;
    let tc = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.position = vec4<f32>(tc.x * 2.0 - 1.0, 1.0 - tc.y * 2.0, 0.0, 1.0);
    out.uv = tc;
    return out;
}

@group(0) @binding(0)
var src_texture: texture_2d<f32>;

@group(0) @binding(1)
var src_sampler: sampler;

@fragment
fn fs_mipgen(input: MipgenVertexOutput) -> @location(0) vec4<f32> {
    return textureSample(src_texture, src_sampler, input.uv);
}
"#;

/// Lanczos-3 fragment shader: samples a 6x6 kernel from the fine texture
const LANCZOS_SHADER_SOURCE: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) fine_uv: vec2<f32>,
    @location(2) coarse_uv: vec2<f32>,
    @location(3) mip_blend: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) fine_uv: vec2<f32>,
    @location(1) coarse_uv: vec2<f32>,
    @location(2) mip_blend: f32,
};

struct Adjustments {
    gamma: f32,
    brightness: f32,
    contrast: f32,
    stain_norm_enabled: f32,
    inv_stain_row0: vec4<f32>,
    inv_stain_row1: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.fine_uv = input.fine_uv;
    output.coarse_uv = input.coarse_uv;
    output.mip_blend = input.mip_blend;
    return output;
}

@group(0) @binding(0)
var fine_texture: texture_2d<f32>;

@group(0) @binding(1)
var coarse_texture: texture_2d<f32>;

@group(0) @binding(2)
var tile_sampler: sampler;

@group(0) @binding(3)
var<uniform> adjustments: Adjustments;

const PI: f32 = 3.14159265358979323846;
const LANCZOS_A: f32 = 3.0;

fn sinc(x: f32) -> f32 {
    if abs(x) < 1e-6 {
        return 1.0;
    }
    let px = PI * x;
    return sin(px) / px;
}

fn lanczos_weight(x: f32) -> f32 {
    if abs(x) >= LANCZOS_A {
        return 0.0;
    }
    return sinc(x) * sinc(x / LANCZOS_A);
}

fn apply_stain_norm(color: vec4<f32>) -> vec4<f32> {
    if adjustments.stain_norm_enabled < 0.5 {
        return color;
    }
    let rgb = max(color.rgb, vec3<f32>(1.0 / 255.0));
    let od = -log(rgb);
    let od_sum = od.r + od.g + od.b;
    if od_sum <= 0.15 || od_sum >= 6.0 {
        return color;
    }
    let c0 = max(dot(adjustments.inv_stain_row0.xyz, od), 0.0);
    let c1 = max(dot(adjustments.inv_stain_row1.xyz, od), 0.0);
    let nc0 = c0 * adjustments.inv_stain_row0.w;
    let nc1 = c1 * adjustments.inv_stain_row1.w;
    let ref_h = vec3<f32>(0.6442, 0.7170, 0.2668);
    let ref_e = vec3<f32>(0.0927, 0.9545, 0.2832);
    let new_od = ref_h * nc0 + ref_e * nc1;
    let new_rgb = exp(-new_od);
    return vec4<f32>(clamp(new_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), color.a);
}

fn apply_adj(color: vec4<f32>) -> vec4<f32> {
    let inv_gamma = 1.0 / max(adjustments.gamma, 0.001);
    let g = vec3<f32>(pow(color.r, inv_gamma), pow(color.g, inv_gamma), pow(color.b, inv_gamma));
    let b = g + vec3<f32>(adjustments.brightness);
    let c = (b - vec3<f32>(0.5)) * adjustments.contrast + vec3<f32>(0.5);
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), color.a);
}

@fragment
fn fs_lanczos(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(fine_texture, 0));
    let uv_pixel = input.fine_uv * tex_size - 0.5;
    let center = floor(uv_pixel);
    let fract_part = uv_pixel - center;

    var color = vec4<f32>(0.0);
    var weight_sum: f32 = 0.0;

    for (var j: i32 = -2; j <= 3; j++) {
        for (var i: i32 = -2; i <= 3; i++) {
            let offset = vec2<f32>(f32(i), f32(j));
            let sample_pos = center + offset;
            let sample_uv = (sample_pos + 0.5) / tex_size;

            let wx = lanczos_weight(f32(i) - fract_part.x);
            let wy = lanczos_weight(f32(j) - fract_part.y);
            let w = wx * wy;

            let sample = textureSampleLevel(fine_texture, tile_sampler, sample_uv, 0.0);
            color += sample * w;
            weight_sum += w;
        }
    }

    if weight_sum > 0.0 {
        color = color / weight_sum;
    }

    return apply_adj(apply_stain_norm(clamp(color, vec4<f32>(0.0), vec4<f32>(1.0))));
}
"#;

#[derive(Clone)]
pub struct TileDraw {
    pub tile: Arc<TileData>,
    pub coarse_tile: Option<Arc<TileData>>,
    pub screen_x: i32,
    pub screen_y: i32,
    pub screen_w: i32,
    pub screen_h: i32,
    pub coarse_uv_min: [f32; 2],
    pub coarse_uv_max: [f32; 2],
    pub mip_blend: f32,
    pub filtering_mode: FilteringMode,
}

#[derive(Clone, Copy)]
pub struct SurfaceSlot(pub usize);

impl SurfaceSlot {
    pub const PRIMARY: Self = Self(0);
    pub const SECONDARY: Self = Self(1);

    fn index(self) -> usize {
        self.0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdjustmentsUniform {
    gamma: f32,
    brightness: f32,
    contrast: f32,
    stain_norm_enabled: f32,
    // Row 0 of inverse source stain matrix + scale_h
    inv_stain_r0: [f32; 4], // [inv[0][0], inv[0][1], inv[0][2], scale_h]
    // Row 1 of inverse source stain matrix + scale_e
    inv_stain_r1: [f32; 4], // [inv[1][0], inv[1][1], inv[1][2], scale_e]
}

#[derive(Clone)]
pub(crate) struct QueuedFrame {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) draws: Vec<TileDraw>,
    pub(crate) gamma: f32,
    pub(crate) brightness: f32,
    pub(crate) contrast: f32,
    pub(crate) stain_norm_enabled: bool,
    pub(crate) inv_stain_r0: [f32; 4],
    pub(crate) inv_stain_r1: [f32; 4],
}

#[derive(Clone)]
struct ImportedSurface {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

#[derive(Clone)]
struct TileTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    mip_levels: u32,
    last_used_frame: u64,
}

struct GpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    lanczos_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    bilinear_sampler: wgpu::Sampler,
    mipgen_pipeline: wgpu::RenderPipeline,
    mipgen_bind_group_layout: wgpu::BindGroupLayout,
    mipgen_sampler: wgpu::Sampler,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    adjustments_buffer: wgpu::Buffer,
    surfaces: HashMap<usize, ImportedSurface>,
}

/// Key for cached bind groups — identifies the (fine tile, coarse tile, sampler) combination.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BindGroupKey {
    fine_coord: TileCoord,
    coarse_coord: TileCoord,
    bilinear: bool,
}

pub struct GpuRenderer {
    runtime: Option<GpuRuntime>,
    pending_frames: HashMap<usize, QueuedFrame>,
    tile_textures: HashMap<TileCoord, TileTexture>,
    bind_group_cache: HashMap<BindGroupKey, wgpu::BindGroup>,
    frame_counter: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    fine_uv: [f32; 2],
    coarse_uv: [f32; 2],
    mip_blend: f32,
}

impl GpuRenderer {
    pub fn new() -> Self {
        Self {
            runtime: None,
            pending_frames: HashMap::new(),
            tile_textures: HashMap::new(),
            bind_group_cache: HashMap::new(),
            frame_counter: 0,
        }
    }

    pub fn install(ui: &AppWindow, renderer: Rc<RefCell<Self>>) -> Result<()> {
        ui.window()
            .set_rendering_notifier(move |state, graphics_api| {
                let mut renderer = renderer.borrow_mut();
                match state {
                    RenderingState::RenderingSetup => renderer.initialize(graphics_api),
                    RenderingState::BeforeRendering => renderer.flush_pending_frames(),
                    RenderingState::RenderingTeardown => renderer.teardown(),
                    RenderingState::AfterRendering => {}
                    _ => {}
                }
            })?;
        Ok(())
    }

    pub fn queue_frame(&mut self, slot: SurfaceSlot, frame: QueuedFrame) -> Option<Image> {
        if frame.width == 0 || frame.height == 0 {
            return None;
        }

        self.runtime.as_ref()?;

        let surface_recreated = self.ensure_surface(slot, frame.width, frame.height)?;
        self.pending_frames.insert(slot.index(), frame);

        if surface_recreated {
            None
        } else {
            self.surface_image(slot)
        }
    }

    fn initialize(&mut self, graphics_api: &GraphicsAPI) {
        if self.runtime.is_some() {
            return;
        }

        let GraphicsAPI::WGPU28 { device, queue, .. } = graphics_api else {
            return;
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-tile-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport-tile-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewport-tile-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            anisotropy_clamp: 1,
            ..Default::default()
        });

        // Bilinear-only sampler (no mipmap blending)
        let bilinear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewport-tile-bilinear-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            anisotropy_clamp: 1,
            ..Default::default()
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport-tile-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-tile-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x2, 3 => Float32],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport-tile-vertex-buffer"),
            size: std::mem::size_of::<Vertex>() as u64 * 6,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let adjustments_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport-adjustments-buffer"),
            size: std::mem::size_of::<AdjustmentsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Lanczos pipeline: same bind group layout, different fragment shader
        let lanczos_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lanczos-shader"),
            source: wgpu::ShaderSource::Wgsl(LANCZOS_SHADER_SOURCE.into()),
        });

        let lanczos_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-lanczos-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &lanczos_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x2, 3 => Float32],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &lanczos_shader,
                entry_point: Some("fs_lanczos"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let mipgen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mipgen-shader"),
            source: wgpu::ShaderSource::Wgsl(MIPGEN_SHADER_SOURCE.into()),
        });

        let mipgen_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mipgen-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let mipgen_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mipgen-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let mipgen_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mipgen-pipeline-layout"),
                bind_group_layouts: &[&mipgen_bind_group_layout],
                immediate_size: 0,
            });

        let mipgen_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mipgen-pipeline"),
            layout: Some(&mipgen_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mipgen_shader,
                entry_point: Some("vs_mipgen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &mipgen_shader,
                entry_point: Some("fs_mipgen"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        self.runtime = Some(GpuRuntime {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            lanczos_pipeline,
            bind_group_layout,
            sampler,
            bilinear_sampler,
            mipgen_pipeline,
            mipgen_bind_group_layout,
            mipgen_sampler,
            vertex_buffer,
            vertex_capacity: 6,
            adjustments_buffer,
            surfaces: HashMap::new(),
        });
    }

    fn teardown(&mut self) {
        self.runtime = None;
        self.pending_frames.clear();
        self.tile_textures.clear();
        self.bind_group_cache.clear();
    }

    fn ensure_surface(&mut self, slot: SurfaceSlot, width: u32, height: u32) -> Option<bool> {
        let runtime = self.runtime.as_mut()?;

        let slot_index = slot.index();
        let needs_recreate = runtime
            .surfaces
            .get(&slot_index)
            .as_ref()
            .is_none_or(|surface| surface.width != width || surface.height != height);

        if !needs_recreate {
            return Some(false);
        }

        let texture = runtime.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pane-viewport-surface"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        runtime.surfaces.insert(
            slot_index,
            ImportedSurface {
                texture,
                view,
                width,
                height,
            },
        );

        Some(true)
    }

    pub fn surface_image(&self, slot: SurfaceSlot) -> Option<Image> {
        self.runtime
            .as_ref()
            .and_then(|runtime| runtime.surfaces.get(&slot.index()))
            .and_then(|surface| Image::try_from(surface.texture.clone()).ok())
    }

    fn flush_pending_frames(&mut self) {
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };

        self.frame_counter = self.frame_counter.wrapping_add(1);
        let frame_id = self.frame_counter;

        let pending_frames = std::mem::take(&mut self.pending_frames);
        for (slot_index, frame) in pending_frames {
            if runtime.surfaces.contains_key(&slot_index) {
                Self::render_frame(
                    runtime,
                    &mut self.tile_textures,
                    &mut self.bind_group_cache,
                    SurfaceSlot(slot_index),
                    frame,
                    frame_id,
                );
            }
        }

        if self.tile_textures.len() > MAX_GPU_TILE_TEXTURES {
            let mut entries: Vec<_> = self
                .tile_textures
                .iter()
                .map(|(coord, tex)| (*coord, tex.last_used_frame))
                .collect();
            entries.sort_by_key(|(_, last_used_frame)| *last_used_frame);
            let remove_count = self.tile_textures.len() - MAX_GPU_TILE_TEXTURES;
            let evicted: HashSet<TileCoord> = entries
                .into_iter()
                .take(remove_count)
                .map(|(coord, _)| coord)
                .collect();
            for coord in &evicted {
                self.tile_textures.remove(coord);
            }
            // Purge bind groups that reference evicted tile textures
            self.bind_group_cache.retain(|key, _| {
                !evicted.contains(&key.fine_coord) && !evicted.contains(&key.coarse_coord)
            });
        }
    }

    fn render_frame(
        runtime: &mut GpuRuntime,
        tile_textures: &mut HashMap<TileCoord, TileTexture>,
        bind_group_cache: &mut HashMap<BindGroupKey, wgpu::BindGroup>,
        slot: SurfaceSlot,
        frame: QueuedFrame,
        frame_id: u64,
    ) {
        let Some(surface_view) = runtime
            .surfaces
            .get(&slot.index())
            .map(|surface| surface.view.clone())
        else {
            return;
        };

        // Phase 1: Ensure all tile textures exist, collecting newly created ones
        let mut needs_mipgen: Vec<TileCoord> = Vec::new();
        for draw in &frame.draws {
            if ensure_tile_texture(runtime, tile_textures, &draw.tile, frame_id) {
                needs_mipgen.push(draw.tile.coord);
            }
            if let Some(coarse) = &draw.coarse_tile
                && ensure_tile_texture(runtime, tile_textures, coarse, frame_id)
            {
                needs_mipgen.push(coarse.coord);
            }
        }

        let vertex_count = frame.draws.len().max(1) * 6;
        if runtime.vertex_capacity < vertex_count {
            runtime.vertex_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("viewport-tile-vertex-buffer"),
                size: std::mem::size_of::<Vertex>() as u64 * vertex_count as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            runtime.vertex_capacity = vertex_count;
        }

        let mut vertices = Vec::with_capacity(frame.draws.len() * 6);
        for draw in &frame.draws {
            vertices.extend_from_slice(&quad_vertices(draw, frame.width, frame.height));
        }

        if !vertices.is_empty() {
            runtime
                .queue
                .write_buffer(&runtime.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }

        // Upload adjustments uniform
        let adj = AdjustmentsUniform {
            gamma: frame.gamma,
            brightness: frame.brightness,
            contrast: frame.contrast,
            stain_norm_enabled: if frame.stain_norm_enabled { 1.0 } else { 0.0 },
            inv_stain_r0: frame.inv_stain_r0,
            inv_stain_r1: frame.inv_stain_r1,
        };
        runtime
            .queue
            .write_buffer(&runtime.adjustments_buffer, 0, bytemuck::bytes_of(&adj));

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("viewport-tile-encoder"),
            });

        // Phase 2: Generate mipmaps for newly created tile textures (async on GPU)
        for coord in &needs_mipgen {
            if let Some(tile_tex) = tile_textures.get(coord)
                && tile_tex.mip_levels > 1
            {
                generate_mipmaps(
                    &mut encoder,
                    runtime,
                    &tile_tex.texture,
                    tile_tex.mip_levels,
                );
            }
        }

        // Phase 3: Main render pass (executes after mipmap generation on GPU)
        // Determine filtering mode from the first draw (all draws share the same mode)
        let filtering_mode = frame
            .draws
            .first()
            .map(|d| d.filtering_mode)
            .unwrap_or(FilteringMode::Bilinear);
        let use_lanczos = matches!(filtering_mode, FilteringMode::Lanczos3);
        let is_bilinear = filtering_mode == FilteringMode::Bilinear;
        let active_sampler = if is_bilinear {
            &runtime.bilinear_sampler
        } else {
            &runtime.sampler
        };
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport-tile-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 30.0 / 255.0,
                            g: 30.0 / 255.0,
                            b: 30.0 / 255.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            if use_lanczos {
                render_pass.set_pipeline(&runtime.lanczos_pipeline);
            } else {
                render_pass.set_pipeline(&runtime.pipeline);
            }
            render_pass.set_vertex_buffer(0, runtime.vertex_buffer.slice(..));

            for (index, draw) in frame.draws.iter().enumerate() {
                let fine_coord = draw.tile.coord;
                let coarse_coord = draw
                    .coarse_tile
                    .as_ref()
                    .map(|t| t.coord)
                    .unwrap_or(fine_coord);

                if !tile_textures.contains_key(&fine_coord) {
                    continue;
                }

                let cache_key = BindGroupKey {
                    fine_coord,
                    coarse_coord,
                    bilinear: is_bilinear,
                };

                let bind_group = bind_group_cache.entry(cache_key).or_insert_with(|| {
                    let fine_view = &tile_textures[&fine_coord].view;
                    let coarse_view = tile_textures
                        .get(&coarse_coord)
                        .map(|t| &t.view)
                        .unwrap_or(fine_view);
                    runtime
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("viewport-tile-bind-group"),
                            layout: &runtime.bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(fine_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(coarse_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(active_sampler),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: runtime.adjustments_buffer.as_entire_binding(),
                                },
                            ],
                        })
                });
                render_pass.set_bind_group(0, &*bind_group, &[]);
                let start = (index * 6) as u32;
                render_pass.draw(start..start + 6, 0..1);
            }
        }

        runtime.queue.submit(Some(encoder.finish()));
    }
}

fn mip_level_count(width: u32, height: u32) -> u32 {
    (width.max(height).max(1) as f32).log2().floor() as u32 + 1
}

/// Ensure a tile texture exists. Returns `true` if the texture was newly created
/// and needs mipmap generation.
fn ensure_tile_texture(
    runtime: &GpuRuntime,
    tile_textures: &mut HashMap<TileCoord, TileTexture>,
    tile: &Arc<TileData>,
    frame_id: u64,
) -> bool {
    let mut is_new = false;
    let entry = tile_textures.entry(tile.coord).or_insert_with(|| {
        is_new = true;
        let width = tile.width.max(1);
        let height = tile.height.max(1);
        let mip_levels = mip_level_count(width, height);

        let texture = runtime.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport-tile-texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        runtime.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &tile.data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        TileTexture {
            texture,
            view,
            mip_levels,
            last_used_frame: frame_id,
        }
    });

    entry.last_used_frame = frame_id;
    is_new
}

fn generate_mipmaps(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &GpuRuntime,
    texture: &wgpu::Texture,
    mip_levels: u32,
) {
    for level in 1..mip_levels {
        let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: level - 1,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: level,
            mip_level_count: Some(1),
            ..Default::default()
        });

        let bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mipgen-bind-group"),
                layout: &runtime.mipgen_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&runtime.mipgen_sampler),
                    },
                ],
            });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mipgen-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dst_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&runtime.mipgen_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn quad_vertices(draw: &TileDraw, width: u32, height: u32) -> [Vertex; 6] {
    let width = width.max(1) as f32;
    let height = height.max(1) as f32;
    let x0 = draw.screen_x as f32;
    let y0 = draw.screen_y as f32;
    let x1 = (draw.screen_x + draw.screen_w) as f32;
    let y1 = (draw.screen_y + draw.screen_h) as f32;

    let left = x0 / width * 2.0 - 1.0;
    let right = x1 / width * 2.0 - 1.0;
    let top = 1.0 - y0 / height * 2.0;
    let bottom = 1.0 - y1 / height * 2.0;
    let coarse_min = draw.coarse_uv_min;
    let coarse_max = draw.coarse_uv_max;
    let mip_blend = draw.mip_blend;

    [
        Vertex {
            position: [left, top],
            fine_uv: [0.0, 0.0],
            coarse_uv: [coarse_min[0], coarse_min[1]],
            mip_blend,
        },
        Vertex {
            position: [right, top],
            fine_uv: [1.0, 0.0],
            coarse_uv: [coarse_max[0], coarse_min[1]],
            mip_blend,
        },
        Vertex {
            position: [left, bottom],
            fine_uv: [0.0, 1.0],
            coarse_uv: [coarse_min[0], coarse_max[1]],
            mip_blend,
        },
        Vertex {
            position: [left, bottom],
            fine_uv: [0.0, 1.0],
            coarse_uv: [coarse_min[0], coarse_max[1]],
            mip_blend,
        },
        Vertex {
            position: [right, top],
            fine_uv: [1.0, 0.0],
            coarse_uv: [coarse_max[0], coarse_min[1]],
            mip_blend,
        },
        Vertex {
            position: [right, bottom],
            fine_uv: [1.0, 1.0],
            coarse_uv: [coarse_max[0], coarse_max[1]],
            mip_blend,
        },
    ]
}

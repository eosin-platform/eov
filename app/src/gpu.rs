use crate::AppWindow;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use eov_common::processing::{GpuColorUniform, wgsl};
use eov_common::{FilteringMode, TileCoord, TileData};
use slint::ComponentHandle;
use slint::wgpu_29::wgpu;
use slint::{GraphicsAPI, Image, RenderingState};
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Arc;

/// Memory budget for the tile texture array (~128 MB).
const MAX_TILE_ARRAY_BYTES: u64 = 128 * 1024 * 1024;

/// Colour of the viewport outside the slide (the same as the CPU path's).
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: crate::color::VIEWPORT_BACKGROUND[0] as f64 / 255.0,
    g: crate::color::VIEWPORT_BACKGROUND[1] as f64 / 255.0,
    b: crate::color::VIEWPORT_BACKGROUND[2] as f64 / 255.0,
    a: 1.0,
};

// ---------------------------------------------------------------------------
// Shaders
//
// The colour stages (stain transform, tone) come verbatim from
// `eov_common::processing::wgsl`, the reference that the CPU pipeline is
// tested against, so the two backends cannot drift apart. Sharpening (stage
// 6) runs as a separate full-screen pass over the composited frame
// (`wgsl::SHARPEN_POST_PASS`), in screen space as on the CPU.
// ---------------------------------------------------------------------------

/// Vertex stage and bindings shared by the tile shaders. The fragment
/// stages use `color_params`, declared by `wgsl::COLOR_PARAMS_STRUCT`.
const TILE_STAGE_WGSL: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) fine_uv: vec2<f32>,
    @location(2) coarse_uv: vec2<f32>,
    @location(3) mip_blend: f32,
    @location(4) fine_layer: i32,
    @location(5) coarse_layer: i32,
    @location(6) fine_tex_size: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) fine_uv: vec2<f32>,
    @location(1) coarse_uv: vec2<f32>,
    @location(2) mip_blend: f32,
    @location(3) @interpolate(flat) fine_layer: i32,
    @location(4) @interpolate(flat) coarse_layer: i32,
    @location(5) fine_tex_size: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.fine_uv = input.fine_uv;
    output.coarse_uv = input.coarse_uv;
    output.mip_blend = input.mip_blend;
    output.fine_layer = input.fine_layer;
    output.coarse_layer = input.coarse_layer;
    output.fine_tex_size = input.fine_tex_size;
    return output;
}

@group(0) @binding(0)
var tile_array: texture_2d_array<f32>;

@group(0) @binding(1)
var tile_sampler: sampler;

@group(0) @binding(2)
var<uniform> color_params: EovColorParams;
"#;

/// Bilinear / trilinear tile sampling.
const BILINEAR_FRAGMENT_WGSL: &str = r#"
fn sample_tile(
    fine_uv: vec2<f32>,
    coarse_uv: vec2<f32>,
    fine_layer: i32,
    coarse_layer: i32,
    mip_blend: f32,
) -> vec4<f32> {
    let fine = textureSampleLevel(tile_array, tile_sampler, fine_uv, fine_layer, 0.0);
    let coarse = textureSampleLevel(tile_array, tile_sampler, coarse_uv, coarse_layer, 0.0);
    return mix(fine, coarse, mip_blend);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = sample_tile(input.fine_uv, input.coarse_uv, input.fine_layer, input.coarse_layer, input.mip_blend);
    return vec4<f32>(eov_color(texel.rgb), texel.a);
}
"#;

/// Lanczos-3: a 6×6 kernel over the fine texture array layer.
const LANCZOS_FRAGMENT_WGSL: &str = r#"
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

fn lanczos_sample(uv: vec2<f32>, layer: i32, tile_size: vec2<f32>) -> vec4<f32> {
    let layer_size = vec2<f32>(textureDimensions(tile_array, 0));
    let uv_pixel = uv * layer_size - 0.5;
    let center = floor(uv_pixel);
    let fract_part = uv_pixel - center;

    var color = vec4<f32>(0.0);
    var weight_sum: f32 = 0.0;

    for (var j: i32 = -2; j <= 3; j++) {
        for (var i: i32 = -2; i <= 3; i++) {
            let offset = vec2<f32>(f32(i), f32(j));
            let sample_pos = clamp(center + offset, vec2<f32>(0.0), tile_size - vec2<f32>(1.0));
            let sample_uv = (sample_pos + 0.5) / layer_size;

            let wx = lanczos_weight(f32(i) - fract_part.x);
            let wy = lanczos_weight(f32(j) - fract_part.y);
            let w = wx * wy;

            let sample = textureSampleLevel(tile_array, tile_sampler, sample_uv, layer, 0.0);
            color += sample * w;
            weight_sum += w;
        }
    }

    if weight_sum > 0.0 {
        color = color / weight_sum;
    }
    return clamp(color, vec4<f32>(0.0), vec4<f32>(1.0));
}

@fragment
fn fs_lanczos(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = lanczos_sample(input.fine_uv, input.fine_layer, input.fine_tex_size);
    return vec4<f32>(eov_color(texel.rgb), texel.a);
}
"#;

/// A complete tile shader: the canonical colour stages plus `fragment`.
fn tile_shader_source(fragment: &str) -> String {
    [
        wgsl::COLOR_PARAMS_STRUCT,
        TILE_STAGE_WGSL,
        wgsl::COLOR_FUNCTIONS,
        fragment,
    ]
    .concat()
}

// ---------------------------------------------------------------------------
// Rust-side data structures
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TileDraw {
    pub tile: Arc<TileData>,
    pub coarse_tile: Option<Arc<TileData>>,
    pub screen_x: f32,
    pub screen_y: f32,
    pub screen_w: f32,
    pub screen_h: f32,
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

#[derive(Clone)]
pub(crate) struct QueuedFrame {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) draws: Vec<TileDraw>,
    /// The pane's colour pipeline as the shaders' `EovColorParams` (stages
    /// 2–4; see `eov_common::processing::ProcessingPipeline::gpu_uniform`).
    pub(crate) color: GpuColorUniform,
    /// Screen-space sharpening of the composited frame (stage 6), 0 = off.
    pub(crate) sharpness: f32,
}

impl QueuedFrame {
    /// Compute a lightweight fingerprint of all render-relevant inputs.
    /// Used to skip redundant vertex rebuilds and GPU render passes.
    fn fingerprint(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.width.hash(&mut hasher);
        self.height.hash(&mut hasher);
        for v in self.color.to_array() {
            v.to_bits().hash(&mut hasher);
        }
        self.sharpness.to_bits().hash(&mut hasher);
        self.draws.len().hash(&mut hasher);
        for draw in &self.draws {
            draw.tile.coord.hash(&mut hasher);
            draw.coarse_tile.as_ref().map(|t| t.coord).hash(&mut hasher);
            draw.screen_x.to_bits().hash(&mut hasher);
            draw.screen_y.to_bits().hash(&mut hasher);
            draw.screen_w.to_bits().hash(&mut hasher);
            draw.screen_h.to_bits().hash(&mut hasher);
            for v in &draw.coarse_uv_min {
                v.to_bits().hash(&mut hasher);
            }
            for v in &draw.coarse_uv_max {
                v.to_bits().hash(&mut hasher);
            }
            draw.mip_blend.to_bits().hash(&mut hasher);
            (draw.filtering_mode as u8).hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Whether the frame needs the sharpening post pass.
    fn sharpens(&self) -> bool {
        self.sharpness > 0.001
    }
}

#[derive(Clone)]
struct ImportedSurface {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    fine_uv: [f32; 2],
    coarse_uv: [f32; 2],
    mip_blend: f32,
    fine_layer: i32,
    coarse_layer: i32,
    fine_tex_size: [f32; 2],
}

/// Parameters for building a tile quad's vertices.
struct QuadParams {
    viewport_size: (u32, u32),
    fine_layer: i32,
    coarse_layer: i32,
    fine_size: (u32, u32),
    coarse_size: (u32, u32),
    layer_size: u32,
    fine_border: u32,
    coarse_border: u32,
}

// ---------------------------------------------------------------------------
// Tile texture array — all tiles live in layers of a single 2D-array texture.
// ---------------------------------------------------------------------------

struct TileSlot {
    layer: u32,
    width: u32,
    height: u32,
    last_used_frame: u64,
}

/// A 2D-array texture used as a tile atlas.  Every layer has the same
/// dimensions (`layer_size × layer_size`); tiles smaller than that are
/// uploaded to the top-left corner and UV coordinates are scaled to the
/// valid sub-region.
struct TileArray {
    texture: wgpu::Texture,
    #[allow(dead_code)] // Kept alive — referenced by the bind groups.
    array_view: wgpu::TextureView,
    layer_size: u32,
    max_layers: u32,
    slots: HashMap<TileCoord, TileSlot>,
    free_list: Vec<u32>,
    /// Pre-built bind groups — one per sampler variant.
    bilinear_bind_group: wgpu::BindGroup,
    trilinear_bind_group: wgpu::BindGroup,
}

impl TileArray {
    fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        bilinear_sampler: &wgpu::Sampler,
        trilinear_sampler: &wgpu::Sampler,
        color_buffer: &wgpu::Buffer,
        layer_size: u32,
        max_layers: u32,
    ) -> Self {
        // No per-layer mipmaps — inter-level filtering is handled
        // explicitly via the fine/coarse trilinear blend.  Generating
        // mipmaps over the full layer_size×layer_size region would
        // contaminate higher mip levels with stale/uninitialized
        // padding data because tiles are smaller than the layer.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tile-array"),
            size: wgpu::Extent3d {
                width: layer_size,
                height: layer_size,
                depth_or_array_layers: max_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let bilinear_bind_group =
            Self::create_bind_group(device, layout, &array_view, bilinear_sampler, color_buffer);
        let trilinear_bind_group =
            Self::create_bind_group(device, layout, &array_view, trilinear_sampler, color_buffer);
        let free_list = (0..max_layers).rev().collect();
        Self {
            texture,
            array_view,
            layer_size,
            max_layers,
            slots: HashMap::new(),
            free_list,
            bilinear_bind_group,
            trilinear_bind_group,
        }
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        array_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        color_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile-array-bind-group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(array_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: color_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Get the layer for a tile, uploading it if necessary.
    /// Returns `(layer_index, is_new)` or `None` if the array is exhausted.
    fn get_or_insert(
        &mut self,
        queue: &wgpu::Queue,
        tile: &Arc<TileData>,
        frame_id: u64,
    ) -> Option<(u32, bool)> {
        if let Some(slot) = self.slots.get_mut(&tile.coord) {
            slot.last_used_frame = frame_id;
            return Some((slot.layer, false));
        }

        // Need a free layer — evict LRU if none available.
        // Protect tiles uploaded this frame so earlier draws' layer
        // references remain valid.
        if self.free_list.is_empty() {
            self.evict_lru(self.max_layers as usize / 4, frame_id);
        }
        let layer = self.free_list.pop()?;

        // Upload the full padded data (including border pixels).
        let data_w = tile.data_width().min(self.layer_size);
        let data_h = tile.data_height().min(self.layer_size);

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &tile.data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tile.data_width() * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: data_w,
                height: data_h,
                depth_or_array_layers: 1,
            },
        );

        self.slots.insert(
            tile.coord,
            TileSlot {
                layer,
                width: data_w,
                height: data_h,
                last_used_frame: frame_id,
            },
        );

        Some((layer, true))
    }

    fn evict_lru(&mut self, count: usize, protect_frame: u64) {
        let mut entries: Vec<_> = self
            .slots
            .iter()
            .filter(|(_, slot)| slot.last_used_frame != protect_frame)
            .map(|(coord, slot)| (*coord, slot.layer, slot.last_used_frame))
            .collect();
        if entries.len() <= count {
            // Evict everything eligible.
            for (coord, layer, _) in &entries {
                self.slots.remove(coord);
                self.free_list.push(*layer);
            }
        } else {
            // Partial sort: partition so the k oldest are at [0..count] in O(n).
            entries.select_nth_unstable_by_key(count - 1, |(_, _, frame)| *frame);
            for (coord, layer, _) in entries[..count].iter() {
                self.slots.remove(coord);
                self.free_list.push(*layer);
            }
        }
    }

    fn tile_dimensions(&self, coord: &TileCoord) -> Option<(u32, u32)> {
        self.slots.get(coord).map(|s| (s.width, s.height))
    }
}

// ---------------------------------------------------------------------------
// GPU runtime — created once at RenderingSetup, holds long-lived GPU objects.
// ---------------------------------------------------------------------------

struct GpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    lanczos_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    bilinear_sampler: wgpu::Sampler,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    /// `EovColorParams` (`GpuColorUniform`, 144 bytes).
    color_buffer: wgpu::Buffer,
    surfaces: HashMap<usize, ImportedSurface>,
    /// Screen-space sharpening over the composited frame (stage 6).
    sharpen_pipeline: wgpu::RenderPipeline,
    sharpen_bind_group_layout: wgpu::BindGroupLayout,
    /// `EovSharpenParams` (the amount, padded to 16 bytes).
    sharpen_buffer: wgpu::Buffer,
    /// Per surface slot: the composited frame the sharpening pass reads.
    frames: HashMap<usize, ImportedSurface>,
}

impl GpuRuntime {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-tile-shader"),
            source: wgpu::ShaderSource::Wgsl(tile_shader_source(BILINEAR_FRAGMENT_WGSL).into()),
        });

        // Bind group layout — shared by bilinear/trilinear and Lanczos pipelines.
        // binding 0: tile_array (texture_2d_array)
        // binding 1: sampler
        // binding 2: colour uniform
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport-tile-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
                0 => Float32x2,  // position
                1 => Float32x2,  // fine_uv
                2 => Float32x2,  // coarse_uv
                3 => Float32,    // mip_blend
                4 => Sint32,     // fine_layer
                5 => Sint32,     // coarse_layer
                6 => Float32x2,  // fine_tex_size
            ],
        };

        let color_target = [Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-tile-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: std::slice::from_ref(&vertex_layout),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &color_target,
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

        let color_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport-color-buffer"),
            size: GpuColorUniform::SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Lanczos pipeline — same bind group layout, different fragment shader
        let lanczos_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lanczos-shader"),
            source: wgpu::ShaderSource::Wgsl(tile_shader_source(LANCZOS_FRAGMENT_WGSL).into()),
        });

        let lanczos_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-lanczos-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &lanczos_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[vertex_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &lanczos_shader,
                entry_point: Some("fs_lanczos"),
                compilation_options: Default::default(),
                targets: &color_target,
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ----- Sharpening post pass (stage 6) -----
        let sharpen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-sharpen-shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl::SHARPEN_POST_PASS.into()),
        });

        let sharpen_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewport-sharpen-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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

        let sharpen_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("viewport-sharpen-pipeline-layout"),
                bind_group_layouts: &[Some(&sharpen_bind_group_layout)],
                immediate_size: 0,
            });

        let sharpen_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-sharpen-pipeline"),
            layout: Some(&sharpen_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sharpen_shader,
                entry_point: Some("vs_eov_sharpen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sharpen_shader,
                entry_point: Some("fs_eov_sharpen"),
                compilation_options: Default::default(),
                targets: &color_target,
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sharpen_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport-sharpen-buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            lanczos_pipeline,
            bind_group_layout,
            sampler,
            bilinear_sampler,
            vertex_buffer,
            vertex_capacity: 6,
            color_buffer,
            surfaces: HashMap::new(),
            sharpen_pipeline,
            sharpen_bind_group_layout,
            sharpen_buffer,
            frames: HashMap::new(),
        }
    }

    /// The composited-frame texture of `slot` (the sharpening pass's input),
    /// (re)created to match its surface.
    fn frame_texture(&mut self, slot: usize, width: u32, height: u32) -> wgpu::TextureView {
        let needs_recreate = self
            .frames
            .get(&slot)
            .is_none_or(|frame| frame.width != width || frame.height != height);
        if needs_recreate {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("pane-viewport-composited-frame"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.frames.insert(
                slot,
                ImportedSurface {
                    texture,
                    view,
                    width,
                    height,
                },
            );
        }
        self.frames[&slot].view.clone()
    }
}

// ---------------------------------------------------------------------------
// Public renderer — owns the runtime, tile array, and pending frame queue.
// ---------------------------------------------------------------------------

pub struct GpuRenderer {
    runtime: Option<GpuRuntime>,
    pending_frames: HashMap<usize, QueuedFrame>,
    /// One `TileArray` per distinct power-of-two layer size (e.g. 256, 512, 1024).
    /// This avoids forcing all tiles into the largest layer size which would
    /// waste most of the memory budget and limit the number of available layers.
    tile_arrays: HashMap<u32, TileArray>,
    frame_counter: u64,
    /// Fingerprint of the last successfully rendered frame per slot.
    /// Used to skip redundant vertex rebuilds and render passes.
    last_rendered_fingerprint: HashMap<usize, u64>,
    /// Last uploaded colour uniform, used to skip redundant uploads.
    last_color: Option<[f32; 36]>,
}

impl GpuRenderer {
    pub fn new() -> Self {
        Self {
            runtime: None,
            pending_frames: HashMap::new(),
            tile_arrays: HashMap::new(),
            frame_counter: 0,
            last_rendered_fingerprint: HashMap::new(),
            last_color: None,
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

        // Skip the vertex rebuild and render pass if the frame is identical
        // to what was last rendered on this slot.
        if !surface_recreated {
            let fp = frame.fingerprint();
            if self
                .last_rendered_fingerprint
                .get(&slot.index())
                .is_some_and(|prev| *prev == fp)
            {
                return self.surface_image(slot);
            }
        }

        self.pending_frames.insert(slot.index(), frame);

        // Return a live handle to the surface texture even on first creation
        // so the UI can keep sampling the same texture across the redraw that
        // flushes the queued frame.
        self.surface_image(slot)
    }

    // -----------------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------------

    fn initialize(&mut self, graphics_api: &GraphicsAPI) {
        if self.runtime.is_some() {
            return;
        }

        let GraphicsAPI::WGPU29 { device, queue, .. } = graphics_api else {
            return;
        };

        self.runtime = Some(GpuRuntime::new(device, queue));
    }

    fn teardown(&mut self) {
        self.runtime = None;
        self.pending_frames.clear();
        self.tile_arrays.clear();
        self.last_rendered_fingerprint.clear();
        self.last_color = None;
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
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

    pub fn read_surface_rgba(&mut self, slot: SurfaceSlot) -> Option<(u32, u32, Vec<u8>)> {
        let runtime = self.runtime.as_mut()?;
        let surface = runtime.surfaces.get(&slot.index())?;

        let bytes_per_row = surface.width.checked_mul(4)?;
        let padded_bytes_per_row = bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer_size = padded_bytes_per_row as u64 * surface.height as u64;

        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pane-viewport-readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pane-viewport-readback-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &surface.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(surface.height),
                },
            },
            wgpu::Extent3d {
                width: surface.width,
                height: surface.height,
                depth_or_array_layers: 1,
            },
        );
        runtime.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = runtime.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok()?.ok()?;

        let mapped = slice.get_mapped_range();
        let mut pixels = vec![0; (surface.width as usize) * (surface.height as usize) * 4];
        let src_stride = padded_bytes_per_row as usize;
        let dst_stride = bytes_per_row as usize;
        for row in 0..surface.height as usize {
            let src_start = row * src_stride;
            let dst_start = row * dst_stride;
            pixels[dst_start..dst_start + dst_stride]
                .copy_from_slice(&mapped[src_start..src_start + dst_stride]);
        }
        drop(mapped);
        readback.unmap();

        Some((surface.width, surface.height, pixels))
    }

    // -----------------------------------------------------------------------
    // Per-frame dispatch
    // -----------------------------------------------------------------------

    fn flush_pending_frames(&mut self) {
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };

        self.frame_counter = self.frame_counter.wrapping_add(1);
        let frame_id = self.frame_counter;

        let pending_frames = std::mem::take(&mut self.pending_frames);

        for (slot_index, frame) in pending_frames {
            if runtime.surfaces.contains_key(&slot_index) {
                let fp = frame.fingerprint();
                Self::render_frame(
                    runtime,
                    &mut self.tile_arrays,
                    &mut self.last_color,
                    SurfaceSlot(slot_index),
                    frame,
                    frame_id,
                );
                self.last_rendered_fingerprint.insert(slot_index, fp);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Core render path — one draw call per tile-array batch per surface,
    // then the sharpening pass when the pane sharpens.
    // -----------------------------------------------------------------------

    fn render_frame(
        runtime: &mut GpuRuntime,
        tile_arrays: &mut HashMap<u32, TileArray>,
        last_color: &mut Option<[f32; 36]>,
        slot: SurfaceSlot,
        frame: QueuedFrame,
        frame_id: u64,
    ) {
        let Some((surface_view, width, height)) = runtime
            .surfaces
            .get(&slot.index())
            .map(|surface| (surface.view.clone(), surface.width, surface.height))
        else {
            return;
        };
        // Tiles are composited straight into the surface, or — when the
        // pane sharpens — into a frame texture that the sharpening pass
        // reads.
        let sharpen = frame.sharpens();
        let tile_target = if sharpen {
            runtime.frame_texture(slot.index(), width, height)
        } else {
            surface_view.clone()
        };

        // Phase 1 & 2: Upload tiles and build vertices, tracking which array
        // each draw belongs to so we can switch bind groups during the pass.
        struct DrawBatch {
            array_ls: u32,
            vertices: [Vertex; 6],
        }
        let mut batches: Vec<DrawBatch> = Vec::with_capacity(frame.draws.len());

        for draw in &frame.draws {
            // Both fine and coarse tiles for a single draw must live in the
            // same texture array (the shader samples both from one binding).
            // Use the larger of the two padded dimensions as the array key.
            let fine_dim = draw.tile.data_width().max(draw.tile.data_height());
            let coarse_dim = draw
                .coarse_tile
                .as_ref()
                .map(|t| t.data_width().max(t.data_height()))
                .unwrap_or(0);
            let array_ls = fine_dim.max(coarse_dim).max(64);

            // Lazily create the array for this layer size.
            let array = tile_arrays.entry(array_ls).or_insert_with(|| {
                let bytes_per_layer = array_ls as u64 * array_ls as u64 * 4;
                let max_layers = (MAX_TILE_ARRAY_BYTES / bytes_per_layer).clamp(16, 256) as u32;
                TileArray::new(
                    &runtime.device,
                    &runtime.bind_group_layout,
                    &runtime.bilinear_sampler,
                    &runtime.sampler,
                    &runtime.color_buffer,
                    array_ls,
                    max_layers,
                )
            });

            // Upload tiles.
            let Some((fine_layer, _)) = array.get_or_insert(&runtime.queue, &draw.tile, frame_id)
            else {
                continue;
            };
            let coarse_layer = draw
                .coarse_tile
                .as_ref()
                .and_then(|t| array.get_or_insert(&runtime.queue, t, frame_id))
                .map(|(l, _)| l)
                .unwrap_or(fine_layer);

            let (fine_w, fine_h) = array
                .tile_dimensions(&draw.tile.coord)
                .unwrap_or((array_ls, array_ls));
            let (coarse_w, coarse_h) = draw
                .coarse_tile
                .as_ref()
                .and_then(|t| array.tile_dimensions(&t.coord))
                .unwrap_or((array_ls, array_ls));

            batches.push(DrawBatch {
                array_ls,
                vertices: quad_vertices(
                    draw,
                    &QuadParams {
                        viewport_size: (frame.width, frame.height),
                        fine_layer: fine_layer as i32,
                        coarse_layer: coarse_layer as i32,
                        fine_size: (fine_w, fine_h),
                        coarse_size: (coarse_w, coarse_h),
                        layer_size: array_ls,
                        fine_border: draw.tile.border,
                        coarse_border: draw.coarse_tile.as_ref().map(|t| t.border).unwrap_or(0),
                    },
                ),
            });
        }

        // Phase 3: Write all vertices into one buffer.
        let total_verts = batches.len() * 6;
        let vertex_count = total_verts.max(6);
        if runtime.vertex_capacity < vertex_count {
            runtime.vertex_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("viewport-tile-vertex-buffer"),
                size: std::mem::size_of::<Vertex>() as u64 * vertex_count as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            runtime.vertex_capacity = vertex_count;
        }

        if total_verts > 0 {
            let flat: Vec<Vertex> = batches
                .iter()
                .flat_map(|b| b.vertices.iter().copied())
                .collect();
            runtime
                .queue
                .write_buffer(&runtime.vertex_buffer, 0, bytemuck::cast_slice(&flat));
        }

        // Upload the colour-pipeline uniform only when values changed.
        let color = frame.color.to_array();
        if last_color.as_ref() != Some(&color) {
            runtime
                .queue
                .write_buffer(&runtime.color_buffer, 0, bytemuck::cast_slice(&color));
            *last_color = Some(color);
        }

        // Build draw ranges — batch consecutive quads that share the same
        // array so we can issue one `draw()` per contiguous run.
        let mut draw_ranges: Vec<(u32, std::ops::Range<u32>)> = Vec::new();
        for (i, b) in batches.iter().enumerate() {
            let start = (i * 6) as u32;
            let end = start + 6;
            if let Some(last) = draw_ranges.last_mut()
                && last.0 == b.array_ls
            {
                last.1.end = end;
                continue;
            }
            draw_ranges.push((b.array_ls, start..end));
        }

        // Phase 4: Render pass — switch bind group per array batch.
        let filtering_mode = frame
            .draws
            .first()
            .map(|d| d.filtering_mode)
            .unwrap_or(FilteringMode::Bilinear);
        let use_lanczos = matches!(filtering_mode, FilteringMode::Lanczos3);
        let is_bilinear = filtering_mode == FilteringMode::Bilinear;

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("viewport-tile-encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport-tile-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &tile_target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
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

            for (ls, range) in &draw_ranges {
                if let Some(array) = tile_arrays.get(ls) {
                    let bg = if is_bilinear {
                        &array.bilinear_bind_group
                    } else {
                        &array.trilinear_bind_group
                    };
                    render_pass.set_bind_group(0, bg, &[]);
                    render_pass.draw(range.clone(), 0..1);
                }
            }
        }

        // ----- Stage 6: screen-space sharpening of the composited frame -----
        if sharpen {
            runtime.queue.write_buffer(
                &runtime.sharpen_buffer,
                0,
                bytemuck::cast_slice(&[frame.sharpness, 0.0, 0.0, 0.0]),
            );
            let bind_group = runtime
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewport-sharpen-bind-group"),
                    layout: &runtime.sharpen_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&tile_target),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: runtime.sharpen_buffer.as_entire_binding(),
                        },
                    ],
                });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport-sharpen-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&runtime.sharpen_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        runtime.queue.submit(Some(encoder.finish()));
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Build the 6 vertices (2 triangles) for a single tile quad.
fn quad_vertices(draw: &TileDraw, params: &QuadParams) -> [Vertex; 6] {
    let vw = params.viewport_size.0.max(1) as f32;
    let vh = params.viewport_size.1.max(1) as f32;
    let fine_layer = params.fine_layer;
    let coarse_layer = params.coarse_layer;
    let (fine_w, fine_h) = params.fine_size;
    let (coarse_w, coarse_h) = params.coarse_size;
    let layer_size = params.layer_size;
    let x0 = draw.screen_x;
    let y0 = draw.screen_y;
    let x1 = draw.screen_x + draw.screen_w;
    let y1 = draw.screen_y + draw.screen_h;

    let left = x0 / vw * 2.0 - 1.0;
    let right = x1 / vw * 2.0 - 1.0;
    let top = 1.0 - y0 / vh * 2.0;
    let bottom = 1.0 - y1 / vh * 2.0;

    // Fine UV: map quad edges to the inner tile region within the padded
    // texture.  With border > 0 the texture contains valid neighbor data so
    // the bilinear/trilinear sampler naturally blends across tile edges.
    let ls = layer_size as f32;
    let fb = params.fine_border as f32;
    let (fu_min, fv_min, fu_max, fv_max) = if fb > 0.0 {
        // Inner tile spans texels [border .. border+inner_w) in the padded data.
        (
            fb / ls,
            fb / ls,
            (fine_w as f32 - fb) / ls,
            (fine_h as f32 - fb) / ls,
        )
    } else {
        // Legacy (border=0): half-texel inset to avoid sampling stale padding.
        let half = 0.5 / ls;
        (
            half,
            half,
            (fine_w as f32 - 0.5) / ls,
            (fine_h as f32 - 0.5) / ls,
        )
    };

    // Coarse UV: the draw's coarse_uv_min/max are in [0,1] of the coarse
    // tile's inner region.  Map them to array-space accounting for border.
    let cb = params.coarse_border as f32;
    let coarse_inner_w = coarse_w as f32 - 2.0 * cb;
    let coarse_inner_h = coarse_h as f32 - 2.0 * cb;
    let coarse_min = [
        (cb + draw.coarse_uv_min[0] * coarse_inner_w) / ls,
        (cb + draw.coarse_uv_min[1] * coarse_inner_h) / ls,
    ];
    let coarse_max = [
        (cb + draw.coarse_uv_max[0] * coarse_inner_w) / ls,
        (cb + draw.coarse_uv_max[1] * coarse_inner_h) / ls,
    ];

    let mip_blend = draw.mip_blend;
    let fine_tex_size = [fine_w as f32, fine_h as f32];

    [
        Vertex {
            position: [left, top],
            fine_uv: [fu_min, fv_min],
            coarse_uv: [coarse_min[0], coarse_min[1]],
            mip_blend,
            fine_layer,
            coarse_layer,
            fine_tex_size,
        },
        Vertex {
            position: [right, top],
            fine_uv: [fu_max, fv_min],
            coarse_uv: [coarse_max[0], coarse_min[1]],
            mip_blend,
            fine_layer,
            coarse_layer,
            fine_tex_size,
        },
        Vertex {
            position: [left, bottom],
            fine_uv: [fu_min, fv_max],
            coarse_uv: [coarse_min[0], coarse_max[1]],
            mip_blend,
            fine_layer,
            coarse_layer,
            fine_tex_size,
        },
        Vertex {
            position: [left, bottom],
            fine_uv: [fu_min, fv_max],
            coarse_uv: [coarse_min[0], coarse_max[1]],
            mip_blend,
            fine_layer,
            coarse_layer,
            fine_tex_size,
        },
        Vertex {
            position: [right, top],
            fine_uv: [fu_max, fv_min],
            coarse_uv: [coarse_max[0], coarse_min[1]],
            mip_blend,
            fine_layer,
            coarse_layer,
            fine_tex_size,
        },
        Vertex {
            position: [right, bottom],
            fine_uv: [fu_max, fv_max],
            coarse_uv: [coarse_max[0], coarse_max[1]],
            mip_blend,
            fine_layer,
            coarse_layer,
            fine_tex_size,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_common::postprocess::ToneAdjustments;
    use eov_common::processing::{ColorSettings, ProcessingPipeline};
    use eov_common::stain::StainChannel;

    fn validate(source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}"));
    }

    #[test]
    fn shaders_are_built_from_the_canonical_wgsl_and_validate() {
        for fragment in [BILINEAR_FRAGMENT_WGSL, LANCZOS_FRAGMENT_WGSL] {
            let source = tile_shader_source(fragment);
            assert!(source.contains(wgsl::COLOR_FUNCTIONS));
            assert!(source.contains(wgsl::COLOR_PARAMS_STRUCT));
            validate(&source);
        }
        validate(wgsl::SHARPEN_POST_PASS);
    }

    fn frame(color: GpuColorUniform, sharpness: f32) -> QueuedFrame {
        QueuedFrame {
            width: 8,
            height: 8,
            draws: Vec::new(),
            color,
            sharpness,
        }
    }

    #[test]
    fn fingerprints_follow_colour_and_sharpening() {
        let neutral = ProcessingPipeline::identity().gpu_uniform();
        let toned = ProcessingPipeline::new(None, ToneAdjustments::new(1.3, 0.0, 1.0), 0.0)
            .unwrap()
            .gpu_uniform();
        let base = frame(neutral, 0.0).fingerprint();
        assert_eq!(base, frame(neutral, 0.0).fingerprint());
        assert_ne!(base, frame(toned, 0.0).fingerprint());
        assert_ne!(base, frame(neutral, 0.5).fingerprint());
        assert!(!frame(neutral, 0.0).sharpens());
        assert!(frame(neutral, 0.5).sharpens());
    }

    // ---- Headless GPU: the shaders against the CPU reference ------------

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        let mut future = std::pin::pin!(future);
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            if let std::task::Poll::Ready(output) = future.as_mut().poll(&mut cx) {
                return output;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "GPU request timed out"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// A headless renderer, or `None` (test skipped) without a usable GPU.
    fn headless_renderer() -> Option<GpuRenderer> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let info = adapter.get_info();
        eprintln!("GPU tests run on {} ({:?})", info.name, info.backend);
        let (device, queue) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
        let mut renderer = GpuRenderer::new();
        renderer.runtime = Some(GpuRuntime::new(&device, &queue));
        Some(renderer)
    }

    const TILE: u32 = 48;

    /// H&E-like test pattern.
    fn pattern(x: u32, y: u32) -> [u8; 3] {
        let t = ((x * 7 + y * 13) % 97) as f32 / 96.0;
        if (x / 6 + y / 6).is_multiple_of(3) {
            // nuclei
            [
                (90.0 + 60.0 * t) as u8,
                (50.0 + 50.0 * t) as u8,
                (140.0 + 50.0 * t) as u8,
            ]
        } else {
            // stroma and glass
            [
                (210.0 + 40.0 * t) as u8,
                (120.0 + 120.0 * t) as u8,
                (170.0 + 70.0 * t) as u8,
            ]
        }
    }

    /// A tile with a one-pixel border (so that it maps 1:1 onto screen
    /// pixels) and the same pixels as tightly packed RGBA.
    fn tile() -> (Arc<TileData>, Vec<u8>) {
        let data_w = TILE + 2;
        let mut data = Vec::with_capacity((data_w * data_w * 4) as usize);
        for y in 0..data_w {
            for x in 0..data_w {
                let px = pattern(
                    x.saturating_sub(1).min(TILE - 1),
                    y.saturating_sub(1).min(TILE - 1),
                );
                data.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
        let inner: Vec<u8> = (0..TILE)
            .flat_map(|y| {
                (0..TILE).flat_map(move |x| {
                    let px = pattern(x, y);
                    [px[0], px[1], px[2], 255]
                })
            })
            .collect();
        (
            Arc::new(TileData::new(
                TileCoord::new(1, 0, 0, 0, TILE),
                data,
                TILE,
                TILE,
                1,
            )),
            inner,
        )
    }

    fn render(renderer: &mut GpuRenderer, width: u32, pipeline: &ProcessingPipeline) -> Vec<u8> {
        let (tile, _) = tile();
        let slot = SurfaceSlot(0);
        renderer.ensure_surface(slot, width, TILE).unwrap();
        renderer.pending_frames.insert(
            slot.index(),
            QueuedFrame {
                width,
                height: TILE,
                draws: vec![TileDraw {
                    tile,
                    coarse_tile: None,
                    screen_x: 0.0,
                    screen_y: 0.0,
                    screen_w: TILE as f32,
                    screen_h: TILE as f32,
                    coarse_uv_min: [0.0, 0.0],
                    coarse_uv_max: [1.0, 1.0],
                    mip_blend: 0.0,
                    filtering_mode: FilteringMode::Bilinear,
                }],
                color: pipeline.gpu_uniform(),
                sharpness: pipeline.sharpness(),
            },
        );
        renderer.flush_pending_frames();
        let (w, h, pixels) = renderer.read_surface_rgba(slot).unwrap();
        assert_eq!((w, h), (width, TILE));
        pixels
    }

    fn max_difference(a: &[u8], b: &[u8]) -> u8 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn gpu_frames_match_the_cpu_pipeline() {
        let Some(mut renderer) = headless_renderer() else {
            eprintln!("no GPU adapter available; skipping the GPU/CPU comparison");
            return;
        };
        let (_, inner) = tile();
        let mut settings = ColorSettings {
            tone: ToneAdjustments::new(1.25, 0.03, 1.15),
            channels: vec![
                StainChannel {
                    intensity: 1.4,
                    visible: true,
                },
                StainChannel {
                    intensity: 0.6,
                    visible: true,
                },
            ],
            ..ColorSettings::default()
        };

        // Stages 2–5: the tile shader against the CPU reference, with the
        // background around the slide.
        let pipeline = settings.pipeline(None);
        let gpu = render(&mut renderer, TILE * 2, &pipeline);
        let mut cpu = vec![0u8; (TILE * 2 * TILE * 4) as usize];
        for y in 0..TILE as usize {
            let row = &inner[y * TILE as usize * 4..(y + 1) * TILE as usize * 4];
            cpu[y * TILE as usize * 8..y * TILE as usize * 8 + row.len()].copy_from_slice(row);
        }
        pipeline.apply_rgba(&mut cpu, TILE * 2, TILE, crate::color::VIEWPORT_BACKGROUND);
        assert!(
            max_difference(&gpu, &cpu) <= 1,
            "GPU and CPU colours differ by {}",
            max_difference(&gpu, &cpu)
        );

        // Isolating a stain uses the same uniform contract.
        settings.isolated = Some(0);
        let isolated = settings.pipeline(None);
        let gpu = render(&mut renderer, TILE, &isolated);
        let mut cpu = inner.clone();
        isolated.apply_rgba(&mut cpu, TILE, TILE, crate::color::VIEWPORT_BACKGROUND);
        assert!(max_difference(&gpu, &cpu) <= 1);
        settings.isolated = None;

        // Stage 6: the sharpening pass is the CPU's screen-space Laplacian
        // over the composited 8-bit frame.
        let unsharpened = render(&mut renderer, TILE * 2, &pipeline);
        settings.sharpness = 0.6;
        let sharpened_pipeline = settings.pipeline(None);
        let gpu = render(&mut renderer, TILE * 2, &sharpened_pipeline);
        let mut cpu = unsharpened;
        sharpened_pipeline.sharpen(&mut cpu, TILE * 2, TILE);
        assert!(
            max_difference(&gpu, &cpu) <= 1,
            "sharpening differs by {}",
            max_difference(&gpu, &cpu)
        );
        // Far from the slide the background stays exactly as cleared.
        let last = gpu.len() - 4;
        assert_eq!(&gpu[last..], &[30, 30, 30, 255]);
    }
}

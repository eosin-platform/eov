use crate::AppWindow;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use common::{TileCoord, TileData};
use slint::ComponentHandle;
use slint::wgpu_28::wgpu;
use slint::wgpu_28::wgpu::util::DeviceExt;
use slint::{GraphicsAPI, Image, RenderingState};
use std::cell::RefCell;
use std::collections::HashMap;
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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let fine = textureSample(fine_texture, tile_sampler, input.fine_uv);
    let coarse = textureSample(coarse_texture, tile_sampler, input.coarse_uv);
    return mix(fine, coarse, input.mip_blend);
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
}

#[derive(Clone, Copy)]
pub enum SurfaceSlot {
    Primary,
    Secondary,
}

#[derive(Clone)]
struct QueuedFrame {
    width: u32,
    height: u32,
    draws: Vec<TileDraw>,
}

#[derive(Clone)]
struct ImportedSurface {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    image: Image,
    width: u32,
    height: u32,
}

#[derive(Clone)]
struct TileTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    last_used_frame: u64,
}

struct GpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    primary_surface: Option<ImportedSurface>,
    secondary_surface: Option<ImportedSurface>,
}

pub struct GpuRenderer {
    runtime: Option<GpuRuntime>,
    pending_primary: Option<QueuedFrame>,
    pending_secondary: Option<QueuedFrame>,
    tile_textures: HashMap<TileCoord, TileTexture>,
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
            pending_primary: None,
            pending_secondary: None,
            tile_textures: HashMap::new(),
            frame_counter: 0,
        }
    }

    pub fn install(ui: &AppWindow, renderer: Rc<RefCell<Self>>) -> Result<()> {
        ui.window().set_rendering_notifier(move |state, graphics_api| {
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

    pub fn is_ready(&self) -> bool {
        self.runtime.is_some()
    }

    pub fn queue_frame(
        &mut self,
        ui: &AppWindow,
        slot: SurfaceSlot,
        width: u32,
        height: u32,
        draws: Vec<TileDraw>,
    ) -> bool {
        if width == 0 || height == 0 {
            return false;
        }

        if self.runtime.is_none() {
            return false;
        }

        if !self.ensure_surface(ui, slot, width, height) {
            return false;
        }

        self.bind_surface_image(ui, slot);

        let frame = QueuedFrame { width, height, draws };
        match slot {
            SurfaceSlot::Primary => self.pending_primary = Some(frame),
            SurfaceSlot::Secondary => self.pending_secondary = Some(frame),
        }

        ui.window().request_redraw();
        true
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
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewport-tile-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
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

        self.runtime = Some(GpuRuntime {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            bind_group_layout,
            sampler,
            vertex_buffer,
            vertex_capacity: 6,
            primary_surface: None,
            secondary_surface: None,
        });
    }

    fn teardown(&mut self) {
        self.runtime = None;
        self.pending_primary = None;
        self.pending_secondary = None;
        self.tile_textures.clear();
    }

    fn ensure_surface(&mut self, ui: &AppWindow, slot: SurfaceSlot, width: u32, height: u32) -> bool {
        let Some(runtime) = self.runtime.as_mut() else {
            return false;
        };

        let surface_ref = match slot {
            SurfaceSlot::Primary => &mut runtime.primary_surface,
            SurfaceSlot::Secondary => &mut runtime.secondary_surface,
        };

        let needs_recreate = surface_ref
            .as_ref()
            .is_none_or(|surface| surface.width != width || surface.height != height);

        if !needs_recreate {
            return true;
        }

        let texture = runtime.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(match slot {
                SurfaceSlot::Primary => "primary-viewport-surface",
                SurfaceSlot::Secondary => "secondary-viewport-surface",
            }),
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
        let image = match Image::try_from(texture.clone()) {
            Ok(image) => image,
            Err(_) => return false,
        };

        *surface_ref = Some(ImportedSurface {
            texture,
            view,
            image,
            width,
            height,
        });

        self.bind_surface_image(ui, slot);
        true
    }

    fn bind_surface_image(&self, ui: &AppWindow, slot: SurfaceSlot) {
        let Some(runtime) = self.runtime.as_ref() else {
            return;
        };

        let image = match slot {
            SurfaceSlot::Primary => runtime.primary_surface.as_ref().map(|surface| surface.image.clone()),
            SurfaceSlot::Secondary => runtime.secondary_surface.as_ref().map(|surface| surface.image.clone()),
        };

        if let Some(image) = image {
            match slot {
                SurfaceSlot::Primary => ui.set_viewport_content(image),
                SurfaceSlot::Secondary => ui.set_secondary_viewport_content(image),
            }
        }
    }

    fn flush_pending_frames(&mut self) {
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };

        self.frame_counter = self.frame_counter.wrapping_add(1);
        let frame_id = self.frame_counter;

        if let Some(frame) = self.pending_primary.take() {
            if runtime.primary_surface.is_some() {
                Self::render_frame(runtime, &mut self.tile_textures, SurfaceSlot::Primary, frame, frame_id);
            }
        }

        if let Some(frame) = self.pending_secondary.take() {
            if runtime.secondary_surface.is_some() {
                Self::render_frame(runtime, &mut self.tile_textures, SurfaceSlot::Secondary, frame, frame_id);
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
            for (coord, _) in entries.into_iter().take(remove_count) {
                self.tile_textures.remove(&coord);
            }
        }
    }

    fn render_frame(
        runtime: &mut GpuRuntime,
        tile_textures: &mut HashMap<TileCoord, TileTexture>,
        slot: SurfaceSlot,
        frame: QueuedFrame,
        frame_id: u64,
    ) {
        let Some(surface_view) = (match slot {
            SurfaceSlot::Primary => runtime.primary_surface.as_ref().map(|surface| surface.view.clone()),
            SurfaceSlot::Secondary => runtime.secondary_surface.as_ref().map(|surface| surface.view.clone()),
        }) else {
            return;
        };

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
            runtime.queue.write_buffer(&runtime.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }

        let mut encoder = runtime.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("viewport-tile-encoder"),
        });

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

            render_pass.set_pipeline(&runtime.pipeline);
            render_pass.set_vertex_buffer(0, runtime.vertex_buffer.slice(..));

            for (index, draw) in frame.draws.iter().enumerate() {
                let fine_texture = ensure_tile_texture(runtime, tile_textures, &draw.tile, frame_id);
                let coarse_texture = draw
                    .coarse_tile
                    .as_ref()
                    .map(|tile| ensure_tile_texture(runtime, tile_textures, tile, frame_id))
                    .unwrap_or_else(|| fine_texture.clone());
                let bind_group = runtime.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewport-tile-bind-group"),
                    layout: &runtime.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&fine_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&coarse_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&runtime.sampler),
                        },
                    ],
                });
                render_pass.set_bind_group(0, &bind_group, &[]);
                let start = (index * 6) as u32;
                render_pass.draw(start..start + 6, 0..1);
            }
        }

        runtime.queue.submit(Some(encoder.finish()));
    }
}

fn ensure_tile_texture(
    runtime: &GpuRuntime,
    tile_textures: &mut HashMap<TileCoord, TileTexture>,
    tile: &Arc<TileData>,
    frame_id: u64,
) -> TileTexture {
    let entry = tile_textures.entry(tile.coord).or_insert_with(|| {
        let texture = runtime.device.create_texture_with_data(
            &runtime.queue,
            &wgpu::TextureDescriptor {
                label: Some("viewport-tile-texture"),
                size: wgpu::Extent3d {
                    width: tile.width.max(1),
                    height: tile.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &tile.data,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        TileTexture {
            texture,
            view,
            last_used_frame: frame_id,
        }
    });

    entry.last_used_frame = frame_id;
    entry.clone()
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

use crate::AppWindow;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use eov_common::color_management::{DisplayTransform, LUT_SIZE};
use eov_common::processing::{GpuColorUniform, wgsl};
use eov_common::viewport::Orientation;
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
use std::sync::atomic::{AtomicU8, Ordering};

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
// The colour stages (stain transform, tone) and the colour management
// (the slide's ICC profile → sRGB, through its 3D table in bind group 1)
// come verbatim from `eov_common::processing::wgsl`, the reference that the
// CPU pipeline is tested against, so the two backends cannot drift apart.
// Sharpening (stage 6) runs as a separate full-screen pass over the
// composited frame (`wgsl::SHARPEN_POST_PASS`), in screen space as on the
// CPU.
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

// The slide's colour management table (`color_params.icc`).
@group(1) @binding(0)
var icc_lut: texture_3d<f32>;
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
    return vec4<f32>(eov_icc(eov_color(texel.rgb)), texel.a);
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
    return vec4<f32>(eov_icc(eov_color(texel.rgb)), texel.a);
}
"#;

/// A complete tile shader: the canonical colour stages and colour
/// management plus `fragment`.
fn tile_shader_source(fragment: &str) -> String {
    [
        wgsl::COLOR_PARAMS_STRUCT,
        TILE_STAGE_WGSL,
        wgsl::COLOR_FUNCTIONS,
        wgsl::ICC_FUNCTIONS,
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
    fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone)]
pub(crate) struct QueuedFrame {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// How the draws, laid out axis-aligned with the slide (see
    /// `Viewport::aligned`), turn onto the `width × height` surface.
    pub(crate) orientation: Orientation,
    pub(crate) draws: Vec<TileDraw>,
    /// The pane's colour pipeline as the shaders' `EovColorParams` (stages
    /// 2–4; see `eov_common::processing::ProcessingPipeline::gpu_uniform`).
    pub(crate) color: GpuColorUniform,
    /// The colour management table `color.icc` refers to.
    pub(crate) icc: Option<Arc<DisplayTransform>>,
    /// Screen-space sharpening of the composited frame (stage 6), 0 = off.
    pub(crate) sharpness: f32,
    /// Composite the background and sharpen (stages 5–6). Off for frames
    /// read back for plugin filters: they stay transparent outside the
    /// slide, and the filter job finishes them after the filters.
    pub(crate) composite: bool,
    /// Read the frame back for the CPU filter chain (RND-6); see
    /// [`GpuRenderer::poll_readbacks`].
    pub(crate) readback: Option<ReadbackTag>,
}

/// Identifies a pane frame read back from the GPU for plugin filters.
#[derive(Clone, Debug)]
pub(crate) struct ReadbackTag {
    pub(crate) pane_uid: u32,
    pub(crate) file_id: i32,
    /// The view the frame shows (in the frame's, physical, pixels).
    pub(crate) viewport: eov_common::Viewport,
    /// Physical pixels of the frame per logical pixel.
    pub(crate) scale: f64,
    /// Increases with every readback a pane requests.
    pub(crate) generation: u64,
    /// What follows the filters: the background and the sharpening.
    pub(crate) finish: crate::render_pool::CpuRenderPostProcess,
}

/// A frame read back from the GPU: tightly packed RGBA.
pub(crate) struct ReadbackFrame {
    pub(crate) tag: ReadbackTag,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: Vec<u8>,
}

/// A surface copy on its way to the CPU.
struct PendingReadback {
    tag: ReadbackTag,
    buffer: wgpu::Buffer,
    padded_bytes_per_row: u32,
    width: u32,
    height: u32,
    /// 0 while mapping, 1 mapped, 2 failed.
    state: Arc<AtomicU8>,
}

impl PendingReadback {
    fn into_frame(self) -> ReadbackFrame {
        let row = self.width as usize * 4;
        let mut pixels = vec![0u8; row * self.height as usize];
        {
            let mapped = self.buffer.slice(..).get_mapped_range();
            for (y, dst) in pixels.chunks_exact_mut(row).enumerate() {
                let src = y * self.padded_bytes_per_row as usize;
                dst.copy_from_slice(&mapped[src..src + row]);
            }
        }
        self.buffer.unmap();
        ReadbackFrame {
            tag: self.tag,
            width: self.width,
            height: self.height,
            pixels,
        }
    }
}

impl QueuedFrame {
    /// Compute a lightweight fingerprint of all render-relevant inputs.
    /// Used to skip redundant vertex rebuilds and GPU render passes.
    fn fingerprint(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.width.hash(&mut hasher);
        self.height.hash(&mut hasher);
        self.orientation.hash(&mut hasher);
        for v in self.color.to_array() {
            v.to_bits().hash(&mut hasher);
        }
        self.sharpness.to_bits().hash(&mut hasher);
        self.composite.hash(&mut hasher);
        self.icc.as_ref().map(|icc| icc.id()).hash(&mut hasher);
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
        self.composite && self.sharpness > 0.001
    }

    /// What the frame is cleared to: the background, or transparent for a
    /// frame that the filter job finishes.
    fn clear_color(&self) -> wgpu::Color {
        if self.composite {
            CLEAR_COLOR
        } else {
            wgpu::Color::TRANSPARENT
        }
    }

    /// The uniform the shaders get. A frame for plugin filters is converted
    /// on the CPU, after the filters (colour management follows them).
    fn shader_color(&self) -> GpuColorUniform {
        let mut color = self.color;
        if !self.composite || self.icc.is_none() {
            color.icc = [0.0; 4];
        }
        color
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
    /// Weight of the coarse level (0 when the draw has none, or it did not
    /// fit into the fine tile's page).
    mip_blend: f32,
    /// How the aligned layout turns onto the surface.
    orientation: Orientation,
}

// ---------------------------------------------------------------------------
// Tile pages — tiles live in the layers of 2D-array textures ("pages"),
// grouped by layer size. A frame's draws get as many pages as they need
// (RND-2); pages that recent frames did not use are freed above a memory
// budget (RND-7).
// ---------------------------------------------------------------------------

/// Layer sizes are multiples of this many texels.
const LAYER_SIZE_STEP: u32 = 32;
/// Size of one page; pages of large tiles hold at least `MIN_PAGE_LAYERS`.
const PAGE_BYTES: u64 = 64 * 1024 * 1024;
const MIN_PAGE_LAYERS: u32 = 4;
/// Tile memory kept between frames: above it, the least recently used
/// pages are freed.
const TILE_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
/// Most tile memory, even when one frame needs more (its remaining draws
/// are then dropped, and counted).
const TILE_MEMORY_LIMIT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// Shortest interval between two reports of dropped draws.
const OVERFLOW_REPORT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);

/// The layer size a tile is stored at: its level's padded tile size, so that
/// a level's edge tiles share its pages, rounded up to `LAYER_SIZE_STEP`.
fn layer_size_class(tile: &TileData) -> u32 {
    let nominal = tile.coord.tile_size.saturating_add(2 * tile.border);
    let actual = tile.data_width().max(tile.data_height());
    nominal.max(actual).max(64).div_ceil(LAYER_SIZE_STEP) * LAYER_SIZE_STEP
}

fn layer_bytes(layer_size: u32) -> u64 {
    u64::from(layer_size) * u64::from(layer_size) * 4
}

/// Layers of a page of `layer_size` layers, within the device's limit.
fn page_layers(layer_size: u32, max_layers: u32) -> u32 {
    (PAGE_BYTES / layer_bytes(layer_size)).clamp(
        u64::from(MIN_PAGE_LAYERS.min(max_layers)),
        u64::from(max_layers),
    ) as u32
}

struct TileSlot {
    layer: u32,
    width: u32,
    height: u32,
    last_used_frame: u64,
}

/// Which layers of a page hold which tiles. Tiles a frame uses are never
/// evicted during that frame, so its earlier draws stay valid.
struct LayerSlots {
    layers: u32,
    slots: HashMap<TileCoord, TileSlot>,
    free: Vec<u32>,
    /// The last frame that used the page, and how many of its tiles it uses.
    frame: u64,
    used_in_frame: u32,
}

impl LayerSlots {
    fn new(layers: u32, frame: u64) -> Self {
        Self {
            layers,
            slots: HashMap::new(),
            free: (0..layers).rev().collect(),
            frame,
            used_in_frame: 0,
        }
    }

    fn contains(&self, coord: &TileCoord) -> bool {
        self.slots.contains_key(coord)
    }

    fn begin(&mut self, frame: u64) {
        if self.frame != frame {
            self.frame = frame;
            self.used_in_frame = 0;
        }
    }

    /// Marks a stored tile as used by `frame`; its layer and size.
    fn touch(&mut self, coord: &TileCoord, frame: u64) -> Option<(u32, u32, u32)> {
        self.begin(frame);
        let slot = self.slots.get_mut(coord)?;
        if slot.last_used_frame != frame {
            slot.last_used_frame = frame;
            self.used_in_frame += 1;
        }
        Some((slot.layer, slot.width, slot.height))
    }

    /// Layers `frame` could still take: free ones and ones holding tiles
    /// it does not use.
    fn room(&self, frame: u64) -> u32 {
        let used = if self.frame == frame {
            self.used_in_frame
        } else {
            0
        };
        self.free.len() as u32 + (self.slots.len() as u32 - used)
    }

    /// A layer for a new tile of `width × height` texels, used by `frame`;
    /// evicts the least recently used tiles `frame` does not use if needed.
    fn allocate(&mut self, coord: TileCoord, width: u32, height: u32, frame: u64) -> Option<u32> {
        self.begin(frame);
        if self.free.is_empty() {
            self.evict_unused((self.layers as usize / 4).max(1), frame);
        }
        let layer = self.free.pop()?;
        if let Some(old) = self.slots.insert(
            coord,
            TileSlot {
                layer,
                width,
                height,
                last_used_frame: frame,
            },
        ) {
            // Re-uploaded (not expected): keep the counts right.
            self.free.push(old.layer);
            if old.last_used_frame == frame {
                self.used_in_frame -= 1;
            }
        }
        self.used_in_frame += 1;
        Some(layer)
    }

    fn evict_unused(&mut self, count: usize, frame: u64) {
        let mut candidates: Vec<(TileCoord, u64)> = self
            .slots
            .iter()
            .filter(|(_, slot)| slot.last_used_frame != frame)
            .map(|(coord, slot)| (*coord, slot.last_used_frame))
            .collect();
        if candidates.len() > count {
            candidates.select_nth_unstable_by_key(count - 1, |(_, last)| *last);
            candidates.truncate(count);
        }
        for (coord, _) in candidates {
            if let Some(slot) = self.slots.remove(&coord) {
                self.free.push(slot.layer);
            }
        }
    }
}

/// A 2D-array texture of `layer_size²` layers. Tiles smaller than that
/// are uploaded to a layer's top-left corner, and their UVs scaled to it.
struct TilePage {
    texture: wgpu::Texture,
    #[allow(dead_code)] // Kept alive — referenced by the bind groups.
    array_view: wgpu::TextureView,
    layer_size: u32,
    slots: LayerSlots,
    /// Pre-built bind groups — one per sampler variant.
    bilinear_bind_group: wgpu::BindGroup,
    trilinear_bind_group: wgpu::BindGroup,
}

impl TilePage {
    /// A new page, or `None` when the device is out of memory.
    fn new(runtime: &GpuRuntime, layer_size: u32, frame: u64) -> Option<Self> {
        let device = &runtime.device;
        let layers = page_layers(layer_size, runtime.max_texture_array_layers);
        let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        // No per-layer mipmaps — inter-level filtering is handled
        // explicitly via the fine/coarse trilinear blend.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tile-page"),
            size: wgpu::Extent3d {
                width: layer_size,
                height: layer_size,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        if let Some(Some(error)) = poll_now(scope.pop()) {
            tracing::warn!("No GPU memory for {layers} tile layers of {layer_size}²: {error}");
            return None;
        }
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let bind_group = |sampler: &wgpu::Sampler| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tile-page-bind-group"),
                layout: &runtime.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&array_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: runtime.color_buffer.as_entire_binding(),
                    },
                ],
            })
        };
        let bilinear_bind_group = bind_group(&runtime.bilinear_sampler);
        let trilinear_bind_group = bind_group(&runtime.sampler);
        Some(Self {
            texture,
            array_view,
            layer_size,
            slots: LayerSlots::new(layers, frame),
            bilinear_bind_group,
            trilinear_bind_group,
        })
    }

    fn bytes(&self) -> u64 {
        layer_bytes(self.layer_size) * u64::from(self.slots.layers)
    }

    /// Uploads a tile (with its border) into a free or evicted layer.
    fn insert(
        &mut self,
        queue: &wgpu::Queue,
        tile: &TileData,
        frame: u64,
    ) -> Option<(u32, u32, u32)> {
        let width = tile.data_width().min(self.layer_size);
        let height = tile.data_height().min(self.layer_size);
        let layer = self.slots.allocate(tile.coord, width, height, frame)?;
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
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Some((layer, width, height))
    }

    /// The tile's layer and size, uploading it when the page lacks it.
    fn get_or_insert(
        &mut self,
        queue: &wgpu::Queue,
        tile: &TileData,
        frame: u64,
    ) -> Option<(u32, u32, u32)> {
        self.slots
            .touch(&tile.coord, frame)
            .or_else(|| self.insert(queue, tile, frame))
    }
}

/// Where a draw's tiles are: a page, and the fine and coarse layers with
/// the texel size of their data.
struct Placement {
    layer_size: u32,
    page: usize,
    fine: (u32, u32, u32),
    coarse: Option<(u32, u32, u32)>,
}

/// All tile pages, by layer size.
#[derive(Default)]
struct TileStore {
    pages: HashMap<u32, Vec<TilePage>>,
    bytes: u64,
    /// Since the last report: draws left out, and draws drawn without their
    /// coarse level.
    dropped: u64,
    unblended: u64,
    last_report: Option<std::time::Instant>,
    /// A frame whose page allocation failed; it does not retry.
    failed_frame: Option<u64>,
}

impl TileStore {
    /// Finds or makes room for a draw's tiles: both in one page, or the
    /// fine tile alone when the coarse one does not fit (then drawn without
    /// the blend). `None` when not even the fine tile fits.
    fn place(
        &mut self,
        runtime: &GpuRuntime,
        fine: &TileData,
        coarse: Option<&TileData>,
        frame: u64,
    ) -> Option<Placement> {
        let layer_size = layer_size_class(fine).max(coarse.map_or(0, layer_size_class));
        if layer_size > runtime.max_texture_dimension_2d {
            return None;
        }
        let queue = &runtime.queue;
        let pages = self.pages.entry(layer_size).or_default();
        let placed = |page: usize, fine, coarse| {
            Some(Placement {
                layer_size,
                page,
                fine,
                coarse,
            })
        };

        // A page that holds the draw's tiles, or one of them and has room
        // for the other.
        for (index, page) in pages.iter_mut().enumerate() {
            let has_fine = page.slots.contains(&fine.coord);
            let has_coarse = coarse.is_some_and(|c| page.slots.contains(&c.coord));
            if !has_fine && !has_coarse {
                continue;
            }
            let fine_slot = has_fine
                .then(|| page.slots.touch(&fine.coord, frame))
                .flatten();
            let coarse_slot = coarse
                .filter(|_| has_coarse)
                .and_then(|c| page.slots.touch(&c.coord, frame));
            match (coarse, fine_slot, coarse_slot) {
                (None, Some(f), _) | (Some(_), Some(f), Some(_)) => {
                    return placed(index, f, coarse_slot);
                }
                (Some(_), Some(f), None) if page.slots.room(frame) >= 1 => {
                    let c = coarse.and_then(|c| page.insert(queue, c, frame));
                    return placed(index, f, c);
                }
                (Some(_), None, Some(c)) if page.slots.room(frame) >= 1 => {
                    if let Some(f) = page.insert(queue, fine, frame) {
                        return placed(index, f, Some(c));
                    }
                }
                _ => {}
            }
        }

        // A page with room for both, or a new one.
        let needed = 1 + u32::from(coarse.is_some());
        let index = match pages
            .iter()
            .position(|page| page.slots.room(frame) >= needed)
        {
            Some(index) => Some(index),
            None => self.add_page(runtime, layer_size, frame),
        };
        let pages = self.pages.entry(layer_size).or_default();
        if let Some(index) = index {
            let page = &mut pages[index];
            if let Some(f) = page.get_or_insert(queue, fine, frame) {
                let c = coarse.and_then(|c| page.get_or_insert(queue, c, frame));
                return placed(index, f, c);
            }
        }

        // Out of tile memory: the fine tile alone.
        if coarse.is_some() {
            for (index, page) in pages.iter_mut().enumerate() {
                if (page.slots.contains(&fine.coord) || page.slots.room(frame) >= 1)
                    && let Some(f) = page.get_or_insert(queue, fine, frame)
                {
                    return placed(index, f, None);
                }
            }
        }
        None
    }

    /// Adds a page of `layer_size` within the memory limit; its index.
    /// (Pages are only freed between frames, so that the indices the
    /// frame's draws refer to stay valid.)
    fn add_page(&mut self, runtime: &GpuRuntime, layer_size: u32, frame: u64) -> Option<usize> {
        if self.failed_frame == Some(frame) {
            return None;
        }
        let bytes = layer_bytes(layer_size)
            * u64::from(page_layers(layer_size, runtime.max_texture_array_layers));
        let page = (self.bytes + bytes <= TILE_MEMORY_LIMIT_BYTES)
            .then(|| TilePage::new(runtime, layer_size, frame))
            .flatten();
        let Some(page) = page else {
            self.failed_frame = Some(frame);
            return None;
        };
        self.bytes += page.bytes();
        let pages = self.pages.entry(layer_size).or_default();
        pages.push(page);
        Some(pages.len() - 1)
    }

    /// Frees the least recently used pages that `frame` does not use until
    /// at most `target` bytes are left.
    fn free_unused(&mut self, frame: u64, target: u64) {
        while self.bytes > target {
            let oldest = self
                .pages
                .iter()
                .flat_map(|(size, pages)| {
                    pages
                        .iter()
                        .enumerate()
                        .filter(|(_, page)| page.slots.frame != frame)
                        .map(move |(index, page)| (page.slots.frame, *size, index))
                })
                .min();
            let Some((_, size, index)) = oldest else {
                break;
            };
            if let Some(pages) = self.pages.get_mut(&size) {
                let page = pages.remove(index);
                self.bytes -= page.bytes();
            }
        }
        self.pages.retain(|_, pages| !pages.is_empty());
    }

    fn page(&self, layer_size: u32, index: usize) -> Option<&TilePage> {
        self.pages.get(&layer_size)?.get(index)
    }

    /// After a frame: frees pages above the budget, and reports draws that
    /// did not fit.
    fn end_frame(&mut self, frame: u64) {
        self.free_unused(frame, TILE_BUDGET_BYTES);
        if self.dropped + self.unblended == 0 {
            return;
        }
        let now = std::time::Instant::now();
        if self
            .last_report
            .is_some_and(|last| now.duration_since(last) < OVERFLOW_REPORT_INTERVAL)
        {
            return;
        }
        tracing::warn!(
            dropped = self.dropped,
            unblended = self.unblended,
            tile_memory_mib = self.bytes / (1024 * 1024),
            "GPU tile memory is full: tile draws were left out or drawn without their coarse level"
        );
        self.dropped = 0;
        self.unblended = 0;
        self.last_report = Some(now);
    }
}

/// The output of a future that is ready at once (wgpu's native error
/// scopes), or `None`.
fn poll_now<F: std::future::Future>(future: F) -> Option<F::Output> {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match future.as_mut().poll(&mut context) {
        std::task::Poll::Ready(output) => Some(output),
        std::task::Poll::Pending => None,
    }
}

// ---------------------------------------------------------------------------
// GPU failures — wgpu errors and a lost device switch the slides to the
// CPU renderer instead of panicking (RND-7).
// ---------------------------------------------------------------------------

static GPU_FAILURE: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

pub(crate) fn record_failure(reason: String) {
    if let Ok(mut failure) = GPU_FAILURE.lock()
        && failure.is_none()
    {
        *failure = Some(reason);
    }
}

/// Why the GPU renderer failed, once: wgpu reported an error or lost the
/// device.
pub(crate) fn take_failure() -> Option<String> {
    GPU_FAILURE.lock().ok()?.take()
}

/// Logs wgpu errors (the default handler panics) and device loss instead
/// of crashing; the render loop then switches to the CPU renderer.
fn install_error_handlers(device: &wgpu::Device) {
    device.on_uncaptured_error(Arc::new(|error: wgpu::Error| {
        tracing::error!("GPU error: {error}");
        record_failure(format!("GPU error: {error}"));
    }));
    device.set_device_lost_callback(|reason, message| {
        if reason != wgpu::DeviceLostReason::Destroyed {
            tracing::error!("The GPU device was lost: {message}");
            record_failure(format!("the GPU device was lost ({message})"));
        }
    });
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
    /// Bind group 1: the colour management table.
    icc_bind_group_layout: wgpu::BindGroupLayout,
    /// A table that is never read (colour management off).
    icc_none: wgpu::BindGroup,
    /// Tables of the slides' profiles, by profile id, with the frame they
    /// were last used in.
    icc_tables: HashMap<u64, (wgpu::BindGroup, u64)>,
    max_texture_array_layers: u32,
    max_texture_dimension_2d: u32,
}

/// Colour management tables kept for slides no frame showed recently.
const KEPT_ICC_TABLES: usize = 8;

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

        let icc_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewport-icc-bind-group-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        let icc_none = icc_bind_group(
            device,
            queue,
            &icc_bind_group_layout,
            2,
            &[[0.0, 0.0, 0.0, 1.0]; 8],
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport-tile-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout), Some(&icc_bind_group_layout)],
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
            icc_bind_group_layout,
            icc_none,
            icc_tables: HashMap::new(),
            max_texture_array_layers: device.limits().max_texture_array_layers.max(1),
            max_texture_dimension_2d: device.limits().max_texture_dimension_2d,
        }
    }

    /// The bind group of `frame`'s colour management table, uploaded the
    /// first time a frame uses it.
    fn icc_table(&mut self, frame: &QueuedFrame, frame_id: u64) -> wgpu::BindGroup {
        let Some(transform) = frame
            .icc
            .as_ref()
            .filter(|_| frame.shader_color().icc[0] > 0.5)
        else {
            return self.icc_none.clone();
        };
        let entry = self.icc_tables.entry(transform.id()).or_insert_with(|| {
            let bind_group = icc_bind_group(
                &self.device,
                &self.queue,
                &self.icc_bind_group_layout,
                LUT_SIZE as u32,
                &transform.gpu_texels(),
            );
            (bind_group, frame_id)
        });
        entry.1 = frame_id;
        let bind_group = entry.0.clone();
        if self.icc_tables.len() > KEPT_ICC_TABLES {
            // Forget the table shown longest ago (never this frame's).
            if let Some(oldest) = self
                .icc_tables
                .iter()
                .filter(|(_, (_, used))| *used != frame_id)
                .min_by_key(|(_, (_, used))| *used)
                .map(|(id, _)| *id)
            {
                self.icc_tables.remove(&oldest);
            }
        }
        bind_group
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

/// A `size³` RGBA32F 3D texture holding `texels` (x fastest, then y, then
/// z), bound for the tile shaders' `icc_lut`.
fn icc_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    size: u32,
    texels: &[[f32; 4]],
) -> wgpu::BindGroup {
    let extent = wgpu::Extent3d {
        width: size,
        height: size,
        depth_or_array_layers: size,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewport-icc-table"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 16),
            rows_per_image: Some(size),
        },
        extent,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("viewport-icc-bind-group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    })
}

// ---------------------------------------------------------------------------
// Public renderer — owns the runtime, tile pages, and pending frame queue.
// ---------------------------------------------------------------------------

pub struct GpuRenderer {
    runtime: Option<GpuRuntime>,
    pending_frames: HashMap<usize, QueuedFrame>,
    /// Tile pages by layer size class.
    tiles: TileStore,
    /// The renderer failed (see [`take_failure`]) and stays off.
    disabled: bool,
    frame_counter: u64,
    /// Fingerprint of the last successfully rendered frame per slot.
    /// Used to skip redundant vertex rebuilds and render passes.
    last_rendered_fingerprint: HashMap<usize, u64>,
    /// Last uploaded colour uniform, used to skip redundant uploads.
    last_color: Option<[f32; 40]>,
    /// Surfaces to read back as they are (their frame did not change).
    readback_requests: Vec<(usize, ReadbackTag)>,
    /// Readbacks waiting for the GPU.
    pending_readbacks: Vec<PendingReadback>,
}

impl GpuRenderer {
    pub fn new() -> Self {
        Self {
            runtime: None,
            pending_frames: HashMap::new(),
            tiles: TileStore::default(),
            disabled: false,
            frame_counter: 0,
            last_rendered_fingerprint: HashMap::new(),
            last_color: None,
            readback_requests: Vec::new(),
            pending_readbacks: Vec::new(),
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
                // The surface already shows this frame; read it back as is.
                if let Some(tag) = frame.readback {
                    self.readback_requests.push((slot.index(), tag));
                }
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
        if self.runtime.is_some() || self.disabled {
            return;
        }

        let GraphicsAPI::WGPU29 { device, queue, .. } = graphics_api else {
            record_failure("the window does not render with wgpu".to_string());
            return;
        };

        install_error_handlers(device);
        self.runtime = Some(GpuRuntime::new(device, queue));
    }

    fn teardown(&mut self) {
        self.pending_readbacks.clear();
        self.readback_requests.clear();
        self.runtime = None;
        self.pending_frames.clear();
        self.tiles = TileStore::default();
        self.last_rendered_fingerprint.clear();
        self.last_color = None;
    }

    /// The largest surface edge the device supports.
    pub(crate) fn max_surface_dimension(&self) -> Option<u32> {
        self.runtime
            .as_ref()
            .map(|runtime| runtime.max_texture_dimension_2d)
    }

    /// Releases everything and stays off: the GPU failed.
    pub(crate) fn disable(&mut self) {
        self.teardown();
        self.disabled = true;
    }

    /// Frees the surfaces of slots `count` and above (closed panes).
    pub(crate) fn retain_slots(&mut self, count: usize) {
        self.pending_frames.retain(|slot, _| *slot < count);
        self.last_rendered_fingerprint
            .retain(|slot, _| *slot < count);
        self.readback_requests.retain(|(slot, _)| *slot < count);
        if let Some(runtime) = self.runtime.as_mut() {
            runtime.surfaces.retain(|slot, _| *slot < count);
            runtime.frames.retain(|slot, _| *slot < count);
        }
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
        if width > runtime.max_texture_dimension_2d || height > runtime.max_texture_dimension_2d {
            tracing::warn!(
                "A {width}×{height} pane exceeds the GPU's {} px limit",
                runtime.max_texture_dimension_2d
            );
            return None;
        }

        let descriptor = wgpu::TextureDescriptor {
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
        };
        let create = |device: &wgpu::Device| {
            let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let texture = device.create_texture(&descriptor);
            match poll_now(scope.pop()) {
                Some(Some(error)) => Err(error),
                _ => Ok(texture),
            }
        };
        let texture = match create(&runtime.device) {
            Ok(texture) => texture,
            Err(_) => {
                // Make room: tile pages are refilled on demand.
                self.tiles = TileStore::default();
                match create(&runtime.device) {
                    Ok(texture) => texture,
                    Err(error) => {
                        record_failure(format!("out of GPU memory ({error})"));
                        return None;
                    }
                }
            }
        };
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
        let mut readbacks = std::mem::take(&mut self.readback_requests);

        for (slot_index, mut frame) in pending_frames {
            if runtime.surfaces.contains_key(&slot_index) {
                let fp = frame.fingerprint();
                if let Some(tag) = frame.readback.take() {
                    readbacks.push((slot_index, tag));
                }
                Self::render_frame(
                    runtime,
                    &mut self.tiles,
                    &mut self.last_color,
                    SurfaceSlot(slot_index),
                    frame,
                    frame_id,
                );
                self.last_rendered_fingerprint.insert(slot_index, fp);
            }
        }
        self.tiles.end_frame(frame_id);
        for (slot_index, tag) in readbacks {
            if let Some(pending) = Self::start_readback(runtime, slot_index, tag) {
                self.pending_readbacks.push(pending);
            }
        }
    }

    /// Copies a surface into a mappable buffer and starts mapping it.
    fn start_readback(
        runtime: &GpuRuntime,
        slot_index: usize,
        tag: ReadbackTag,
    ) -> Option<PendingReadback> {
        let surface = runtime.surfaces.get(&slot_index)?;
        let bytes_per_row = surface.width.checked_mul(4)?;
        let padded_bytes_per_row = bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pane-filter-readback"),
            size: u64::from(padded_bytes_per_row) * u64::from(surface.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pane-filter-readback-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &surface.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
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
        let state = Arc::new(AtomicU8::new(0));
        let flag = Arc::clone(&state);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                flag.store(if result.is_ok() { 1 } else { 2 }, Ordering::Release);
            });
        Some(PendingReadback {
            tag,
            buffer,
            padded_bytes_per_row,
            width: surface.width,
            height: surface.height,
            state,
        })
    }

    /// Frames read back since the last call, without blocking: readbacks
    /// still on the GPU stay pending.
    pub(crate) fn poll_readbacks(&mut self) -> Vec<ReadbackFrame> {
        if self.pending_readbacks.is_empty() {
            return Vec::new();
        }
        let Some(runtime) = self.runtime.as_ref() else {
            self.pending_readbacks.clear();
            return Vec::new();
        };
        let _ = runtime.device.poll(wgpu::PollType::Poll);
        let mut done = Vec::new();
        let mut index = 0;
        while index < self.pending_readbacks.len() {
            match self.pending_readbacks[index].state.load(Ordering::Acquire) {
                0 => index += 1,
                1 => {
                    let pending = self.pending_readbacks.swap_remove(index);
                    done.push(pending.into_frame());
                }
                _ => {
                    let pending = self.pending_readbacks.swap_remove(index);
                    tracing::warn!(
                        "reading pane {} back from the GPU failed",
                        pending.tag.pane_uid
                    );
                }
            }
        }
        done
    }

    /// Whether readbacks are still on their way.
    pub(crate) fn readbacks_pending(&self) -> bool {
        !self.pending_readbacks.is_empty() || !self.readback_requests.is_empty()
    }

    // -----------------------------------------------------------------------
    // Core render path — one draw call per tile-array batch per surface,
    // then the sharpening pass when the pane sharpens.
    // -----------------------------------------------------------------------

    fn render_frame(
        runtime: &mut GpuRuntime,
        tiles: &mut TileStore,
        last_color: &mut Option<[f32; 40]>,
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

        // Phase 1 & 2: Upload tiles and build vertices, tracking which page
        // each draw uses so that the pass can switch bind groups.
        struct DrawBatch {
            page: (u32, usize),
            vertices: [Vertex; 6],
        }
        let mut batches: Vec<DrawBatch> = Vec::with_capacity(frame.draws.len());
        let runtime_ref: &GpuRuntime = runtime;

        for draw in &frame.draws {
            // A draw's fine and coarse tiles share a page: the shader samples
            // both from one binding.
            let Some(placement) = tiles.place(
                runtime_ref,
                &draw.tile,
                draw.coarse_tile.as_deref(),
                frame_id,
            ) else {
                tiles.dropped += 1;
                continue;
            };
            let (fine_layer, fine_w, fine_h) = placement.fine;
            let (coarse_layer, coarse_size, coarse_border, mip_blend) =
                match (placement.coarse, &draw.coarse_tile) {
                    (Some((layer, w, h)), Some(coarse)) => {
                        (layer, (w, h), coarse.border, draw.mip_blend)
                    }
                    (_, coarse) => {
                        if coarse.is_some() {
                            tiles.unblended += 1;
                        }
                        (fine_layer, (fine_w, fine_h), draw.tile.border, 0.0)
                    }
                };

            batches.push(DrawBatch {
                page: (placement.layer_size, placement.page),
                vertices: quad_vertices(
                    draw,
                    &QuadParams {
                        viewport_size: (frame.width, frame.height),
                        fine_layer: fine_layer as i32,
                        coarse_layer: coarse_layer as i32,
                        fine_size: (fine_w, fine_h),
                        coarse_size,
                        layer_size: placement.layer_size,
                        fine_border: draw.tile.border,
                        coarse_border,
                        mip_blend,
                        orientation: frame.orientation,
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
        let color = frame.shader_color().to_array();
        let icc_table = runtime.icc_table(&frame, frame_id);
        if last_color.as_ref() != Some(&color) {
            runtime
                .queue
                .write_buffer(&runtime.color_buffer, 0, bytemuck::cast_slice(&color));
            *last_color = Some(color);
        }

        // Build draw ranges — batch consecutive quads that share a page so
        // that one `draw()` covers each contiguous run.
        let mut draw_ranges: Vec<((u32, usize), std::ops::Range<u32>)> = Vec::new();
        for (i, b) in batches.iter().enumerate() {
            let start = (i * 6) as u32;
            let end = start + 6;
            if let Some(last) = draw_ranges.last_mut()
                && last.0 == b.page
            {
                last.1.end = end;
                continue;
            }
            draw_ranges.push((b.page, start..end));
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
                        load: wgpu::LoadOp::Clear(frame.clear_color()),
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
            render_pass.set_bind_group(1, &icc_table, &[]);

            for ((layer_size, index), range) in &draw_ranges {
                if let Some(page) = tiles.page(*layer_size, *index) {
                    let bg = if is_bilinear {
                        &page.bilinear_bind_group
                    } else {
                        &page.trilinear_bind_group
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

    // Draws are laid out in the aligned frame; turn each corner onto the
    // surface about the frame centres.
    let orientation = params.orientation;
    let (aligned_w, aligned_h) = orientation.swap(vw, vh);
    let ndc = |x: f32, y: f32| -> [f32; 2] {
        let (px, py) = orientation.to_screen_xy(
            f64::from(x - aligned_w / 2.0),
            f64::from(y - aligned_h / 2.0),
        );
        let (sx, sy) = (px as f32 + vw / 2.0, py as f32 + vh / 2.0);
        [sx / vw * 2.0 - 1.0, 1.0 - sy / vh * 2.0]
    };
    let top_left = ndc(x0, y0);
    let top_right = ndc(x1, y0);
    let bottom_left = ndc(x0, y1);
    let bottom_right = ndc(x1, y1);

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

    let mip_blend = params.mip_blend;
    let fine_tex_size = [fine_w as f32, fine_h as f32];

    [
        Vertex {
            position: top_left,
            fine_uv: [fu_min, fv_min],
            coarse_uv: [coarse_min[0], coarse_min[1]],
            mip_blend,
            fine_layer,
            coarse_layer,
            fine_tex_size,
        },
        Vertex {
            position: top_right,
            fine_uv: [fu_max, fv_min],
            coarse_uv: [coarse_max[0], coarse_min[1]],
            mip_blend,
            fine_layer,
            coarse_layer,
            fine_tex_size,
        },
        Vertex {
            position: bottom_left,
            fine_uv: [fu_min, fv_max],
            coarse_uv: [coarse_min[0], coarse_max[1]],
            mip_blend,
            fine_layer,
            coarse_layer,
            fine_tex_size,
        },
        Vertex {
            position: bottom_left,
            fine_uv: [fu_min, fv_max],
            coarse_uv: [coarse_min[0], coarse_max[1]],
            mip_blend,
            fine_layer,
            coarse_layer,
            fine_tex_size,
        },
        Vertex {
            position: top_right,
            fine_uv: [fu_max, fv_min],
            coarse_uv: [coarse_max[0], coarse_min[1]],
            mip_blend,
            fine_layer,
            coarse_layer,
            fine_tex_size,
        },
        Vertex {
            position: bottom_right,
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
            orientation: Orientation::IDENTITY,
            draws: Vec::new(),
            color,
            icc: None,
            sharpness,
            composite: true,
            readback: None,
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
        // Frames for plugin filters are finished on the CPU.
        let mut for_filters = frame(neutral, 0.5);
        for_filters.composite = false;
        assert!(!for_filters.sharpens());
        assert_ne!(for_filters.fingerprint(), frame(neutral, 0.5).fingerprint());
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
                orientation: Orientation::IDENTITY,
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
                icc: pipeline.color_management().cloned(),
                sharpness: pipeline.sharpness(),
                composite: true,
                readback: None,
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

    /// A turned frame on the GPU is the upright frame turned (as the CPU
    /// turns its frames).
    #[test]
    fn turned_frames_turn_the_composite() {
        let Some(mut renderer) = headless_renderer() else {
            eprintln!("no GPU adapter available; skipping the turned-frame test");
            return;
        };
        let pipeline = ProcessingPipeline::identity();
        let upright = render(&mut renderer, TILE * 2, &pipeline);
        for orientation in Orientation::all() {
            let (tile, _) = tile();
            let (width, height) = orientation.swap(TILE * 2, TILE);
            let slot = SurfaceSlot(1);
            renderer.ensure_surface(slot, width, height).unwrap();
            renderer.pending_frames.insert(
                slot.index(),
                QueuedFrame {
                    width,
                    height,
                    orientation,
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
                    icc: None,
                    sharpness: 0.0,
                    composite: true,
                    readback: None,
                },
            );
            renderer.flush_pending_frames();
            let (_, _, turned) = renderer.read_surface_rgba(slot).unwrap();
            let mut expected = vec![0u8; upright.len()];
            eov_common::blitter::orient_frame(&upright, TILE * 2, TILE, orientation, &mut expected);
            assert!(
                max_difference(&turned, &expected) <= 1,
                "{orientation:?} differs by {}",
                max_difference(&turned, &expected)
            );
        }
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

    /// The colour management on the GPU (the slide's table, interpolated in
    /// the tile shader) against the CPU reference.
    #[test]
    fn colour_managed_gpu_frames_match_the_cpu_pipeline() {
        let Some(mut renderer) = headless_renderer() else {
            eprintln!("no GPU adapter available; skipping the colour management comparison");
            return;
        };
        let (_, inner) = tile();
        let transform = Arc::new(
            DisplayTransform::from_icc(&eov_common::color_management::test_profiles::adobe_rgb())
                .unwrap(),
        );
        for settings in [
            ColorSettings::default(),
            ColorSettings {
                tone: ToneAdjustments::new(1.25, 0.03, 1.15),
                ..ColorSettings::default()
            },
        ] {
            let pipeline = settings
                .pipeline(None)
                .with_color_management(Some(Arc::clone(&transform)));
            let gpu = render(&mut renderer, TILE * 2, &pipeline);
            let mut cpu = vec![0u8; (TILE * 2 * TILE * 4) as usize];
            for y in 0..TILE as usize {
                let row = &inner[y * TILE as usize * 4..(y + 1) * TILE as usize * 4];
                cpu[y * TILE as usize * 8..y * TILE as usize * 8 + row.len()].copy_from_slice(row);
            }
            pipeline.apply_rgba(&mut cpu, TILE * 2, TILE, crate::color::VIEWPORT_BACKGROUND);
            assert!(
                max_difference(&gpu, &cpu) <= 1,
                "colour-managed GPU and CPU frames differ by {}",
                max_difference(&gpu, &cpu)
            );
            // The conversion changed the slide, not the background.
            let unmanaged = render(&mut renderer, TILE * 2, &settings.pipeline(None));
            assert!(max_difference(&gpu[..16], &unmanaged[..16]) > 1);
            let last = gpu.len() - 4;
            assert_eq!(&gpu[last..], &[30, 30, 30, 255]);
        }
    }

    fn tagged_frame(width: u32, generation: u64) -> QueuedFrame {
        let (tile, _) = tile();
        QueuedFrame {
            width,
            height: TILE,
            orientation: Orientation::IDENTITY,
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
            color: ProcessingPipeline::identity().gpu_uniform(),
            icc: None,
            sharpness: 0.0,
            composite: false,
            readback: Some(ReadbackTag {
                pane_uid: 7,
                file_id: 1,
                viewport: eov_common::Viewport::new(width as f64, TILE as f64, 100.0, 100.0),
                scale: 1.0,
                generation,
                finish: crate::render_pool::CpuRenderPostProcess {
                    pipeline: Arc::new(ProcessingPipeline::identity()),
                    background: crate::color::VIEWPORT_BACKGROUND,
                },
            }),
        }
    }

    fn wait_for_readbacks(renderer: &mut GpuRenderer) -> Vec<ReadbackFrame> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let frames = renderer.poll_readbacks();
            if !frames.is_empty() || std::time::Instant::now() > deadline {
                return frames;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    #[test]
    fn edge_tiles_share_their_level_s_layer_size() {
        let tile = |size: u32, w: u32, h: u32| {
            TileData::new(
                TileCoord::new(1, 0, 0, 0, size),
                vec![0; ((w + 2) * (h + 2) * 4) as usize],
                w,
                h,
                1,
            )
        };
        assert_eq!(layer_size_class(&tile(256, 256, 256)), 288);
        assert_eq!(layer_size_class(&tile(256, 100, 256)), 288);
        assert_eq!(layer_size_class(&tile(256, 37, 12)), 288);
        assert_eq!(layer_size_class(&tile(1024, 1024, 1024)), 1056);
        assert_eq!(layer_size_class(&tile(8, 8, 8)), 64);
        // Pages hold 64 MiB, within the device's layer limit.
        assert_eq!(page_layers(288, 256), 202);
        assert_eq!(page_layers(1056, 256), 15);
        assert_eq!(page_layers(64, 256), 256);
        assert_eq!(page_layers(4096, 256), MIN_PAGE_LAYERS);
    }

    #[test]
    fn layers_the_current_frame_uses_are_never_evicted() {
        let coord = |x| TileCoord::new(1, 0, x, 0, 256);
        let mut slots = LayerSlots::new(4, 1);
        for x in 0..4 {
            assert!(slots.allocate(coord(x), 258, 258, 1).is_some());
        }
        assert_eq!(slots.room(1), 0);
        assert!(
            slots.allocate(coord(9), 258, 258, 1).is_none(),
            "all in use"
        );

        // The next frame uses two of them; the other two can make room.
        assert!(slots.touch(&coord(0), 2).is_some());
        assert!(slots.touch(&coord(1), 2).is_some());
        assert_eq!(slots.room(2), 2);
        assert!(slots.allocate(coord(9), 258, 258, 2).is_some());
        assert!(slots.contains(&coord(0)) && slots.contains(&coord(1)));
        assert!(slots.contains(&coord(9)));
        assert_eq!(slots.room(2), 1);
    }

    /// RND-2: a frame needing more tiles than one texture array holds gets
    /// more pages, and draws every tile.
    #[test]
    fn frames_with_more_tiles_than_a_page_draw_them_all() {
        let Some(mut renderer) = headless_renderer() else {
            eprintln!("no GPU adapter available; skipping the tile page test");
            return;
        };
        const SIDE: u32 = 8;
        let (cols, rows) = (24u32, 16u32);
        let color = |i: u32| {
            [
                (i * 37 % 251) as u8,
                (i * 11 % 241) as u8,
                (i / 3 % 239) as u8,
            ]
        };
        let draws: Vec<TileDraw> = (0..cols * rows)
            .map(|i| {
                let (x, y) = (i % cols, i / cols);
                let c = color(i);
                let data = (0..(SIDE + 2) * (SIDE + 2))
                    .flat_map(|_| [c[0], c[1], c[2], 255])
                    .collect();
                TileDraw {
                    tile: Arc::new(TileData::new(
                        TileCoord::new(1, 0, x.into(), y.into(), SIDE),
                        data,
                        SIDE,
                        SIDE,
                        1,
                    )),
                    coarse_tile: None,
                    screen_x: (x * SIDE) as f32,
                    screen_y: (y * SIDE) as f32,
                    screen_w: SIDE as f32,
                    screen_h: SIDE as f32,
                    coarse_uv_min: [0.0, 0.0],
                    coarse_uv_max: [1.0, 1.0],
                    mip_blend: 0.0,
                    filtering_mode: FilteringMode::Bilinear,
                }
            })
            .collect();
        let (width, height) = (cols * SIDE, rows * SIDE);
        let slot = SurfaceSlot(0);
        renderer.ensure_surface(slot, width, height).unwrap();
        renderer.pending_frames.insert(
            slot.index(),
            QueuedFrame {
                width,
                height,
                orientation: Orientation::IDENTITY,
                draws,
                color: ProcessingPipeline::identity().gpu_uniform(),
                icc: None,
                sharpness: 0.0,
                composite: true,
                readback: None,
            },
        );
        renderer.flush_pending_frames();
        assert!(
            renderer.tiles.pages[&64].len() >= 2,
            "{} tiles need two pages",
            cols * rows
        );
        let (_, _, pixels) = renderer.read_surface_rgba(slot).unwrap();
        for i in 0..cols * rows {
            let x = (i % cols) * SIDE + SIDE / 2;
            let y = (i / cols) * SIDE + SIDE / 2;
            let at = ((y * width + x) * 4) as usize;
            assert!(
                max_difference(&pixels[at..at + 3], &color(i)) <= 1,
                "tile {i} is {:?}, not {:?}",
                &pixels[at..at + 3],
                color(i)
            );
        }

        // Closing the pane frees its surface.
        renderer.retain_slots(0);
        assert!(renderer.surface_image(slot).is_none());
    }

    #[test]
    fn rendered_frames_are_read_back_without_blocking() {
        let Some(mut renderer) = headless_renderer() else {
            eprintln!("no GPU adapter available; skipping the readback test");
            return;
        };
        let slot = SurfaceSlot(0);
        let width = TILE + 16;
        renderer.ensure_surface(slot, width, TILE).unwrap();
        renderer
            .pending_frames
            .insert(slot.index(), tagged_frame(width, 1));
        renderer.flush_pending_frames();
        assert!(renderer.readbacks_pending());
        let frames = wait_for_readbacks(&mut renderer);
        assert_eq!(frames.len(), 1);
        let frame = &frames[0];
        assert_eq!((frame.tag.pane_uid, frame.tag.generation), (7, 1));
        assert_eq!((frame.width, frame.height), (width, TILE));
        let (_, _, surface) = renderer.read_surface_rgba(slot).unwrap();
        assert_eq!(frame.pixels, surface, "rows are unpadded");
        assert!(!renderer.readbacks_pending());
        // Frames for plugin filters stay transparent outside the slide.
        let row_end = width as usize * 4;
        assert_eq!(&frame.pixels[row_end - 4..row_end], &[0, 0, 0, 0]);
        assert_eq!(frame.pixels[3], 255, "the slide is opaque");

        // An unchanged frame is not rendered again, but still read back.
        let _ = renderer.queue_frame(slot, tagged_frame(width, 2));
        assert!(renderer.pending_frames.is_empty(), "nothing to render");
        renderer.flush_pending_frames();
        let frames = wait_for_readbacks(&mut renderer);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].tag.generation, 2);
        assert_eq!(frames[0].pixels, surface);
    }
}

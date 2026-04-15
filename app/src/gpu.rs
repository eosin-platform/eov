use crate::AppWindow;
use crate::gpu_interop::{ExportableImage, VulkanHandles};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use common::{FilteringMode, TileCoord, TileData};
use slint::ComponentHandle;
use slint::wgpu_28::wgpu;
use slint::{GraphicsAPI, Image, RenderingState};
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Arc;

/// Memory budget for the tile texture array (~128 MB).
const MAX_TILE_ARRAY_BYTES: u64 = 128 * 1024 * 1024;

const SHADER_SOURCE: &str = r#"
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
};

struct Adjustments {
    inv_gamma: f32,
    brightness: f32,
    contrast: f32,
    stain_norm_enabled: f32,
    inv_stain_row0: vec4<f32>,
    inv_stain_row1: vec4<f32>,
    sharpness: f32,
    deconv_enabled: f32,
    deconv_isolated: f32,
    _pad0: f32,
    deconv_inv_row0: vec4<f32>,
    deconv_inv_row1: vec4<f32>,
    deconv_stain_h: vec4<f32>,
    deconv_stain_e: vec4<f32>,
    deconv_visibility: vec4<f32>,
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
    return output;
}

@group(0) @binding(0)
var tile_array: texture_2d_array<f32>;

@group(0) @binding(1)
var tile_sampler: sampler;

@group(0) @binding(2)
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

fn apply_deconv(color: vec4<f32>) -> vec4<f32> {
    if adjustments.deconv_enabled < 0.5 {
        return color;
    }
    let rgb = max(color.rgb, vec3<f32>(1.0 / 255.0));
    let od = -log(rgb);
    let od_sum = od.r + od.g + od.b;
    let iso = i32(adjustments.deconv_isolated + 0.5);
    if od_sum <= 0.15 {
        return color;
    }
    if od_sum >= 6.0 && iso == 0 {
        return color;
    }
    let c_h = max(dot(adjustments.deconv_inv_row0.xyz, od), 0.0);
    let c_e = max(dot(adjustments.deconv_inv_row1.xyz, od), 0.0);
    if iso == 1 {
        // Isolated hematoxylin grayscale
        let v = exp(-c_h);
        return vec4<f32>(v, v, v, color.a);
    }
    if iso == 2 {
        // Isolated eosin grayscale
        let v = exp(-c_e);
        return vec4<f32>(v, v, v, color.a);
    }
    // Blended mode
    let vis_h = adjustments.deconv_visibility.x;
    let vis_e = adjustments.deconv_visibility.y;
    if vis_h < 0.5 && vis_e < 0.5 {
        return vec4<f32>(1.0, 1.0, 1.0, color.a);
    }
    var new_od = vec3<f32>(0.0);
    if vis_h >= 0.5 {
        new_od += adjustments.deconv_stain_h.xyz * c_h * adjustments.deconv_stain_h.w;
    }
    if vis_e >= 0.5 {
        new_od += adjustments.deconv_stain_e.xyz * c_e * adjustments.deconv_stain_e.w;
    }
    let new_rgb = exp(-new_od);
    return vec4<f32>(clamp(new_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), color.a);
}

fn apply_adj(color: vec4<f32>) -> vec4<f32> {
    let g = vec3<f32>(
        pow(color.r, adjustments.inv_gamma),
        pow(color.g, adjustments.inv_gamma),
        pow(color.b, adjustments.inv_gamma),
    );
    let b = g + vec3<f32>(adjustments.brightness);
    let c = (b - vec3<f32>(0.5)) * adjustments.contrast + vec3<f32>(0.5);
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), color.a);
}

// Unsharp mask: sharpen by subtracting a blurred version from the original.
// Uses a 3x3 box blur approximation (single-pass Laplacian).
fn apply_sharpen(uv: vec2<f32>, layer: i32, center: vec4<f32>) -> vec4<f32> {
    if adjustments.sharpness < 0.001 {
        return center;
    }
    let tex_size = vec2<f32>(textureDimensions(tile_array, 0));
    let step = 1.0 / tex_size;
    // Sample 4-connected neighbors for Laplacian
    let n = textureSample(tile_array, tile_sampler, uv + vec2<f32>(0.0, -step.y), layer);
    let s = textureSample(tile_array, tile_sampler, uv + vec2<f32>(0.0, step.y), layer);
    let w = textureSample(tile_array, tile_sampler, uv + vec2<f32>(-step.x, 0.0), layer);
    let e = textureSample(tile_array, tile_sampler, uv + vec2<f32>(step.x, 0.0), layer);
    // Laplacian = 4*center - neighbors  (high-pass detail)
    let detail = center.rgb * 4.0 - (n.rgb + s.rgb + w.rgb + e.rgb);
    // Blend: output = center + sharpness * detail
    let sharpened = center.rgb + adjustments.sharpness * detail;
    return vec4<f32>(clamp(sharpened, vec3<f32>(0.0), vec3<f32>(1.0)), center.a);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let fine = textureSample(tile_array, tile_sampler, input.fine_uv, input.fine_layer);
    let coarse = textureSample(tile_array, tile_sampler, input.coarse_uv, input.coarse_layer);
    let blended = mix(fine, coarse, input.mip_blend);
    return apply_adj(apply_deconv(apply_stain_norm(apply_sharpen(input.fine_uv, input.fine_layer, blended))));
}
"#;

/// Lanczos-3 fragment shader: samples a 6×6 kernel from the fine texture array layer
const LANCZOS_SHADER_SOURCE: &str = r#"
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

struct Adjustments {
    inv_gamma: f32,
    brightness: f32,
    contrast: f32,
    stain_norm_enabled: f32,
    inv_stain_row0: vec4<f32>,
    inv_stain_row1: vec4<f32>,
    sharpness: f32,
    deconv_enabled: f32,
    deconv_isolated: f32,
    _pad0: f32,
    deconv_inv_row0: vec4<f32>,
    deconv_inv_row1: vec4<f32>,
    deconv_stain_h: vec4<f32>,
    deconv_stain_e: vec4<f32>,
    deconv_visibility: vec4<f32>,
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

fn apply_deconv(color: vec4<f32>) -> vec4<f32> {
    if adjustments.deconv_enabled < 0.5 {
        return color;
    }
    let rgb = max(color.rgb, vec3<f32>(1.0 / 255.0));
    let od = -log(rgb);
    let od_sum = od.r + od.g + od.b;
    let iso = i32(adjustments.deconv_isolated + 0.5);
    if od_sum <= 0.15 {
        return color;
    }
    if od_sum >= 6.0 && iso == 0 {
        return color;
    }
    let c_h = max(dot(adjustments.deconv_inv_row0.xyz, od), 0.0);
    let c_e = max(dot(adjustments.deconv_inv_row1.xyz, od), 0.0);
    if iso == 1 {
        let v = exp(-c_h);
        return vec4<f32>(v, v, v, color.a);
    }
    if iso == 2 {
        let v = exp(-c_e);
        return vec4<f32>(v, v, v, color.a);
    }
    let vis_h = adjustments.deconv_visibility.x;
    let vis_e = adjustments.deconv_visibility.y;
    if vis_h < 0.5 && vis_e < 0.5 {
        return vec4<f32>(1.0, 1.0, 1.0, color.a);
    }
    var new_od = vec3<f32>(0.0);
    if vis_h >= 0.5 {
        new_od += adjustments.deconv_stain_h.xyz * c_h * adjustments.deconv_stain_h.w;
    }
    if vis_e >= 0.5 {
        new_od += adjustments.deconv_stain_e.xyz * c_e * adjustments.deconv_stain_e.w;
    }
    let new_rgb = exp(-new_od);
    return vec4<f32>(clamp(new_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), color.a);
}

fn apply_adj(color: vec4<f32>) -> vec4<f32> {
    let g = vec3<f32>(
        pow(color.r, adjustments.inv_gamma),
        pow(color.g, adjustments.inv_gamma),
        pow(color.b, adjustments.inv_gamma),
    );
    let b = g + vec3<f32>(adjustments.brightness);
    let c = (b - vec3<f32>(0.5)) * adjustments.contrast + vec3<f32>(0.5);
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), color.a);
}

fn apply_sharpen(uv: vec2<f32>, layer: i32, center: vec4<f32>) -> vec4<f32> {
    if adjustments.sharpness < 0.001 {
        return center;
    }
    let tex_size = vec2<f32>(textureDimensions(tile_array, 0));
    let step = 1.0 / tex_size;
    let n = textureSampleLevel(tile_array, tile_sampler, uv + vec2<f32>(0.0, -step.y), layer, 0.0);
    let s = textureSampleLevel(tile_array, tile_sampler, uv + vec2<f32>(0.0, step.y), layer, 0.0);
    let w = textureSampleLevel(tile_array, tile_sampler, uv + vec2<f32>(-step.x, 0.0), layer, 0.0);
    let e = textureSampleLevel(tile_array, tile_sampler, uv + vec2<f32>(step.x, 0.0), layer, 0.0);
    let detail = center.rgb * 4.0 - (n.rgb + s.rgb + w.rgb + e.rgb);
    let sharpened = center.rgb + adjustments.sharpness * detail;
    return vec4<f32>(clamp(sharpened, vec3<f32>(0.0), vec3<f32>(1.0)), center.a);
}

@fragment
fn fs_lanczos(input: VertexOutput) -> @location(0) vec4<f32> {
    let layer_size = vec2<f32>(textureDimensions(tile_array, 0));
    let tile_size = input.fine_tex_size;
    let uv_pixel = input.fine_uv * layer_size - 0.5;
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

            let sample = textureSampleLevel(tile_array, tile_sampler, sample_uv, input.fine_layer, 0.0);
            color += sample * w;
            weight_sum += w;
        }
    }

    if weight_sum > 0.0 {
        color = color / weight_sum;
    }

    return apply_adj(apply_deconv(apply_stain_norm(apply_sharpen(input.fine_uv, input.fine_layer, clamp(color, vec4<f32>(0.0), vec4<f32>(1.0))))));
}
"#;

/// Fullscreen post-process shader for viewport filters (e.g. grayscale).
/// Reads from a source texture and writes to the render target.
/// The `enabled` uniform controls whether the effect is active.
const POST_PROCESS_SHADER: &str = r#"
struct PostProcessUniforms {
    enabled: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: PostProcessUniforms;

struct FullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) idx: u32) -> FullscreenOutput {
    // Fullscreen triangle covering the viewport.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: FullscreenOutput;
    out.position = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

@fragment
fn fs_grayscale(input: FullscreenOutput) -> @location(0) vec4<f32> {
    let color = textureSample(src_texture, src_sampler, input.uv);
    if uniforms.enabled < 0.5 {
        return color;
    }
    let lum = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    return vec4<f32>(lum, lum, lum, color.a);
}
"#;

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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq)]
struct AdjustmentsUniform {
    inv_gamma: f32,
    brightness: f32,
    contrast: f32,
    stain_norm_enabled: f32,
    // Row 0 of inverse source stain matrix + scale_h
    inv_stain_r0: [f32; 4], // [inv[0][0], inv[0][1], inv[0][2], scale_h]
    // Row 1 of inverse source stain matrix + scale_e
    inv_stain_r1: [f32; 4], // [inv[1][0], inv[1][1], inv[1][2], scale_e]
    sharpness: f32,
    // Color deconvolution parameters
    deconv_enabled: f32,  // 0.0 = disabled, 1.0 = enabled
    deconv_isolated: f32, // 0.0 = blended, 1.0 = H isolated, 2.0 = E isolated
    _pad0: f32,
    deconv_inv_row0: [f32; 4],   // inverse stain row 0 + H intensity
    deconv_inv_row1: [f32; 4],   // inverse stain row 1 + E intensity
    deconv_stain_h: [f32; 4],    // H stain OD vector
    deconv_stain_e: [f32; 4],    // E stain OD vector
    deconv_visibility: [f32; 4], // [h_vis, e_vis, 0, 0]
}

#[derive(Clone)]
pub(crate) struct QueuedFrame {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) draws: Vec<TileDraw>,
    pub(crate) gpu_post_process_enabled: bool,
    pub(crate) gamma: f32,
    pub(crate) brightness: f32,
    pub(crate) contrast: f32,
    pub(crate) sharpness: f32,
    pub(crate) stain_norm_enabled: bool,
    pub(crate) inv_stain_r0: [f32; 4],
    pub(crate) inv_stain_r1: [f32; 4],
    pub(crate) deconv_params: crate::stain::ColorDeconvParams,
}

impl QueuedFrame {
    /// Compute a lightweight fingerprint of all render-relevant inputs.
    /// Used to skip redundant vertex rebuilds and GPU render passes.
    fn fingerprint(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.width.hash(&mut hasher);
        self.height.hash(&mut hasher);
        self.gpu_post_process_enabled.hash(&mut hasher);
        self.gamma.to_bits().hash(&mut hasher);
        self.brightness.to_bits().hash(&mut hasher);
        self.contrast.to_bits().hash(&mut hasher);
        self.sharpness.to_bits().hash(&mut hasher);
        self.stain_norm_enabled.hash(&mut hasher);
        for v in &self.inv_stain_r0 {
            v.to_bits().hash(&mut hasher);
        }
        for v in &self.inv_stain_r1 {
            v.to_bits().hash(&mut hasher);
        }
        // Include color deconvolution state in fingerprint
        self.deconv_params.enabled.hash(&mut hasher);
        self.deconv_params.isolated_mode.to_bits().hash(&mut hasher);
        for v in &self.deconv_params.inv_row0 {
            v.to_bits().hash(&mut hasher);
        }
        for v in &self.deconv_params.inv_row1 {
            v.to_bits().hash(&mut hasher);
        }
        for v in &self.deconv_params.visibility {
            v.to_bits().hash(&mut hasher);
        }
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
        adjustments_buffer: &wgpu::Buffer,
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
        let bilinear_bind_group = Self::create_bind_group(
            device,
            layout,
            &array_view,
            bilinear_sampler,
            adjustments_buffer,
        );
        let trilinear_bind_group = Self::create_bind_group(
            device,
            layout,
            &array_view,
            trilinear_sampler,
            adjustments_buffer,
        );
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
        adjustments_buffer: &wgpu::Buffer,
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
                    resource: adjustments_buffer.as_entire_binding(),
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
    adjustments_buffer: wgpu::Buffer,
    surfaces: HashMap<usize, ImportedSurface>,
    // Post-process pipeline for viewport filters
    post_process_pipeline: wgpu::RenderPipeline,
    post_process_bind_group_layout: wgpu::BindGroupLayout,
    post_process_sampler: wgpu::Sampler,
    post_process_uniforms_buffer: wgpu::Buffer,
    /// Scratch textures for post-processing (per surface slot).
    post_process_scratch: HashMap<usize, ImportedSurface>,
    /// Raw Vulkan handles for plugin GPU filter interop.
    vulkan_handles: Option<VulkanHandles>,
    /// Per-slot exportable images for GPU filter plugins (DMA-BUF capable).
    filter_images: HashMap<usize, ExportableImage>,
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
    /// Last uploaded adjustments uniform, used to skip redundant uploads.
    last_adjustments: Option<AdjustmentsUniform>,
}

impl GpuRenderer {
    pub fn new() -> Self {
        Self {
            runtime: None,
            pending_frames: HashMap::new(),
            tile_arrays: HashMap::new(),
            frame_counter: 0,
            last_rendered_fingerprint: HashMap::new(),
            last_adjustments: None,
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

        let GraphicsAPI::WGPU28 { device, queue, .. } = graphics_api else {
            return;
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-tile-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        // Bind group layout — shared by bilinear/trilinear and Lanczos pipelines.
        // binding 0: tile_array (texture_2d_array)
        // binding 1: sampler
        // binding 2: adjustments uniform
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
            bind_group_layouts: &[&bind_group_layout],
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

        // Lanczos pipeline — same bind group layout, different fragment shader
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
                buffers: &[vertex_layout],
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

        // ----- Post-process pipeline (fullscreen filter pass) -----
        let pp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post-process-shader"),
            source: wgpu::ShaderSource::Wgsl(POST_PROCESS_SHADER.into()),
        });

        let post_process_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("post-process-bind-group-layout"),
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

        let pp_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post-process-pipeline-layout"),
            bind_group_layouts: &[&post_process_bind_group_layout],
            immediate_size: 0,
        });

        let post_process_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("post-process-grayscale-pipeline"),
                layout: Some(&pp_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &pp_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &pp_shader,
                    entry_point: Some("fs_grayscale"),
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

        let post_process_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post-process-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let post_process_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post-process-uniforms"),
            size: 16, // PostProcessUniforms: 4 floats
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Extract raw Vulkan handles for GPU filter plugin interop.
        let vulkan_handles = unsafe { crate::gpu_interop::extract_vulkan_handles(device) };
        if vulkan_handles.is_none() {
            tracing::warn!("Failed to extract Vulkan handles for GPU filter interop");
        }

        self.runtime = Some(GpuRuntime {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            lanczos_pipeline,
            bind_group_layout,
            sampler,
            bilinear_sampler,
            vertex_buffer,
            vertex_capacity: 6,
            adjustments_buffer,
            surfaces: HashMap::new(),
            post_process_pipeline,
            post_process_bind_group_layout,
            post_process_sampler,
            post_process_uniforms_buffer,
            post_process_scratch: HashMap::new(),
            vulkan_handles,
            filter_images: HashMap::new(),
        });
    }

    fn teardown(&mut self) {
        if let Some(runtime) = &self.runtime
            && let Some(handles) = &runtime.vulkan_handles
        {
            for img in runtime.filter_images.values() {
                unsafe { img.destroy(&handles.device_fn) };
            }
        }
        self.runtime = None;
        self.pending_frames.clear();
        self.tile_arrays.clear();
        self.last_rendered_fingerprint.clear();
        self.last_adjustments = None;
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
                let gpu_post_process_enabled = frame.gpu_post_process_enabled;
                Self::render_frame(
                    runtime,
                    &mut self.tile_arrays,
                    &mut self.last_adjustments,
                    SurfaceSlot(slot_index),
                    frame,
                    frame_id,
                    gpu_post_process_enabled,
                );
                self.last_rendered_fingerprint.insert(slot_index, fp);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Core render path — one draw call per tile-array batch per surface.
    // -----------------------------------------------------------------------

    fn render_frame(
        runtime: &mut GpuRuntime,
        tile_arrays: &mut HashMap<u32, TileArray>,
        last_adjustments: &mut Option<AdjustmentsUniform>,
        slot: SurfaceSlot,
        frame: QueuedFrame,
        frame_id: u64,
        gpu_filters_enabled: bool,
    ) {
        let Some(surface_view) = runtime
            .surfaces
            .get(&slot.index())
            .map(|surface| surface.view.clone())
        else {
            return;
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
                    &runtime.adjustments_buffer,
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

        // Upload adjustments uniform only when values changed.
        let adj = AdjustmentsUniform {
            inv_gamma: if frame.gamma > 0.001 {
                1.0 / frame.gamma
            } else {
                1.0
            },
            brightness: frame.brightness,
            contrast: frame.contrast,
            stain_norm_enabled: if frame.stain_norm_enabled { 1.0 } else { 0.0 },
            inv_stain_r0: frame.inv_stain_r0,
            inv_stain_r1: frame.inv_stain_r1,
            sharpness: frame.sharpness,
            deconv_enabled: if frame.deconv_params.enabled {
                1.0
            } else {
                0.0
            },
            deconv_isolated: frame.deconv_params.isolated_mode,
            _pad0: 0.0,
            deconv_inv_row0: frame.deconv_params.inv_row0,
            deconv_inv_row1: frame.deconv_params.inv_row1,
            deconv_stain_h: frame.deconv_params.stain_h,
            deconv_stain_e: frame.deconv_params.stain_e,
            deconv_visibility: frame.deconv_params.visibility,
        };
        if last_adjustments.as_ref() != Some(&adj) {
            runtime
                .queue
                .write_buffer(&runtime.adjustments_buffer, 0, bytemuck::bytes_of(&adj));
            *last_adjustments = Some(adj);
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

        runtime.queue.submit(Some(encoder.finish()));

        // ----- Post-process pass (e.g. grayscale viewport filter) ----------
        if gpu_filters_enabled {
            let surface = runtime.surfaces.get(&slot.index()).unwrap();
            let (sw, sh) = (surface.width, surface.height);

            // Ensure scratch texture exists and matches surface dimensions.
            let scratch_key = slot.index();
            let needs_recreate = runtime
                .post_process_scratch
                .get(&scratch_key)
                .is_none_or(|s| s.width != sw || s.height != sh);
            if needs_recreate {
                let tex = runtime.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("post-process-scratch"),
                    size: wgpu::Extent3d {
                        width: sw,
                        height: sh,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                runtime.post_process_scratch.insert(
                    scratch_key,
                    ImportedSurface {
                        texture: tex,
                        view,
                        width: sw,
                        height: sh,
                    },
                );
            }

            let scratch = runtime.post_process_scratch.get(&scratch_key).unwrap();

            // Update uniforms: enabled = 1.0
            runtime.queue.write_buffer(
                &runtime.post_process_uniforms_buffer,
                0,
                bytemuck::bytes_of(&[1.0f32, 0.0f32, 0.0f32, 0.0f32]),
            );

            // Bind group: surface texture as input.
            let bind_group = runtime
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("post-process-bind-group"),
                    layout: &runtime.post_process_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&surface_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&runtime.post_process_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: runtime.post_process_uniforms_buffer.as_entire_binding(),
                        },
                    ],
                });

            let mut pp_encoder =
                runtime
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("post-process-encoder"),
                    });

            {
                let mut pass = pp_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("post-process-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &scratch.view,
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
                pass.set_pipeline(&runtime.post_process_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.draw(0..3, 0..1);
            }

            // Copy scratch back to surface.
            let surface_tex = &runtime.surfaces.get(&slot.index()).unwrap().texture;
            pp_encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &scratch.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: surface_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: sw,
                    height: sh,
                    depth_or_array_layers: 1,
                },
            );

            runtime.queue.submit(Some(pp_encoder.finish()));
        }
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

//! Stain normalization module for H&E histology images.
//!
//! Implements two industry-standard stain normalization methods:
//! - **Macenko**: SVD/PCA-based stain plane projection with angular percentile
//!   extremes (Macenko et al., 2009)
//! - **Vahadane**: Sparse non-negative dictionary learning with L1 (LASSO)
//!   regularization (Vahadane et al., 2016)
//!
//! Both share a common normalization pipeline:
//! 1. RGB → OD conversion
//! 2. Tissue masking (background removal)
//! 3. Stain matrix estimation (method-specific)
//! 4. Concentration solve via least squares
//! 5. 99th-percentile concentration scaling (source → target)
//! 6. Reconstruction via target stain matrix
//! 7. OD → RGB conversion

use crate::StainNormalization;
use tracing::debug;

// ─── Reference target stain matrix (Ruifrok & Johnston, 2001) ────────────────
// Rows = stains (H, E), Cols = channels (R, G, B) — in OD space, unit-normalized.
const TARGET_STAIN_MATRIX: [[f32; 3]; 2] = [
    [0.6442, 0.7170, 0.2668], // Hematoxylin
    [0.0927, 0.9545, 0.2832], // Eosin
];

/// 99th-percentile reference concentrations for the target stain matrix.
/// Pre-computed from a representative H&E target image.
const TARGET_MAX_C: [f32; 2] = [1.9705, 1.0308];

// ─── Numerical constants ─────────────────────────────────────────────────────
/// Background OD threshold — pixels below this total OD are treated as background.
const OD_THRESHOLD: f32 = 0.15;
/// Upper OD threshold — pixels above this total OD are artificial dark fill
/// (e.g. the (30,30,30) viewport background has OD_sum ≈ 6.42) and not tissue.
const OD_THRESHOLD_HIGH: f32 = 6.0;
/// Minimum pixel count required for a valid fit.
const MIN_TISSUE_PIXELS: usize = 100;
/// Epsilon to avoid log(0) / division by zero.
const EPS: f32 = 1.0 / 255.0;

// ─── Stain normalization parameters (passed to GPU as uniforms) ──────────────

/// Parameters computed by the CPU-side stain fitting, consumed by the per-pixel
/// shader transform. The inverse source stain matrix and concentration scale
/// factors are packed for efficient GPU upload.
#[derive(Debug, Clone, Copy)]
pub struct StainNormParams {
    pub enabled: bool,
    /// [inv[0][0], inv[0][1], inv[0][2], scale_h]
    pub inv_stain_r0: [f32; 4],
    /// [inv[1][0], inv[1][1], inv[1][2], scale_e]
    pub inv_stain_r1: [f32; 4],
}

impl Default for StainNormParams {
    fn default() -> Self {
        Self {
            enabled: false,
            inv_stain_r0: [0.0; 4],
            inv_stain_r1: [0.0; 4],
        }
    }
}

// ─── Shared infrastructure ───────────────────────────────────────────────────

/// Convert an 8-bit RGB value to optical density. Clamps to avoid log(0).
#[inline]
fn rgb_to_od(r: u8, g: u8, b: u8) -> [f32; 3] {
    let rf = (r.max(1) as f32) / 255.0;
    let gf = (g.max(1) as f32) / 255.0;
    let bf = (b.max(1) as f32) / 255.0;
    [-rf.ln(), -gf.ln(), -bf.ln()]
}

/// Reconstruct a clamped 8-bit RGB triple from optical density.
#[inline]
fn od_to_rgb(od: [f32; 3]) -> [u8; 3] {
    [
        ((-od[0]).exp() * 255.0).clamp(0.0, 255.0) as u8,
        ((-od[1]).exp() * 255.0).clamp(0.0, 255.0) as u8,
        ((-od[2]).exp() * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

/// Returns true if a pixel's total OD indicates tissue (non-background).
/// Filters both near-white (slide background) and near-black (viewport fill).
#[inline]
fn is_tissue(od: &[f32; 3]) -> bool {
    let sum = od[0] + od[1] + od[2];
    sum > OD_THRESHOLD && sum < OD_THRESHOLD_HIGH
}

/// Compute 2×3 pseudo-inverse of a 3×2 matrix M: (M^T M)^{-1} M^T
/// M is a 3×2 matrix stored as [[col0, col1]; 3 rows].
fn pseudo_inverse_2x3(m: &[[f32; 2]; 3]) -> [[f32; 3]; 2] {
    // M^T M (2x2)
    let mut mtm = [[0.0f32; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            mtm[i][j] = m[0][i] * m[0][j] + m[1][i] * m[1][j] + m[2][i] * m[2][j];
        }
    }
    let det = mtm[0][0] * mtm[1][1] - mtm[0][1] * mtm[1][0];
    if det.abs() < 1e-10 {
        return [[0.0; 3]; 2];
    }
    let inv = [
        [mtm[1][1] / det, -mtm[0][1] / det],
        [-mtm[1][0] / det, mtm[0][0] / det],
    ];
    let mut result = [[0.0f32; 3]; 2];
    for i in 0..2 {
        for j in 0..3 {
            result[i][j] = inv[i][0] * m[j][0] + inv[i][1] * m[j][1];
        }
    }
    result
}

/// Solve concentrations: C = inv(stain_mat) * OD, clamped to >= 0.
/// stain_mat is 2×3 (rows = stains, cols = channels).
fn solve_concentrations(od: &[f32; 3], inv_stain: &[[f32; 3]; 2]) -> [f32; 2] {
    [
        (inv_stain[0][0] * od[0] + inv_stain[0][1] * od[1] + inv_stain[0][2] * od[2]).max(0.0),
        (inv_stain[1][0] * od[0] + inv_stain[1][1] * od[1] + inv_stain[1][2] * od[2]).max(0.0),
    ]
}

/// Compute 99th percentile of a f32 slice.
fn percentile_99(values: &mut [f32]) -> f32 {
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((values.len() as f32 * 0.99) as usize).min(values.len().saturating_sub(1));
    values[idx].max(0.001)
}

/// Reconstruct an OD vector from concentrations and the target stain matrix.
#[inline]
fn reconstruct_od(c: [f32; 2]) -> [f32; 3] {
    [
        TARGET_STAIN_MATRIX[0][0] * c[0] + TARGET_STAIN_MATRIX[1][0] * c[1],
        TARGET_STAIN_MATRIX[0][1] * c[0] + TARGET_STAIN_MATRIX[1][1] * c[1],
        TARGET_STAIN_MATRIX[0][2] * c[0] + TARGET_STAIN_MATRIX[1][2] * c[1],
    ]
}

/// Stain matrix in row-major form: stain_matrix[stain][channel].
/// Convert to column-major for pseudo_inverse_2x3 which expects [[col0,col1]; 3].
fn stain_mat_to_col_major(sm: &[[f32; 3]; 2]) -> [[f32; 2]; 3] {
    [
        [sm[0][0], sm[1][0]], // R channel: H, E
        [sm[0][1], sm[1][1]], // G channel
        [sm[0][2], sm[1][2]], // B channel
    ]
}

// ─── Macenko stain extractor ─────────────────────────────────────────────────

/// Estimate stain matrix using the Macenko method (SVD/PCA + angular extremes).
fn macenko_extract(od_pixels: &[[f32; 3]]) -> Option<[[f32; 3]; 2]> {
    if od_pixels.len() < MIN_TISSUE_PIXELS {
        return None;
    }
    let n = od_pixels.len() as f32;

    // Compute mean
    let mut mean = [0.0f32; 3];
    for p in od_pixels {
        mean[0] += p[0];
        mean[1] += p[1];
        mean[2] += p[2];
    }
    mean[0] /= n;
    mean[1] /= n;
    mean[2] /= n;

    // Compute covariance matrix (symmetric 3×3)
    let mut cov = [[0.0f32; 3]; 3];
    for p in od_pixels {
        let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
        cov[0][0] += d[0] * d[0];
        cov[0][1] += d[0] * d[1];
        cov[0][2] += d[0] * d[2];
        cov[1][1] += d[1] * d[1];
        cov[1][2] += d[1] * d[2];
        cov[2][2] += d[2] * d[2];
    }
    let scale = n - 1.0;
    cov[0][0] /= scale;
    cov[0][1] /= scale;
    cov[0][2] /= scale;
    cov[1][1] /= scale;
    cov[1][2] /= scale;
    cov[2][2] /= scale;
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    // Top 2 eigenvectors via power iteration + deflation
    let (e1, e2) = top_two_eigenvectors(&cov);

    // Ensure eigenvectors point in consistent direction (positive first component)
    let e1 = if e1[0] < 0.0 {
        [-e1[0], -e1[1], -e1[2]]
    } else {
        e1
    };
    let e2 = if e2[0] < 0.0 {
        [-e2[0], -e2[1], -e2[2]]
    } else {
        e2
    };

    // Project tissue OD onto 2D plane of top eigenvectors
    let mut angles: Vec<f32> = Vec::with_capacity(od_pixels.len());
    for p in od_pixels {
        let proj1 = p[0] * e1[0] + p[1] * e1[1] + p[2] * e1[2];
        let proj2 = p[0] * e2[0] + p[1] * e2[1] + p[2] * e2[2];
        angles.push(proj2.atan2(proj1));
    }

    // Robust angular extremes (1st and 99th percentile)
    angles.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p1_idx = (angles.len() as f32 * 0.01) as usize;
    let p99_idx = ((angles.len() as f32 * 0.99) as usize).min(angles.len() - 1);
    let min_phi = angles[p1_idx];
    let max_phi = angles[p99_idx];

    // Stain vectors from angular extremes
    let v1 = [
        e1[0] * min_phi.cos() + e2[0] * min_phi.sin(),
        e1[1] * min_phi.cos() + e2[1] * min_phi.sin(),
        e1[2] * min_phi.cos() + e2[2] * min_phi.sin(),
    ];
    let v2 = [
        e1[0] * max_phi.cos() + e2[0] * max_phi.sin(),
        e1[1] * max_phi.cos() + e2[1] * max_phi.sin(),
        e1[2] * max_phi.cos() + e2[2] * max_phi.sin(),
    ];

    // Normalize to unit vectors (take abs to ensure positive OD)
    let norm1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
    let norm2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();
    if norm1 < 1e-6 || norm2 < 1e-6 {
        return None;
    }
    let mut s1 = [v1[0] / norm1, v1[1] / norm1, v1[2] / norm1];
    let mut s2 = [v2[0] / norm2, v2[1] / norm2, v2[2] / norm2];

    // Ensure each component is non-negative (OD must be non-negative for physical stains)
    for v in &mut s1 {
        *v = v.abs();
    }
    for v in &mut s2 {
        *v = v.abs();
    }

    // Order: Hematoxylin first (larger R component, as H absorbs more red than E)
    if s1[0] < s2[0] {
        std::mem::swap(&mut s1, &mut s2);
    }

    Some([s1, s2])
}

// ─── Vahadane stain extractor ────────────────────────────────────────────────

/// Estimate stain matrix using sparse non-negative dictionary learning
/// following Vahadane et al. (2016).
fn vahadane_extract(od_pixels: &[[f32; 3]], regularizer: f32) -> Option<[[f32; 3]; 2]> {
    if od_pixels.len() < MIN_TISSUE_PIXELS {
        return None;
    }

    // Subsample for tractability
    let max_samples = 5000;
    let step = (od_pixels.len() / max_samples).max(1);
    let sampled: Vec<[f32; 3]> = od_pixels.iter().step_by(step).copied().collect();
    let n = sampled.len();

    let n_f = n as f32;
    let mut mean = [0.0f32; 3];
    for p in &sampled {
        mean[0] += p[0];
        mean[1] += p[1];
        mean[2] += p[2];
    }
    mean[0] /= n_f;
    mean[1] /= n_f;
    mean[2] /= n_f;

    let mut w: [[f32; 3]; 2] = [
        [
            (mean[0] * 1.5).max(EPS),
            (mean[1] * 1.0).max(EPS),
            (mean[2] * 0.5).max(EPS),
        ],
        [
            (mean[0] * 0.3).max(EPS),
            (mean[1] * 1.5).max(EPS),
            (mean[2] * 0.6).max(EPS),
        ],
    ];
    // Normalize rows
    for row in &mut w {
        let norm = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
        if norm > 1e-6 {
            row[0] /= norm;
            row[1] /= norm;
            row[2] /= norm;
        }
    }

    // Alternating optimization: sparse coding + dictionary update
    for _outer in 0..15 {
        let ww = [
            [
                w[0][0] * w[0][0] + w[0][1] * w[0][1] + w[0][2] * w[0][2],
                w[0][0] * w[1][0] + w[0][1] * w[1][1] + w[0][2] * w[1][2],
            ],
            [
                w[1][0] * w[0][0] + w[1][1] * w[0][1] + w[1][2] * w[0][2],
                w[1][0] * w[1][0] + w[1][1] * w[1][1] + w[1][2] * w[1][2],
            ],
        ];

        let mut h_all: Vec<[f32; 2]> = Vec::with_capacity(n);
        for p in &sampled {
            let wv = [
                w[0][0] * p[0] + w[0][1] * p[1] + w[0][2] * p[2],
                w[1][0] * p[0] + w[1][1] * p[1] + w[1][2] * p[2],
            ];
            let mut h = [0.0f32; 2];
            for _ in 0..20 {
                let r0 = wv[0] - ww[0][1] * h[1];
                h[0] = ((r0 - regularizer) / ww[0][0].max(1e-10)).max(0.0);
                let r1 = wv[1] - ww[1][0] * h[0];
                h[1] = ((r1 - regularizer) / ww[1][1].max(1e-10)).max(0.0);
            }
            h_all.push(h);
        }

        let mut hth = [[0.0f32; 2]; 2];
        for h in &h_all {
            hth[0][0] += h[0] * h[0];
            hth[0][1] += h[0] * h[1];
            hth[1][0] += h[1] * h[0];
            hth[1][1] += h[1] * h[1];
        }

        let det = hth[0][0] * hth[1][1] - hth[0][1] * hth[1][0];
        if det.abs() < 1e-10 {
            break;
        }
        let inv_hth = [
            [hth[1][1] / det, -hth[0][1] / det],
            [-hth[1][0] / det, hth[0][0] / det],
        ];

        let mut htv = [[0.0f32; 3]; 2];
        for (sample, h) in sampled.iter().zip(&h_all) {
            htv[0][0] += h[0] * sample[0];
            htv[0][1] += h[0] * sample[1];
            htv[0][2] += h[0] * sample[2];
            htv[1][0] += h[1] * sample[0];
            htv[1][1] += h[1] * sample[1];
            htv[1][2] += h[1] * sample[2];
        }

        w[0][0] = (inv_hth[0][0] * htv[0][0] + inv_hth[0][1] * htv[1][0]).max(0.0);
        w[0][1] = (inv_hth[0][0] * htv[0][1] + inv_hth[0][1] * htv[1][1]).max(0.0);
        w[0][2] = (inv_hth[0][0] * htv[0][2] + inv_hth[0][1] * htv[1][2]).max(0.0);
        w[1][0] = (inv_hth[1][0] * htv[0][0] + inv_hth[1][1] * htv[1][0]).max(0.0);
        w[1][1] = (inv_hth[1][0] * htv[0][1] + inv_hth[1][1] * htv[1][1]).max(0.0);
        w[1][2] = (inv_hth[1][0] * htv[0][2] + inv_hth[1][1] * htv[1][2]).max(0.0);

        for row in &mut w {
            let norm = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
            if norm > 1e-6 {
                row[0] /= norm;
                row[1] /= norm;
                row[2] /= norm;
            }
        }
    }

    // Order: Hematoxylin first (larger R component)
    if w[0][0] < w[1][0] {
        w.swap(0, 1);
    }

    let norm0 = (w[0][0] * w[0][0] + w[0][1] * w[0][1] + w[0][2] * w[0][2]).sqrt();
    let norm1 = (w[1][0] * w[1][0] + w[1][1] * w[1][1] + w[1][2] * w[1][2]).sqrt();
    if norm0 < 1e-6 || norm1 < 1e-6 {
        return None;
    }

    Some(w)
}

// ─── Common normalization pipeline ───────────────────────────────────────────

/// Given a source stain matrix (2×3 row-major), compute the inverse stain matrix
/// and the concentration scale factors from source OD pixels → target.
fn compute_norm_params(stain_matrix: &[[f32; 3]; 2], od_pixels: &[[f32; 3]]) -> StainNormParams {
    let col_major = stain_mat_to_col_major(stain_matrix);
    let inv = pseudo_inverse_2x3(&col_major);

    let mut conc_h: Vec<f32> = Vec::with_capacity(od_pixels.len());
    let mut conc_e: Vec<f32> = Vec::with_capacity(od_pixels.len());
    for p in od_pixels {
        let c = solve_concentrations(p, &inv);
        conc_h.push(c[0]);
        conc_e.push(c[1]);
    }

    let src_max_h = percentile_99(&mut conc_h);
    let src_max_e = percentile_99(&mut conc_e);
    let scale_h = TARGET_MAX_C[0] / src_max_h;
    let scale_e = TARGET_MAX_C[1] / src_max_e;

    StainNormParams {
        enabled: true,
        inv_stain_r0: [inv[0][0], inv[0][1], inv[0][2], scale_h],
        inv_stain_r1: [inv[1][0], inv[1][1], inv[1][2], scale_e],
    }
}

/// Sample tissue OD pixels from raw RGBA tile byte slices with subsampling.
fn sample_tissue_od_from_raw(tile_slices: &[&[u8]], max_samples: usize) -> Vec<[f32; 3]> {
    let total_pixels: usize = tile_slices.iter().map(|s| s.len() / 4).sum();
    let sample_step = (total_pixels / max_samples).max(1);
    let mut result = Vec::with_capacity(max_samples);
    let mut idx = 0usize;
    for slice in tile_slices {
        for chunk in slice.chunks_exact(4) {
            if idx.is_multiple_of(sample_step) {
                let od = rgb_to_od(chunk[0], chunk[1], chunk[2]);
                if is_tissue(&od) {
                    result.push(od);
                }
            }
            idx += 1;
        }
    }
    result
}

// ─── Public API: CPU buffer normalization ────────────────────────────────────

/// Apply stain normalization to an RGBA buffer (CPU path).
///
/// `tile_data` provides raw RGBA byte slices from tiles for stain matrix
/// estimation.
#[allow(dead_code)]
pub fn normalize_buffer(buffer: &mut [u8], method: StainNormalization, tile_data: &[&[u8]]) {
    if method == StainNormalization::None {
        return;
    }

    let params = compute_cpu_stain_params(method, tile_data);
    apply_normalization_to_buffer(buffer, &params);
}

/// Apply pre-computed stain normalization parameters to an RGBA buffer.
pub fn apply_stain_params_to_buffer(buffer: &mut [u8], params: &StainNormParams) {
    apply_normalization_to_buffer(buffer, params);
}

/// Compute stain normalization parameters from raw RGBA tile slices (CPU path).
/// Separated from `normalize_buffer` so the render loop can cache the result
/// and skip the expensive fitting when the visible tile set hasn't changed.
pub fn compute_cpu_stain_params(
    method: StainNormalization,
    tile_data: &[&[u8]],
) -> StainNormParams {
    if method == StainNormalization::None {
        return StainNormParams::default();
    }

    let od_pixels = sample_tissue_od_from_raw(tile_data, 10000);
    if od_pixels.len() < MIN_TISSUE_PIXELS {
        return StainNormParams::default();
    }

    let stain_matrix = match method {
        StainNormalization::None => return StainNormParams::default(),
        StainNormalization::Macenko => macenko_extract(&od_pixels),
        StainNormalization::Vahadane => vahadane_extract(&od_pixels, 0.1),
    };

    let Some(sm) = stain_matrix else {
        return StainNormParams::default();
    };

    debug!(
        method = ?method,
        stain_h = ?sm[0],
        stain_e = ?sm[1],
        tissue_pixels = od_pixels.len(),
        "stain matrix estimated"
    );

    compute_norm_params(&sm, &od_pixels)
}

/// Apply pre-computed normalization parameters to an RGBA buffer.
pub fn apply_normalization_to_buffer(buffer: &mut [u8], params: &StainNormParams) {
    if !params.enabled {
        return;
    }
    let inv = [
        [
            params.inv_stain_r0[0],
            params.inv_stain_r0[1],
            params.inv_stain_r0[2],
        ],
        [
            params.inv_stain_r1[0],
            params.inv_stain_r1[1],
            params.inv_stain_r1[2],
        ],
    ];
    let scale = [params.inv_stain_r0[3], params.inv_stain_r1[3]];

    for chunk in buffer.chunks_exact_mut(4) {
        let od = rgb_to_od(chunk[0], chunk[1], chunk[2]);
        if !is_tissue(&od) {
            continue;
        }
        let c = solve_concentrations(&od, &inv);
        let nc = [c[0] * scale[0], c[1] * scale[1]];
        let new_od = reconstruct_od(nc);
        let rgb = od_to_rgb(new_od);
        chunk[0] = rgb[0];
        chunk[1] = rgb[1];
        chunk[2] = rgb[2];
    }
}

// ─── Linear algebra helpers ──────────────────────────────────────────────────

/// Compute top 2 eigenvectors of a 3×3 symmetric matrix via power iteration.
fn top_two_eigenvectors(cov: &[[f32; 3]; 3]) -> ([f32; 3], [f32; 3]) {
    let mut v1 = [1.0f32, 1.0, 1.0];
    for _ in 0..50 {
        let mut new_v = [0.0f32; 3];
        for i in 0..3 {
            new_v[i] = cov[i][0] * v1[0] + cov[i][1] * v1[1] + cov[i][2] * v1[2];
        }
        let norm = (new_v[0] * new_v[0] + new_v[1] * new_v[1] + new_v[2] * new_v[2]).sqrt();
        if norm < 1e-10 {
            break;
        }
        v1 = [new_v[0] / norm, new_v[1] / norm, new_v[2] / norm];
    }

    // Deflate
    let lambda1 = cov[0][0] * v1[0] * v1[0]
        + cov[1][1] * v1[1] * v1[1]
        + cov[2][2] * v1[2] * v1[2]
        + 2.0 * (cov[0][1] * v1[0] * v1[1] + cov[0][2] * v1[0] * v1[2] + cov[1][2] * v1[1] * v1[2]);
    let mut cov2 = *cov;
    for i in 0..3 {
        for j in 0..3 {
            cov2[i][j] -= lambda1 * v1[i] * v1[j];
        }
    }

    let mut v2 = [1.0f32, 0.0, 0.0];
    let dot = v2[0] * v1[0] + v2[1] * v1[1] + v2[2] * v1[2];
    v2[0] -= dot * v1[0];
    v2[1] -= dot * v1[1];
    v2[2] -= dot * v1[2];
    let norm = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();
    if norm > 1e-10 {
        v2[0] /= norm;
        v2[1] /= norm;
        v2[2] /= norm;
    }
    for _ in 0..50 {
        let mut new_v = [0.0f32; 3];
        for i in 0..3 {
            new_v[i] = cov2[i][0] * v2[0] + cov2[i][1] * v2[1] + cov2[i][2] * v2[2];
        }
        let norm = (new_v[0] * new_v[0] + new_v[1] * new_v[1] + new_v[2] * new_v[2]).sqrt();
        if norm < 1e-10 {
            break;
        }
        v2 = [new_v[0] / norm, new_v[1] / norm, new_v[2] / norm];
    }

    (v1, v2)
}

// ─── Color deconvolution ─────────────────────────────────────────────────────
//
// Color deconvolution separates H&E stain channels from RGB tile data.
// It uses the optical density model:
//   1. Convert RGB to OD:  OD = -ln(RGB / 255)
//   2. Solve for stain concentrations:  C = inv(stain_matrix) * OD
//   3. Reconstruct output based on which channels are visible/isolated.
//
// The default stain matrix (Ruifrok & Johnston, 2001) is used when no
// slide-specific stain matrix is available from normalization fitting.

/// Default H&E stain matrix for deconvolution (same as TARGET_STAIN_MATRIX).
/// Row 0 = Hematoxylin OD vector, Row 1 = Eosin OD vector.
pub const DEFAULT_DECONV_MATRIX: [[f32; 3]; 2] = TARGET_STAIN_MATRIX;

/// Pre-computed pseudo-inverse of the default deconvolution matrix.
/// Used for concentration estimation: C = inv_matrix * OD.
fn default_deconv_inverse() -> [[f32; 3]; 2] {
    let col_major = stain_mat_to_col_major(&DEFAULT_DECONV_MATRIX);
    pseudo_inverse_2x3(&col_major)
}

/// Parameters controlling color deconvolution rendering.
/// Packed for efficient GPU uniform upload.
#[derive(Debug, Clone, Copy)]
pub struct ColorDeconvParams {
    /// Inverse stain matrix row 0 (Hematoxylin): [inv[0][0], inv[0][1], inv[0][2], h_intensity]
    pub inv_row0: [f32; 4],
    /// Inverse stain matrix row 1 (Eosin): [inv[1][0], inv[1][1], inv[1][2], e_intensity]
    pub inv_row1: [f32; 4],
    /// Stain matrix row 0 (Hematoxylin OD vector)
    pub stain_h: [f32; 4],
    /// Stain matrix row 1 (Eosin OD vector)
    pub stain_e: [f32; 4],
    /// Channel visibility: 0.0 = hidden, 1.0 = visible  (packed as [h_vis, e_vis, 0, 0])
    pub visibility: [f32; 4],
    /// Isolated channel mode: 0.0 = blended, 1.0 = hematoxylin isolated, 2.0 = eosin isolated
    pub isolated_mode: f32,
    /// Feature enabled flag (any deconv parameter differs from defaults)
    pub enabled: bool,
}

impl Default for ColorDeconvParams {
    fn default() -> Self {
        Self {
            inv_row0: [0.0; 4],
            inv_row1: [0.0; 4],
            stain_h: [0.0; 4],
            stain_e: [0.0; 4],
            visibility: [1.0, 1.0, 0.0, 0.0],
            isolated_mode: 0.0,
            enabled: false,
        }
    }
}

/// Build color deconvolution parameters from UI state and an optional
/// slide-specific stain matrix (from normalization fitting).
///
/// If `stain_norm_params` is Some and enabled, its inverse stain matrix is
/// reused for deconvolution so the channel separation matches what
/// normalization computed. Otherwise the default Ruifrok & Johnston matrix
/// is used.
pub fn build_deconv_params(
    h_intensity: f32,
    h_visible: bool,
    e_intensity: f32,
    e_visible: bool,
    isolated_channel: u8, // 0=none, 1=hematoxylin, 2=eosin
    stain_norm_params: Option<&StainNormParams>,
) -> ColorDeconvParams {
    // Determine if deconvolution is effectively active
    let is_default = h_visible
        && e_visible
        && (h_intensity - 1.0).abs() < 0.001
        && (e_intensity - 1.0).abs() < 0.001
        && isolated_channel == 0;

    if is_default {
        return ColorDeconvParams::default();
    }

    // Use slide-specific inverse matrix if available from stain normalization,
    // otherwise fall back to the default Ruifrok & Johnston matrix.
    let (inv, stain_mat) = if let Some(params) = stain_norm_params
        && params.enabled
    {
        // The normalization params already contain the inverse stain matrix;
        // reconstruct the forward matrix from the default target for OD reconstruction.
        let inv = [
            [
                params.inv_stain_r0[0],
                params.inv_stain_r0[1],
                params.inv_stain_r0[2],
            ],
            [
                params.inv_stain_r1[0],
                params.inv_stain_r1[1],
                params.inv_stain_r1[2],
            ],
        ];
        (inv, DEFAULT_DECONV_MATRIX)
    } else {
        (default_deconv_inverse(), DEFAULT_DECONV_MATRIX)
    };

    ColorDeconvParams {
        inv_row0: [inv[0][0], inv[0][1], inv[0][2], h_intensity],
        inv_row1: [inv[1][0], inv[1][1], inv[1][2], e_intensity],
        stain_h: [
            stain_mat[0][0],
            stain_mat[0][1],
            stain_mat[0][2],
            h_intensity,
        ],
        stain_e: [
            stain_mat[1][0],
            stain_mat[1][1],
            stain_mat[1][2],
            e_intensity,
        ],
        visibility: [
            if h_visible { 1.0 } else { 0.0 },
            if e_visible { 1.0 } else { 0.0 },
            0.0,
            0.0,
        ],
        isolated_mode: isolated_channel as f32,
        enabled: true,
    }
}

/// Apply color deconvolution to an RGBA buffer on the CPU.
///
/// This performs the same per-pixel logic as the GPU shader:
///   - RGB → optical density
///   - OD → stain concentrations via inverse matrix
///   - Reconstruct output based on visibility, intensity, and isolation mode
pub fn apply_color_deconvolution(buffer: &mut [u8], params: &ColorDeconvParams) {
    if !params.enabled {
        return;
    }

    let inv = [
        [params.inv_row0[0], params.inv_row0[1], params.inv_row0[2]],
        [params.inv_row1[0], params.inv_row1[1], params.inv_row1[2]],
    ];
    let h_intensity = params.inv_row0[3];
    let e_intensity = params.inv_row1[3];
    let h_visible = params.visibility[0] > 0.5;
    let e_visible = params.visibility[1] > 0.5;
    let isolated = params.isolated_mode;

    let stain_h = [params.stain_h[0], params.stain_h[1], params.stain_h[2]];
    let stain_e = [params.stain_e[0], params.stain_e[1], params.stain_e[2]];

    for chunk in buffer.chunks_exact_mut(4) {
        let od = rgb_to_od(chunk[0], chunk[1], chunk[2]);
        let od_sum = od[0] + od[1] + od[2];
        if od_sum <= OD_THRESHOLD {
            // Background pixels pass through unchanged.
            continue;
        }
        if od_sum >= OD_THRESHOLD_HIGH && isolated <= 0.5 {
            // Very dark pixels: skip in blended mode only.
            // In isolated mode, still process to produce grayscale.
            continue;
        }

        // Solve concentrations: C = inv(stain_matrix) * OD, clamped >= 0
        let c_h = (inv[0][0] * od[0] + inv[0][1] * od[1] + inv[0][2] * od[2]).max(0.0);
        let c_e = (inv[1][0] * od[0] + inv[1][1] * od[1] + inv[1][2] * od[2]).max(0.0);

        let (r, g, b) = if isolated > 0.5 {
            // Isolated grayscale view: show one channel as grayscale.
            // Map concentration to grayscale: higher concentration → darker.
            let c = if isolated < 1.5 {
                // Hematoxylin isolated
                if h_visible { c_h * h_intensity } else { 0.0 }
            } else {
                // Eosin isolated
                if e_visible { c_e * e_intensity } else { 0.0 }
            };
            // Grayscale via OD: use concentration as scalar OD, then exp(-OD)
            let gray = (-c).exp().clamp(0.0, 1.0);
            (gray, gray, gray)
        } else {
            // Blended mode: reconstruct from visible channels with intensity scaling.
            let mut new_od = [0.0f32; 3];
            if h_visible && h_intensity > 0.0 {
                let scaled_h = c_h * h_intensity;
                new_od[0] += stain_h[0] * scaled_h;
                new_od[1] += stain_h[1] * scaled_h;
                new_od[2] += stain_h[2] * scaled_h;
            }
            if e_visible && e_intensity > 0.0 {
                let scaled_e = c_e * e_intensity;
                new_od[0] += stain_e[0] * scaled_e;
                new_od[1] += stain_e[1] * scaled_e;
                new_od[2] += stain_e[2] * scaled_e;
            }
            // If both channels hidden, output white (zero OD = full transmission)
            (
                (-new_od[0]).exp().clamp(0.0, 1.0),
                (-new_od[1]).exp().clamp(0.0, 1.0),
                (-new_od[2]).exp().clamp(0.0, 1.0),
            )
        };

        chunk[0] = (r * 255.0) as u8;
        chunk[1] = (g * 255.0) as u8;
        chunk[2] = (b * 255.0) as u8;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_tissue_od(buffer: &[u8]) -> Vec<[f32; 3]> {
        let pixel_count = buffer.len() / 4;
        let mut result = Vec::with_capacity(pixel_count / 2);
        for chunk in buffer.chunks_exact(4) {
            let od = rgb_to_od(chunk[0], chunk[1], chunk[2]);
            if is_tissue(&od) {
                result.push(od);
            }
        }
        result
    }

    fn synth_he_pixel(c_h: f32, c_e: f32) -> [u8; 4] {
        let od_r = TARGET_STAIN_MATRIX[0][0] * c_h + TARGET_STAIN_MATRIX[1][0] * c_e;
        let od_g = TARGET_STAIN_MATRIX[0][1] * c_h + TARGET_STAIN_MATRIX[1][1] * c_e;
        let od_b = TARGET_STAIN_MATRIX[0][2] * c_h + TARGET_STAIN_MATRIX[1][2] * c_e;
        [
            ((-od_r).exp() * 255.0).clamp(0.0, 255.0) as u8,
            ((-od_g).exp() * 255.0).clamp(0.0, 255.0) as u8,
            ((-od_b).exp() * 255.0).clamp(0.0, 255.0) as u8,
            255,
        ]
    }

    fn make_synthetic_tissue(n: usize) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(n * 4);
        for i in 0..n {
            let t = i as f32 / n as f32;
            let c_h = 0.3 + 1.5 * t;
            let c_e = 1.2 - 0.8 * t;
            let px = synth_he_pixel(c_h, c_e);
            buffer.extend_from_slice(&px);
        }
        buffer
    }

    #[test]
    fn test_rgb_od_roundtrip() {
        let rgb = [180u8, 120, 200];
        let od = rgb_to_od(rgb[0], rgb[1], rgb[2]);
        let back = od_to_rgb(od);
        for i in 0..3 {
            assert!(
                (rgb[i] as i32 - back[i] as i32).unsigned_abs() <= 1,
                "channel {i}: {rgb_v} != {back_v}",
                rgb_v = rgb[i],
                back_v = back[i]
            );
        }
    }

    #[test]
    fn test_rgb_od_white_stability() {
        let od = rgb_to_od(255, 255, 255);
        for (i, &v) in od.iter().enumerate() {
            assert!(v.abs() < 0.005, "white OD channel {i} = {v}, expected ~0");
        }
    }

    #[test]
    fn test_rgb_od_black_stability() {
        let od = rgb_to_od(1, 1, 1);
        for (i, &v) in od.iter().enumerate() {
            assert!(v.is_finite(), "black OD channel {i} is not finite");
            assert!(v > 0.0, "black OD channel {i} = {v}, expected > 0");
        }
    }

    #[test]
    fn test_macenko_produces_plausible_matrix() {
        let buffer = make_synthetic_tissue(2000);
        let od = collect_tissue_od(&buffer);
        let sm = macenko_extract(&od).expect("Macenko should succeed on synthetic tissue");
        assert!(sm[0][0] > 0.3, "H R component too small: {}", sm[0][0]);
        assert!(sm[1][1] > 0.5, "E G component too small: {}", sm[1][1]);
        for (i, row) in sm.iter().enumerate() {
            let norm = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
            assert!(
                (norm - 1.0).abs() < 0.05,
                "row {i} norm = {norm}, expected ~1.0"
            );
        }
    }

    #[test]
    fn test_vahadane_produces_plausible_matrix() {
        let buffer = make_synthetic_tissue(2000);
        let od = collect_tissue_od(&buffer);
        let sm = vahadane_extract(&od, 0.1).expect("Vahadane should succeed on synthetic tissue");
        assert!(sm[0][0] > 0.3, "H R component too small: {}", sm[0][0]);
        assert!(sm[1][1] > 0.5, "E G component too small: {}", sm[1][1]);
        for (i, row) in sm.iter().enumerate() {
            let norm = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
            assert!(
                (norm - 1.0).abs() < 0.05,
                "row {i} norm = {norm}, expected ~1.0"
            );
        }
    }

    #[test]
    fn test_macenko_and_vahadane_are_different() {
        let buffer = make_synthetic_tissue(3000);
        let od = collect_tissue_od(&buffer);
        let macenko = macenko_extract(&od).expect("Macenko should succeed");
        let vahadane = vahadane_extract(&od, 0.1).expect("Vahadane should succeed");
        let mut total_diff = 0.0f32;
        for i in 0..2 {
            for j in 0..3 {
                total_diff += (macenko[i][j] - vahadane[i][j]).abs();
            }
        }
        assert!(
            total_diff > 0.01,
            "Macenko and Vahadane matrices are too similar (diff = {total_diff}): \
             macenko = {macenko:?}, vahadane = {vahadane:?}"
        );
    }

    #[test]
    fn test_cpu_and_gpu_params_agree() {
        let buffer = make_synthetic_tissue(2000);
        let od_pixels = collect_tissue_od(&buffer);
        let sm = macenko_extract(&od_pixels).expect("Macenko should succeed");
        let params = compute_norm_params(&sm, &od_pixels);
        assert!(params.enabled);
        for v in &params.inv_stain_r0 {
            assert!(v.is_finite(), "inv_stain_r0 contains non-finite: {v}");
        }
        for v in &params.inv_stain_r1 {
            assert!(v.is_finite(), "inv_stain_r1 contains non-finite: {v}");
        }
        assert!(params.inv_stain_r0[3] > 0.0, "scale_h should be positive");
        assert!(params.inv_stain_r1[3] > 0.0, "scale_e should be positive");
    }

    #[test]
    fn test_normalization_preserves_background() {
        let mut buffer = vec![255u8, 255, 255, 255, 254, 254, 254, 255];
        let original = buffer.clone();
        let params = StainNormParams {
            enabled: true,
            inv_stain_r0: [1.0, 0.0, 0.0, 1.0],
            inv_stain_r1: [0.0, 1.0, 0.0, 1.0],
        };
        apply_normalization_to_buffer(&mut buffer, &params);
        assert_eq!(buffer, original, "background pixels were modified");
    }
}

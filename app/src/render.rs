//! Rendering utilities for the WSI viewer
//!
//! This module contains helper functions for rendering tiles and compositing
//! the viewport image.

use crate::AppWindow;
use crate::blitter;
use crate::gpu::{SurfaceSlot, TileDraw};
use crate::state::{
    AppState, FilteringMode, OpenFile, PaneId, RenderBackend, StainNormalization,
    TileRequestSignature,
};
use crate::tile_loader::calculate_wanted_tiles;
use crate::tools;
use common::{TileCache, TileCoord, TileManager, Viewport, WsiFile};
use parking_lot::RwLock;
use slint::{ComponentHandle, Image};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::debug;

/// Adaptive Lanczos-to-Trilinear blending weight based on zoom level.
///
/// At very low zoom, trilinear filtering produces better results than Lanczos.
/// This function returns a weight in \[0.0, 1.0\] controlling the mix:
///   zoom >= 0.12 → 1.0 (100% Lanczos)
///   0.07–0.12   → linear blend Lanczos → Trilinear
///   zoom < 0.07 → 0.0 (100% Trilinear)
fn lanczos_adaptive_weight(zoom: f64) -> f64 {
    const LANCZOS_FULL: f64 = 0.12;
    const TRILINEAR_FULL: f64 = 0.07;
    if zoom >= LANCZOS_FULL {
        1.0
    } else if zoom <= TRILINEAR_FULL {
        0.0
    } else {
        (zoom - TRILINEAR_FULL) / (LANCZOS_FULL - TRILINEAR_FULL)
    }
}

/// Apply gamma, brightness, and contrast adjustments to an RGBA pixel buffer.
fn apply_adjustments(buffer: &mut [u8], gamma: f32, brightness: f32, contrast: f32) {
    // Pre-compute a 256-entry lookup table for the combined transformation.
    // Pipeline: input → gamma → brightness → contrast → clamp
    let inv_gamma = if gamma > 0.001 { 1.0 / gamma } else { 1.0 };
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let normalized = i as f32 / 255.0;
        // Apply gamma
        let g = normalized.powf(inv_gamma);
        // Apply brightness (additive)
        let b = g + brightness;
        // Apply contrast (multiply around midpoint 0.5)
        let c = (b - 0.5) * contrast + 0.5;
        *entry = (c * 255.0).clamp(0.0, 255.0) as u8;
    }
    // Apply LUT to RGB channels (skip alpha every 4th byte)
    for chunk in buffer.chunks_exact_mut(4) {
        chunk[0] = lut[chunk[0] as usize];
        chunk[1] = lut[chunk[1] as usize];
        chunk[2] = lut[chunk[2] as usize];
    }
}

/// Standard H&E reference stain matrix (optical density space).
/// Columns are [Hematoxylin, Eosin] stain vectors, rows are [R, G, B].
/// From Macenko et al. and Ruifrok & Johnston (2001).
const REF_STAIN_MATRIX: [[f32; 2]; 3] = [
    [0.6442, 0.0927],  // R: Hematoxylin, Eosin
    [0.7170, 0.9545],  // G
    [0.2668, 0.2832],  // B
];

/// Maximum stain concentrations for the reference (99th percentile, H&E).
const REF_MAX_CONC: [f32; 2] = [1.9705, 1.0308];

/// Apply Macenko stain normalization to an RGBA buffer.
/// Uses a simplified per-tile approach: estimate stain vectors via
/// robust percentile thresholding in OD space, then normalize concentrations
/// to the reference.
#[allow(clippy::needless_range_loop)]
fn apply_macenko_normalization(buffer: &mut [u8]) {
    let pixel_count = buffer.len() / 4;
    if pixel_count == 0 {
        return;
    }

    // Step 1: Convert to optical density and collect non-background pixels
    let mut od_pixels: Vec<[f32; 3]> = Vec::with_capacity(pixel_count);
    for chunk in buffer.chunks_exact(4) {
        let r = chunk[0].max(1) as f32 / 255.0;
        let g = chunk[1].max(1) as f32 / 255.0;
        let b = chunk[2].max(1) as f32 / 255.0;
        let od_r = -r.ln();
        let od_g = -g.ln();
        let od_b = -b.ln();
        let od_sum = od_r + od_g + od_b;
        // Filter out background (low OD) pixels
        if od_sum > 0.15 {
            od_pixels.push([od_r, od_g, od_b]);
        }
    }

    if od_pixels.len() < 100 {
        return; // Not enough tissue pixels
    }

    // Step 2: Estimate stain vectors using angle-based approach
    // Compute covariance matrix of OD values
    let n = od_pixels.len() as f32;
    let mut mean = [0.0f32; 3];
    for p in &od_pixels {
        mean[0] += p[0];
        mean[1] += p[1];
        mean[2] += p[2];
    }
    mean[0] /= n;
    mean[1] /= n;
    mean[2] /= n;

    // Covariance (upper triangle of 3x3 symmetric matrix)
    let mut cov = [[0.0f32; 3]; 3];
    for p in &od_pixels {
        let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
        for i in 0..3 {
            for j in i..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    for i in 0..3 {
        for j in i..3 {
            cov[i][j] /= n - 1.0;
            if j > i {
                cov[j][i] = cov[i][j];
            }
        }
    }

    // Power iteration for top 2 eigenvectors
    let (v1, v2) = top_two_eigenvectors(&cov);

    // Project OD pixels onto the plane spanned by v1, v2
    let mut projections: Vec<f32> = Vec::with_capacity(od_pixels.len());
    for p in &od_pixels {
        let proj1 = p[0] * v1[0] + p[1] * v1[1] + p[2] * v1[2];
        let proj2 = p[0] * v2[0] + p[1] * v2[1] + p[2] * v2[2];
        projections.push(proj2.atan2(proj1));
    }

    // Find robust angular extremes (1st and 99th percentile)
    let mut sorted_angles = projections.clone();
    sorted_angles.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p1_idx = (sorted_angles.len() as f32 * 0.01) as usize;
    let p99_idx = ((sorted_angles.len() as f32 * 0.99) as usize).min(sorted_angles.len() - 1);
    let angle_min = sorted_angles[p1_idx];
    let angle_max = sorted_angles[p99_idx];

    // Stain vectors from extreme angles
    let mut stain1 = [
        v1[0] * angle_min.cos() + v2[0] * angle_min.sin(),
        v1[1] * angle_min.cos() + v2[1] * angle_min.sin(),
        v1[2] * angle_min.cos() + v2[2] * angle_min.sin(),
    ];
    let mut stain2 = [
        v1[0] * angle_max.cos() + v2[0] * angle_max.sin(),
        v1[1] * angle_max.cos() + v2[1] * angle_max.sin(),
        v1[2] * angle_max.cos() + v2[2] * angle_max.sin(),
    ];

    // Ensure positive OD
    for v in &mut stain1 {
        *v = v.abs();
    }
    for v in &mut stain2 {
        *v = v.abs();
    }

    // Normalize
    let norm1 = (stain1[0] * stain1[0] + stain1[1] * stain1[1] + stain1[2] * stain1[2]).sqrt();
    let norm2 = (stain2[0] * stain2[0] + stain2[1] * stain2[1] + stain2[2] * stain2[2]).sqrt();
    if norm1 < 1e-6 || norm2 < 1e-6 {
        return;
    }
    for v in &mut stain1 {
        *v /= norm1;
    }
    for v in &mut stain2 {
        *v /= norm2;
    }

    // Ensure Hematoxylin is the one with higher blue component
    if stain1[2] < stain2[2] {
        std::mem::swap(&mut stain1, &mut stain2);
    }

    // Step 3: Deconvolve and normalize
    // Build 2x3 inverse stain matrix (pseudo-inverse for 2 stains)
    let src_mat = [[stain1[0], stain2[0]], [stain1[1], stain2[1]], [stain1[2], stain2[2]]];
    let inv = pseudo_inverse_2x3(&src_mat);

    // Find max concentrations (99th percentile)
    let mut conc_h: Vec<f32> = Vec::with_capacity(od_pixels.len());
    let mut conc_e: Vec<f32> = Vec::with_capacity(od_pixels.len());
    for p in &od_pixels {
        let c0 = inv[0][0] * p[0] + inv[0][1] * p[1] + inv[0][2] * p[2];
        let c1 = inv[1][0] * p[0] + inv[1][1] * p[1] + inv[1][2] * p[2];
        conc_h.push(c0);
        conc_e.push(c1);
    }
    conc_h.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    conc_e.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let src_max_h = conc_h[((conc_h.len() as f32 * 0.99) as usize).min(conc_h.len() - 1)].max(0.001);
    let src_max_e = conc_e[((conc_e.len() as f32 * 0.99) as usize).min(conc_e.len() - 1)].max(0.001);

    // Apply normalization to all pixels
    let scale_h = REF_MAX_CONC[0] / src_max_h;
    let scale_e = REF_MAX_CONC[1] / src_max_e;

    for chunk in buffer.chunks_exact_mut(4) {
        let r = chunk[0].max(1) as f32 / 255.0;
        let g = chunk[1].max(1) as f32 / 255.0;
        let b = chunk[2].max(1) as f32 / 255.0;
        let od = [-r.ln(), -g.ln(), -b.ln()];

        // Deconvolve
        let c0 = (inv[0][0] * od[0] + inv[0][1] * od[1] + inv[0][2] * od[2]).max(0.0);
        let c1 = (inv[1][0] * od[0] + inv[1][1] * od[1] + inv[1][2] * od[2]).max(0.0);

        // Scale to reference
        let nc0 = c0 * scale_h;
        let nc1 = c1 * scale_e;

        // Reconstruct with reference stain matrix
        let new_od_r = REF_STAIN_MATRIX[0][0] * nc0 + REF_STAIN_MATRIX[0][1] * nc1;
        let new_od_g = REF_STAIN_MATRIX[1][0] * nc0 + REF_STAIN_MATRIX[1][1] * nc1;
        let new_od_b = REF_STAIN_MATRIX[2][0] * nc0 + REF_STAIN_MATRIX[2][1] * nc1;

        chunk[0] = ((-new_od_r).exp() * 255.0).clamp(0.0, 255.0) as u8;
        chunk[1] = ((-new_od_g).exp() * 255.0).clamp(0.0, 255.0) as u8;
        chunk[2] = ((-new_od_b).exp() * 255.0).clamp(0.0, 255.0) as u8;
    }
}

/// Apply Vahadane stain normalization to an RGBA buffer.
/// Uses dictionary learning with sparsity constraints (SNMF-like).
/// Approximated with iterative alternating non-negative least squares.
#[allow(clippy::needless_range_loop)]
fn apply_vahadane_normalization(buffer: &mut [u8]) {
    let pixel_count = buffer.len() / 4;
    if pixel_count == 0 {
        return;
    }

    // Step 1: Convert to optical density, filter background
    let mut od_pixels: Vec<[f32; 3]> = Vec::with_capacity(pixel_count);
    let mut od_indices: Vec<usize> = Vec::with_capacity(pixel_count);
    for (i, chunk) in buffer.chunks_exact(4).enumerate() {
        let r = chunk[0].max(1) as f32 / 255.0;
        let g = chunk[1].max(1) as f32 / 255.0;
        let b = chunk[2].max(1) as f32 / 255.0;
        let od_r = -r.ln();
        let od_g = -g.ln();
        let od_b = -b.ln();
        if od_r + od_g + od_b > 0.15 {
            od_pixels.push([od_r, od_g, od_b]);
            od_indices.push(i);
        }
    }

    if od_pixels.len() < 100 {
        return;
    }

    // Step 2: Dictionary learning - initialize with Macenko-style SVD
    let n = od_pixels.len() as f32;
    let mut mean = [0.0f32; 3];
    for p in &od_pixels {
        mean[0] += p[0];
        mean[1] += p[1];
        mean[2] += p[2];
    }
    mean[0] /= n;
    mean[1] /= n;
    mean[2] /= n;

    let mut cov = [[0.0f32; 3]; 3];
    for p in &od_pixels {
        let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
        for i in 0..3 {
            for j in i..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    for i in 0..3 {
        for j in i..3 {
            cov[i][j] /= n - 1.0;
            if j > i {
                cov[j][i] = cov[i][j];
            }
        }
    }

    let (v1, v2) = top_two_eigenvectors(&cov);

    // Initialize dictionary W (3x2) from extreme angles (same as Macenko init)
    let mut projections: Vec<f32> = Vec::with_capacity(od_pixels.len());
    for p in &od_pixels {
        let proj1 = p[0] * v1[0] + p[1] * v1[1] + p[2] * v1[2];
        let proj2 = p[0] * v2[0] + p[1] * v2[1] + p[2] * v2[2];
        projections.push(proj2.atan2(proj1));
    }
    let mut sorted_angles = projections;
    sorted_angles.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p1_idx = (sorted_angles.len() as f32 * 0.01) as usize;
    let p99_idx = ((sorted_angles.len() as f32 * 0.99) as usize).min(sorted_angles.len() - 1);
    let angle_min = sorted_angles[p1_idx];
    let angle_max = sorted_angles[p99_idx];

    let mut w = [[0.0f32; 2]; 3]; // W[channel][stain]
    for i in 0..3 {
        w[i][0] = (v1[i] * angle_min.cos() + v2[i] * angle_min.sin()).abs();
        w[i][1] = (v1[i] * angle_max.cos() + v2[i] * angle_max.sin()).abs();
    }

    // Normalize columns
    for s in 0..2 {
        let norm = (w[0][s] * w[0][s] + w[1][s] * w[1][s] + w[2][s] * w[2][s]).sqrt();
        if norm > 1e-6 {
            for i in 0..3 {
                w[i][s] /= norm;
            }
        }
    }

    // Ensure stain 0 = Hematoxylin (higher blue)
    if w[2][0] < w[2][1] {
        for row in &mut w {
            row.swap(0, 1);
        }
    }

    // Step 3: Alternating NNLS iterations to refine W
    // Use a subset for efficiency
    let sample_step = (od_pixels.len() / 5000).max(1);
    let sampled: Vec<&[f32; 3]> = od_pixels.iter().step_by(sample_step).collect();

    for _iter in 0..5 {
        // Solve for H given W: H = W^+ * OD (non-negative)
        let w_mat = [[w[0][0], w[1][0], w[2][0]], [w[0][1], w[1][1], w[2][1]]];
        // W^T * W
        let mut wtw = [[0.0f32; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                wtw[i][j] = w_mat[i][0] * w_mat[j][0] + w_mat[i][1] * w_mat[j][1] + w_mat[i][2] * w_mat[j][2];
            }
        }
        let det = wtw[0][0] * wtw[1][1] - wtw[0][1] * wtw[1][0];
        if det.abs() < 1e-10 {
            break;
        }
        let inv_wtw = [
            [wtw[1][1] / det, -wtw[0][1] / det],
            [-wtw[1][0] / det, wtw[0][0] / det],
        ];

        // Accumulate for dictionary update
        let mut sum_h = [[0.0f32; 3]; 2]; // sum_h[stain][channel] = sum(h_s * od_c)
        let mut sum_hh = [[0.0f32; 2]; 2]; // sum_hh[s1][s2] = sum(h_s1 * h_s2)
        for p in &sampled {
            let wt_od = [
                w_mat[0][0] * p[0] + w_mat[0][1] * p[1] + w_mat[0][2] * p[2],
                w_mat[1][0] * p[0] + w_mat[1][1] * p[1] + w_mat[1][2] * p[2],
            ];
            let h0 = (inv_wtw[0][0] * wt_od[0] + inv_wtw[0][1] * wt_od[1]).max(0.0);
            let h1 = (inv_wtw[1][0] * wt_od[0] + inv_wtw[1][1] * wt_od[1]).max(0.0);

            for c in 0..3 {
                sum_h[0][c] += h0 * p[c];
                sum_h[1][c] += h1 * p[c];
            }
            sum_hh[0][0] += h0 * h0;
            sum_hh[0][1] += h0 * h1;
            sum_hh[1][0] += h1 * h0;
            sum_hh[1][1] += h1 * h1;
        }

        // Update W given H: W = OD * H^T * (H * H^T)^-1 (non-negative)
        let det2 = sum_hh[0][0] * sum_hh[1][1] - sum_hh[0][1] * sum_hh[1][0];
        if det2.abs() < 1e-10 {
            break;
        }
        let inv_hht = [
            [sum_hh[1][1] / det2, -sum_hh[0][1] / det2],
            [-sum_hh[1][0] / det2, sum_hh[0][0] / det2],
        ];

        for c in 0..3 {
            w[c][0] = (sum_h[0][c] * inv_hht[0][0] + sum_h[1][c] * inv_hht[1][0]).max(0.0);
            w[c][1] = (sum_h[0][c] * inv_hht[0][1] + sum_h[1][c] * inv_hht[1][1]).max(0.0);
        }

        // Normalize columns
        for s in 0..2 {
            let norm = (w[0][s] * w[0][s] + w[1][s] * w[1][s] + w[2][s] * w[2][s]).sqrt();
            if norm > 1e-6 {
                for i in 0..3 {
                    w[i][s] /= norm;
                }
            }
        }

        // Ensure stain 0 = Hematoxylin (higher blue)
        if w[2][0] < w[2][1] {
            for row in &mut w {
                row.swap(0, 1);
            }
        }
    }

    // Step 4: Deconvolve all tissue pixels and normalize
    let src_mat = [[w[0][0], w[0][1]], [w[1][0], w[1][1]], [w[2][0], w[2][1]]];
    let inv = pseudo_inverse_2x3(&src_mat);

    // Find max concentrations (99th percentile)
    let mut conc_h: Vec<f32> = Vec::with_capacity(od_pixels.len());
    let mut conc_e: Vec<f32> = Vec::with_capacity(od_pixels.len());
    for p in &od_pixels {
        let c0 = (inv[0][0] * p[0] + inv[0][1] * p[1] + inv[0][2] * p[2]).max(0.0);
        let c1 = (inv[1][0] * p[0] + inv[1][1] * p[1] + inv[1][2] * p[2]).max(0.0);
        conc_h.push(c0);
        conc_e.push(c1);
    }
    conc_h.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    conc_e.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let src_max_h = conc_h[((conc_h.len() as f32 * 0.99) as usize).min(conc_h.len() - 1)].max(0.001);
    let src_max_e = conc_e[((conc_e.len() as f32 * 0.99) as usize).min(conc_e.len() - 1)].max(0.001);

    let scale_h = REF_MAX_CONC[0] / src_max_h;
    let scale_e = REF_MAX_CONC[1] / src_max_e;

    // Apply to all pixels (not just tissue)
    for chunk in buffer.chunks_exact_mut(4) {
        let r = chunk[0].max(1) as f32 / 255.0;
        let g = chunk[1].max(1) as f32 / 255.0;
        let b = chunk[2].max(1) as f32 / 255.0;
        let od = [-r.ln(), -g.ln(), -b.ln()];
        let od_sum = od[0] + od[1] + od[2];

        if od_sum <= 0.15 {
            continue; // Skip background
        }

        let c0 = (inv[0][0] * od[0] + inv[0][1] * od[1] + inv[0][2] * od[2]).max(0.0);
        let c1 = (inv[1][0] * od[0] + inv[1][1] * od[1] + inv[1][2] * od[2]).max(0.0);

        let nc0 = c0 * scale_h;
        let nc1 = c1 * scale_e;

        let new_od_r = REF_STAIN_MATRIX[0][0] * nc0 + REF_STAIN_MATRIX[0][1] * nc1;
        let new_od_g = REF_STAIN_MATRIX[1][0] * nc0 + REF_STAIN_MATRIX[1][1] * nc1;
        let new_od_b = REF_STAIN_MATRIX[2][0] * nc0 + REF_STAIN_MATRIX[2][1] * nc1;

        chunk[0] = ((-new_od_r).exp() * 255.0).clamp(0.0, 255.0) as u8;
        chunk[1] = ((-new_od_g).exp() * 255.0).clamp(0.0, 255.0) as u8;
        chunk[2] = ((-new_od_b).exp() * 255.0).clamp(0.0, 255.0) as u8;
    }
}

/// Stain normalization parameters computed from source image statistics.
/// These are passed as GPU uniform data for per-pixel shader normalization.
pub struct StainNormParams {
    /// Whether stain normalization is enabled
    pub enabled: bool,
    /// Row 0 of inverse source stain matrix + scale_h: [inv[0][0], inv[0][1], inv[0][2], scale_h]
    pub inv_stain_r0: [f32; 4],
    /// Row 1 of inverse source stain matrix + scale_e: [inv[1][0], inv[1][1], inv[1][2], scale_e]
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

/// Compute stain normalization parameters from tile pixel data.
/// Samples pixels from the provided tiles to estimate source stain vectors
/// via the Macenko (SVD) approach, then returns the inverse stain matrix
/// and scale factors needed for the GPU shader.
///
/// For Vahadane mode, additionally refines stain vectors via alternating NNLS.
#[allow(clippy::needless_range_loop)]
fn compute_stain_params(
    draws: &[TileDraw],
    method: StainNormalization,
) -> StainNormParams {
    if method == StainNormalization::None || draws.is_empty() {
        return StainNormParams::default();
    }

    // Sample pixels from tiles (skip every N pixels for performance)
    let total_pixels: usize = draws.iter().map(|d| d.tile.data.len() / 4).sum();
    let sample_step = (total_pixels / 10000).max(1);

    let mut od_pixels: Vec<[f32; 3]> = Vec::with_capacity(10000);
    let mut pixel_idx = 0usize;
    for draw in draws {
        for chunk in draw.tile.data.chunks_exact(4) {
            if pixel_idx.is_multiple_of(sample_step) {
                let r = chunk[0].max(1) as f32 / 255.0;
                let g = chunk[1].max(1) as f32 / 255.0;
                let b = chunk[2].max(1) as f32 / 255.0;
                let od_r = -r.ln();
                let od_g = -g.ln();
                let od_b = -b.ln();
                if od_r + od_g + od_b > 0.15 {
                    od_pixels.push([od_r, od_g, od_b]);
                }
            }
            pixel_idx += 1;
        }
    }

    if od_pixels.len() < 100 {
        return StainNormParams::default();
    }

    // Compute covariance matrix
    let n = od_pixels.len() as f32;
    let mut mean = [0.0f32; 3];
    for p in &od_pixels {
        mean[0] += p[0];
        mean[1] += p[1];
        mean[2] += p[2];
    }
    mean[0] /= n;
    mean[1] /= n;
    mean[2] /= n;

    let mut cov = [[0.0f32; 3]; 3];
    for p in &od_pixels {
        let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
        for i in 0..3 {
            for j in i..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    for i in 0..3 {
        for j in i..3 {
            cov[i][j] /= n - 1.0;
            if j > i {
                cov[j][i] = cov[i][j];
            }
        }
    }

    let (v1, v2) = top_two_eigenvectors(&cov);

    // Project and find angular extremes
    let mut angles: Vec<f32> = Vec::with_capacity(od_pixels.len());
    for p in &od_pixels {
        let proj1 = p[0] * v1[0] + p[1] * v1[1] + p[2] * v1[2];
        let proj2 = p[0] * v2[0] + p[1] * v2[1] + p[2] * v2[2];
        angles.push(proj2.atan2(proj1));
    }
    angles.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p1_idx = (angles.len() as f32 * 0.01) as usize;
    let p99_idx = ((angles.len() as f32 * 0.99) as usize).min(angles.len() - 1);
    let angle_min = angles[p1_idx];
    let angle_max = angles[p99_idx];

    let mut stain1 = [
        (v1[0] * angle_min.cos() + v2[0] * angle_min.sin()).abs(),
        (v1[1] * angle_min.cos() + v2[1] * angle_min.sin()).abs(),
        (v1[2] * angle_min.cos() + v2[2] * angle_min.sin()).abs(),
    ];
    let mut stain2 = [
        (v1[0] * angle_max.cos() + v2[0] * angle_max.sin()).abs(),
        (v1[1] * angle_max.cos() + v2[1] * angle_max.sin()).abs(),
        (v1[2] * angle_max.cos() + v2[2] * angle_max.sin()).abs(),
    ];

    // Normalize
    let norm1 = (stain1[0] * stain1[0] + stain1[1] * stain1[1] + stain1[2] * stain1[2]).sqrt();
    let norm2 = (stain2[0] * stain2[0] + stain2[1] * stain2[1] + stain2[2] * stain2[2]).sqrt();
    if norm1 < 1e-6 || norm2 < 1e-6 {
        return StainNormParams::default();
    }
    for v in &mut stain1 {
        *v /= norm1;
    }
    for v in &mut stain2 {
        *v /= norm2;
    }

    // Ensure stain1 = Hematoxylin (higher blue)
    if stain1[2] < stain2[2] {
        std::mem::swap(&mut stain1, &mut stain2);
    }

    // For Vahadane: refine via alternating NNLS
    if method == StainNormalization::Vahadane {
        let mut w = [
            [stain1[0], stain2[0]],
            [stain1[1], stain2[1]],
            [stain1[2], stain2[2]],
        ];
        let sampled = &od_pixels;
        for _iter in 0..5 {
            let w_mat = [[w[0][0], w[1][0], w[2][0]], [w[0][1], w[1][1], w[2][1]]];
            let mut wtw = [[0.0f32; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    wtw[i][j] =
                        w_mat[i][0] * w_mat[j][0] + w_mat[i][1] * w_mat[j][1] + w_mat[i][2] * w_mat[j][2];
                }
            }
            let det = wtw[0][0] * wtw[1][1] - wtw[0][1] * wtw[1][0];
            if det.abs() < 1e-10 {
                break;
            }
            let inv_wtw = [
                [wtw[1][1] / det, -wtw[0][1] / det],
                [-wtw[1][0] / det, wtw[0][0] / det],
            ];
            let mut sum_h = [[0.0f32; 3]; 2];
            let mut sum_hh = [[0.0f32; 2]; 2];
            for p in sampled {
                let wt_od = [
                    w_mat[0][0] * p[0] + w_mat[0][1] * p[1] + w_mat[0][2] * p[2],
                    w_mat[1][0] * p[0] + w_mat[1][1] * p[1] + w_mat[1][2] * p[2],
                ];
                let h0 = (inv_wtw[0][0] * wt_od[0] + inv_wtw[0][1] * wt_od[1]).max(0.0);
                let h1 = (inv_wtw[1][0] * wt_od[0] + inv_wtw[1][1] * wt_od[1]).max(0.0);
                for c in 0..3 {
                    sum_h[0][c] += h0 * p[c];
                    sum_h[1][c] += h1 * p[c];
                }
                sum_hh[0][0] += h0 * h0;
                sum_hh[0][1] += h0 * h1;
                sum_hh[1][0] += h1 * h0;
                sum_hh[1][1] += h1 * h1;
            }
            let det2 = sum_hh[0][0] * sum_hh[1][1] - sum_hh[0][1] * sum_hh[1][0];
            if det2.abs() < 1e-10 {
                break;
            }
            let inv_hht = [
                [sum_hh[1][1] / det2, -sum_hh[0][1] / det2],
                [-sum_hh[1][0] / det2, sum_hh[0][0] / det2],
            ];
            for c in 0..3 {
                w[c][0] = (sum_h[0][c] * inv_hht[0][0] + sum_h[1][c] * inv_hht[1][0]).max(0.0);
                w[c][1] = (sum_h[0][c] * inv_hht[0][1] + sum_h[1][c] * inv_hht[1][1]).max(0.0);
            }
            for s in 0..2 {
                let norm =
                    (w[0][s] * w[0][s] + w[1][s] * w[1][s] + w[2][s] * w[2][s]).sqrt();
                if norm > 1e-6 {
                    for i in 0..3 {
                        w[i][s] /= norm;
                    }
                }
            }
            if w[2][0] < w[2][1] {
                for row in &mut w {
                    row.swap(0, 1);
                }
            }
        }
        stain1 = [w[0][0], w[1][0], w[2][0]];
        stain2 = [w[0][1], w[1][1], w[2][1]];
    }

    // Compute pseudo-inverse and scale factors
    let src_mat = [
        [stain1[0], stain2[0]],
        [stain1[1], stain2[1]],
        [stain1[2], stain2[2]],
    ];
    let inv = pseudo_inverse_2x3(&src_mat);

    // Find 99th percentile concentrations
    let mut conc_h: Vec<f32> = Vec::with_capacity(od_pixels.len());
    let mut conc_e: Vec<f32> = Vec::with_capacity(od_pixels.len());
    for p in &od_pixels {
        let c0 = (inv[0][0] * p[0] + inv[0][1] * p[1] + inv[0][2] * p[2]).max(0.0);
        let c1 = (inv[1][0] * p[0] + inv[1][1] * p[1] + inv[1][2] * p[2]).max(0.0);
        conc_h.push(c0);
        conc_e.push(c1);
    }
    conc_h.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    conc_e.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let src_max_h =
        conc_h[((conc_h.len() as f32 * 0.99) as usize).min(conc_h.len() - 1)].max(0.001);
    let src_max_e =
        conc_e[((conc_e.len() as f32 * 0.99) as usize).min(conc_e.len() - 1)].max(0.001);

    let scale_h = REF_MAX_CONC[0] / src_max_h;
    let scale_e = REF_MAX_CONC[1] / src_max_e;

    StainNormParams {
        enabled: true,
        inv_stain_r0: [inv[0][0], inv[0][1], inv[0][2], scale_h],
        inv_stain_r1: [inv[1][0], inv[1][1], inv[1][2], scale_e],
    }
}

/// Compute top 2 eigenvectors of a 3x3 symmetric matrix via power iteration.
fn top_two_eigenvectors(cov: &[[f32; 3]; 3]) -> ([f32; 3], [f32; 3]) {
    // First eigenvector
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

    // Deflate: cov2 = cov - lambda1 * v1 * v1^T
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

    // Second eigenvector
    let mut v2 = [1.0f32, 0.0, 0.0];
    // Start orthogonal to v1
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

/// 2x3 pseudo-inverse of a 3x2 matrix M: (M^T M)^-1 M^T
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
    let inv_mtm = [
        [mtm[1][1] / det, -mtm[0][1] / det],
        [-mtm[1][0] / det, mtm[0][0] / det],
    ];

    // (M^T M)^-1 M^T
    let mut result = [[0.0f32; 3]; 2];
    for i in 0..2 {
        for j in 0..3 {
            result[i][j] = inv_mtm[i][0] * m[j][0] + inv_mtm[i][1] * m[j][1];
        }
    }
    result
}

/// Result of trilinear level calculation
#[derive(Debug, Clone, Copy)]
pub struct TrilinearLevels {
    /// The higher resolution (lower index) level
    pub level_fine: u32,
    /// The lower resolution (higher index) level
    pub level_coarse: u32,
    /// Blend factor: 0.0 = use level_fine, 1.0 = use level_coarse
    pub blend: f64,
}

/// Calculate the two mip levels to blend for trilinear filtering
pub fn calculate_trilinear_levels(wsi: &WsiFile, target_downsample: f64) -> TrilinearLevels {
    let level_count = wsi.level_count();

    if level_count == 0 {
        return TrilinearLevels {
            level_fine: 0,
            level_coarse: 0,
            blend: 0.0,
        };
    }

    if level_count == 1 {
        return TrilinearLevels {
            level_fine: 0,
            level_coarse: 0,
            blend: 0.0,
        };
    }

    // Find the best level (where pixel density is closest to 1:1)
    let best_level = wsi.best_level_for_downsample(target_downsample);

    let best_info = match wsi.level(best_level) {
        Some(info) => info,
        None => {
            return TrilinearLevels {
                level_fine: 0,
                level_coarse: 0,
                blend: 0.0,
            };
        }
    };

    // Determine if we should blend with the next finer or coarser level
    // If target_downsample > best_level's downsample, we're between best and next coarser
    // If target_downsample < best_level's downsample, we're between best and next finer
    let (level_fine, level_coarse) = if target_downsample >= best_info.downsample {
        // Blend between best (fine) and next coarser level
        if best_level + 1 < level_count {
            (best_level, best_level + 1)
        } else {
            // At coarsest level, no blending
            return TrilinearLevels {
                level_fine: best_level,
                level_coarse: best_level,
                blend: 0.0,
            };
        }
    } else {
        // Blend between previous finer level and best (coarse)
        if best_level > 0 {
            (best_level - 1, best_level)
        } else {
            // At finest level, no blending
            return TrilinearLevels {
                level_fine: 0,
                level_coarse: 0,
                blend: 0.0,
            };
        }
    };

    // Calculate blend factor using log space for perceptually linear transitions
    let fine_info = wsi.level(level_fine);
    let coarse_info = wsi.level(level_coarse);

    let (fine_ds, coarse_ds) = match (fine_info, coarse_info) {
        (Some(f), Some(c)) => (f.downsample, c.downsample),
        _ => {
            return TrilinearLevels {
                level_fine,
                level_coarse,
                blend: 0.0,
            };
        }
    };

    // Log-space interpolation for smooth transitions
    let log_target = target_downsample.ln();
    let log_fine = fine_ds.ln();
    let log_coarse = coarse_ds.ln();

    let blend = if (log_coarse - log_fine).abs() < 0.001 {
        0.0
    } else {
        ((log_target - log_fine) / (log_coarse - log_fine)).clamp(0.0, 1.0)
    };

    TrilinearLevels {
        level_fine,
        level_coarse,
        blend,
    }
}

/// Render statistics for performance monitoring
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct RenderStats {
    /// Number of tiles rendered this frame
    pub tiles_rendered: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Frame render time in microseconds
    pub render_time_us: u64,
}

/// Calculate the optimal level for a given zoom
#[allow(dead_code)]
pub fn optimal_level(wsi: &WsiFile, zoom: f64) -> u32 {
    // Target: find level where pixel density is close to 1:1
    let target_downsample = 1.0 / zoom;
    wsi.best_level_for_downsample(target_downsample)
}

/// Calculate visible tile range for the viewport
#[allow(dead_code)]
pub fn visible_tile_range(
    viewport: &Viewport,
    level: u32,
    wsi: &WsiFile,
    tile_size: u32,
) -> Option<TileRange> {
    let level_info = wsi.level(level)?;
    let bounds = viewport.bounds();

    // Convert viewport bounds to level coordinates
    let level_left = bounds.left / level_info.downsample;
    let level_top = bounds.top / level_info.downsample;
    let level_right = bounds.right / level_info.downsample;
    let level_bottom = bounds.bottom / level_info.downsample;

    // Calculate tile indices
    let ts = tile_size as f64;
    let start_x = ((level_left / ts).floor() as i64 - 1).max(0) as u64;
    let start_y = ((level_top / ts).floor() as i64 - 1).max(0) as u64;
    let end_x = ((level_right / ts).ceil() as u64 + 1).min(level_info.tiles_x(tile_size));
    let end_y = ((level_bottom / ts).ceil() as u64 + 1).min(level_info.tiles_y(tile_size));

    Some(TileRange {
        level,
        start_x,
        start_y,
        end_x,
        end_y,
    })
}

/// Range of tiles to render
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TileRange {
    pub level: u32,
    pub start_x: u64,
    pub start_y: u64,
    pub end_x: u64,
    pub end_y: u64,
}

#[allow(dead_code)]
impl TileRange {
    /// Iterate over all tile coordinates in this range
    pub fn iter(&self) -> impl Iterator<Item = TileCoord> + '_ {
        let tile_size = 256;
        (self.start_y..self.end_y).flat_map(move |y| {
            (self.start_x..self.end_x).map(move |x| TileCoord::new(self.level, x, y, tile_size))
        })
    }

    /// Get the total number of tiles in this range
    pub fn tile_count(&self) -> usize {
        ((self.end_x - self.start_x) * (self.end_y - self.start_y)) as usize
    }
}

/// Bilinear interpolation for pixel sampling
#[allow(dead_code)]
pub fn bilinear_sample(data: &[u8], width: u32, height: u32, x: f64, y: f64) -> [u8; 4] {
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let get_pixel = |px: u32, py: u32| -> [f64; 4] {
        let idx = ((py * width + px) * 4) as usize;
        if idx + 3 < data.len() {
            [
                data[idx] as f64,
                data[idx + 1] as f64,
                data[idx + 2] as f64,
                data[idx + 3] as f64,
            ]
        } else {
            [0.0, 0.0, 0.0, 255.0]
        }
    };

    let p00 = get_pixel(x0, y0);
    let p10 = get_pixel(x1, y0);
    let p01 = get_pixel(x0, y1);
    let p11 = get_pixel(x1, y1);

    let mut result = [0u8; 4];
    for i in 0..4 {
        let top = p00[i] * (1.0 - fx) + p10[i] * fx;
        let bottom = p01[i] * (1.0 - fx) + p11[i] * fx;
        result[i] = (top * (1.0 - fy) + bottom * fy).clamp(0.0, 255.0) as u8;
    }

    result
}

type CoarseBlendData = (Arc<common::TileData>, [f32; 2], [f32; 2], f32);

#[derive(Default)]
struct PaneRenderOutcome {
    image: Option<Image>,
    keep_running: bool,
    rendered: bool,
}

fn tile_request_signature(
    tile_manager: &TileManager,
    viewport: &Viewport,
    level: u32,
    margin_tiles: i32,
) -> Option<TileRequestSignature> {
    let level_info = tile_manager.wsi().level(level)?;
    let bounds = viewport.bounds();
    let tile_size = tile_manager.tile_size_for_level(level);
    let tile_size_f64 = tile_size as f64;
    let level_left = bounds.left / level_info.downsample;
    let level_top = bounds.top / level_info.downsample;
    let level_right = bounds.right / level_info.downsample;
    let level_bottom = bounds.bottom / level_info.downsample;
    let margin_tiles_i64 = margin_tiles.max(0) as i64;

    Some(TileRequestSignature {
        level,
        margin_tiles,
        start_x: ((level_left / tile_size_f64).floor() as i64 - margin_tiles_i64).max(0) as u64,
        start_y: ((level_top / tile_size_f64).floor() as i64 - margin_tiles_i64).max(0) as u64,
        end_x: ((level_right / tile_size_f64).ceil() as u64 + margin_tiles_i64 as u64)
            .min(level_info.tiles_x(tile_size)),
        end_y: ((level_bottom / tile_size_f64).ceil() as u64 + margin_tiles_i64 as u64)
            .min(level_info.tiles_y(tile_size)),
        tile_size,
    })
}

pub(crate) fn thumbnail_image_for_file(file: &OpenFile) -> Option<Image> {
    let thumb_data = file.thumbnail.as_ref()?;

    let level = file.wsi.level_count().saturating_sub(1);
    let level_info = file.wsi.level(level)?;
    let aspect = level_info.width as f64 / level_info.height as f64;
    let (width, height) = if aspect > 1.0 {
        (150u32, (150.0 / aspect) as u32)
    } else {
        ((150.0 * aspect) as u32, 150u32)
    };

    blitter::create_image_buffer(thumb_data, width.max(1), height.max(1)).map(Image::from_rgba8)
}

pub(crate) fn update_and_render(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
) -> bool {
    let mut state = state.write();
    let render_backend = state.render_backend;
    let filtering_mode = state.filtering_mode;
    let pane_count = state.panes.len();
    let active_file_ids: Vec<Option<i32>> = state
        .panes
        .iter()
        .enumerate()
        .map(|(pane_index, _)| state.active_file_id_for_pane(PaneId(pane_index)))
        .collect();
    if active_file_ids.iter().all(Option::is_none) {
        for pane_index in 0..pane_count {
            crate::clear_cached_pane(PaneId(pane_index));
            state.set_last_rendered_file_id(PaneId(pane_index), None);
        }
        crate::update_tabs(ui, &state);
        return false;
    }

    let render_requested = std::mem::take(&mut state.needs_render);
    state.ant_offset = (state.ant_offset + 0.5) % 16.0;

    let content_width = (ui.get_content_area_width() as f64).max(100.0);
    let content_height = (ui.get_content_area_height() as f64 - 35.0).max(100.0);
    let pane_gap = 6.0;
    let pane_width = ((content_width - pane_gap * (pane_count.saturating_sub(1) as f64))
        / pane_count.max(1) as f64)
        .max(100.0);

    let mut wanted_tiles_by_file: HashMap<i32, HashSet<common::TileCoord>> = HashMap::new();
    for (pane_index, file_id) in active_file_ids.iter().copied().enumerate() {
        let Some(file_id) = file_id else {
            continue;
        };
        let pane = PaneId(pane_index);
        let Some(file_index) = state.open_files.iter().position(|f| f.id == file_id) else {
            continue;
        };

        let Some((viewport_state, last_request)) = ({
            let file = &mut state.open_files[file_index];
            match file.pane_state_mut(pane) {
                Some(pane_state) => {
                    pane_state.viewport.set_size(pane_width, content_height);
                    Some((pane_state.viewport.clone(), pane_state.last_request))
                }
                None => None,
            }
        }) else {
            continue;
        };

        let vp = &viewport_state.viewport;
        let margin_tiles = if viewport_state.is_moving() { 1 } else { 0 };
        let request = {
            let file = &state.open_files[file_index];
            let trilinear = calculate_trilinear_levels(&file.wsi, vp.effective_downsample());
            let level = trilinear.level_fine;
            tile_request_signature(&file.tile_manager, vp, level, margin_tiles).map(|signature| {
                let wanted = calculate_wanted_tiles(
                    &file.tile_manager,
                    level,
                    vp.bounds().left,
                    vp.bounds().top,
                    vp.bounds().right,
                    vp.bounds().bottom,
                    margin_tiles,
                );
                (signature, wanted)
            })
        };
        if let Some((signature, wanted)) = request {
            if last_request != Some(signature)
                && let Some(file) = state.open_files.get_mut(file_index)
                && let Some(pane_state) = file.pane_state_mut(pane)
            {
                pane_state.last_request = Some(signature);
            }
            wanted_tiles_by_file
                .entry(file_id)
                .or_default()
                .extend(wanted);
        }
    }

    for (file_id, wanted_tiles) in wanted_tiles_by_file {
        if let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) {
            file.tile_loader
                .set_wanted_tiles(wanted_tiles.into_iter().collect());
        }
    }

    let mut keep_running = render_requested;
    let mut rendered_frame = false;

    for (pane_index, file_id) in active_file_ids.into_iter().enumerate() {
        let pane = PaneId(pane_index);
        let file_switched = state.last_rendered_file_id(pane) != file_id;
        keep_running |= file_switched;

        let Some(file_id) = file_id else {
            crate::clear_cached_pane(pane);
            state.set_last_rendered_file_id(pane, None);
            continue;
        };

        let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
            crate::clear_cached_pane(pane);
            state.set_last_rendered_file_id(pane, None);
            continue;
        };

        let minimap_missing = crate::with_pane_render_cache(pane_count, |cache| {
            cache
                .get(pane.0)
                .and_then(|entry| entry.minimap_thumbnail.as_ref())
                .is_none()
        });
        let content_missing = crate::with_pane_render_cache(pane_count, |cache| {
            cache
                .get(pane.0)
                .and_then(|entry| entry.content.as_ref())
                .is_none()
        });
        let force_render = file_switched || content_missing;

        if pane.0 == 1 {
            debug!(
                pane = pane.0,
                file_id,
                file_switched,
                force_render,
                content_missing,
                minimap_missing,
                "pane cache state before render"
            );
        }

        if file_switched && let Some(pane_state) = file.pane_state_mut(pane) {
            pane_state.frame_count = 0;
        }
        if file_switched || minimap_missing {
            crate::set_cached_pane_minimap(pane, thumbnail_image_for_file(file));
        }

        let outcome = render_pane_to_image(
            ui,
            file,
            pane,
            tile_cache,
            pane_width,
            content_height,
            force_render,
            render_backend,
            filtering_mode,
        );

        if pane.0 == 1 {
            debug!(
                pane = pane.0,
                file_id,
                rendered = outcome.rendered,
                image = outcome.image.is_some(),
                keep_running = outcome.keep_running,
                "pane render outcome"
            );
        }

        keep_running |= outcome.keep_running;
        rendered_frame |= outcome.rendered;
        if let Some(image) = outcome.image {
            crate::set_cached_pane_content(pane, image);
            state.set_last_rendered_file_id(pane, Some(file_id));
        } else {
            state.set_last_rendered_file_id(pane, Some(file_id));
            keep_running |= content_missing;
        }
    }

    crate::update_tabs(ui, &state);
    keep_running |= tools::has_active_roi_overlay(&state);

    if rendered_frame {
        state.update_fps();
        ui.set_fps(state.current_fps);
    }

    keep_running
}

#[allow(clippy::too_many_arguments)]
fn render_pane_to_image(
    ui: &AppWindow,
    file: &mut OpenFile,
    pane: PaneId,
    tile_cache: &Arc<TileCache>,
    target_width: f64,
    target_height: f64,
    force_render: bool,
    render_backend: RenderBackend,
    filtering_mode: FilteringMode,
) -> PaneRenderOutcome {
    let (
        animating,
        viewport_state,
        frame_count,
        last_render_zoom,
        last_render_center_x,
        last_render_center_y,
        last_render_width,
        last_render_height,
        last_render_level,
        previous_tiles_loaded,
        last_seen_tile_epoch,
        hud_gamma,
        hud_brightness,
        hud_contrast,
        hud_stain_normalization,
        last_render_gamma,
        last_render_brightness,
        last_render_contrast,
        last_render_stain_normalization,
    ) = {
        let Some(pane_state) = file.pane_state_mut(pane) else {
            return PaneRenderOutcome::default();
        };

        let animating = pane_state.viewport.update();
        pane_state.viewport.set_size(target_width, target_height);
        (
            animating,
            pane_state.viewport.clone(),
            pane_state.frame_count,
            pane_state.last_render_zoom,
            pane_state.last_render_center_x,
            pane_state.last_render_center_y,
            pane_state.last_render_width,
            pane_state.last_render_height,
            pane_state.last_render_level,
            pane_state.tiles_loaded_since_render,
            pane_state.last_seen_tile_epoch,
            pane_state.hud.gamma,
            pane_state.hud.brightness,
            pane_state.hud.contrast,
            pane_state.hud.stain_normalization,
            pane_state.last_render_gamma,
            pane_state.last_render_brightness,
            pane_state.last_render_contrast,
            pane_state.last_render_stain_normalization,
        )
    };

    let vp = &viewport_state.viewport;
    let vp_zoom = vp.zoom;
    let vp_center_x = vp.center.x;
    let vp_center_y = vp.center.y;
    let vp_width = vp.width;
    let vp_height = vp.height;
    let is_first_frame = frame_count == 0;
    let viewport_changed = animating
        || (last_render_zoom - vp_zoom).abs() > 0.001
        || (last_render_center_x - vp_center_x).abs() > 1.0
        || (last_render_center_y - vp_center_y).abs() > 1.0
        || (last_render_width - vp_width).abs() > 1.0
        || (last_render_height - vp_height).abs() > 1.0;

    let trilinear = calculate_trilinear_levels(&file.wsi, vp.effective_downsample());
    let level = trilinear.level_fine;
    let margin_tiles = if viewport_state.is_moving() { 1 } else { 0 };
    let level_changed = level != last_render_level;

    let bounds = vp.bounds();
    let visible_tiles = file.tile_manager.visible_tiles_with_margin(
        level,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
        margin_tiles,
    );
    let visible_tiles: Vec<_> = visible_tiles.into_iter().take(500).collect();
    let cached_tiles: Vec<_> = visible_tiles
        .iter()
        .filter_map(|coord| tile_cache.peek(coord).map(|data| (*coord, data)))
        .collect();
    let cached_count = cached_tiles.len() as u32;

    // Only fetch coarse tiles for trilinear blending
    let lanczos_weight = if filtering_mode == FilteringMode::Lanczos3 {
        lanczos_adaptive_weight(vp_zoom)
    } else {
        1.0
    };
    // Trilinear blend needed for explicit Trilinear mode or adaptive Lanczos at low zoom
    let use_trilinear_blend = filtering_mode == FilteringMode::Trilinear
        || (filtering_mode == FilteringMode::Lanczos3 && lanczos_weight < 1.0);
    let cached_coarse_tiles: Vec<_> = if use_trilinear_blend
        && trilinear.level_fine != trilinear.level_coarse
        && trilinear.blend > 0.01
    {
        file.tile_manager
            .visible_tiles_with_margin(
                trilinear.level_coarse,
                bounds.left,
                bounds.top,
                bounds.right,
                bounds.bottom,
                margin_tiles,
            )
            .into_iter()
            .filter_map(|coord| tile_cache.peek(&coord).map(|data| (coord, data)))
            .collect()
    } else {
        Vec::new()
    };

    let loaded_tile_epoch = file.tile_loader.loaded_epoch();
    let tile_epoch_advanced = loaded_tile_epoch > last_seen_tile_epoch;
    let new_tiles_loaded = cached_count
        > if level_changed {
            0
        } else {
            previous_tiles_loaded
        }
        || !cached_coarse_tiles.is_empty()
        || tile_epoch_advanced;
    let tiles_pending = file.tile_loader.pending_count() > 0;

    let keep_running = animating || new_tiles_loaded || tiles_pending;

    if pane.0 == 1 {
        debug!(
            pane = pane.0,
            zoom = vp_zoom,
            width = vp_width,
            height = vp_height,
            level,
            cached_count,
            coarse_tiles = cached_coarse_tiles.len(),
            previous_tiles_loaded,
            tile_epoch_advanced,
            tiles_pending,
            viewport_changed,
            level_changed,
            force_render,
            keep_running,
            "pane render decision inputs"
        );
    }

    let adjustments_changed = (hud_gamma - last_render_gamma).abs() > 0.001
        || (hud_brightness - last_render_brightness).abs() > 0.001
        || (hud_contrast - last_render_contrast).abs() > 0.001
        || hud_stain_normalization != last_render_stain_normalization;

    if !force_render
        && !is_first_frame
        && !viewport_changed
        && !level_changed
        && !new_tiles_loaded
        && !adjustments_changed
    {
        return PaneRenderOutcome {
            image: None,
            keep_running,
            rendered: false,
        };
    }

    if let Some(pane_state) = file.pane_state_mut(pane) {
        pane_state.frame_count += 1;
        pane_state.last_render_time = std::time::Instant::now();
        pane_state.last_render_zoom = vp_zoom;
        pane_state.last_render_center_x = vp_center_x;
        pane_state.last_render_center_y = vp_center_y;
        pane_state.last_render_width = vp_width;
        pane_state.last_render_height = vp_height;
        pane_state.last_render_level = level;
        pane_state.tiles_loaded_since_render = cached_count;
        pane_state.last_seen_tile_epoch = loaded_tile_epoch;
        pane_state.last_render_gamma = hud_gamma;
        pane_state.last_render_brightness = hud_brightness;
        pane_state.last_render_contrast = hud_contrast;
        pane_state.last_render_stain_normalization = hud_stain_normalization;
    }

    let render_width = vp_width as u32;
    let render_height = vp_height.max(1.0) as u32;
    if render_width == 0 || render_height == 0 {
        return PaneRenderOutcome {
            image: None,
            keep_running,
            rendered: false,
        };
    }

    if render_backend == RenderBackend::Gpu {
        // Adaptive Lanczos: at low zoom, switch to trilinear on GPU
        let gpu_filtering = if filtering_mode == FilteringMode::Lanczos3 && lanczos_weight < 1.0 {
            FilteringMode::Trilinear
        } else {
            filtering_mode
        };
        let gpu_use_trilinear = gpu_filtering == FilteringMode::Trilinear;
        let gpu_trilinear = if gpu_use_trilinear {
            trilinear
        } else {
            TrilinearLevels {
                level_fine: trilinear.level_fine,
                level_coarse: trilinear.level_fine,
                blend: 0.0,
            }
        };
        let draws = collect_tile_draws(file, tile_cache, vp, gpu_trilinear, gpu_filtering);
        let stain_params = compute_stain_params(&draws, hud_stain_normalization);
        let slot = match pane {
            PaneId::PRIMARY => SurfaceSlot::PRIMARY,
            PaneId::SECONDARY => SurfaceSlot::SECONDARY,
            _ => SurfaceSlot(pane.0),
        };
        let image = crate::with_gpu_renderer(|renderer| {
            renderer.borrow_mut().queue_frame(
                slot,
                render_width,
                render_height,
                draws,
                hud_gamma,
                hud_brightness,
                hud_contrast,
                stain_params.enabled,
                stain_params.inv_stain_r0,
                stain_params.inv_stain_r1,
            )
        })
        .flatten();

        if image.is_some() {
            ui.window().request_redraw();
        } else if let Some(pane_state) = file.pane_state_mut(pane) {
            pane_state.frame_count = 0;
            pane_state.last_render_level = u32::MAX;
            pane_state.tiles_loaded_since_render = 0;
        }
        let rendered = image.is_some();

        return PaneRenderOutcome {
            image,
            keep_running: keep_running || !rendered,
            rendered,
        };
    }

    let level_info = match file.wsi.level(level) {
        Some(info) => info.clone(),
        None => {
            return PaneRenderOutcome {
                image: None,
                keep_running,
                rendered: false,
            };
        }
    };

    let level_count = file.wsi.level_count();
    let mut fallback_blits = Vec::new();
    for fallback_level in (0..level_count).rev() {
        if fallback_level <= level {
            continue;
        }

        let Some(fallback_level_info) = file.wsi.level(fallback_level).cloned() else {
            continue;
        };

        let fallback_tiles = file.tile_manager.visible_tiles(
            fallback_level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
        );

        for fb_coord in fallback_tiles.iter().take(100) {
            let Some(fallback_tile) = tile_cache.peek(fb_coord) else {
                continue;
            };

            let fb_image_x =
                fb_coord.x as f64 * fb_coord.tile_size as f64 * fallback_level_info.downsample;
            let fb_image_y =
                fb_coord.y as f64 * fb_coord.tile_size as f64 * fallback_level_info.downsample;
            let fb_image_x_end =
                fb_image_x + fallback_tile.width as f64 * fallback_level_info.downsample;
            let fb_image_y_end =
                fb_image_y + fallback_tile.height as f64 * fallback_level_info.downsample;

            let screen_x = ((fb_image_x - bounds.left) * vp.zoom).round() as i32;
            let screen_y = ((fb_image_y - bounds.top) * vp.zoom).round() as i32;
            let screen_x_end = ((fb_image_x_end - bounds.left) * vp.zoom).round() as i32;
            let screen_y_end = ((fb_image_y_end - bounds.top) * vp.zoom).round() as i32;
            let screen_w = screen_x_end - screen_x;
            let screen_h = screen_y_end - screen_y;

            if screen_w <= 0 || screen_h <= 0 {
                continue;
            }

            fallback_blits.push((fallback_tile, screen_x, screen_y, screen_w, screen_h));
        }
    }

    let Some(_pane_state) = file.pane_state_mut(pane) else {
        return PaneRenderOutcome::default();
    };
    // Render directly into SharedPixelBuffer to avoid an intermediate copy.
    let mut pixel_buffer =
        slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(render_width, render_height);
    let buffer = pixel_buffer.make_mut_bytes();
    blitter::fast_fill_rgba(buffer, 30, 30, 30, 255);

    // Helper: blit all fallback + fine tiles into a buffer with a given blit function
    #[allow(clippy::type_complexity)]
    let blit_all_tiles =
        |buf: &mut [u8], blit_fn: fn(&mut [u8], u32, u32, &[u8], u32, u32, i32, i32, i32, i32)| {
            for (fallback_tile, sx, sy, sw, sh) in &fallback_blits {
                blit_fn(
                    buf,
                    render_width,
                    render_height,
                    &fallback_tile.data,
                    fallback_tile.width,
                    fallback_tile.height,
                    *sx,
                    *sy,
                    *sw,
                    *sh,
                );
            }
            for (coord, tile_data) in cached_tiles.iter() {
                let image_x = coord.x as f64 * coord.tile_size as f64 * level_info.downsample;
                let image_y = coord.y as f64 * coord.tile_size as f64 * level_info.downsample;
                let image_x_end = image_x + tile_data.width as f64 * level_info.downsample;
                let image_y_end = image_y + tile_data.height as f64 * level_info.downsample;
                let screen_x = ((image_x - bounds.left) * vp.zoom).round() as i32;
                let screen_y = ((image_y - bounds.top) * vp.zoom).round() as i32;
                let screen_x_end = ((image_x_end - bounds.left) * vp.zoom).round() as i32;
                let screen_y_end = ((image_y_end - bounds.top) * vp.zoom).round() as i32;
                blit_fn(
                    buf,
                    render_width,
                    render_height,
                    &tile_data.data,
                    tile_data.width,
                    tile_data.height,
                    screen_x,
                    screen_y,
                    screen_x_end - screen_x,
                    screen_y_end - screen_y,
                );
            }
        };

    // Helper: apply trilinear coarse-level blend into a buffer
    let apply_trilinear_coarse = |buf: &mut [u8]| {
        if !use_trilinear_blend || trilinear.blend <= 0.01 || cached_coarse_tiles.is_empty() {
            return;
        }
        let Some(coarse_info) = file.wsi.level(trilinear.level_coarse) else {
            return;
        };
        let mut coarse_buffer = vec![0u8; (render_width * render_height * 4) as usize];
        blitter::fast_fill_rgba(&mut coarse_buffer, 30, 30, 30, 255);
        for (coord, tile_data) in cached_coarse_tiles.iter() {
            let image_x = coord.x as f64 * coord.tile_size as f64 * coarse_info.downsample;
            let image_y = coord.y as f64 * coord.tile_size as f64 * coarse_info.downsample;
            let image_x_end = image_x + tile_data.width as f64 * coarse_info.downsample;
            let image_y_end = image_y + tile_data.height as f64 * coarse_info.downsample;
            let screen_x = ((image_x - bounds.left) * vp.zoom).round() as i32;
            let screen_y = ((image_y - bounds.top) * vp.zoom).round() as i32;
            let screen_x_end = ((image_x_end - bounds.left) * vp.zoom).round() as i32;
            let screen_y_end = ((image_y_end - bounds.top) * vp.zoom).round() as i32;
            blitter::blit_tile(
                &mut coarse_buffer,
                render_width,
                render_height,
                &tile_data.data,
                tile_data.width,
                tile_data.height,
                screen_x,
                screen_y,
                screen_x_end - screen_x,
                screen_y_end - screen_y,
            );
        }
        blitter::blend_buffers(buf, &coarse_buffer, trilinear.blend);
    };

    let is_adaptive_lanczos = filtering_mode == FilteringMode::Lanczos3;

    if is_adaptive_lanczos && lanczos_weight > 0.0 && lanczos_weight < 1.0 {
        // ADAPTIVE BLEND ZONE: render Lanczos and Trilinear, then cross-fade
        // 1. Render Lanczos into main buffer
        blit_all_tiles(buffer, blitter::blit_tile_lanczos3);

        // 2. Render Trilinear into temp buffer (bilinear blit + coarse mip blend)
        let mut tri_buffer = vec![0u8; (render_width * render_height * 4) as usize];
        blitter::fast_fill_rgba(&mut tri_buffer, 30, 30, 30, 255);
        blit_all_tiles(&mut tri_buffer, blitter::blit_tile);
        apply_trilinear_coarse(&mut tri_buffer);

        // 3. Cross-fade: buffer = lanczos_weight * lanczos + (1 - lanczos_weight) * trilinear
        blitter::blend_buffers(buffer, &tri_buffer, 1.0 - lanczos_weight);
    } else if is_adaptive_lanczos && lanczos_weight >= 1.0 {
        // Pure Lanczos (high zoom)
        blit_all_tiles(buffer, blitter::blit_tile_lanczos3);
    } else if is_adaptive_lanczos {
        // Pure Trilinear (very low zoom, adaptive Lanczos fully faded out)
        blit_all_tiles(buffer, blitter::blit_tile);
        apply_trilinear_coarse(buffer);
    } else {
        // Non-Lanczos modes: Bilinear or explicit Trilinear
        blit_all_tiles(buffer, blitter::blit_tile);
        apply_trilinear_coarse(buffer);
    }

    // Post-processing: apply stain normalization if enabled
    match hud_stain_normalization {
        StainNormalization::None => {}
        StainNormalization::Macenko => apply_macenko_normalization(buffer),
        StainNormalization::Vahadane => apply_vahadane_normalization(buffer),
    }

    // Post-processing: apply gamma, brightness, contrast if they differ from defaults
    let has_adjustments = (hud_gamma - 1.0).abs() > 0.001
        || hud_brightness.abs() > 0.001
        || (hud_contrast - 1.0).abs() > 0.001;
    if has_adjustments {
        apply_adjustments(buffer, hud_gamma, hud_brightness, hud_contrast);
    }

    PaneRenderOutcome {
        image: Some(Image::from_rgba8(pixel_buffer)),
        keep_running,
        rendered: true,
    }
}

fn collect_tile_draws(
    file: &OpenFile,
    tile_cache: &Arc<TileCache>,
    vp: &Viewport,
    trilinear: TrilinearLevels,
    filtering_mode: FilteringMode,
) -> Vec<TileDraw> {
    let mut draws = Vec::new();
    let bounds = vp.bounds();
    let level_count = file.wsi.level_count();

    for fallback_level in (0..level_count).rev() {
        if fallback_level <= trilinear.level_fine {
            continue;
        }

        let Some(fallback_level_info) = file.wsi.level(fallback_level) else {
            continue;
        };

        let fallback_tiles = file.tile_manager.visible_tiles(
            fallback_level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
        );

        for coord in fallback_tiles.iter().take(100) {
            let Some(tile_data) = tile_cache.peek(coord) else {
                continue;
            };
            if let Some(draw) = tile_draw_from_tile(
                vp,
                bounds.left,
                bounds.top,
                fallback_level_info.downsample,
                *coord,
                tile_data,
                None,
                filtering_mode,
            ) {
                draws.push(draw);
            }
        }
    }

    let Some(level_info) = file.wsi.level(trilinear.level_fine) else {
        return draws;
    };

    let visible_tiles = file.tile_manager.visible_tiles_with_margin(
        trilinear.level_fine,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
        0,
    );

    for coord in visible_tiles.iter().take(500) {
        let Some(tile_data) = tile_cache.peek(coord) else {
            continue;
        };
        let coarse_blend = coarse_blend_for_tile(
            file,
            tile_cache,
            trilinear,
            level_info.downsample,
            *coord,
            &tile_data,
        );
        if let Some(draw) = tile_draw_from_tile(
            vp,
            bounds.left,
            bounds.top,
            level_info.downsample,
            *coord,
            tile_data,
            coarse_blend,
            filtering_mode,
        ) {
            draws.push(draw);
        }
    }

    draws
}

fn coarse_blend_for_tile(
    file: &OpenFile,
    tile_cache: &Arc<TileCache>,
    trilinear: TrilinearLevels,
    fine_downsample: f64,
    fine_coord: common::TileCoord,
    fine_tile: &Arc<common::TileData>,
) -> Option<CoarseBlendData> {
    const COARSE_BOUNDARY_EPSILON: f64 = 1e-3;

    if trilinear.level_fine == trilinear.level_coarse || trilinear.blend <= 0.01 {
        return None;
    }

    let coarse_info = file.wsi.level(trilinear.level_coarse)?;
    let image_x = fine_coord.x as f64 * fine_coord.tile_size as f64 * fine_downsample;
    let image_y = fine_coord.y as f64 * fine_coord.tile_size as f64 * fine_downsample;
    let fine_image_w = fine_tile.width as f64 * fine_downsample;
    let fine_image_h = fine_tile.height as f64 * fine_downsample;
    let coarse_tile_size = file
        .tile_manager
        .tile_size_for_level(trilinear.level_coarse) as f64;
    let image_x_end = image_x + fine_image_w;
    let image_y_end = image_y + fine_image_h;

    let coarse_tile_x = image_x / coarse_info.downsample;
    let coarse_tile_y = image_y / coarse_info.downsample;
    let coarse_tile_x_end =
        ((image_x_end - COARSE_BOUNDARY_EPSILON) / coarse_info.downsample).max(coarse_tile_x);
    let coarse_tile_y_end =
        ((image_y_end - COARSE_BOUNDARY_EPSILON) / coarse_info.downsample).max(coarse_tile_y);
    let coarse_start_tile_x = (coarse_tile_x / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_start_tile_y = (coarse_tile_y / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_end_tile_x = (coarse_tile_x_end / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_end_tile_y = (coarse_tile_y_end / coarse_tile_size).floor().max(0.0) as u64;

    if coarse_start_tile_x != coarse_end_tile_x || coarse_start_tile_y != coarse_end_tile_y {
        return None;
    }

    let coarse_coord = common::TileCoord::new(
        trilinear.level_coarse,
        coarse_start_tile_x,
        coarse_start_tile_y,
        coarse_tile_size as u32,
    );

    let coarse_tile = tile_cache.peek(&coarse_coord)?;
    let coarse_origin_x = coarse_coord.x as f64 * coarse_coord.tile_size as f64;
    let coarse_origin_y = coarse_coord.y as f64 * coarse_coord.tile_size as f64;
    let coarse_src_x = (coarse_tile_x - coarse_origin_x).max(0.0);
    let coarse_src_y = (coarse_tile_y - coarse_origin_y).max(0.0);
    let coarse_src_w = (fine_image_w / coarse_info.downsample).max(0.0);
    let coarse_src_h = (fine_image_h / coarse_info.downsample).max(0.0);
    let coarse_src_x_end = coarse_src_x + coarse_src_w;
    let coarse_src_y_end = coarse_src_y + coarse_src_h;

    if coarse_src_x_end <= coarse_src_x
        || coarse_src_y_end <= coarse_src_y
        || coarse_src_x_end > coarse_tile.width as f64 + COARSE_BOUNDARY_EPSILON
        || coarse_src_y_end > coarse_tile.height as f64 + COARSE_BOUNDARY_EPSILON
    {
        return None;
    }

    let coarse_width = coarse_tile.width as f64;
    let coarse_height = coarse_tile.height as f64;

    Some((
        coarse_tile,
        [
            (coarse_src_x / coarse_width) as f32,
            (coarse_src_y / coarse_height) as f32,
        ],
        [
            (coarse_src_x_end / coarse_width) as f32,
            (coarse_src_y_end / coarse_height) as f32,
        ],
        trilinear.blend as f32,
    ))
}

#[allow(clippy::too_many_arguments)]
fn tile_draw_from_tile(
    vp: &Viewport,
    bounds_left: f64,
    bounds_top: f64,
    downsample: f64,
    coord: common::TileCoord,
    tile_data: Arc<common::TileData>,
    coarse_blend: Option<CoarseBlendData>,
    filtering_mode: FilteringMode,
) -> Option<TileDraw> {
    let image_x = coord.x as f64 * coord.tile_size as f64 * downsample;
    let image_y = coord.y as f64 * coord.tile_size as f64 * downsample;
    let image_x_end = image_x + tile_data.width as f64 * downsample;
    let image_y_end = image_y + tile_data.height as f64 * downsample;

    let screen_x = ((image_x - bounds_left) * vp.zoom).round() as i32;
    let screen_y = ((image_y - bounds_top) * vp.zoom).round() as i32;
    let screen_x_end = ((image_x_end - bounds_left) * vp.zoom).round() as i32;
    let screen_y_end = ((image_y_end - bounds_top) * vp.zoom).round() as i32;
    let screen_w = screen_x_end - screen_x;
    let screen_h = screen_y_end - screen_y;

    if screen_w <= 0 || screen_h <= 0 {
        return None;
    }

    let (coarse_tile, coarse_uv_min, coarse_uv_max, mip_blend) = coarse_blend
        .map(|(coarse_tile, coarse_uv_min, coarse_uv_max, mip_blend)| {
            (Some(coarse_tile), coarse_uv_min, coarse_uv_max, mip_blend)
        })
        .unwrap_or((None, [0.0, 0.0], [1.0, 1.0], 0.0));

    Some(TileDraw {
        tile: tile_data,
        coarse_tile,
        screen_x,
        screen_y,
        screen_w,
        screen_h,
        coarse_uv_min,
        coarse_uv_max,
        mip_blend,
        filtering_mode,
    })
}

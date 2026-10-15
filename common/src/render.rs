//! Render-level selection shared by the CPU, GPU and export paths.

use crate::WsiFile;
use crate::wsi::{WsiLevel, best_level_for_downsample};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderBackend {
    #[default]
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilteringMode {
    Bilinear,
    #[default]
    Trilinear,
    Lanczos3,
}

// ─── Trilinear mip level selection ───────────────────────────────────────────

/// Conservative negative LOD bias for the trilinear path, in **octaves**
/// (log2 of the downsample): the target downsample is multiplied by
/// `2^TRILINEAR_LOD_BIAS` (about 0.84) before levels are chosen, which
/// offsets a slight tendency to choose overly coarse blends.
///
/// It used to be added to the level *index*, which on 4x pyramids (Aperio:
/// one index step is two octaves) amounted to half an octave and made the
/// full-resolution level the "fine" partner for downsamples 4-5.66.
pub const TRILINEAR_LOD_BIAS: f64 = -0.25;

/// Default [`TrilinearOptions::min_coarse_weight`].
pub const TRILINEAR_MIN_COARSE_WEIGHT: f64 = 0.02;

/// Default finer-level weight below which it is dropped when neighbouring
/// levels are at least 3x apart, so the finer level costs at least 9x the
/// pixels of the coarser one.
pub const TRILINEAR_MIN_FINE_WEIGHT_STEEP: f64 = 0.15;

/// Default finer-level weight below which it is dropped for gentler
/// pyramids (e.g. 2x).
pub const TRILINEAR_MIN_FINE_WEIGHT: f64 = 0.05;

/// Result of trilinear level calculation
#[derive(Debug, Clone, Copy)]
pub struct TrilinearLevels {
    /// The higher resolution (lower index) level
    pub level_fine: u32,
    /// The lower resolution (higher index) level
    pub level_coarse: u32,
    /// Blend factor: 0.0 = use level_fine, 1.0 = use level_coarse
    pub blend: f64,
    /// Continuous LOD (level index + fraction, interpolated in log
    /// downsample) of the target before the bias.
    pub lod_before_bias: f64,
    /// Continuous LOD of the biased target.
    pub lod_after_bias: f64,
}

/// Options for [`trilinear_levels`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrilinearOptions {
    /// LOD bias in octaves; negative values pick sharper levels.
    pub lod_bias_octaves: f64,
    /// The finer level is dropped (only the coarser one is drawn) when its
    /// blend weight is below this. `None` uses
    /// [`TRILINEAR_MIN_FINE_WEIGHT_STEEP`] for levels at least 3x apart and
    /// [`TRILINEAR_MIN_FINE_WEIGHT`] otherwise.
    pub min_fine_weight: Option<f64>,
    /// The coarser level is dropped when its blend weight is below this.
    pub min_coarse_weight: f64,
    /// `(viewport_pixels, budget)`: drop the finer level when drawing a
    /// viewport of `viewport_pixels` screen pixels from it would need more
    /// than `budget` level pixels.
    pub fine_pixel_budget: Option<(f64, f64)>,
}

impl Default for TrilinearOptions {
    fn default() -> Self {
        Self {
            lod_bias_octaves: TRILINEAR_LOD_BIAS,
            min_fine_weight: None,
            min_coarse_weight: TRILINEAR_MIN_COARSE_WEIGHT,
            fine_pixel_budget: None,
        }
    }
}

impl TrilinearOptions {
    /// Default options without the LOD bias.
    pub fn unbiased() -> Self {
        Self {
            lod_bias_octaves: 0.0,
            ..Self::default()
        }
    }
}

pub fn single_level_trilinear(level: u32) -> TrilinearLevels {
    let lod = level as f64;
    TrilinearLevels {
        level_fine: level,
        level_coarse: level,
        blend: 0.0,
        lod_before_bias: lod,
        lod_after_bias: lod,
    }
}

/// Continuous LOD of `downsample`: the index of the level at or below it
/// plus the fraction towards the next level, interpolated in log space.
pub fn continuous_lod(levels: &[WsiLevel], downsample: f64) -> f64 {
    if levels.is_empty() {
        return 0.0;
    }
    let fine = best_level_for_downsample(levels, downsample) as usize;
    let Some(coarse) = levels.get(fine + 1) else {
        return fine as f64;
    };
    let ds_fine = levels[fine].downsample;
    let span = (coarse.downsample / ds_fine).ln();
    if span.abs() < 1e-9 || downsample <= ds_fine {
        return fine as f64;
    }
    fine as f64 + ((downsample / ds_fine).ln() / span).clamp(0.0, 1.0)
}

/// Index-space LOD to levels, applying [`TRILINEAR_LOD_BIAS`] to the level
/// *index*.
#[deprecated(
    since = "1.0.0",
    note = "applies the bias in level-index units, which is wrong for pyramids that are not \
            2x per level; use `trilinear_levels` / `calculate_trilinear_levels`"
)]
pub fn finalize_trilinear_levels(
    level_count: u32,
    lod_before_bias: f64,
    apply_lod_bias: bool,
) -> TrilinearLevels {
    let max_lod = level_count.saturating_sub(1) as f64;
    let lod_after_bias = if apply_lod_bias {
        (lod_before_bias + TRILINEAR_LOD_BIAS).clamp(0.0, max_lod)
    } else {
        lod_before_bias.clamp(0.0, max_lod)
    };
    let level_fine = lod_after_bias.floor() as u32;
    let level_coarse = lod_after_bias.ceil() as u32;

    TrilinearLevels {
        level_fine,
        level_coarse,
        blend: lod_after_bias - level_fine as f64,
        lod_before_bias,
        lod_after_bias,
    }
}

/// The two levels to blend for trilinear filtering of `target_downsample`
/// (1 / zoom), and the blend factor.
///
/// The bias is applied in log2(downsample) space. A level whose weight is
/// negligible (see [`TrilinearOptions`]) or whose pixels exceed the budget
/// is dropped, in which case `level_fine == level_coarse`.
pub fn trilinear_levels(
    levels: &[WsiLevel],
    target_downsample: f64,
    options: &TrilinearOptions,
) -> TrilinearLevels {
    if levels.len() <= 1 {
        return single_level_trilinear(0);
    }
    let target = if target_downsample.is_finite() && target_downsample > 0.0 {
        target_downsample
    } else {
        levels[0].downsample
    };
    let biased = target * options.lod_bias_octaves.exp2();
    let lod_before_bias = continuous_lod(levels, target);
    let lod_after_bias = continuous_lod(levels, biased);
    let with_lods = |fine: u32, coarse: u32, blend: f64| TrilinearLevels {
        level_fine: fine,
        level_coarse: coarse,
        blend,
        lod_before_bias,
        lod_after_bias,
    };

    let fine = best_level_for_downsample(levels, biased);
    let Some(coarse_info) = levels.get(fine as usize + 1) else {
        return with_lods(fine, fine, 0.0);
    };
    let coarse = fine + 1;
    let ds_fine = levels[fine as usize].downsample;
    let ratio = coarse_info.downsample / ds_fine;
    if ratio <= 1.0 + 1e-9 || biased <= ds_fine {
        return with_lods(fine, fine, 0.0);
    }
    let blend = ((biased / ds_fine).ln() / ratio.ln()).clamp(0.0, 1.0);

    let min_fine_weight = options.min_fine_weight.unwrap_or(if ratio >= 3.0 {
        TRILINEAR_MIN_FINE_WEIGHT_STEEP
    } else {
        TRILINEAR_MIN_FINE_WEIGHT
    });
    let fine_too_big = options.fine_pixel_budget.is_some_and(|(viewport, budget)| {
        let scale = target / ds_fine;
        viewport * scale * scale > budget
    });
    if 1.0 - blend < min_fine_weight || fine_too_big {
        return with_lods(coarse, coarse, 0.0);
    }
    if blend < options.min_coarse_weight {
        return with_lods(fine, fine, 0.0);
    }
    with_lods(fine, coarse, blend)
}

/// Calculate the two mip levels to blend for trilinear filtering.
///
/// Given the target downsample factor (1.0 / viewport zoom) and the WSI file,
/// returns the fine and coarse levels plus the blend factor. With
/// `apply_lod_bias`, [`TRILINEAR_LOD_BIAS`] is applied (in octaves). See
/// [`trilinear_levels`].
pub fn calculate_trilinear_levels(
    wsi: &WsiFile,
    target_downsample: f64,
    apply_lod_bias: bool,
) -> TrilinearLevels {
    let options = if apply_lod_bias {
        TrilinearOptions::default()
    } else {
        TrilinearOptions::unbiased()
    };
    trilinear_levels(&wsi.properties().levels, target_downsample, &options)
}

/// [`trilinear_levels`] for a [`WsiFile`] with explicit options.
pub fn calculate_trilinear_levels_with(
    wsi: &WsiFile,
    target_downsample: f64,
    options: &TrilinearOptions,
) -> TrilinearLevels {
    trilinear_levels(&wsi.properties().levels, target_downsample, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn levels(downsamples: &[f64]) -> Vec<WsiLevel> {
        downsamples
            .iter()
            .enumerate()
            .map(|(i, &downsample)| WsiLevel {
                level: i as u32,
                width: (100_000.0 / downsample) as u64,
                height: (80_000.0 / downsample) as u64,
                downsample,
                tile_width: None,
                tile_height: None,
            })
            .collect()
    }

    /// C3L-00088-22.svs, the audit's example.
    fn aperio() -> Vec<WsiLevel> {
        levels(&[1.0, 4.0003, 16.004])
    }

    #[test]
    fn bias_is_applied_in_octaves_on_4x_pyramids() {
        let levels = aperio();
        let options = TrilinearOptions::default();
        // Downsample 4..5.66 used to pick level 0 as the fine partner.
        for target in [4.0, 4.5, 5.0, 5.5, 5.66] {
            let t = trilinear_levels(&levels, target, &options);
            assert_eq!(t.level_fine, 1, "target {target}: {t:?}");
        }
        let t = trilinear_levels(&levels, 5.0, &options);
        assert_eq!(t.level_coarse, 2);
        let expected = (5.0 * 2f64.powf(-0.25) / 4.0003).ln() / (16.004f64 / 4.0003).ln();
        assert!((t.blend - expected).abs() < 1e-9);
        assert!((t.lod_before_bias - (1.0 + (5.0f64 / 4.0003).ln() / 4.0007f64.ln())).abs() < 1e-3);
        assert!((t.lod_after_bias - (1.0 + expected)).abs() < 1e-9);
    }

    #[test]
    fn negligible_levels_are_dropped() {
        let levels = aperio();
        // Just below level 1 the finer level 0 would get under 15 %.
        let t = trilinear_levels(&levels, 3.9, &TrilinearOptions::default());
        assert_eq!((t.level_fine, t.level_coarse, t.blend), (1, 1, 0.0));
        // With a meaningful weight it is kept.
        let t = trilinear_levels(&levels, 2.5, &TrilinearOptions::default());
        assert_eq!((t.level_fine, t.level_coarse), (0, 1));
        // Exactly at a level: single level.
        let t = trilinear_levels(&levels, 16.004 * 2f64.powf(0.25), &TrilinearOptions::default());
        assert_eq!((t.level_fine, t.level_coarse), (2, 2));
        // Beyond either end.
        let t = trilinear_levels(&levels, 0.25, &TrilinearOptions::default());
        assert_eq!((t.level_fine, t.level_coarse), (0, 0));
        let t = trilinear_levels(&levels, 1000.0, &TrilinearOptions::default());
        assert_eq!((t.level_fine, t.level_coarse), (2, 2));
        let t = trilinear_levels(&levels, f64::NAN, &TrilinearOptions::default());
        assert_eq!((t.level_fine, t.level_coarse), (0, 0));
    }

    #[test]
    fn two_x_pyramids_blend_as_before_away_from_level_edges() {
        let levels = levels(&[1.0, 2.0, 4.0, 8.0]);
        let t = trilinear_levels(&levels, 3.0, &TrilinearOptions::default());
        assert_eq!((t.level_fine, t.level_coarse), (1, 2));
        let expected = (3.0 * 2f64.powf(-0.25) / 2.0).log2();
        assert!((t.blend - expected).abs() < 1e-9);
        // Without the bias, downsample 2 is level 1 exactly.
        let t = trilinear_levels(&levels, 2.0, &TrilinearOptions::unbiased());
        assert_eq!((t.level_fine, t.level_coarse, t.blend), (1, 1, 0.0));
    }

    #[test]
    fn fine_levels_over_the_pixel_budget_are_dropped() {
        let levels = levels(&[1.0, 4.0]);
        let options = TrilinearOptions {
            fine_pixel_budget: Some((1000.0, 4000.0)),
            ..TrilinearOptions::unbiased()
        };
        // Sampling level 0 at 3x needs 9000 pixels for a 1000-pixel view.
        let t = trilinear_levels(&levels, 3.0, &options);
        assert_eq!((t.level_fine, t.level_coarse), (1, 1));
        let t = trilinear_levels(&levels, 1.5, &options);
        assert_eq!((t.level_fine, t.level_coarse), (0, 1));
    }

    #[test]
    fn single_level_slides_never_blend() {
        let levels = levels(&[1.0]);
        let t = trilinear_levels(&levels, 50.0, &TrilinearOptions::default());
        assert_eq!((t.level_fine, t.level_coarse, t.blend), (0, 0, 0.0));
        assert_eq!(continuous_lod(&levels, 50.0), 0.0);
        assert_eq!(continuous_lod(&[], 2.0), 0.0);
    }

    #[test]
    #[allow(deprecated)]
    fn the_index_space_helper_keeps_its_old_behaviour() {
        let t = finalize_trilinear_levels(6, 2.5, true);
        assert_eq!((t.level_fine, t.level_coarse), (2, 3));
        assert!((t.blend - 0.25).abs() < 1e-9);
    }
}

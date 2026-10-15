//! Stain-vector estimation (Macenko, Vahadane) with tissue masking, fit
//! sanity checks and deterministic fallbacks.
//!
//! Every estimator here is deterministic: the same pixels and options give
//! bit-identical results on every run and thread count, because samples are
//! chosen with a fixed stride and all parallel reductions combine per-chunk
//! partial results in chunk order.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::basis::{Stain, StainBasis, StainPreset};
use super::linalg::{self, Vec3};
use super::od::{BackgroundEstimate, BackgroundOptions, OdLut, estimate_background};

/// Pixels per parallel work item. Fixed (not derived from the thread count)
/// so results do not depend on the machine.
const CHUNK_PIXELS: usize = 1 << 15;

/// Stain-vector estimation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StainFitMethod {
    /// Macenko et al. (2009): plane of the two principal OD directions, with
    /// robust angular extremes.
    Macenko,
    /// Vahadane et al. (2016): sparse non-negative factorisation. This is a
    /// simplified, deterministic alternating LASSO / least-squares scheme
    /// initialised from the Macenko estimate.
    Vahadane,
}

impl StainFitMethod {
    pub fn label(self) -> &'static str {
        match self {
            Self::Macenko => "Macenko",
            Self::Vahadane => "Vahadane",
        }
    }

    /// Maps the HUD/export enum to a fit method (`None` → no normalisation).
    pub fn from_normalization(value: crate::StainNormalization) -> Option<Self> {
        match value {
            crate::StainNormalization::None => None,
            crate::StainNormalization::Macenko => Some(Self::Macenko),
            crate::StainNormalization::Vahadane => Some(Self::Vahadane),
        }
    }
}

/// Which pixels may contribute to a stain fit (the tissue mask, CLR-2).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PixelFilter {
    /// Macenko's β: a pixel is used only if the OD of **every** channel is at
    /// least this large. Applying β to each channel (not to the OD sum)
    /// keeps glass at 225–240 out of the fit.
    pub beta: f32,
    /// Pixels whose OD is above this in all three channels (near-black ink,
    /// folds, out-of-slide fill) are excluded.
    pub dark_od: f32,
    /// Exclude pixels whose green channel is the brightest (green/cyan pen
    /// and ink). No common brightfield stain transmits green best.
    pub exclude_green_dominant: bool,
}

impl Default for PixelFilter {
    fn default() -> Self {
        Self {
            beta: 0.15,
            dark_od: 1.6,
            exclude_green_dominant: true,
        }
    }
}

impl PixelFilter {
    /// Whether an opaque pixel with the given RGB and OD is tissue for the
    /// purposes of a stain fit.
    #[inline(always)]
    pub fn accepts(&self, rgb: [u8; 3], od: [f32; 3]) -> bool {
        if od[0] < self.beta || od[1] < self.beta || od[2] < self.beta {
            return false;
        }
        if od[0] > self.dark_od && od[1] > self.dark_od && od[2] > self.dark_od {
            return false;
        }
        if self.exclude_green_dominant && rgb[1] > rgb[0] && rgb[1] > rgb[2] {
            return false;
        }
        true
    }
}

/// Sanity checks applied to every fit (CLR-6). A fit that fails is replaced
/// by the reference vectors (see [`FitFallback`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FitChecks {
    /// Minimum angle between any two fitted stain vectors.
    pub min_angle_deg: f32,
    /// Maximum 2-norm condition number of the fitted stain matrix.
    pub max_condition_number: f32,
    /// Maximum angle between a fitted vector and the corresponding reference
    /// vector. Catches fits dominated by pen marks, ink or non-H&E stains.
    pub max_reference_deviation_deg: f32,
    /// Normalisation scale factors (target / source concentration) are
    /// clamped to `[min_scale, max_scale]`.
    pub min_scale: f32,
    pub max_scale: f32,
}

impl Default for FitChecks {
    fn default() -> Self {
        Self {
            min_angle_deg: 10.0,
            max_condition_number: 15.0,
            max_reference_deviation_deg: 35.0,
            min_scale: 1.0 / 3.0,
            max_scale: 3.0,
        }
    }
}

/// Options for stain estimation. The defaults suit a slide-level fit; see
/// [`StainEstimateOptions::for_region`] for small regions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StainEstimateOptions {
    /// Pixels to read for a slide or region fit. The finest pyramid level
    /// whose image (or region) fits in this budget is used; slides without a
    /// small enough level are sampled sparsely.
    pub target_pixels: u64,
    /// Maximum number of tissue pixels used by the fit (taken with a fixed
    /// stride, so the result is deterministic).
    pub max_samples: usize,
    /// Maximum samples for the (iterative) Vahadane fit.
    pub vahadane_max_samples: usize,
    /// Tissue mask.
    pub filter: PixelFilter,
    /// Minimum fraction of the examined pixels that must be tissue.
    pub min_tissue_fraction: f32,
    /// Minimum number of tissue pixels.
    pub min_tissue_pixels: usize,
    /// Fixed background intensity. `None` estimates it from the pixels
    /// (falling back to 255 when no glass is visible).
    pub background: Option<[f32; 3]>,
    /// Background estimation settings.
    pub background_options: BackgroundOptions,
    /// Fit sanity checks.
    pub checks: FitChecks,
    /// Reference vectors used for plausibility checks and as the fallback.
    pub reference: StainPreset,
    /// Macenko angular percentile α (in percent).
    pub macenko_alpha: f32,
    /// Vahadane sparsity regulariser λ (OD units).
    pub vahadane_lambda: f32,
    /// Percentile of the concentration distribution used as each stain's
    /// "maximum" concentration (Macenko uses 99).
    pub concentration_percentile: f32,
}

impl Default for StainEstimateOptions {
    fn default() -> Self {
        Self {
            target_pixels: 4_000_000,
            max_samples: 100_000,
            vahadane_max_samples: 20_000,
            filter: PixelFilter::default(),
            min_tissue_fraction: 0.001,
            min_tissue_pixels: 5_000,
            background: None,
            background_options: BackgroundOptions::default(),
            checks: FitChecks::default(),
            reference: StainPreset::HematoxylinEosinMacenko,
            macenko_alpha: 1.0,
            vahadane_lambda: 0.1,
            concentration_percentile: 99.0,
        }
    }
}

impl StainEstimateOptions {
    /// Defaults for a user-selected region ("fit to ROI"): the region must be
    /// mostly tissue, and a smaller absolute pixel count suffices.
    pub fn for_region() -> Self {
        Self {
            min_tissue_fraction: 0.05,
            min_tissue_pixels: 1_000,
            ..Self::default()
        }
    }
}

/// Why a fit was replaced by the reference vectors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "kebab-case")]
pub enum FitFallback {
    /// Too little tissue was visible. The reference vectors and the
    /// reference maximum concentrations are used (normalisation is then the
    /// identity on tissue colours).
    InsufficientTissue {
        tissue_pixels: u64,
        tissue_fraction: f32,
    },
    /// The fitted vectors were (nearly) collinear.
    IllConditioned {
        min_angle_deg: f32,
        condition_number: f32,
    },
    /// A fitted vector was implausibly far from the reference.
    Implausible { deviation_deg: f32 },
    /// The estimator did not converge to a usable basis.
    FitFailed,
}

impl FitFallback {
    pub fn describe(&self) -> String {
        match self {
            Self::InsufficientTissue {
                tissue_pixels,
                tissue_fraction,
            } => format!(
                "insufficient tissue ({tissue_pixels} px, {:.2}%)",
                tissue_fraction * 100.0
            ),
            Self::IllConditioned {
                min_angle_deg,
                condition_number,
            } => format!(
                "ill-conditioned fit (stain vectors {min_angle_deg:.1}° apart, condition {condition_number:.1})"
            ),
            Self::Implausible { deviation_deg } => {
                format!("implausible stain vectors ({deviation_deg:.1}° from the reference)")
            }
            Self::FitFailed => "the fit did not converge".to_string(),
        }
    }
}

/// Where the pixels of an estimate came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum EstimateSource {
    /// Pixels supplied directly by the caller.
    Pixels,
    /// A whole slide, read at `level` (`sparse` when a regular grid of
    /// blocks was sampled instead of the full level).
    Slide {
        level: u32,
        downsample: f64,
        sparse: bool,
    },
    /// A level-0 rectangle of a slide, read at `level`.
    Region {
        x: i64,
        y: i64,
        width: u64,
        height: u64,
        level: u32,
        downsample: f64,
    },
}

/// Current [`StainEstimate`] file format version.
pub const STAIN_ESTIMATE_FORMAT_VERSION: u32 = 1;

/// The result of a stain fit: source stain vectors, robust maximum
/// concentrations and background intensity, plus fit diagnostics.
///
/// Serialisable so that a fit can be locked, saved and re-used (for example
/// as a reference slide via [`NormalizationTarget::from_estimate`](super::NormalizationTarget::from_estimate)).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StainEstimate {
    pub format_version: u32,
    pub method: StainFitMethod,
    /// Source stain vectors (hematoxylin first), or the reference vectors if
    /// [`fallback`](Self::fallback) is set.
    pub basis: StainBasis,
    /// Robust maximum concentration of each stain (same order as `basis`).
    pub max_concentrations: Vec<f32>,
    /// Background intensity per channel.
    pub i0: [f32; 3],
    /// Whether `i0` was measured from visible glass (vs. assumed/fixed).
    pub background_found: bool,
    /// Opaque pixels examined.
    pub examined_pixels: u64,
    /// Pixels that passed the tissue mask.
    pub tissue_pixels: u64,
    /// `tissue_pixels / examined_pixels`.
    pub tissue_fraction: f32,
    /// Samples used by the fit.
    pub samples: u64,
    /// Smallest angle between the fitted vectors (before any fallback).
    pub min_angle_deg: f32,
    /// Condition number of the fitted matrix (before any fallback).
    pub condition_number: f32,
    /// Set when the fit was rejected and reference vectors are used.
    pub fallback: Option<FitFallback>,
    pub source: EstimateSource,
    /// Optional caller-supplied slide identity (fingerprint), recorded so a
    /// saved estimate can be matched to its slide.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slide_identity: Option<String>,
}

impl StainEstimate {
    /// `true` if the reference vectors replaced the fit.
    pub fn is_fallback(&self) -> bool {
        self.fallback.is_some()
    }

    /// One-line human-readable summary for provenance and logs.
    pub fn summary(&self) -> String {
        let source = match &self.source {
            EstimateSource::Pixels => "supplied pixels".to_string(),
            EstimateSource::Slide { level, sparse, .. } => {
                if *sparse {
                    format!("slide, sparse sample of level {level}")
                } else {
                    format!("slide level {level}")
                }
            }
            EstimateSource::Region {
                width,
                height,
                level,
                ..
            } => format!("{width}×{height} px region, level {level}"),
        };
        match &self.fallback {
            None => format!(
                "{} fit ({source}; {:.1}% tissue; stains {:.1}° apart)",
                self.method.label(),
                self.tissue_fraction * 100.0,
                self.basis.min_angle_deg()
            ),
            Some(reason) => format!(
                "{} fit rejected ({}); reference stain vectors used ({source})",
                self.method.label(),
                reason.describe()
            ),
        }
    }

    /// Serialises to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("stain estimates are serialisable")
    }

    /// Parses JSON produced by [`to_json`](Self::to_json), validating the
    /// format version and stain vectors.
    pub fn from_json(json: &str) -> crate::Result<Self> {
        let estimate: Self = serde_json::from_str(json)
            .map_err(|e| crate::Error::InvalidFormat(format!("invalid stain estimate: {e}")))?;
        if estimate.format_version > STAIN_ESTIMATE_FORMAT_VERSION {
            return Err(crate::Error::InvalidFormat(format!(
                "stain estimate format {} is newer than supported ({STAIN_ESTIMATE_FORMAT_VERSION})",
                estimate.format_version
            )));
        }
        if estimate.max_concentrations.len() != estimate.basis.len()
            || estimate
                .max_concentrations
                .iter()
                .any(|c| !c.is_finite() || *c <= 0.0)
            || estimate
                .i0
                .iter()
                .any(|v| !v.is_finite() || *v < 1.0 || *v > 255.0)
        {
            return Err(crate::Error::InvalidFormat(
                "stain estimate has invalid concentrations or background".into(),
            ));
        }
        Ok(estimate)
    }

    /// Writes the estimate as JSON ("lock / save stain vectors").
    pub fn save_json(&self, path: &std::path::Path) -> crate::Result<()> {
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, self.to_json())?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Reads an estimate written by [`save_json`](Self::save_json).
    pub fn load_json(path: &std::path::Path) -> crate::Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }
}

// ─── Pixel access ────────────────────────────────────────────────────────────

/// A logical concatenation of RGBA slices, split into fixed-size chunks.
struct PixelChunks<'a> {
    chunks: Vec<&'a [u8]>,
}

impl<'a> PixelChunks<'a> {
    fn new(slices: &[&'a [u8]]) -> Self {
        let mut chunks = Vec::new();
        for slice in slices {
            let usable = slice.len() / 4 * 4;
            for chunk in slice[..usable].chunks(CHUNK_PIXELS * 4) {
                chunks.push(chunk);
            }
        }
        Self { chunks }
    }

    /// Iterates opaque pixels in order (pixels with alpha < 255 are "no
    /// data" or partially covered edges and never count as tissue or glass).
    fn opaque(&self) -> impl Iterator<Item = [u8; 3]> + Clone + '_ {
        self.chunks.iter().flat_map(|chunk| {
            chunk
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|px| px[3] == 255)
                .map(|px| [px[0], px[1], px[2]])
        })
    }
}

struct Samples {
    od: Vec<Vec3>,
    examined: u64,
    tissue: u64,
}

fn collect_samples(
    pixels: &PixelChunks<'_>,
    lut: &OdLut,
    filter: &PixelFilter,
    max_samples: usize,
) -> Samples {
    // Pass 1: per-chunk counts (ordered).
    let counts: Vec<(u64, u64)> = pixels
        .chunks
        .par_iter()
        .map(|chunk| {
            let mut examined = 0u64;
            let mut tissue = 0u64;
            for px in chunk.as_chunks::<4>().0 {
                if px[3] != 255 {
                    continue;
                }
                examined += 1;
                let rgb = [px[0], px[1], px[2]];
                if filter.accepts(rgb, lut.od(rgb)) {
                    tissue += 1;
                }
            }
            (examined, tissue)
        })
        .collect();
    let examined: u64 = counts.iter().map(|c| c.0).sum();
    let tissue: u64 = counts.iter().map(|c| c.1).sum();
    if tissue == 0 {
        return Samples {
            od: Vec::new(),
            examined,
            tissue,
        };
    }
    let stride = tissue.div_ceil(max_samples.max(1) as u64).max(1);
    let mut starts = Vec::with_capacity(counts.len());
    let mut acc = 0u64;
    for (_, t) in &counts {
        starts.push(acc);
        acc += t;
    }
    // Pass 2: take every `stride`-th tissue pixel, by global tissue index.
    let parts: Vec<Vec<Vec3>> = pixels
        .chunks
        .par_iter()
        .zip(starts.par_iter())
        .map(|(chunk, &start)| {
            let mut out = Vec::new();
            let mut index = start;
            for px in chunk.as_chunks::<4>().0 {
                if px[3] != 255 {
                    continue;
                }
                let rgb = [px[0], px[1], px[2]];
                let od = lut.od(rgb);
                if filter.accepts(rgb, od) {
                    if index.is_multiple_of(stride) {
                        out.push(od);
                    }
                    index += 1;
                }
            }
            out
        })
        .collect();
    Samples {
        od: parts.into_iter().flatten().collect(),
        examined,
        tissue,
    }
}

/// Deterministic parallel sum over fixed chunks of `items`.
fn chunked_sum<T, const N: usize>(items: &[T], f: impl Fn(&T) -> [f64; N] + Sync) -> [f64; N]
where
    T: Sync,
{
    let partials: Vec<[f64; N]> = items
        .par_chunks(4096)
        .map(|chunk| {
            let mut acc = [0.0f64; N];
            for item in chunk {
                let v = f(item);
                for k in 0..N {
                    acc[k] += v[k];
                }
            }
            acc
        })
        .collect();
    let mut total = [0.0f64; N];
    for p in partials {
        for k in 0..N {
            total[k] += p[k];
        }
    }
    total
}

fn percentile(values: &mut [f32], pct: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f32::total_cmp);
    let rank = (pct.clamp(0.0, 100.0) / 100.0) * (values.len() - 1) as f32;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let t = rank - lo as f32;
    values[lo] * (1.0 - t) + values[hi] * t
}

/// Orders two stain vectors hematoxylin-first (H absorbs more red) and makes
/// them physically valid (non-negative, unit length).
fn finish_pair(a: Vec3, b: Vec3) -> Option<[Vec3; 2]> {
    let a = linalg::normalize(a.map(|v| v.max(0.0)))?;
    let b = linalg::normalize(b.map(|v| v.max(0.0)))?;
    Some(if a[0] >= b[0] { [a, b] } else { [b, a] })
}

// ─── Macenko ─────────────────────────────────────────────────────────────────

fn macenko(samples: &[Vec3], alpha: f32) -> Option<[Vec3; 2]> {
    if samples.len() < 3 {
        return None;
    }
    let n = samples.len() as f64;
    let sums = chunked_sum(samples, |p| {
        let p = p.map(|v| v as f64);
        [
            p[0],
            p[1],
            p[2],
            p[0] * p[0],
            p[0] * p[1],
            p[0] * p[2],
            p[1] * p[1],
            p[1] * p[2],
            p[2] * p[2],
        ]
    });
    let mean = [sums[0] / n, sums[1] / n, sums[2] / n];
    let cov = |s: f64, i: usize, j: usize| (s - n * mean[i] * mean[j]) / (n - 1.0).max(1.0);
    let c = [
        [cov(sums[3], 0, 0), cov(sums[4], 0, 1), cov(sums[5], 0, 2)],
        [cov(sums[4], 0, 1), cov(sums[6], 1, 1), cov(sums[7], 1, 2)],
        [cov(sums[5], 0, 2), cov(sums[7], 1, 2), cov(sums[8], 2, 2)],
    ];
    let (_, vectors) = linalg::symmetric_eigen3(c);
    let mut e1 = vectors[0];
    let e2 = vectors[1];
    // Orient the principal direction along the tissue (all points then have
    // positive projections and the angles do not wrap around ±π).
    if e1[0] * mean[0] + e1[1] * mean[1] + e1[2] * mean[2] < 0.0 {
        e1 = e1.map(|v| -v);
    }
    let e1f = e1.map(|v| v as f32);
    let e2f = e2.map(|v| v as f32);
    let mut angles: Vec<f32> = samples
        .par_iter()
        .map(|p| linalg::dot(*p, e2f).atan2(linalg::dot(*p, e1f)))
        .collect();
    let min_phi = percentile(&mut angles, alpha);
    let max_phi = percentile(&mut angles, 100.0 - alpha);
    let along = |phi: f32| -> Vec3 {
        let (s, c) = phi.sin_cos();
        [
            e1f[0] * c + e2f[0] * s,
            e1f[1] * c + e2f[1] * s,
            e1f[2] * c + e2f[2] * s,
        ]
    };
    finish_pair(along(min_phi), along(max_phi))
}

// ─── Vahadane ────────────────────────────────────────────────────────────────

fn vahadane(samples: &[Vec3], init: [Vec3; 2], lambda: f32) -> Option<[Vec3; 2]> {
    if samples.len() < 3 {
        return None;
    }
    let mut w = init;
    for _outer in 0..30 {
        // Sparse coding: per-sample non-negative LASSO by coordinate descent.
        let ww = [
            [linalg::dot(w[0], w[0]), linalg::dot(w[0], w[1])],
            [linalg::dot(w[1], w[0]), linalg::dot(w[1], w[1])],
        ];
        let codes: Vec<[f32; 2]> = samples
            .par_iter()
            .map(|p| {
                let wv = [linalg::dot(w[0], *p), linalg::dot(w[1], *p)];
                let mut h = [0.0f32; 2];
                for _ in 0..25 {
                    h[0] = ((wv[0] - ww[0][1] * h[1] - lambda) / ww[0][0].max(1e-10)).max(0.0);
                    h[1] = ((wv[1] - ww[1][0] * h[0] - lambda) / ww[1][1].max(1e-10)).max(0.0);
                }
                h
            })
            .collect();
        // Dictionary update: least squares W = (HᵀH)⁻¹ HᵀV, projected onto
        // non-negative unit vectors.
        let indices: Vec<usize> = (0..samples.len()).collect();
        let s = chunked_sum(&indices, |&i| {
            let h = codes[i];
            let v = samples[i];
            [
                (h[0] * h[0]) as f64,
                (h[0] * h[1]) as f64,
                (h[1] * h[1]) as f64,
                (h[0] * v[0]) as f64,
                (h[0] * v[1]) as f64,
                (h[0] * v[2]) as f64,
                (h[1] * v[0]) as f64,
                (h[1] * v[1]) as f64,
                (h[1] * v[2]) as f64,
            ]
        });
        let det = s[0] * s[2] - s[1] * s[1];
        if det.abs() < 1e-12 {
            break;
        }
        let inv = [[s[2] / det, -s[1] / det], [-s[1] / det, s[0] / det]];
        let htv = [[s[3], s[4], s[5]], [s[6], s[7], s[8]]];
        let mut next = [[0.0f32; 3]; 2];
        for (r, row) in next.iter_mut().enumerate() {
            for (k, value) in row.iter_mut().enumerate() {
                *value = ((inv[r][0] * htv[0][k] + inv[r][1] * htv[1][k]) as f32).max(0.0);
            }
        }
        let (Some(a), Some(b)) = (linalg::normalize(next[0]), linalg::normalize(next[1])) else {
            break;
        };
        let change = linalg::angle_deg(a, w[0]).max(linalg::angle_deg(b, w[1]));
        w = [a, b];
        if change < 1e-3 {
            break;
        }
    }
    finish_pair(w[0], w[1])
}

// ─── Fit pipeline ────────────────────────────────────────────────────────────

fn max_concentrations(samples: &[Vec3], basis: &StainBasis, pct: f32) -> Vec<f32> {
    let pinv = basis.pseudo_inverse();
    (0..basis.len())
        .map(|i| {
            let mut c: Vec<f32> = samples
                .par_iter()
                .map(|od| linalg::dot(pinv[i], *od))
                .collect();
            percentile(&mut c, pct).max(1e-3)
        })
        .collect()
}

fn two_stain_basis(pair: [Vec3; 2], reference: &StainBasis) -> Option<StainBasis> {
    let names = reference.names();
    let name = |i: usize, fallback: &str| -> String {
        names
            .get(i)
            .map(|s| s.to_string())
            .unwrap_or_else(|| fallback.to_string())
    };
    StainBasis::new(vec![
        Stain::new(name(0, "Stain 1"), pair[0]),
        Stain::new(name(1, "Stain 2"), pair[1]),
    ])
    .ok()
}

/// Estimates stain vectors from RGBA pixel slices.
///
/// This is the core used by the slide-level and region estimators; it can
/// also be called with any in-memory pixels. Pixels with alpha < 255 are
/// ignored.
pub fn estimate_stain_from_rgba(
    slices: &[&[u8]],
    method: StainFitMethod,
    options: &StainEstimateOptions,
) -> StainEstimate {
    estimate_with_source(slices, method, options, EstimateSource::Pixels)
}

pub(crate) fn estimate_with_source(
    slices: &[&[u8]],
    method: StainFitMethod,
    options: &StainEstimateOptions,
    source: EstimateSource,
) -> StainEstimate {
    let pixels = PixelChunks::new(slices);
    let background = match options.background {
        Some(i0) => BackgroundEstimate {
            i0: i0.map(|v| v.clamp(1.0, 255.0)),
            found: false,
        },
        None => estimate_background(pixels.opaque(), &options.background_options),
    };
    let lut = OdLut::new(background.i0);
    let max_samples = match method {
        StainFitMethod::Macenko => options.max_samples,
        StainFitMethod::Vahadane => options.max_samples.min(options.vahadane_max_samples),
    };
    let samples = collect_samples(&pixels, &lut, &options.filter, max_samples);
    let reference = StainBasis::preset(options.reference);
    let tissue_fraction = if samples.examined == 0 {
        0.0
    } else {
        samples.tissue as f32 / samples.examined as f32
    };

    let mut estimate = StainEstimate {
        format_version: STAIN_ESTIMATE_FORMAT_VERSION,
        method,
        basis: reference.clone(),
        max_concentrations: Vec::new(),
        i0: background.i0,
        background_found: background.found,
        examined_pixels: samples.examined,
        tissue_pixels: samples.tissue,
        tissue_fraction,
        samples: samples.od.len() as u64,
        min_angle_deg: 0.0,
        condition_number: f32::INFINITY,
        fallback: None,
        source,
        slide_identity: None,
    };

    if samples.tissue < options.min_tissue_pixels as u64
        || tissue_fraction < options.min_tissue_fraction
        || samples.od.len() < 3
    {
        estimate.fallback = Some(FitFallback::InsufficientTissue {
            tissue_pixels: samples.tissue,
            tissue_fraction,
        });
        estimate.max_concentrations = reference_max_concentrations(&reference);
        return estimate;
    }

    let macenko_fit = macenko(&samples.od, options.macenko_alpha);
    let fit = match method {
        StainFitMethod::Macenko => macenko_fit,
        StainFitMethod::Vahadane => {
            let init = macenko_fit.unwrap_or_else(|| {
                let v = reference.vectors();
                [v[0], v[1]]
            });
            vahadane(&samples.od, init, options.vahadane_lambda)
        }
    };

    let mut chosen = None;
    match fit {
        Some(pair) => {
            let vectors = [pair[0], pair[1]];
            estimate.min_angle_deg = linalg::min_pairwise_angle_deg(&vectors);
            estimate.condition_number = linalg::condition_number(&vectors);
            match validate(pair, &reference, &options.checks) {
                Ok(()) => match two_stain_basis(pair, &reference) {
                    Some(basis) => chosen = Some(basis),
                    None => estimate.fallback = Some(FitFallback::FitFailed),
                },
                Err(reason) => estimate.fallback = Some(reason),
            }
        }
        None => estimate.fallback = Some(FitFallback::FitFailed),
    }
    estimate.basis = chosen.unwrap_or_else(|| reference.clone());
    estimate.max_concentrations = max_concentrations(
        &samples.od,
        &estimate.basis,
        options.concentration_percentile,
    );
    estimate
}

fn validate(
    pair: [Vec3; 2],
    reference: &StainBasis,
    checks: &FitChecks,
) -> Result<(), FitFallback> {
    let vectors = [pair[0], pair[1]];
    let min_angle = linalg::min_pairwise_angle_deg(&vectors);
    let condition = linalg::condition_number(&vectors);
    let well_conditioned = min_angle.is_finite()
        && condition.is_finite()
        && min_angle >= checks.min_angle_deg
        && condition <= checks.max_condition_number;
    if !well_conditioned {
        return Err(FitFallback::IllConditioned {
            min_angle_deg: min_angle,
            condition_number: condition,
        });
    }
    let refs = reference.vectors();
    if refs.len() == 2 {
        let deviation =
            linalg::angle_deg(pair[0], refs[0]).max(linalg::angle_deg(pair[1], refs[1]));
        if deviation > checks.max_reference_deviation_deg {
            return Err(FitFallback::Implausible {
                deviation_deg: deviation,
            });
        }
    }
    Ok(())
}

/// Maximum concentrations that make a reference basis map onto itself when
/// it is also the normalisation target.
pub(crate) fn reference_max_concentrations(reference: &StainBasis) -> Vec<f32> {
    if reference.matching_preset() == Some(StainPreset::HematoxylinEosinMacenko) {
        super::transform::MACENKO_REFERENCE_MAX_C.to_vec()
    } else {
        vec![1.0; reference.len()]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Deterministic pseudo-random noise in [-1, 1).
    pub(crate) fn noise(seed: u64) -> f32 {
        let mut x = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x6A09_E667_F3BC_C909);
        x ^= x >> 31;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 29;
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// Synthesises an RGBA pixel from H and E concentrations using `basis`
    /// and background `i0`, with a little OD noise.
    pub(crate) fn synth_pixel(basis: &[Vec3], conc: &[f32], i0: [f32; 3], seed: u64) -> [u8; 4] {
        let mut out = [0u8; 4];
        for c in 0..3 {
            let mut od: f32 = basis.iter().zip(conc).map(|(v, k)| v[c] * k).sum();
            od += 0.01 * noise(seed * 3 + c as u64);
            out[c] = (i0[c] * (-od).exp()).round().clamp(0.0, 255.0) as u8;
        }
        out[3] = 255;
        out
    }

    /// A field of synthetic H&E tissue (and optionally glass) pixels.
    pub(crate) fn synthetic_field(
        basis: &[Vec3],
        i0: [f32; 3],
        tissue: usize,
        glass: usize,
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity((tissue + glass) * 4);
        for i in 0..tissue {
            let t = (noise(i as u64 + 17) + 1.0) / 2.0;
            let u = (noise(i as u64 + 91_919) + 1.0) / 2.0;
            // Nuclei (H-rich, a little eosin), stroma (E-rich, a little
            // hematoxylin) and mixed cytoplasm.
            let (ch, ce) = match i % 10 {
                0..=2 => (0.6 + 1.0 * t, 0.15 * u),
                3..=7 => (0.12 * t, 0.3 + 0.7 * u),
                _ => (0.2 + 0.6 * t, 0.2 + 0.6 * u),
            };
            buf.extend_from_slice(&synth_pixel(basis, &[ch, ce], i0, i as u64));
        }
        for i in 0..glass {
            let j = (noise(i as u64 + 5) * 2.0).round();
            buf.extend_from_slice(&[
                (i0[0] + j).clamp(0.0, 255.0) as u8,
                (i0[1] + j).clamp(0.0, 255.0) as u8,
                (i0[2] + j).clamp(0.0, 255.0) as u8,
                255,
            ]);
        }
        buf
    }

    fn rj_basis() -> Vec<Vec3> {
        StainPreset::HematoxylinEosin.basis().vectors()
    }

    #[test]
    fn macenko_recovers_source_vectors_and_rejects_glass() {
        let src = rj_basis();
        for bg in [255.0f32, 240.0, 232.0, 225.0] {
            let i0 = [bg, bg - 2.0, bg];
            let pixels = synthetic_field(&src, i0, 30_000, 30_000);
            let est = estimate_stain_from_rgba(
                &[&pixels],
                StainFitMethod::Macenko,
                &StainEstimateOptions::default(),
            );
            assert!(est.fallback.is_none(), "bg {bg}: {:?}", est.fallback);
            assert!(est.background_found, "bg {bg}");
            // Macenko's per-channel β drops the palest pure-eosin pixels
            // (eosin barely absorbs red), which biases E by a few degrees;
            // the reference target was derived the same way.
            let fitted = est.basis.vectors();
            for (k, (f, s)) in fitted.iter().zip(&src).enumerate() {
                let err = linalg::angle_deg(*f, *s);
                assert!(err < 5.0, "bg {bg}: stain {k} off by {err}°");
            }
            // No glass pixel may pass the mask (pale stroma legitimately
            // fails β as well, so fewer than all 30 000 tissue pixels count).
            assert!(
                est.tissue_pixels <= 30_000 && est.tissue_pixels > 10_000,
                "bg {bg}: {} tissue pixels",
                est.tissue_pixels
            );
        }
    }

    #[test]
    fn dark_glass_needs_the_estimated_background() {
        // On a scanner whose glass is ~210, assuming I₀ = 255 makes every
        // glass pixel pass β (OD ≈ 0.19 per channel) and the fit is dragged
        // towards neutral grey (CLR-2). Estimating I₀ removes the glass.
        let src = rj_basis();
        let i0 = [210.0, 208.0, 212.0];
        let pixels = synthetic_field(&src, i0, 20_000, 60_000);
        let assumed = StainEstimateOptions {
            background: Some([255.0; 3]),
            ..StainEstimateOptions::default()
        };
        let bad = estimate_stain_from_rgba(&[&pixels], StainFitMethod::Macenko, &assumed);
        let good = estimate_stain_from_rgba(
            &[&pixels],
            StainFitMethod::Macenko,
            &StainEstimateOptions::default(),
        );
        assert!(
            bad.tissue_pixels > 60_000,
            "glass counted as tissue: {}",
            bad.tissue_pixels
        );
        assert!(good.fallback.is_none(), "{:?}", good.fallback);
        assert!(
            good.tissue_pixels <= 20_000 && good.tissue_pixels > 5_000,
            "{}",
            good.tissue_pixels
        );
        let err = |est: &StainEstimate| {
            est.basis
                .vectors()
                .iter()
                .zip(&src)
                .map(|(f, s)| linalg::angle_deg(*f, *s))
                .fold(0.0f32, f32::max)
        };
        assert!(err(&good) < 5.0, "{}", err(&good));
        assert!(bad.is_fallback() || err(&bad) > 2.0 * err(&good));
    }

    #[test]
    fn vahadane_recovers_source_vectors() {
        let src = rj_basis();
        let pixels = synthetic_field(&src, [240.0; 3], 20_000, 10_000);
        let est = estimate_stain_from_rgba(
            &[&pixels],
            StainFitMethod::Vahadane,
            &StainEstimateOptions::default(),
        );
        assert!(est.fallback.is_none(), "{:?}", est.fallback);
        for (f, s) in est.basis.vectors().iter().zip(&src) {
            assert!(linalg::angle_deg(*f, *s) < 6.0, "{f:?} vs {s:?}");
        }
    }

    #[test]
    fn estimation_is_deterministic() {
        let src = rj_basis();
        let pixels = synthetic_field(&src, [242.0; 3], 120_000, 40_000);
        for method in [StainFitMethod::Macenko, StainFitMethod::Vahadane] {
            let a = estimate_stain_from_rgba(&[&pixels], method, &StainEstimateOptions::default());
            let b = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap()
                .install(|| {
                    estimate_stain_from_rgba(&[&pixels], method, &StainEstimateOptions::default())
                });
            assert_eq!(a, b, "{method:?} differs between thread counts");
        }
    }

    #[test]
    fn glass_only_views_fall_back_to_the_reference() {
        let pixels = synthetic_field(&rj_basis(), [238.0, 236.0, 240.0], 50, 100_000);
        let est = estimate_stain_from_rgba(
            &[&pixels],
            StainFitMethod::Macenko,
            &StainEstimateOptions::default(),
        );
        assert!(matches!(
            est.fallback,
            Some(FitFallback::InsufficientTissue { .. })
        ));
        assert_eq!(est.basis, StainPreset::HematoxylinEosinMacenko.basis());
        assert_eq!(est.max_concentrations, MACENKO_REFERENCE_MAX_C_VEC.to_vec());
    }

    const MACENKO_REFERENCE_MAX_C_VEC: [f32; 2] = super::super::transform::MACENKO_REFERENCE_MAX_C;

    #[test]
    fn uniform_regions_are_rejected_as_ill_conditioned() {
        // A single colour repeated: coincident stain vectors (CLR-6).
        let mut pixels = Vec::new();
        for i in 0..20_000u64 {
            let j = (noise(i) * 1.5) as i32;
            pixels.extend_from_slice(&[(150 + j) as u8, (90 + j) as u8, (170 + j) as u8, 255]);
        }
        let est = estimate_stain_from_rgba(
            &[&pixels],
            StainFitMethod::Macenko,
            &StainEstimateOptions::default(),
        );
        assert!(est.is_fallback(), "{:?}", est.basis);
        // Fallback still measures the tissue's concentrations in the
        // reference basis, so intensity normalisation remains meaningful.
        assert!(est.max_concentrations.iter().all(|c| *c > 0.01));
    }

    #[test]
    fn green_pen_and_black_ink_are_masked() {
        let filter = PixelFilter::default();
        let lut = OdLut::new([255.0; 3]);
        let green_pen = [60, 140, 80];
        let black_ink = [40, 40, 45];
        let nucleus = [115, 86, 175];
        let glass = [236, 234, 239];
        assert!(!filter.accepts(green_pen, lut.od(green_pen)));
        assert!(!filter.accepts(black_ink, lut.od(black_ink)));
        assert!(filter.accepts(nucleus, lut.od(nucleus)));
        // Per-channel β: glass at ~236 is not tissue even with I₀ = 255.
        assert!(!filter.accepts(glass, lut.od(glass)));
    }

    #[test]
    fn transparent_pixels_are_ignored() {
        let src = rj_basis();
        let mut pixels = synthetic_field(&src, [240.0; 3], 10_000, 0);
        // Append a large block of transparent "no data".
        pixels.extend(std::iter::repeat_n(0u8, 4 * 200_000));
        let est = estimate_stain_from_rgba(
            &[&pixels],
            StainFitMethod::Macenko,
            &StainEstimateOptions::default(),
        );
        assert_eq!(est.examined_pixels, 10_000);
        assert!(est.fallback.is_none());
    }

    #[test]
    fn estimates_round_trip_through_json() {
        let pixels = synthetic_field(&rj_basis(), [240.0; 3], 20_000, 5_000);
        let mut est = estimate_stain_from_rgba(
            &[&pixels],
            StainFitMethod::Macenko,
            &StainEstimateOptions::default(),
        );
        est.slide_identity = Some("abc123".into());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stain.json");
        est.save_json(&path).unwrap();
        let back = StainEstimate::load_json(&path).unwrap();
        assert_eq!(back, est);
        let mut bad = est.clone();
        bad.format_version = STAIN_ESTIMATE_FORMAT_VERSION + 1;
        assert!(StainEstimate::from_json(&bad.to_json()).is_err());
    }
}

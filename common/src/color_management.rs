//! ICC colour management (AUDIT §14.1): a slide's colours converted from
//! its scanner's colour profile to sRGB, for display and export.
//!
//! The conversion is sampled once per profile into a [`LUT_SIZE`]³ lookup
//! table of unclamped sRGB values (computed by `moxcms` with the profile's
//! perceptual intent), which the CPU and the GPU interpolate identically:
//! tetrahedrally, in `f32`, from the quantised 8-bit pixel, then clamped and
//! rounded. [`crate::processing::wgsl::ICC_FUNCTIONS`] is the shader
//! counterpart and [`crate::processing::wgsl::emulate_icc`] its Rust
//! transcription, for the golden tests.
//!
//! On real H&E pixels the table is within 2 levels of the direct
//! conversion (mean 0.07 for an Aperio "ScanScope v1" profile). Keeping the
//! nodes unclamped lets the interpolation follow the conversion up to the
//! edge of the sRGB gamut; beyond it (saturated greens and cyans, which
//! stained tissue does not have) colours are only approximated.
//!
//! Supported are RGB profiles that `moxcms` reads: matrix/TRC shapers and
//! LUT-based profiles (`lut8`, `lut16`, `mAB`) connecting through XYZ or
//! Lab, which covers scanner profiles such as Aperio's "ScanScope v1" and
//! Roche's DP 200 ones. Grey, CMYK and other colour spaces, and named-colour
//! or device-link profiles, are refused; such a slide is shown as scanned.
//! The display is taken to be sRGB: eov does not read the monitor's
//! profile.

use std::fmt;
use std::hash::{Hash, Hasher};

use moxcms::{ColorProfile, DataColorSpace, Layout, ProfileText, TransformOptions};

/// Nodes per axis of the lookup table.
pub const LUT_SIZE: usize = 33;

/// Grid units per 8-bit level: an 8-bit value `v` sits at `v · GRID_SCALE`.
pub const GRID_SCALE: f32 = (LUT_SIZE - 1) as f32 / 255.0;

/// Why a profile cannot be used.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ColorProfileError {
    #[error("the colour profile cannot be read ({0})")]
    Unreadable(String),
    #[error("the colour profile is not an RGB profile ({0})")]
    NotRgb(String),
    #[error("the colour profile cannot be converted to sRGB ({0})")]
    Unconvertible(String),
}

/// A slide's colour profile → sRGB, ready to apply.
pub struct DisplayTransform {
    description: String,
    /// `LUT_SIZE³` nodes, red outermost and blue innermost: node
    /// `(r, g, b)` is at `(r · N + g) · N + b`.
    lut: Box<[[f32; 3]]>,
    identity: bool,
    id: u64,
}

impl fmt::Debug for DisplayTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DisplayTransform")
            .field("description", &self.description)
            .field("identity", &self.identity)
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

fn text(value: &Option<ProfileText>) -> Option<String> {
    let text = match value.as_ref()? {
        ProfileText::PlainString(text) => text.clone(),
        ProfileText::Localizable(strings) => strings
            .iter()
            .find(|s| s.language.eq_ignore_ascii_case("en"))
            .or_else(|| strings.first())?
            .value
            .clone(),
        ProfileText::Description(description) => description.ascii_string.clone(),
    };
    let text = text.trim_matches(|c: char| c == '\0' || c.is_whitespace());
    (!text.is_empty()).then(|| text.to_string())
}

/// How the table is sampled: the profile's perceptual intent, out-of-gamut
/// values kept for matrix/TRC profiles too.
fn transform_options() -> TransformOptions {
    TransformOptions {
        allow_extended_range_rgb_xyz: true,
        ..TransformOptions::default()
    }
}

/// The node value of grid index `i` (`0..LUT_SIZE`), in `[0, 1]`.
fn node(i: usize) -> f32 {
    i as f32 / (LUT_SIZE - 1) as f32
}

impl DisplayTransform {
    /// Samples the conversion of `profile` (the bytes of an ICC profile) to
    /// sRGB.
    pub fn from_icc(profile: &[u8]) -> Result<Self, ColorProfileError> {
        let source = ColorProfile::new_from_slice(profile)
            .map_err(|error| ColorProfileError::Unreadable(format!("{error:?}")))?;
        if source.color_space != DataColorSpace::Rgb {
            return Err(ColorProfileError::NotRgb(format!(
                "{:?}",
                source.color_space
            )));
        }
        let srgb = ColorProfile::new_srgb();
        let transform = source
            .create_transform_f32(Layout::Rgb, &srgb, Layout::Rgb, transform_options())
            .map_err(|error| ColorProfileError::Unconvertible(format!("{error:?}")))?;
        let n = LUT_SIZE;
        let mut grid = Vec::with_capacity(n * n * n * 3);
        for r in 0..n {
            for g in 0..n {
                for b in 0..n {
                    grid.extend_from_slice(&[node(r), node(g), node(b)]);
                }
            }
        }
        let mut out = vec![0f32; grid.len()];
        transform
            .transform(&grid, &mut out)
            .map_err(|error| ColorProfileError::Unconvertible(format!("{error:?}")))?;
        if out.iter().any(|v| !v.is_finite()) {
            return Err(ColorProfileError::Unconvertible(
                "the conversion is not finite".to_string(),
            ));
        }
        let lut: Box<[[f32; 3]]> = out.as_chunks::<3>().0.into();
        // A profile that changes nothing (sRGB itself) is not applied.
        let identity = lut.iter().enumerate().all(|(i, value)| {
            let expected = [node(i / (n * n)), node(i / n % n), node(i % n)];
            value
                .iter()
                .zip(expected)
                .all(|(v, e)| (v - e).abs() < 0.25 / 255.0)
        });
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        profile.hash(&mut hasher);
        Ok(Self {
            description: text(&source.description).unwrap_or_else(|| "unnamed profile".into()),
            lut,
            identity,
            id: hasher.finish(),
        })
    }

    /// The profile's own name, e.g. `"ScanScope v1"`.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Identifies the profile (a hash of its bytes), for caches.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The profile is sRGB already: converting changes nothing.
    pub fn is_identity(&self) -> bool {
        self.identity
    }

    /// The lookup table (see the struct docs for its layout).
    pub fn lut(&self) -> &[[f32; 3]] {
        &self.lut
    }

    /// The table as `RGBA32F` texels of a `LUT_SIZE³` 3D texture whose x
    /// axis is blue, y green and z red (the layout the shader reads).
    pub fn gpu_texels(&self) -> Vec<[f32; 4]> {
        self.lut.iter().map(|&[r, g, b]| [r, g, b, 1.0]).collect()
    }

    #[inline(always)]
    fn at(&self, r: usize, g: usize, b: usize) -> [f32; 3] {
        self.lut[(r * LUT_SIZE + g) * LUT_SIZE + b]
    }

    /// One pixel, interpolated tetrahedrally (clamped to `[0, 1]`, not
    /// rounded).
    pub fn convert(&self, rgb: [u8; 3]) -> [f32; 3] {
        let p = rgb.map(|v| f32::from(v) * GRID_SCALE);
        let i = p.map(|v| (v.floor() as usize).min(LUT_SIZE - 2));
        let (fx, fy, fz) = (p[0] - i[0] as f32, p[1] - i[1] as f32, p[2] - i[2] as f32);
        let (r, g, b) = (i[0], i[1], i[2]);
        let c000 = self.at(r, g, b);
        let c111 = self.at(r + 1, g + 1, b + 1);
        // The tetrahedron of the cube that holds the point, and the
        // barycentric weights of its corners.
        let (c1, c2, w0, w1, w2, w3) = if fx >= fy {
            if fy >= fz {
                (
                    self.at(r + 1, g, b),
                    self.at(r + 1, g + 1, b),
                    1.0 - fx,
                    fx - fy,
                    fy - fz,
                    fz,
                )
            } else if fx >= fz {
                (
                    self.at(r + 1, g, b),
                    self.at(r + 1, g, b + 1),
                    1.0 - fx,
                    fx - fz,
                    fz - fy,
                    fy,
                )
            } else {
                (
                    self.at(r, g, b + 1),
                    self.at(r + 1, g, b + 1),
                    1.0 - fz,
                    fz - fx,
                    fx - fy,
                    fy,
                )
            }
        } else if fz >= fy {
            (
                self.at(r, g, b + 1),
                self.at(r, g + 1, b + 1),
                1.0 - fz,
                fz - fy,
                fy - fx,
                fx,
            )
        } else if fz >= fx {
            (
                self.at(r, g + 1, b),
                self.at(r, g + 1, b + 1),
                1.0 - fy,
                fy - fz,
                fz - fx,
                fx,
            )
        } else {
            (
                self.at(r, g + 1, b),
                self.at(r + 1, g + 1, b),
                1.0 - fy,
                fy - fx,
                fx - fz,
                fz,
            )
        };
        [0, 1, 2].map(|c| (w0 * c000[c] + w1 * c1[c] + w2 * c2[c] + w3 * c111[c]).clamp(0.0, 1.0))
    }

    /// One pixel, converted and rounded.
    #[inline]
    pub fn apply_pixel(&self, rgb: [u8; 3]) -> [u8; 3] {
        self.convert(rgb).map(|v| (v * 255.0).round() as u8)
    }

    /// An RGBA buffer, in parallel; pixels with alpha 0 (outside the
    /// slide) are left as they are.
    pub fn apply_rgba(&self, buffer: &mut [u8]) {
        use rayon::prelude::*;
        if self.identity {
            return;
        }
        buffer.par_chunks_mut(4 * 16_384).for_each(|chunk| {
            for px in chunk.as_chunks_mut::<4>().0 {
                if px[3] == 0 {
                    continue;
                }
                let out = self.apply_pixel([px[0], px[1], px[2]]);
                px[..3].copy_from_slice(&out);
            }
        });
    }
}

/// Synthetic profiles for tests.
#[cfg(any(test, feature = "test-support"))]
pub mod test_profiles {
    use moxcms::ColorProfile;

    /// An Adobe RGB (1998) profile: its colours are more saturated than
    /// sRGB's, so converting visibly changes them.
    pub fn adobe_rgb() -> Vec<u8> {
        ColorProfile::new_adobe_rgb()
            .encode()
            .expect("moxcms encodes its own profiles")
    }

    /// Display P3.
    pub fn display_p3() -> Vec<u8> {
        ColorProfile::new_display_p3()
            .encode()
            .expect("moxcms encodes its own profiles")
    }

    /// sRGB itself.
    pub fn srgb() -> Vec<u8> {
        ColorProfile::new_srgb()
            .encode()
            .expect("moxcms encodes its own profiles")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moxcms::ColorProfile;

    /// moxcms's own conversion of `pixels`, clamped and rounded.
    fn direct(profile: &[u8], pixels: &[[u8; 3]]) -> Vec<[u8; 3]> {
        let source = ColorProfile::new_from_slice(profile).unwrap();
        let transform = source
            .create_transform_f32(
                Layout::Rgb,
                &ColorProfile::new_srgb(),
                Layout::Rgb,
                transform_options(),
            )
            .unwrap();
        let input: Vec<f32> = pixels
            .iter()
            .flatten()
            .map(|&v| f32::from(v) / 255.0)
            .collect();
        let mut out = vec![0f32; input.len()];
        transform.transform(&input, &mut out).unwrap();
        out.as_chunks::<3>()
            .0
            .iter()
            .map(|px| px.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8))
            .collect()
    }

    /// Pseudo-random stained-tissue colours: pinks and purples (green the
    /// weakest channel) up to glass.
    fn tissue_colours(count: usize) -> Vec<[u8; 3]> {
        let mut seed = 7u32;
        let mut next = move |lo: u32, hi: u32| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            lo + (seed >> 8) % (hi - lo + 1)
        };
        (0..count)
            .map(|_| {
                let (r, b) = (next(110, 255), next(110, 255));
                let g = next(30, r.min(b) - 5);
                [r as u8, g as u8, b as u8]
            })
            .collect()
    }

    #[test]
    fn profiles_convert_through_the_table_like_the_transform() {
        let profile = test_profiles::adobe_rgb();
        let transform = DisplayTransform::from_icc(&profile).unwrap();
        assert!(!transform.is_identity());
        assert_eq!(transform.lut().len(), LUT_SIZE.pow(3));
        let pixels = tissue_colours(20_000);
        let expected = direct(&profile, &pixels);
        let mut worst = 0u8;
        for (rgb, expected) in pixels.iter().zip(&expected) {
            let got = transform.apply_pixel(*rgb);
            for c in 0..3 {
                worst = worst.max(got[c].abs_diff(expected[c]));
            }
        }
        assert!(worst <= 2, "the table is off by {worst} levels");
        // Adobe RGB's green is outside sRGB: converted, it clips.
        let green = transform.apply_pixel([0, 255, 0]);
        assert!(
            green[0] == 0 && green[1] == 255 && green[2] < 80,
            "{green:?}"
        );
        // Neutral greys stay neutral.
        let grey = transform.apply_pixel([128, 128, 128]);
        assert!(grey[0].abs_diff(grey[1]) <= 1 && grey[1].abs_diff(grey[2]) <= 1);
    }

    #[test]
    fn grid_nodes_are_reproduced_exactly() {
        let transform = DisplayTransform::from_icc(&test_profiles::display_p3()).unwrap();
        // 8-bit values on a node: 0 and 255.
        for rgb in [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 255]] {
            let node = transform.at(
                if rgb[0] == 255 { LUT_SIZE - 1 } else { 0 },
                if rgb[1] == 255 { LUT_SIZE - 1 } else { 0 },
                if rgb[2] == 255 { LUT_SIZE - 1 } else { 0 },
            );
            let expected = node.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
            assert_eq!(transform.apply_pixel(rgb), expected, "{rgb:?}");
        }
    }

    #[test]
    fn srgb_profiles_change_nothing() {
        let transform = DisplayTransform::from_icc(&test_profiles::srgb()).unwrap();
        assert!(transform.is_identity());
        let mut pixels = vec![12, 200, 99, 255];
        transform.apply_rgba(&mut pixels);
        assert_eq!(pixels, [12, 200, 99, 255]);
    }

    #[test]
    fn transparent_pixels_are_left_alone() {
        let transform = DisplayTransform::from_icc(&test_profiles::adobe_rgb()).unwrap();
        let mut pixels = vec![10, 200, 30, 255, 10, 200, 30, 0];
        transform.apply_rgba(&mut pixels);
        assert_eq!(&pixels[..3], &transform.apply_pixel([10, 200, 30]));
        assert_eq!(&pixels[4..], &[10, 200, 30, 0]);
    }

    #[test]
    fn unusable_profiles_are_refused() {
        assert!(matches!(
            DisplayTransform::from_icc(b"not a profile"),
            Err(ColorProfileError::Unreadable(_))
        ));
        let grey = ColorProfile::new_gray_with_gamma(2.2).encode().unwrap();
        assert!(matches!(
            DisplayTransform::from_icc(&grey),
            Err(ColorProfileError::NotRgb(_))
        ));
    }

    #[test]
    fn profiles_are_named_and_identified() {
        let adobe = DisplayTransform::from_icc(&test_profiles::adobe_rgb()).unwrap();
        let p3 = DisplayTransform::from_icc(&test_profiles::display_p3()).unwrap();
        assert!(!adobe.description().is_empty());
        assert_ne!(adobe.id(), p3.id());
        assert_eq!(
            adobe.id(),
            DisplayTransform::from_icc(&test_profiles::adobe_rgb())
                .unwrap()
                .id()
        );
        assert_eq!(adobe.gpu_texels().len(), LUT_SIZE.pow(3));
    }
}

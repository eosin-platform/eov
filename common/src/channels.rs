//! Channels of multichannel slides (fluorescence, multiplexed
//! immunofluorescence) and how they are displayed.
//!
//! A multichannel slide has one grey-value plane per channel (a stain, a
//! marker, a fluorophore). eov shows them composited to RGB: every visible
//! channel adds its colour, scaled by its value inside the channel's
//! display window, and the sum is clamped. Reads ([`WsiFile::read_region`],
//! tiles, thumbnails, exports, plugins) all return that composite, so
//! everything downstream of the slide works unchanged.
//!
//! * [`ChannelSet`]: what the file says about its channels, with sample
//!   statistics of a low-resolution level ([`ChannelStats`]).
//! * [`ChannelDisplay`]: how each channel is drawn (visibility, colour,
//!   window, gamma). [`WsiFile::set_channel_display`] changes it for every
//!   reader of the slide; cached tiles must then be dropped.
//! * [`Palette`]: channel colour presets, including colour-blind-safe ones.
//!
//! [`WsiFile::read_region`]: crate::WsiFile::read_region
//! [`WsiFile::set_channel_display`]: crate::WsiFile::set_channel_display

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub use crate::ome_tiff::tiff::SampleType;

/// Percentile of a channel's samples that auto contrast maps to black.
pub const AUTO_CONTRAST_LOW: f64 = 0.1;
/// Percentile that auto contrast maps to the full channel colour.
pub const AUTO_CONTRAST_HIGH: f64 = 99.9;

/// Channels shown when a slide opens without saved settings; composites of
/// more channels are hard to read (and slower to draw).
pub const DEFAULT_VISIBLE_CHANNELS: usize = 6;

/// Bins of [`ChannelStats::histogram`].
pub const HISTOGRAM_BINS: usize = 256;

/// Accepted range of [`ChannelSettings::gamma`].
pub const GAMMA_RANGE: (f64, f64) = (0.1, 10.0);

/// Statistics of a channel's samples, from a low-resolution level.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChannelStats {
    /// Samples counted (0 when the level could not be read).
    pub samples: u64,
    pub min: f64,
    pub max: f64,
    /// The [`AUTO_CONTRAST_LOW`] percentile (zeros, which mark areas
    /// without data in stitched scans, are left out).
    pub low: f64,
    /// The [`AUTO_CONTRAST_HIGH`] percentile.
    pub high: f64,
    /// Counts of [`HISTOGRAM_BINS`] equal bins over `min..=max`.
    pub histogram: Vec<u32>,
}

impl ChannelStats {
    /// Statistics of exact counts per value: `counts[i]` samples have the
    /// value `offset + i` (8- and 16-bit types).
    pub(crate) fn from_counts(counts: &[u64], offset: f64, unsigned: bool) -> Self {
        let samples: u64 = counts.iter().sum();
        let Some(first) = counts.iter().position(|&n| n > 0) else {
            return Self::default();
        };
        let last = counts.iter().rposition(|&n| n > 0).unwrap_or(first);
        // Zeros of unsigned data mark "no data" in stitched scans: leave
        // them out of the percentiles when anything else exists.
        let skip_zero = unsigned && first == 0 && last > 0;
        let counted: u64 = samples - if skip_zero { counts[0] } else { 0 };
        let percentile = |p: f64| -> f64 {
            let target = ((p / 100.0) * counted as f64).ceil().max(1.0) as u64;
            let mut seen = 0u64;
            for (i, &n) in counts.iter().enumerate() {
                if skip_zero && i == 0 {
                    continue;
                }
                seen += n;
                if seen >= target {
                    return offset + i as f64;
                }
            }
            offset + last as f64
        };
        let (min, max) = (offset + first as f64, offset + last as f64);
        let mut histogram = vec![0u32; HISTOGRAM_BINS];
        let span = (max - min).max(1.0);
        for (i, &n) in counts.iter().enumerate().take(last + 1).skip(first) {
            let bin = (((i - first) as f64 / span) * (HISTOGRAM_BINS - 1) as f64).round() as usize;
            let bin = bin.min(HISTOGRAM_BINS - 1);
            histogram[bin] = histogram[bin].saturating_add(u32::try_from(n).unwrap_or(u32::MAX));
        }
        Self {
            samples,
            min,
            max,
            low: percentile(AUTO_CONTRAST_LOW),
            high: percentile(AUTO_CONTRAST_HIGH),
            histogram,
        }
    }

    /// Statistics of arbitrary values (32-bit and floating-point types).
    pub(crate) fn from_values(values: &mut [f64]) -> Self {
        values.sort_unstable_by(f64::total_cmp);
        let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        let (Some(&min), Some(&max)) = (finite.first(), finite.last()) else {
            return Self::default();
        };
        let nonzero: Vec<f64> = finite.iter().copied().filter(|&v| v != 0.0).collect();
        let ranked = if min >= 0.0 && !nonzero.is_empty() {
            &nonzero
        } else {
            &finite
        };
        let percentile = |p: f64| -> f64 {
            let rank = ((p / 100.0) * ranked.len() as f64).ceil().max(1.0) as usize;
            ranked[rank.min(ranked.len()) - 1]
        };
        let mut histogram = vec![0u32; HISTOGRAM_BINS];
        let span = max - min;
        for &v in &finite {
            let bin = if span > 0.0 {
                (((v - min) / span) * (HISTOGRAM_BINS - 1) as f64).round() as usize
            } else {
                0
            };
            let bin = bin.min(HISTOGRAM_BINS - 1);
            histogram[bin] = histogram[bin].saturating_add(1);
        }
        Self {
            samples: finite.len() as u64,
            min,
            max,
            low: percentile(AUTO_CONTRAST_LOW),
            high: percentile(AUTO_CONTRAST_HIGH),
            histogram,
        }
    }

    /// The auto-contrast window: the percentiles, widened when they meet.
    pub fn auto_window(&self, sample_type: SampleType) -> (f64, f64) {
        if self.samples == 0 {
            return sample_type.range();
        }
        let (low, high) = (self.low, self.high.max(self.low));
        if high > low {
            return (low, high);
        }
        // A flat channel: put its value mid-window.
        let step = if sample_type.is_float() {
            (low.abs() * 0.01).max(1e-6)
        } else {
            1.0
        };
        (low - step, high + step)
    }
}

/// What a slide says about one channel.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelInfo {
    pub index: usize,
    /// The channel's name (`Channel@Name`), or "Channel N".
    pub name: String,
    /// The colour the file gives it (`Channel@Color`), if any.
    pub color: Option<[u8; 3]>,
    /// Fluorophore (`Channel@Fluor`).
    pub fluor: Option<String>,
    pub emission_nm: Option<f64>,
    pub excitation_nm: Option<f64>,
    pub stats: ChannelStats,
}

/// The channels of a multichannel slide.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelSet {
    pub channels: Vec<ChannelInfo>,
    pub sample_type: SampleType,
    /// Bits actually used (`Pixels@SignificantBits`, e.g. 12 of 16).
    pub significant_bits: Option<u32>,
    /// Focal planes and time points in the file; eov shows the first.
    pub size_z: u64,
    pub size_t: u64,
}

impl ChannelSet {
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    /// The values a display window can span: the sample type's range,
    /// narrowed to the significant bits; for floats, the values seen.
    pub fn value_range(&self) -> (f64, f64) {
        if self.sample_type.is_float() {
            let mut range: Option<(f64, f64)> = None;
            for stats in self
                .channels
                .iter()
                .map(|c| &c.stats)
                .filter(|s| s.samples > 0)
            {
                range = Some(match range {
                    Some((lo, hi)) => (lo.min(stats.min), hi.max(stats.max)),
                    None => (stats.min, stats.max),
                });
            }
            return match range {
                Some((lo, hi)) if hi > lo => (lo, hi),
                Some((lo, _)) => (lo - 0.5, lo + 0.5),
                None => (0.0, 1.0),
            };
        }
        let (lo, hi) = self.sample_type.range();
        match self.significant_bits {
            Some(bits) if (1..=31).contains(&bits) && lo == 0.0 => {
                (0.0, hi.min(f64::from((1u32 << bits) - 1)))
            }
            _ => (lo, hi),
        }
    }

    /// The display a slide opens with: colours from the file (when they
    /// tell the channels apart), else from the channel names and emission
    /// wavelengths, else from the standard palette; auto-contrast windows;
    /// the first [`DEFAULT_VISIBLE_CHANNELS`] channels shown.
    pub fn default_display(&self) -> ChannelDisplay {
        let colors = default_colors(&self.channels);
        let channels = self
            .channels
            .iter()
            .zip(colors)
            .map(|(info, color)| {
                let (min, max) = info.stats.auto_window(self.sample_type);
                ChannelSettings {
                    visible: info.index < DEFAULT_VISIBLE_CHANNELS,
                    color,
                    min,
                    max,
                    gamma: 1.0,
                }
            })
            .collect();
        ChannelDisplay { channels }
    }
}

/// How one channel is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChannelSettings {
    pub visible: bool,
    /// The colour of the channel at full intensity.
    pub color: [u8; 3],
    /// Display window in sample values: `min` and below are black, `max`
    /// and above the full colour.
    pub min: f64,
    pub max: f64,
    /// Intensity inside the window is raised to `1 / gamma`, so values
    /// above 1 brighten dim signal (as the HUD's gamma does).
    pub gamma: f64,
}

impl ChannelSettings {
    /// The settings with a valid window and gamma.
    fn sanitized(mut self) -> Self {
        if !self.min.is_finite() {
            self.min = 0.0;
        }
        if !self.max.is_finite() || self.max <= self.min {
            self.max = self.min + 1.0;
        }
        self.gamma = if self.gamma.is_finite() {
            self.gamma.clamp(GAMMA_RANGE.0, GAMMA_RANGE.1)
        } else {
            1.0
        };
        self
    }

    /// Display intensity (0..=1) of a sample value.
    pub fn intensity(&self, value: f64) -> f64 {
        let t = ((value - self.min) / (self.max - self.min)).clamp(0.0, 1.0);
        if (self.gamma - 1.0).abs() < 1e-9 || t == 0.0 || t == 1.0 {
            t
        } else {
            t.powf(1.0 / self.gamma)
        }
    }
}

/// How every channel of a slide is drawn, in channel order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelDisplay {
    pub channels: Vec<ChannelSettings>,
}

impl ChannelDisplay {
    /// The display checked against `set`: one entry per channel, with valid
    /// windows and gammas.
    pub fn validated(self, set: &ChannelSet) -> Result<Self> {
        if self.channels.len() != set.len() {
            return Err(Error::InvalidArgument(format!(
                "display settings for {} channels, the slide has {}",
                self.channels.len(),
                set.len()
            )));
        }
        Ok(Self {
            channels: self
                .channels
                .into_iter()
                .map(ChannelSettings::sanitized)
                .collect(),
        })
    }

    /// Sets every window to the channel's auto-contrast window.
    pub fn auto_contrast(&mut self, set: &ChannelSet) {
        for (settings, info) in self.channels.iter_mut().zip(&set.channels) {
            let (min, max) = info.stats.auto_window(set.sample_type);
            settings.min = min;
            settings.max = max;
        }
    }

    /// Recolours every channel from `palette`, in channel order.
    pub fn apply_palette(&mut self, palette: Palette) {
        for (index, settings) in self.channels.iter_mut().enumerate() {
            settings.color = palette.color(index);
        }
    }

    pub fn visible_count(&self) -> usize {
        self.channels.iter().filter(|c| c.visible).count()
    }
}

/// Channel colour presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Palette {
    /// The usual fluorescence colours: blue, green, red, then cyan,
    /// magenta, yellow, orange and white.
    Standard,
    /// Distinguishable with red-green colour blindness: magenta and green
    /// instead of red and green, with blue, cyan and yellow.
    ColorBlindSafe,
    /// The Okabe–Ito palette (Okabe and Ito, 2008), designed to be told
    /// apart with every common form of colour blindness.
    OkabeIto,
}

impl Palette {
    pub const ALL: [Palette; 3] = [
        Palette::Standard,
        Palette::ColorBlindSafe,
        Palette::OkabeIto,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::ColorBlindSafe => "Colour-blind safe",
            Self::OkabeIto => "Okabe–Ito",
        }
    }

    /// A stable identifier, for settings files and the UI.
    pub fn id(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::ColorBlindSafe => "colour-blind-safe",
            Self::OkabeIto => "okabe-ito",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.id() == id)
    }

    pub fn colors(self) -> &'static [[u8; 3]] {
        match self {
            Self::Standard => &[
                [0, 0, 255],
                [0, 255, 0],
                [255, 0, 0],
                [0, 255, 255],
                [255, 0, 255],
                [255, 255, 0],
                [255, 128, 0],
                [255, 255, 255],
            ],
            Self::ColorBlindSafe => &[
                [0, 114, 255],
                [0, 255, 0],
                [255, 0, 255],
                [0, 255, 255],
                [255, 255, 0],
                [255, 255, 255],
            ],
            Self::OkabeIto => &[
                [0, 114, 178],
                [0, 158, 115],
                [213, 94, 0],
                [86, 180, 233],
                [204, 121, 167],
                [240, 228, 66],
                [230, 159, 0],
                [255, 255, 255],
            ],
        }
    }

    /// The colour of channel `index` (palettes repeat).
    pub fn color(self, index: usize) -> [u8; 3] {
        let colors = self.colors();
        colors[index % colors.len()]
    }
}

/// Whether a channel name denotes a DNA counterstain (DAPI, Hoechst…),
/// conventionally shown in blue.
pub fn is_nuclear_stain(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    [
        "dapi", "hoechst", "dna", "nuclei", "nuclear", "draq5", "sytox",
    ]
    .iter()
    .any(|word| lower.contains(word))
}

/// The colour light of `nm` nanometres appears in, saturated, for channels
/// that only record their emission wavelength.
pub fn color_for_wavelength(nm: f64) -> [u8; 3] {
    match nm {
        x if x < 490.0 => [0, 0, 255],
        x if x < 510.0 => [0, 255, 255],
        x if x < 560.0 => [0, 255, 0],
        x if x < 590.0 => [255, 255, 0],
        x if x < 630.0 => [255, 128, 0],
        x if x < 700.0 => [255, 0, 0],
        _ => [255, 0, 255],
    }
}

/// Default colours: the file's, if they tell the channels apart;
/// otherwise blue for DNA stains, the emission colour when every other
/// channel records a distinct wavelength, and the standard palette.
fn default_colors(channels: &[ChannelInfo]) -> Vec<[u8; 3]> {
    let from_file: Vec<Option<[u8; 3]>> = channels
        .iter()
        .map(|c| c.color.filter(|rgb| *rgb != [0, 0, 0]))
        .collect();
    let distinct = from_file
        .iter()
        .flatten()
        .collect::<std::collections::HashSet<_>>()
        .len();
    if channels.len() == 1 {
        return vec![from_file[0].unwrap_or([255, 255, 255])];
    }
    if from_file.iter().all(Option::is_some) && distinct == channels.len() {
        return from_file.into_iter().flatten().collect();
    }
    let wavelengths: Vec<Option<f64>> = channels.iter().map(|c| c.emission_nm).collect();
    let others: Vec<f64> = channels
        .iter()
        .zip(&wavelengths)
        .filter(|(c, _)| !is_nuclear_stain(&c.name))
        .filter_map(|(_, nm)| *nm)
        .collect();
    let use_wavelengths = !others.is_empty()
        && others.len()
            == channels
                .iter()
                .filter(|c| !is_nuclear_stain(&c.name))
                .count()
        && {
            let colors: std::collections::HashSet<[u8; 3]> =
                others.iter().map(|&nm| color_for_wavelength(nm)).collect();
            colors.len() == others.len()
        };
    // Blue goes to the DNA stain; other channels skip it.
    let any_nuclear = channels.iter().any(|c| is_nuclear_stain(&c.name));
    let palette: Vec<[u8; 3]> = Palette::Standard
        .colors()
        .iter()
        .copied()
        .filter(|rgb| !(any_nuclear && *rgb == [0, 0, 255]))
        .collect();
    let mut next = 0;
    channels
        .iter()
        .zip(wavelengths)
        .map(|(channel, nm)| {
            if is_nuclear_stain(&channel.name) {
                [0, 0, 255]
            } else if let (true, Some(nm)) = (use_wavelengths, nm) {
                color_for_wavelength(nm)
            } else {
                let color = palette[next % palette.len()];
                next += 1;
                color
            }
        })
        .collect()
}

/// Colour contributions of one channel, by sample value.
pub(crate) enum ChannelLut {
    /// Indexed by `value - offset` (8- and 16-bit samples).
    Table { offset: i64, table: Vec<[u8; 3]> },
    /// [`WINDOW_STEPS`] steps across the window, for 32-bit and float
    /// samples; values outside it clamp.
    Window {
        min: f64,
        steps_per_unit: f64,
        table: Vec<[u8; 3]>,
    },
}

/// Resolution of [`ChannelLut::Window`].
const WINDOW_STEPS: usize = 4096;

impl ChannelLut {
    fn contribution(settings: &ChannelSettings, value: f64) -> [u8; 3] {
        let intensity = settings.intensity(value);
        settings
            .color
            .map(|c| (f64::from(c) * intensity).round().clamp(0.0, 255.0) as u8)
    }

    pub(crate) fn new(settings: &ChannelSettings, sample_type: SampleType) -> Self {
        let table_for = |offset: i64, len: usize| {
            let mut table = vec![[0u8; 3]; len];
            for (i, entry) in table.iter_mut().enumerate() {
                let value = (offset + i as i64) as f64;
                // Outside the window the result is constant: skip powf.
                *entry = if value <= settings.min {
                    [0, 0, 0]
                } else if value >= settings.max {
                    settings.color
                } else {
                    Self::contribution(settings, value)
                };
            }
            ChannelLut::Table { offset, table }
        };
        match sample_type {
            SampleType::U8 => table_for(0, 256),
            SampleType::I8 => table_for(-128, 256),
            SampleType::U16 => table_for(0, 65_536),
            SampleType::I16 => table_for(-32_768, 65_536),
            _ => {
                let span = settings.max - settings.min;
                let table = (0..WINDOW_STEPS)
                    .map(|i| {
                        let value = settings.min + span * (i as f64 / (WINDOW_STEPS - 1) as f64);
                        Self::contribution(settings, value)
                    })
                    .collect();
                ChannelLut::Window {
                    min: settings.min,
                    steps_per_unit: (WINDOW_STEPS - 1) as f64 / span,
                    table,
                }
            }
        }
    }

    #[inline(always)]
    pub(crate) fn lookup_index(&self, index: i64) -> [u8; 3] {
        match self {
            ChannelLut::Table { offset, table } => {
                table[((index - offset).max(0) as usize).min(table.len() - 1)]
            }
            ChannelLut::Window { .. } => self.lookup_value(index as f64),
        }
    }

    #[inline(always)]
    pub(crate) fn lookup_value(&self, value: f64) -> [u8; 3] {
        match self {
            ChannelLut::Table { offset, table } => {
                let index = (value.round() as i64 - offset).clamp(0, table.len() as i64 - 1);
                table[index as usize]
            }
            ChannelLut::Window {
                min,
                steps_per_unit,
                table,
            } => {
                let step = ((value - min) * steps_per_unit).round();
                // NaN maps to the first entry.
                let step = if step.is_nan() { 0.0 } else { step };
                table[(step.max(0.0) as usize).min(table.len() - 1)]
            }
        }
    }
}

/// A display with its lookup tables, as reads use it.
pub(crate) struct Compositor {
    pub display: ChannelDisplay,
    /// One per channel; `None` for hidden channels.
    pub luts: Vec<Option<Arc<ChannelLut>>>,
}

impl Compositor {
    /// Builds the tables of `display`, reusing those of `previous` for
    /// channels whose settings did not change (a slider moves one channel).
    pub(crate) fn new(
        display: ChannelDisplay,
        sample_type: SampleType,
        previous: Option<&Compositor>,
    ) -> Self {
        let luts = display
            .channels
            .iter()
            .enumerate()
            .map(|(index, settings)| {
                if !settings.visible {
                    return None;
                }
                if let Some(previous) = previous
                    && previous.display.channels.get(index) == Some(settings)
                    && let Some(Some(lut)) = previous.luts.get(index)
                {
                    return Some(Arc::clone(lut));
                }
                Some(Arc::new(ChannelLut::new(settings, sample_type)))
            })
            .collect();
        Self { display, luts }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(
        index: usize,
        name: &str,
        color: Option<[u8; 3]>,
        emission: Option<f64>,
    ) -> ChannelInfo {
        ChannelInfo {
            index,
            name: name.to_string(),
            color,
            fluor: None,
            emission_nm: emission,
            excitation_nm: None,
            stats: ChannelStats::default(),
        }
    }

    #[test]
    fn exact_histograms_give_percentiles_without_the_zeros() {
        // 1000 samples: 500 zeros (no data), then 1..=500.
        let mut counts = vec![0u64; 65_536];
        counts[0] = 500;
        counts[1..=500].fill(1);
        let stats = ChannelStats::from_counts(&counts, 0.0, true);
        assert_eq!(stats.samples, 1000);
        assert_eq!((stats.min, stats.max), (0.0, 500.0));
        assert_eq!(stats.low, 1.0, "zeros are left out");
        assert_eq!(stats.high, 500.0);
        assert_eq!(stats.histogram.len(), HISTOGRAM_BINS);
        assert_eq!(
            stats.histogram.iter().map(|&n| u64::from(n)).sum::<u64>(),
            1000
        );
        assert_eq!(stats.auto_window(SampleType::U16), (1.0, 500.0));

        // Signed data keeps its zeros.
        let mut counts = vec![0u64; 256];
        counts[128] = 10; // value 0 of int8
        counts[200] = 10;
        let stats = ChannelStats::from_counts(&counts, -128.0, false);
        assert_eq!((stats.min, stats.max), (0.0, 72.0));
        assert_eq!(stats.low, 0.0);

        // Flat and empty channels get a usable window.
        let mut counts = vec![0u64; 256];
        counts[7] = 99;
        let flat = ChannelStats::from_counts(&counts, 0.0, true);
        assert_eq!(flat.auto_window(SampleType::U8), (6.0, 8.0));
        assert_eq!(
            ChannelStats::default().auto_window(SampleType::U16),
            (0.0, 65_535.0)
        );
    }

    #[test]
    fn float_statistics_ignore_non_finite_values() {
        let mut values: Vec<f64> = (0..1000).map(|i| f64::from(i) / 1000.0).collect();
        values.push(f64::NAN);
        values.push(f64::INFINITY);
        let stats = ChannelStats::from_values(&mut values);
        assert_eq!(stats.samples, 1000);
        assert_eq!(stats.min, 0.0);
        assert!((stats.max - 0.999).abs() < 1e-12);
        assert!(stats.high > 0.99 && stats.high <= 0.999);
        assert!(stats.low > 0.0 && stats.low < 0.01);
    }

    #[test]
    fn default_colours_come_from_the_file_names_or_the_palette() {
        let file = [
            info(0, "a", Some([255, 0, 0]), None),
            info(1, "b", Some([0, 255, 0]), None),
        ];
        assert_eq!(default_colors(&file), [[255, 0, 0], [0, 255, 0]]);
        // All white (a common writer default) says nothing.
        let white = [
            info(0, "DAPI", Some([255; 3]), None),
            info(1, "CD45", Some([255; 3]), None),
            info(2, "CD8", Some([255; 3]), None),
        ];
        assert_eq!(
            default_colors(&white),
            [[0, 0, 255], [0, 255, 0], [255, 0, 0]],
            "DNA is blue; the rest skip blue"
        );
        let waves = [
            info(0, "Hoechst", None, Some(461.0)),
            info(1, "x", None, Some(520.0)),
            info(2, "y", None, Some(670.0)),
        ];
        assert_eq!(
            default_colors(&waves),
            [[0, 0, 255], [0, 255, 0], [255, 0, 0]]
        );
        assert_eq!(default_colors(&[info(0, "only", None, None)]), [[255; 3]]);
    }

    #[test]
    fn luts_map_windows_colours_and_gamma() {
        let settings = ChannelSettings {
            visible: true,
            color: [255, 128, 0],
            min: 100.0,
            max: 200.0,
            gamma: 1.0,
        };
        let lut = ChannelLut::new(&settings, SampleType::U16);
        assert_eq!(lut.lookup_index(0), [0, 0, 0]);
        assert_eq!(lut.lookup_index(100), [0, 0, 0]);
        assert_eq!(lut.lookup_index(150), [128, 64, 0]);
        assert_eq!(lut.lookup_index(200), [255, 128, 0]);
        assert_eq!(lut.lookup_index(65_535), [255, 128, 0]);
        let bright = ChannelLut::new(
            &ChannelSettings {
                gamma: 2.0,
                ..settings
            },
            SampleType::U16,
        );
        // 0.5^(1/2) = 0.707.
        assert_eq!(bright.lookup_index(150), [180, 91, 0]);

        let float = ChannelLut::new(
            &ChannelSettings {
                min: 0.0,
                max: 1.0,
                ..settings
            },
            SampleType::F32,
        );
        assert_eq!(float.lookup_value(-5.0), [0, 0, 0]);
        assert_eq!(float.lookup_value(0.5), [128, 64, 0]);
        assert_eq!(float.lookup_value(9.0), [255, 128, 0]);
        assert_eq!(float.lookup_value(f64::NAN), [0, 0, 0]);
        let signed = ChannelLut::new(
            &ChannelSettings {
                min: -100.0,
                max: 100.0,
                ..settings
            },
            SampleType::I16,
        );
        assert_eq!(signed.lookup_index(-32_768), [0, 0, 0]);
        assert_eq!(signed.lookup_index(0), [128, 64, 0]);
    }

    #[test]
    fn displays_are_validated_and_tables_reused() {
        let set = ChannelSet {
            channels: vec![info(0, "a", None, None), info(1, "b", None, None)],
            sample_type: SampleType::U8,
            significant_bits: None,
            size_z: 1,
            size_t: 1,
        };
        let mut display = set.default_display();
        assert_eq!(display.visible_count(), 2);
        assert!(ChannelDisplay { channels: vec![] }.validated(&set).is_err());
        display.channels[0].max = f64::NAN;
        display.channels[1].gamma = 1000.0;
        let display = display.validated(&set).unwrap();
        assert!(display.channels[0].max > display.channels[0].min);
        assert_eq!(display.channels[1].gamma, GAMMA_RANGE.1);

        let first = Compositor::new(display.clone(), set.sample_type, None);
        let mut changed = display.clone();
        changed.channels[1].color = [1, 2, 3];
        changed.channels[0].visible = true;
        let second = Compositor::new(changed, set.sample_type, Some(&first));
        assert!(Arc::ptr_eq(
            first.luts[0].as_ref().unwrap(),
            second.luts[0].as_ref().unwrap()
        ));
        assert!(!Arc::ptr_eq(
            first.luts[1].as_ref().unwrap(),
            second.luts[1].as_ref().unwrap()
        ));
        let mut hidden = display;
        hidden.channels[0].visible = false;
        assert!(Compositor::new(hidden, set.sample_type, None).luts[0].is_none());
    }

    #[test]
    fn palettes_and_ranges() {
        for palette in Palette::ALL {
            assert_eq!(Palette::from_id(palette.id()), Some(palette));
            assert!(!palette.colors().is_empty());
            assert_eq!(palette.color(palette.colors().len()), palette.color(0));
        }
        let mut set = ChannelSet {
            channels: vec![],
            sample_type: SampleType::U16,
            significant_bits: Some(12),
            size_z: 1,
            size_t: 1,
        };
        assert_eq!(set.value_range(), (0.0, 4095.0));
        set.significant_bits = None;
        assert_eq!(set.value_range(), (0.0, 65_535.0));
        assert!(is_nuclear_stain("DAPI (DNA)"));
        assert!(is_nuclear_stain("Hoechst 33342"));
        assert!(!is_nuclear_stain("CD45"));
    }
}

//! The Channels panel (AUDIT §14.3): how each channel of a multichannel
//! slide (fluorescence, multiplexed immunofluorescence) is shown —
//! visibility, colour, display window and gamma — with auto contrast,
//! palettes (colour-blind-safe ones too) and one channel alone.
//!
//! Every change applies to the slide at once
//! ([`WsiFile::set_channel_display`]): later reads, by the viewer, exports
//! and plugins alike, composite with it. Tiles already decoded show the
//! old display, and the GPU keeps tiles by coordinate, so changed pixels
//! need new coordinates: the tiles on screen are rebuilt under a new tile
//! file id in the background (decoding nothing, the reader caches raw
//! channel samples) and swapped in when ready, so the view never goes
//! blank. One rebuild runs per slide at a time; changes made meanwhile are
//! picked up by the next. The minimap is redrawn the same way. Settings
//! are saved per slide identity ([`crate::channel_store`]).

use crate::host_ui::UiHost;
use crate::state::PaneId;
use crate::tile_loader::TileLoader;
use crate::{AppWindow, ChannelRow};
use eov_common::channels::{
    ChannelDisplay, ChannelSet, ChannelSettings, ChannelStats, GAMMA_RANGE, Palette, SampleType,
};
use eov_common::{TileCoord, TileManager, WsiFile};
use rayon::prelude::*;
use slint::{Color, Image, Model, ModelRc, Rgba8Pixel, SharedPixelBuffer, Timer, VecModel};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

/// Tiles rebuilt under the new display before it is shown: the most
/// recently used ones, which are the last frames' (the renderer touches
/// what it draws). The loader fetches anything else after the switch.
const REBUILT_TILES: usize = 128;

/// Quiet time after the last change before the settings are saved.
const SAVE_DELAY: Duration = Duration::from_millis(600);

/// Range of the gamma slider (logarithmic, 1 in the middle).
const GAMMA_SLIDER: (f64, f64) = (0.2, 5.0);

/// Size of the histogram drawn behind each channel's window slider.
pub(crate) const HISTOGRAM_SIZE: (usize, usize) = (240, 26);

/// What the panel shows of one channel.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RowView {
    pub name: String,
    pub detail: String,
    pub color: [u8; 3],
    pub visible: bool,
    pub solo: bool,
    pub min_text: String,
    pub max_text: String,
    pub gamma_text: String,
    /// Window ends as positions (0..=1) on the slider.
    pub low: f64,
    pub high: f64,
    pub gamma_position: f64,
}

/// The panel's state for one slide, without any UI: what the edits do.
#[derive(Debug, Clone)]
pub(crate) struct PanelState {
    set: ChannelSet,
    defaults: ChannelDisplay,
    display: ChannelDisplay,
    /// The channel shown alone, and the visibility to restore after.
    solo: Option<(usize, Vec<bool>)>,
}

/// One edit from the panel.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Edit {
    Visible(usize, bool),
    Solo(usize),
    ShowAll,
    Color(usize, [u8; 3]),
    /// A window end moved on the slider: `high` or not, to a position.
    Window {
        index: usize,
        high: bool,
        position: f64,
    },
    Min(usize, String),
    Max(usize, String),
    /// The gamma slider, at a position.
    GammaPosition(usize, f64),
    Gamma(usize, String),
    AutoContrast,
    Reset,
    Palette(Palette),
}

fn parse_number(text: &str) -> Option<f64> {
    text.trim()
        .replace(',', ".")
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
}

fn format_value(value: f64, sample_type: SampleType) -> String {
    if sample_type.is_float() {
        eov_common::format_significant(value, 4)
    } else {
        format!("{}", value.round() as i64)
    }
}

fn gamma_from_position(position: f64) -> f64 {
    let (lo, hi) = GAMMA_SLIDER;
    lo * (hi / lo).powf(position.clamp(0.0, 1.0))
}

fn gamma_position(gamma: f64) -> f64 {
    let (lo, hi) = GAMMA_SLIDER;
    ((gamma / lo).ln() / (hi / lo).ln()).clamp(0.0, 1.0)
}

impl PanelState {
    pub(crate) fn new(set: ChannelSet, defaults: ChannelDisplay, display: ChannelDisplay) -> Self {
        Self {
            set,
            defaults,
            display,
            solo: None,
        }
    }

    pub(crate) fn display(&self) -> &ChannelDisplay {
        &self.display
    }

    pub(crate) fn len(&self) -> usize {
        self.display.channels.len()
    }

    /// The smallest step between window ends.
    fn step(&self) -> f64 {
        if self.set.sample_type.is_float() {
            let (lo, hi) = self.set.value_range();
            ((hi - lo) / 10_000.0).max(f64::EPSILON)
        } else {
            1.0
        }
    }

    /// The values the window slider of channel `index` spans: from 0 (or
    /// the lowest value seen) to twice the auto-contrast maximum, so hot
    /// pixels do not squeeze the useful range into a corner; always
    /// including the current window. Typed values may go beyond.
    pub(crate) fn slider_range(&self, index: usize) -> (f64, f64) {
        let stats = &self.set.channels[index].stats;
        let settings = &self.display.channels[index];
        let (type_lo, type_hi) = self.set.value_range();
        let (mut lo, mut hi) = if stats.samples == 0 {
            (type_lo, type_hi)
        } else {
            let lo = if type_lo >= 0.0 { 0.0 } else { stats.min };
            let hi = (stats.high * 2.0 - lo)
                .max(stats.high)
                .min(stats.max.max(stats.high));
            (lo, hi)
        };
        lo = lo.min(settings.min);
        hi = hi.max(settings.max);
        if hi <= lo {
            hi = lo + self.step();
        }
        (lo, hi)
    }

    fn position(&self, index: usize, value: f64) -> f64 {
        let (lo, hi) = self.slider_range(index);
        ((value - lo) / (hi - lo)).clamp(0.0, 1.0)
    }

    fn set_min(&mut self, index: usize, value: f64) {
        let step = self.step();
        let settings = &mut self.display.channels[index];
        settings.min = value;
        if settings.max <= value {
            settings.max = value + step;
        }
    }

    fn set_max(&mut self, index: usize, value: f64) {
        let step = self.step();
        let settings = &mut self.display.channels[index];
        settings.max = value;
        if settings.min >= value {
            settings.min = value - step;
        }
    }

    /// Applies an edit; returns whether the display changed.
    pub(crate) fn apply(&mut self, edit: Edit) -> bool {
        let before = self.display.clone();
        let in_range = |index: usize| index < self.display.channels.len();
        match edit {
            Edit::Visible(index, visible) if in_range(index) => {
                self.solo = None;
                self.display.channels[index].visible = visible;
            }
            Edit::Solo(index) if in_range(index) => match self.solo.take() {
                // Clicking the solo channel again restores the others.
                Some((current, before)) if current == index => {
                    for (settings, visible) in self.display.channels.iter_mut().zip(before) {
                        settings.visible = visible;
                    }
                }
                previous => {
                    // Moving from one solo channel to another keeps the
                    // visibility from before the first.
                    let before = previous.map(|(_, before)| before).unwrap_or_else(|| {
                        self.display.channels.iter().map(|c| c.visible).collect()
                    });
                    for (i, settings) in self.display.channels.iter_mut().enumerate() {
                        settings.visible = i == index;
                    }
                    self.solo = Some((index, before));
                }
            },
            Edit::ShowAll => {
                self.solo = None;
                for settings in &mut self.display.channels {
                    settings.visible = true;
                }
            }
            Edit::Color(index, color) if in_range(index) => {
                self.display.channels[index].color = color;
            }
            Edit::Window {
                index,
                high,
                position,
            } if in_range(index) => {
                let (lo, hi) = self.slider_range(index);
                let mut value = lo + position.clamp(0.0, 1.0) * (hi - lo);
                if !self.set.sample_type.is_float() {
                    value = value.round();
                }
                if high {
                    self.set_max(index, value);
                } else {
                    self.set_min(index, value);
                }
            }
            Edit::Min(index, text) if in_range(index) => match parse_number(&text) {
                Some(value) => self.set_min(index, value),
                None => return false,
            },
            Edit::Max(index, text) if in_range(index) => match parse_number(&text) {
                Some(value) => self.set_max(index, value),
                None => return false,
            },
            Edit::GammaPosition(index, position) if in_range(index) => {
                let gamma = gamma_from_position(position);
                // Snap to 1 near the middle, where most users want it.
                self.display.channels[index].gamma = if (gamma - 1.0).abs() < 0.04 {
                    1.0
                } else {
                    (gamma * 100.0).round() / 100.0
                };
            }
            Edit::Gamma(index, text) if in_range(index) => match parse_number(&text) {
                Some(gamma) if gamma > 0.0 => {
                    self.display.channels[index].gamma = gamma.clamp(GAMMA_RANGE.0, GAMMA_RANGE.1);
                }
                _ => return false,
            },
            Edit::AutoContrast => self.display.auto_contrast(&self.set),
            Edit::Reset => {
                self.solo = None;
                self.display = self.defaults.clone();
            }
            Edit::Palette(palette) => self.display.apply_palette(palette),
            _ => return false,
        }
        self.display != before
    }

    pub(crate) fn rows(&self) -> Vec<RowView> {
        let sample_type = self.set.sample_type;
        self.display
            .channels
            .iter()
            .zip(&self.set.channels)
            .enumerate()
            .map(|(index, (settings, info))| {
                let mut detail = Vec::new();
                if let Some(fluor) = &info.fluor
                    && fluor != &info.name
                {
                    detail.push(fluor.clone());
                }
                if let Some(nm) = info.emission_nm {
                    detail.push(format!("{} nm", nm.round()));
                }
                RowView {
                    name: info.name.clone(),
                    detail: detail.join(" · "),
                    color: settings.color,
                    visible: settings.visible,
                    solo: self.solo.as_ref().is_some_and(|(solo, _)| *solo == index),
                    min_text: format_value(settings.min, sample_type),
                    max_text: format_value(settings.max, sample_type),
                    gamma_text: format!("{:.2}", settings.gamma),
                    low: self.position(index, settings.min),
                    high: self.position(index, settings.max),
                    gamma_position: gamma_position(settings.gamma),
                }
            })
            .collect()
    }

    /// One line about the channels: how many are shown, the sample type,
    /// and the planes not shown.
    pub(crate) fn summary(&self) -> String {
        let mut parts = vec![format!(
            "{} of {} channels shown",
            self.display.visible_count(),
            self.len()
        )];
        parts.push(self.set.sample_type.name().to_string());
        if let Some(bits) = self.set.significant_bits {
            parts.push(format!("{bits} significant bits"));
        }
        if self.set.size_z > 1 {
            parts.push(format!("focal plane 1 of {}", self.set.size_z));
        }
        if self.set.size_t > 1 {
            parts.push(format!("time point 1 of {}", self.set.size_t));
        }
        parts.join(" · ")
    }

    /// The histogram behind channel `index`'s slider, as RGBA pixels of
    /// [`HISTOGRAM_SIZE`].
    pub(crate) fn histogram_pixels(&self, index: usize) -> Vec<u8> {
        histogram_pixels(
            &self.set.channels[index].stats,
            self.slider_range(index),
            &self.display.channels[index],
        )
    }
}

/// A channel's histogram over `range`: log-scaled bars, in the channel's
/// colour inside its window and grey outside, over a faint window band.
pub(crate) fn histogram_pixels(
    stats: &ChannelStats,
    (lo, hi): (f64, f64),
    settings: &ChannelSettings,
) -> Vec<u8> {
    let (width, height) = HISTOGRAM_SIZE;
    let mut columns = vec![0u64; width];
    let bins = stats.histogram.len();
    if stats.samples > 0 && bins > 0 && hi > lo {
        for (i, &count) in stats.histogram.iter().enumerate() {
            let value = if bins > 1 {
                stats.min + (stats.max - stats.min) * i as f64 / (bins - 1) as f64
            } else {
                stats.min
            };
            let position = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
            let x = ((position * (width - 1) as f64).round() as usize).min(width - 1);
            columns[x] += u64::from(count);
        }
    }
    let peak = columns.iter().copied().max().unwrap_or(0);
    let scale = if peak > 0 {
        (peak as f64 + 1.0).ln()
    } else {
        1.0
    };
    let position = |value: f64| ((value - lo) / (hi - lo)).clamp(0.0, 1.0) * (width - 1) as f64;
    let (window_x0, window_x1) = (position(settings.min), position(settings.max));
    let [r, g, b] = settings.color;
    let mut pixels = vec![0u8; width * height * 4];
    for (x, &count) in columns.iter().enumerate() {
        let inside = (x as f64) >= window_x0.floor() && (x as f64) <= window_x1.ceil();
        let bar = if count == 0 {
            0
        } else {
            (((count as f64 + 1.0).ln() / scale) * (height - 2) as f64).ceil() as usize + 1
        };
        for y in 0..height {
            let at = (y * width + x) * 4;
            let filled = height - y <= bar;
            let px: [u8; 4] = match (filled, inside) {
                (true, true) => [r.max(60), g.max(60), b.max(60), 230],
                (true, false) => [150, 150, 150, 160],
                (false, true) => [r, g, b, 40],
                (false, false) => [0, 0, 0, 0],
            };
            pixels[at..at + 4].copy_from_slice(&px);
        }
    }
    pixels
}

fn histogram_image(pixels: &[u8]) -> Image {
    let (width, height) = HISTOGRAM_SIZE;
    let buffer =
        SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(pixels, width as u32, height as u32);
    Image::from_rgba8(buffer)
}

fn slint_color([r, g, b]: [u8; 3]) -> Color {
    Color::from_rgb_u8(r, g, b)
}

fn slint_row(index: usize, row: &RowView, histogram: Image) -> ChannelRow {
    ChannelRow {
        index: index as i32,
        name: row.name.clone().into(),
        detail: row.detail.clone().into(),
        color: slint_color(row.color),
        visible: row.visible,
        solo: row.solo,
        min_text: row.min_text.clone().into(),
        max_text: row.max_text.clone().into(),
        gamma_text: row.gamma_text.clone().into(),
        low: row.low as f32,
        high: row.high as f32,
        gamma_position: row.gamma_position as f32,
        histogram,
    }
}

/// The colours offered for a channel: every palette's, once each.
pub(crate) fn chip_colors() -> Vec<[u8; 3]> {
    let mut colors: Vec<[u8; 3]> = Vec::new();
    for palette in Palette::ALL {
        for &color in palette.colors() {
            if !colors.contains(&color) {
                colors.push(color);
            }
        }
    }
    colors
}

/// The panel open for one slide.
struct Open {
    file_id: i32,
    wsi: WsiFile,
    file_name: String,
    state: PanelState,
    rows: Rc<VecModel<ChannelRow>>,
}

/// One slide's tile rebuilds: whether one runs, and whether the display
/// changed again since it started.
#[derive(Default)]
struct Rebuild {
    running: bool,
    again: bool,
}

thread_local! {
    static PANEL: RefCell<Option<Open>> = const { RefCell::new(None) };
    static REBUILDS: RefCell<HashMap<i32, Rebuild>> = RefCell::new(HashMap::new());
    static SAVE_TIMER: Timer = Timer::default();
}

/// Applies the display saved for a slide when it opens (worker thread),
/// unless the slide is open already and its display was changed in this
/// session (the handle is shared, and its display is newer).
pub(crate) fn restore_saved_display(wsi: &WsiFile) {
    let Some(set) = wsi.channels() else {
        return;
    };
    if wsi.channel_display() != wsi.default_channel_display() {
        return;
    }
    let identity = match wsi.identity() {
        Ok(identity) => identity,
        Err(error) => {
            tracing::debug!("No slide identity, so no saved channel display: {error}");
            return;
        }
    };
    if let Some(saved) = crate::channel_store::load(identity.key(), set)
        && let Err(error) = wsi.set_channel_display(saved)
    {
        tracing::warn!("Could not apply the saved channel display: {error}");
    }
}

/// Opens the panel for the slide shown in the focused pane.
pub(crate) fn open_for_focused_pane(host: &UiHost) {
    let file_id = {
        let state = host.state.read();
        state.active_file_id_for_pane(state.active_viewport_pane())
    };
    match file_id {
        Some(file_id) => open(host, file_id),
        None => host.status("Open a slide to adjust its channels"),
    }
}

/// Opens the panel for slide `file_id`.
pub(crate) fn open(host: &UiHost, file_id: i32) {
    let Some(ui) = host.ui() else {
        return;
    };
    let (wsi, file_name) = {
        let state = host.state.read();
        let Some(file) = state.get_file(file_id) else {
            return;
        };
        (file.wsi.clone(), file.filename.clone())
    };
    let (Some(set), Some(display), Some(defaults)) = (
        wsi.channels().cloned(),
        wsi.channel_display(),
        wsi.default_channel_display(),
    ) else {
        host.status(
            "This slide has no channels to adjust: it is shown as its colours were scanned",
        );
        return;
    };
    let state = PanelState::new(set, defaults, display);
    let rows: Vec<ChannelRow> = state
        .rows()
        .iter()
        .enumerate()
        .map(|(index, row)| slint_row(index, row, histogram_image(&state.histogram_pixels(index))))
        .collect();
    let rows = Rc::new(VecModel::from(rows));
    ui.set_channels_title(format!("Channels · {file_name}").into());
    ui.set_channels_summary(state.summary().into());
    ui.set_channel_rows(ModelRc::from(Rc::clone(&rows)));
    let chips: Vec<Color> = chip_colors().into_iter().map(slint_color).collect();
    ui.set_channel_chip_colors(ModelRc::from(Rc::new(VecModel::from(chips))));
    PANEL.with(|panel| {
        *panel.borrow_mut() = Some(Open {
            file_id,
            wsi,
            file_name,
            state,
            rows,
        });
    });
    ui.set_slide_info_visible(false);
    ui.set_channels_panel_visible(true);
}

/// Applies an edit from the panel to its slide.
pub(crate) fn edit(host: &UiHost, edit: Edit) {
    let Some(ui) = host.ui() else {
        return;
    };
    let closed = PANEL.with(|panel| {
        panel
            .borrow()
            .as_ref()
            .is_none_or(|open| host.state.read().get_file(open.file_id).is_none())
    });
    if closed {
        // The slide was closed under the panel.
        PANEL.with(|panel| panel.borrow_mut().take());
        ui.set_channels_panel_visible(false);
        return;
    }
    let bulk = matches!(
        edit,
        Edit::Solo(_) | Edit::ShowAll | Edit::AutoContrast | Edit::Reset | Edit::Palette(_)
    );
    let index = match &edit {
        Edit::Visible(i, _)
        | Edit::Color(i, _)
        | Edit::Min(i, _)
        | Edit::Max(i, _)
        | Edit::GammaPosition(i, _)
        | Edit::Gamma(i, _) => Some(*i),
        Edit::Window { index, .. } => Some(*index),
        _ => None,
    };
    let Some((file_id, wsi, display)) = PANEL.with(|panel| {
        let mut panel = panel.borrow_mut();
        let open = panel.as_mut()?;
        let changed = open.state.apply(edit);
        // Refresh the rows that changed (in place: a slider being dragged
        // keeps its pointer grab), even when the edit was refused, so a
        // typed value that did not parse snaps back.
        let rows = open.state.rows();
        let update = |i: usize| {
            let histogram = histogram_image(&open.state.histogram_pixels(i));
            open.rows.set_row_data(i, slint_row(i, &rows[i], histogram));
        };
        match index {
            Some(i) if !bulk && i < rows.len() => update(i),
            _ => (0..rows.len()).for_each(update),
        }
        changed.then(|| (open.file_id, open.wsi.clone(), open.state.display().clone()))
    }) else {
        return;
    };
    ui.set_channels_summary(
        PANEL
            .with(|panel| panel.borrow().as_ref().map(|open| open.state.summary()))
            .unwrap_or_default()
            .into(),
    );
    if let Err(error) = wsi.set_channel_display(display) {
        host.status(&format!("Could not change the channels: {error}"));
        return;
    }
    // Every open file reading through this slide's handle shows the new
    // display (normally just the panel's own).
    let sharing: Vec<i32> = host
        .state
        .read()
        .open_files
        .iter()
        .filter(|file| file.id == file_id || file.wsi.shares_handle_with(&wsi))
        .map(|file| file.id)
        .collect();
    for id in sharing {
        schedule_rebuild(host, id);
    }
    schedule_save();
}

/// Saves the open panel's display after a quiet moment.
fn schedule_save() {
    let Some((wsi, name, display, defaults)) = PANEL.with(|panel| {
        panel.borrow().as_ref().map(|open| {
            (
                open.wsi.clone(),
                open.file_name.clone(),
                open.state.display().clone(),
                open.state.defaults.clone(),
            )
        })
    }) else {
        return;
    };
    SAVE_TIMER.with(|timer| {
        timer.start(slint::TimerMode::SingleShot, SAVE_DELAY, move || {
            let (wsi, name, display, defaults) =
                (wsi.clone(), name.clone(), display.clone(), defaults.clone());
            // The identity is usually known by now; computing it reads
            // the file, so not on the UI thread.
            let _ = std::thread::Builder::new()
                .name("eov-channels-save".into())
                .spawn(move || {
                    let Some(set) = wsi.channels() else {
                        return;
                    };
                    let result = wsi.identity().map_err(|e| e.to_string()).and_then(|id| {
                        if display == defaults {
                            crate::channel_store::forget(id.key())
                        } else {
                            crate::channel_store::save(id.key(), &name, &display, set)
                        }
                    });
                    if let Err(error) = result {
                        tracing::warn!("Could not save the channel display of {name}: {error}");
                    }
                });
        });
    });
}

/// Rebuilds the tiles of slide `file_id` with its current display, unless a
/// rebuild runs already (it is then repeated once it finishes).
fn schedule_rebuild(host: &UiHost, file_id: i32) {
    let start = REBUILDS.with(|rebuilds| {
        let mut rebuilds = rebuilds.borrow_mut();
        let rebuild = rebuilds.entry(file_id).or_default();
        if rebuild.running {
            rebuild.again = true;
            false
        } else {
            rebuild.running = true;
            true
        }
    });
    if start {
        start_rebuild(host, file_id);
    }
}

/// What a rebuild works on, taken on the UI thread: a tile manager with a
/// new tile file id, the tiles to build with it, and the slide.
struct RebuildJob {
    manager: Arc<TileManager>,
    coords: Vec<TileCoord>,
    wsi: WsiFile,
}

fn prepare_rebuild(host: &UiHost, file_id: i32) -> Option<RebuildJob> {
    let mut state = host.state.write();
    let tile_id = state.allocate_file_id();
    let file = state.get_file(file_id)?;
    let old = Arc::clone(&file.tile_manager);
    let manager = Arc::new(
        TileManager::new(old.wsi().clone(), tile_id).with_tile_cache(Arc::clone(&host.tile_cache)),
    );
    let coords = host
        .tile_cache
        .recent_coords(old.file_id(), REBUILT_TILES)
        .into_iter()
        .map(|coord| TileCoord {
            file_id: tile_id,
            ..coord
        })
        .collect();
    Some(RebuildJob {
        manager,
        coords,
        wsi: file.wsi.clone(),
    })
}

/// Builds the job's tiles (into the cache, under the new id) and the
/// minimap (worker thread).
fn run_rebuild(job: &RebuildJob) -> Option<crate::state::MinimapThumbnail> {
    let started = std::time::Instant::now();
    job.coords.par_iter().for_each(|coord| {
        if let Err(error) = job.manager.load_tile_cached(*coord) {
            tracing::trace!("Tile {coord:?} not rebuilt: {error}");
        }
    });
    let thumbnail = crate::file_ops::minimap_thumbnail(&job.wsi);
    tracing::debug!(
        "Rebuilt {} tiles and the minimap for a channel display in {:?}",
        job.coords.len(),
        started.elapsed()
    );
    thumbnail
}

fn start_rebuild(host: &UiHost, file_id: i32) {
    let Some(job) = prepare_rebuild(host, file_id) else {
        rebuild_done(host, file_id);
        return;
    };
    let spawned = std::thread::Builder::new()
        .name("eov-channels".into())
        .spawn(move || {
            let thumbnail = run_rebuild(&job);
            let manager = job.manager;
            let _ = slint::invoke_from_event_loop(move || {
                crate::host_ui::with_host(|host| finish_rebuild(host, file_id, manager, thumbnail));
            });
        });
    if let Err(error) = spawned {
        tracing::warn!("Could not rebuild the tiles: {error}");
        rebuild_done(host, file_id);
    }
}

/// Swaps the rebuilt tiles in (UI thread).
fn finish_rebuild(
    host: &UiHost,
    file_id: i32,
    manager: Arc<TileManager>,
    thumbnail: Option<crate::state::MinimapThumbnail>,
) {
    let panes = {
        let mut state = host.state.write();
        let panes: Vec<PaneId> = (0..state.panes.len())
            .map(PaneId)
            .filter(|&pane| state.active_file_id_for_pane(pane) == Some(file_id))
            .collect();
        match state.get_file_mut(file_id) {
            Some(file) => {
                manager.pin_fallback_levels();
                let loader = Arc::new(TileLoader::new(
                    Arc::clone(&manager),
                    Arc::clone(&host.tile_cache),
                ));
                let old_manager = std::mem::replace(&mut file.tile_manager, manager);
                let old_loader = std::mem::replace(&mut file.tile_loader, loader);
                // Retire first: then nothing re-caches the old tiles.
                old_manager.retire();
                old_loader.retire();
                if thumbnail.is_some() {
                    file.thumbnail = thumbnail;
                }
                file.invalidate_render_state();
                state.request_render();
                Some(panes)
            }
            None => {
                // Closed meanwhile.
                manager.retire();
                host.tile_cache.remove_file(manager.file_id());
                None
            }
        }
    };
    for pane in panes.unwrap_or_default() {
        // The render loop draws the new thumbnail into a pane without one.
        crate::set_cached_pane_minimap(pane, None);
    }
    host.request_render();
    rebuild_done(host, file_id);
}

fn rebuild_done(host: &UiHost, file_id: i32) {
    let again = REBUILDS.with(|rebuilds| {
        let mut rebuilds = rebuilds.borrow_mut();
        match rebuilds.get_mut(&file_id) {
            Some(rebuild) if rebuild.again => {
                rebuild.again = false;
                true
            }
            _ => {
                rebuilds.remove(&file_id);
                false
            }
        }
    });
    if again {
        start_rebuild(host, file_id);
    }
}

/// The panel was closed.
pub(crate) fn closed(ui: &AppWindow) {
    ui.set_channels_panel_visible(false);
    PANEL.with(|panel| panel.borrow_mut().take());
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_common::channels::{ChannelInfo, ChannelStats};

    fn stats(min: f64, max: f64, low: f64, high: f64) -> ChannelStats {
        ChannelStats {
            samples: 1000,
            min,
            max,
            low,
            high,
            histogram: vec![10; 256],
        }
    }

    fn panel() -> PanelState {
        let channels = ["DAPI", "CD45", "Pan-CK"]
            .iter()
            .enumerate()
            .map(|(index, name)| ChannelInfo {
                index,
                name: name.to_string(),
                color: None,
                fluor: (index == 1).then(|| "AF488".to_string()),
                emission_nm: (index == 1).then_some(520.0),
                excitation_nm: None,
                stats: stats(0.0, 65_535.0, 100.0, 2000.0),
            })
            .collect();
        let set = ChannelSet {
            channels,
            sample_type: SampleType::U16,
            significant_bits: None,
            size_z: 3,
            size_t: 1,
        };
        let defaults = set.default_display();
        PanelState::new(set, defaults.clone(), defaults)
    }

    #[test]
    fn rows_show_names_windows_and_slider_positions() {
        let state = panel();
        let rows = state.rows();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].name, "DAPI");
        assert_eq!(rows[0].color, [0, 0, 255]);
        assert_eq!(rows[1].detail, "AF488 · 520 nm");
        assert_eq!(
            (rows[0].min_text.as_str(), rows[0].max_text.as_str()),
            ("100", "2000")
        );
        assert_eq!(rows[0].gamma_text, "1.00");
        // The slider spans 0..4000 (twice the auto maximum), not the
        // 65535 the hot pixels reach.
        assert_eq!(state.slider_range(0), (0.0, 4000.0));
        assert!((rows[0].low - 0.025).abs() < 1e-9);
        assert!((rows[0].high - 0.5).abs() < 1e-9);
        assert!((rows[0].gamma_position - 0.5).abs() < 1e-9);
        assert_eq!(
            state.summary(),
            "3 of 3 channels shown · uint16 · focal plane 1 of 3"
        );
    }

    #[test]
    fn window_edits_keep_min_below_max() {
        let mut state = panel();
        assert!(state.apply(Edit::Window {
            index: 0,
            high: true,
            position: 0.25,
        }));
        assert_eq!(state.display().channels[0].max, 1000.0);
        // Dragging min past max pushes max along.
        assert!(state.apply(Edit::Window {
            index: 0,
            high: false,
            position: 0.5,
        }));
        let channel = state.display().channels[0];
        assert_eq!((channel.min, channel.max), (2000.0, 2001.0));
        // Typed values, with a decimal comma; nonsense is ignored.
        assert!(state.apply(Edit::Max(0, " 30000,0 ".into())));
        assert_eq!(state.display().channels[0].max, 30_000.0);
        assert!(!state.apply(Edit::Min(0, "abc".into())));
        assert!(state.apply(Edit::Min(0, "40000".into())));
        let channel = state.display().channels[0];
        assert_eq!((channel.min, channel.max), (40_000.0, 40_001.0));
        // The slider grows to include the window.
        assert!(state.slider_range(0).1 >= 40_001.0);
        assert!(!state.apply(Edit::Max(7, "1".into())), "no such channel");
    }

    #[test]
    fn solo_shows_one_channel_and_restores_the_others() {
        let mut state = panel();
        state.apply(Edit::Visible(2, false));
        assert!(state.apply(Edit::Solo(1)));
        let visible: Vec<bool> = state.display().channels.iter().map(|c| c.visible).collect();
        assert_eq!(visible, [false, true, false]);
        assert!(state.rows()[1].solo);
        // Another channel alone; then the same one again restores the
        // visibility from before the first solo.
        state.apply(Edit::Solo(0));
        assert_eq!(
            state
                .display()
                .channels
                .iter()
                .map(|c| c.visible)
                .collect::<Vec<_>>(),
            [true, false, false]
        );
        state.apply(Edit::Solo(0));
        assert_eq!(
            state
                .display()
                .channels
                .iter()
                .map(|c| c.visible)
                .collect::<Vec<_>>(),
            [true, true, false]
        );
        assert!(!state.rows().iter().any(|row| row.solo));
        state.apply(Edit::ShowAll);
        assert_eq!(state.display().visible_count(), 3);
        assert!(state.summary().starts_with("3 of 3"));
    }

    #[test]
    fn colours_palettes_gamma_auto_contrast_and_reset() {
        let mut state = panel();
        let defaults = state.display().clone();
        assert!(state.apply(Edit::Color(1, [255, 0, 255])));
        assert_eq!(state.rows()[1].color, [255, 0, 255]);
        assert!(state.apply(Edit::Palette(Palette::OkabeIto)));
        assert_eq!(
            state.display().channels[2].color,
            Palette::OkabeIto.color(2)
        );
        assert!(state.apply(Edit::GammaPosition(0, 1.0)));
        assert_eq!(state.display().channels[0].gamma, 5.0);
        assert!(state.apply(Edit::GammaPosition(0, 0.51)));
        assert_eq!(state.display().channels[0].gamma, 1.0, "snaps to 1");
        assert!(state.apply(Edit::Gamma(0, "0.5".into())));
        assert_eq!(state.rows()[0].gamma_text, "0.50");
        assert!(!state.apply(Edit::Gamma(0, "-1".into())));
        state.apply(Edit::Min(0, "7".into()));
        assert!(state.apply(Edit::AutoContrast));
        assert_eq!(state.display().channels[0].min, 100.0);
        assert!(state.apply(Edit::Reset));
        assert_eq!(state.display(), &defaults);
        assert!(!state.apply(Edit::Reset), "nothing left to reset");
    }

    /// The panel in the real (headless) window: its callbacks change the
    /// slide, its rows follow, and a rebuild swaps in tiles with the new
    /// display under a new tile id.
    #[test]
    fn the_panel_drives_the_slide_and_swaps_rebuilt_tiles_in() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let setup = crate::test_support::HostWithSlide::new();
            let (ui, host, state) = (&setup.ui, &setup.host, &setup.state);
            let (tile_cache, file_id, wsi) = (&setup.tile_cache, setup.file_id, &setup.wsi);
            let old = Arc::clone(&state.read().get_file(file_id).unwrap().tile_manager);
            for coord in old.visible_tiles(0, 0.0, 0.0, 512.0, 384.0) {
                old.load_tile_cached(coord).unwrap();
            }

            open(host, file_id);
            assert!(ui.get_channels_panel_visible());
            assert_eq!(ui.get_channel_rows().row_count(), 3);
            assert_eq!(ui.get_channel_rows().row_data(1).unwrap().name, "CD45");

            // The window's callbacks reach the slide and the rows.
            ui.invoke_channels_solo_toggled(1);
            let visible: Vec<bool> = wsi
                .channel_display()
                .unwrap()
                .channels
                .iter()
                .map(|c| c.visible)
                .collect();
            assert_eq!(visible, [false, true, false]);
            assert!(ui.get_channel_rows().row_data(1).unwrap().solo);
            assert!(ui.get_channels_summary().starts_with("1 of 3"));
            ui.invoke_channels_color_chosen(1, Color::from_rgb_u8(255, 0, 255));
            assert_eq!(
                wsi.channel_display().unwrap().channels[1].color,
                [255, 0, 255]
            );
            let max_before = wsi.channel_display().unwrap().channels[1].max;
            ui.invoke_channels_max_entered(1, "nonsense".into());
            assert_eq!(wsi.channel_display().unwrap().channels[1].max, max_before);
            ui.invoke_channels_max_entered(1, "300".into());
            assert_eq!(wsi.channel_display().unwrap().channels[1].max, 300.0);
            assert_eq!(ui.get_channel_rows().row_data(1).unwrap().max_text, "300");

            // A rebuild, run here rather than on its worker.
            let job = prepare_rebuild(host, file_id).unwrap();
            assert!(!job.coords.is_empty());
            let thumbnail = run_rebuild(&job);
            assert!(thumbnail.is_some());
            let new_id = job.manager.file_id();
            assert_ne!(new_id, old.file_id());
            finish_rebuild(host, file_id, job.manager, thumbnail);
            {
                let state = state.read();
                let file = state.get_file(file_id).unwrap();
                assert_eq!(file.tile_manager.file_id(), new_id);
                assert!(file.thumbnail.is_some());
            }
            assert!(old.is_retired());
            assert!(tile_cache.recent_coords(old.file_id(), 1000).is_empty());
            let rebuilt = tile_cache.recent_coords(new_id, 1000);
            assert!(!rebuilt.is_empty());
            // They show the new display: CD45 alone, in magenta.
            let tile = tile_cache.peek(&rebuilt[0]).unwrap();
            assert!(
                tile.data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|p| p[0] == p[2] && p[1] == 0 && p[3] == 255)
            );
            assert!(tile.data.as_chunks::<4>().0.iter().any(|p| p[0] > 0));

            closed(ui);
            assert!(!ui.get_channels_panel_visible());
        });
    }

    #[test]
    fn saved_displays_come_back_when_a_slide_opens() {
        use eov_common::test_support::{OmeTiffSpec, write_ome_tiff};
        crate::paths::isolate_for_tests();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("saved.ome.tif");
        let spec = OmeTiffSpec::channels(64, 64, 32, 1, &[("a", [255, 0, 0]), ("b", [0, 255, 0])]);
        write_ome_tiff(&path, &spec, |c, x, _| f64::from(x) + c as f64).unwrap();
        let first = WsiFile::open_exclusive(&path).unwrap();
        let mut display = first.channel_display().unwrap();
        display.channels[0].visible = false;
        display.channels[1].color = [1, 2, 3];
        let key = first.identity().unwrap().key().to_string();
        crate::channel_store::save(&key, "saved.ome.tif", &display, first.channels().unwrap())
            .unwrap();

        let reopened = WsiFile::open_exclusive(&path).unwrap();
        assert_ne!(reopened.channel_display(), Some(display.clone()));
        restore_saved_display(&reopened);
        assert_eq!(reopened.channel_display(), Some(display));
        // A display already changed in this session wins over the file.
        let mut newer = reopened.channel_display().unwrap();
        newer.channels[1].color = [9, 9, 9];
        reopened.set_channel_display(newer.clone()).unwrap();
        restore_saved_display(&reopened);
        assert_eq!(reopened.channel_display(), Some(newer));
        crate::channel_store::forget(&key).unwrap();
    }

    #[test]
    fn histograms_mark_the_window_in_the_channel_colour() {
        let settings = ChannelSettings {
            visible: true,
            color: [255, 0, 0],
            min: 1000.0,
            max: 2000.0,
            gamma: 1.0,
        };
        let pixels = histogram_pixels(&stats(0.0, 4000.0, 0.0, 4000.0), (0.0, 4000.0), &settings);
        let (width, height) = HISTOGRAM_SIZE;
        assert_eq!(pixels.len(), width * height * 4);
        let at = |x: usize, y: usize| &pixels[(y * width + x) * 4..(y * width + x) * 4 + 4];
        // Bottom row: bars everywhere, red inside the window, grey outside.
        assert_eq!(at(width / 2 - 20, height - 1)[0], 255);
        assert_eq!(at(5, height - 1), [150, 150, 150, 160]);
        // An empty channel draws nothing but the window band.
        let empty = histogram_pixels(&ChannelStats::default(), (0.0, 4000.0), &settings);
        assert!(empty.as_chunks::<4>().0.iter().all(|p| p[3] <= 40));
        assert!(chip_colors().contains(&[0, 255, 0]));
        assert!((gamma_from_position(gamma_position(2.0)) - 2.0).abs() < 1e-9);
    }
}

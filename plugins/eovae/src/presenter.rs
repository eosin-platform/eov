//! The presenter thread: everything the user sees, pushed at a bounded
//! rate.
//!
//! Workers and event handlers only set dirty flags; this thread wakes at
//! most a few times per second, compares with what it last sent and pushes
//! the differences: sidebar properties, overlay layers and rasters, HUD
//! button states, the status-bar text and the saved settings (VAE-13). It
//! never holds the plugin's locks while calling the host.
//!
//! # Overlay layers
//!
//! Every pane chooses its own overlay of the slide it shows. A slide has an
//! error-map layer (the heat map, as chunk rasters so that new results
//! re-send only the chunks they touch), a tile-grid layer and a highlight
//! layer; every pane showing a reconstruction or difference has its own
//! mosaic layer. Layers are hidden in the panes that do not show them
//! ([`Layer::set_visible_in_pane`]). Reconstructions get their pane's
//! display adjustments, so they compare with the slide as the pane shows it
//! (VAE-17).

use crate::engine::Backend;
use crate::plugin::{
    Core, DirtyFlags, HUD_GRID, HUD_VIEW, PaneOverlay, RunContext, grid_layer, heat_layer, lock,
    mark_layer, mosaic_layer, run_context, view_menu,
};
use crate::render::{self, DisplayTone, HeatLayout, HeatMap, Mode, MosaicFrame, MosaicTile};
use crate::results::{Counts, ResultKey, TileState};
use crate::stats::{self, Bin, Summary};
use crate::tiling::{TileCoord, TileGrid};
use eov_plugin_api::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// Shortest time between two refreshes.
const MIN_INTERVAL: Duration = Duration::from_millis(200);
/// Settings are saved this long after the last change.
const SAVE_DELAY: Duration = Duration::from_millis(700);
/// How long a region stays highlighted after a jump.
pub const PULSE: Duration = Duration::from_millis(1600);
/// While a job writes a slide's results, its rasters are refreshed at most
/// this often; the final state follows the job's end.
const RUNNING_UPLOAD_INTERVAL: Duration = Duration::from_secs(1);
/// How often display adjustments are re-read (the host sends no event).
const DISPLAY_CHECK: Duration = Duration::from_secs(1);
/// Side of a heat-map chunk, in heat-map pixels.
pub const HEAT_CHUNK: u32 = 64;

/// Statistics of one run, recomputed when its results change.
#[derive(Clone, Debug, Default)]
pub struct RunStats {
    pub revision: u64,
    pub counts: Counts,
    pub summary: Option<Summary>,
    pub bins: Vec<Bin>,
    pub top: Vec<(f32, TileCoord)>,
}

impl RunStats {
    /// The colour range of the error map: p05–p95 of the analysed tiles.
    pub fn heat_range(&self) -> (f32, f32) {
        match &self.summary {
            Some(s) if s.count >= 2 && s.p95 > s.p05 => (s.p05 as f32, s.p95 as f32),
            Some(s) => (0.0, (s.max as f32).max(1e-6)),
            None => (0.0, 1.0),
        }
    }

    /// The per-pixel error at the top of the Difference colour ramp.
    pub fn pixel_range(&self) -> f32 {
        self.summary
            .map(|s| (s.p95 as f32 * 4.0).clamp(0.01, 0.5))
            .unwrap_or(0.1)
    }
}

/// The heat map as last sent.
struct HeatShown {
    run: Arc<str>,
    revision: u64,
    range: (f32, f32),
    layout: HeatLayout,
    /// Content hash of every chunk object on the layer.
    chunks: HashMap<(u32, u32), u64>,
}

/// A pane's mosaic as last sent.
struct MosaicShown {
    run: Arc<str>,
    revision: u64,
    mode: Mode,
    frame: MosaicFrame,
    tone: Option<DisplayTone>,
}

#[derive(Default)]
struct FileOverlay {
    /// The error-map and highlight layers exist.
    layers: bool,
    heat: Option<HeatShown>,
    heat_uploaded: Option<Instant>,
    /// Panes with a mosaic layer, and what it shows.
    mosaics: HashMap<PaneId, Option<MosaicShown>>,
    mosaic_uploaded: Option<Instant>,
    grid_layer: bool,
    /// The lines on the grid layer.
    grid: Option<String>,
    mark: Option<TileCoord>,
    /// The panes each layer is hidden in, as told to the host.
    hidden: HashMap<String, BTreeSet<PaneId>>,
}

#[derive(Default)]
pub struct Memo {
    sidebar_id: Option<String>,
    pub sidebar_sent: Map<String, Value>,
    stats: HashMap<ResultKey, Arc<RunStats>>,
    files: HashMap<FileId, FileOverlay>,
    hud: HashMap<PaneId, PaneOverlay>,
    status: Option<String>,
    /// The panes of the last snapshot.
    panes: BTreeSet<PaneId>,
    /// Each pane's display adjustments, as last read.
    displays: HashMap<PaneId, DisplayAdjustments>,
    /// The host does not report display adjustments.
    pub displays_unsupported: bool,
    displays_checked: Option<Instant>,
}

impl Memo {
    /// Statistics of a run, cached by its revision.
    pub fn stats<B: Backend>(&mut self, core: &Core<B>, key: &ResultKey) -> Option<Arc<RunStats>> {
        let (revision, scores, counts) = {
            let results = lock(&core.results);
            let run = results.run(key)?;
            if let Some(cached) = self.stats.get(key)
                && cached.revision == run.revision
            {
                return Some(Arc::clone(cached));
            }
            (run.revision, results.scores(key, None), run.counts())
        };
        let mut values: Vec<f32> = scores.iter().map(|(mae, _)| *mae).collect();
        let summary = stats::summarize(&mut values);
        let bins = stats::histogram(&values, 12);
        let top = stats::top_k(scores, 12);
        let computed = Arc::new(RunStats {
            revision,
            counts,
            summary,
            bins,
            top,
        });
        self.stats.insert(key.clone(), Arc::clone(&computed));
        Some(computed)
    }

    /// The error-map colour range drawn for a slide (the legend shows it).
    pub fn heat_range(&self, file_id: FileId) -> Option<(f32, f32)> {
        self.files.get(&file_id)?.heat.as_ref().map(|h| h.range)
    }

    /// How a pane displays its slide; `None` when the host cannot say.
    pub fn display<B: Backend>(
        &mut self,
        core: &Core<B>,
        pane: PaneId,
    ) -> Option<DisplayAdjustments> {
        if self.displays_unsupported {
            return None;
        }
        if let Some(display) = self.displays.get(&pane) {
            return Some(display.clone());
        }
        match core.host.display_adjustments(Some(pane)) {
            Ok(display) => {
                self.displays.insert(pane, display.clone());
                Some(display)
            }
            Err(error) => {
                if error.is_unsupported() {
                    self.displays_unsupported = true;
                }
                None
            }
        }
    }
}

pub fn run<B: Backend>(core: Arc<Core<B>>) {
    let mut memo = Memo::default();
    let mut last: Option<Instant> = None;
    loop {
        let mut flags = core.dirty.wait(Duration::from_millis(400));
        if core.stopping.load(Ordering::SeqCst) {
            break;
        }
        // Coalesce bursts.
        let since = last.map_or(MIN_INTERVAL, |at| at.elapsed());
        if since < MIN_INTERVAL {
            std::thread::sleep(MIN_INTERVAL - since);
            let more = core.dirty.wait(Duration::ZERO);
            flags.sidebar |= more.sidebar;
            flags.hud |= more.hud;
            flags.status |= more.status;
            flags.all_overlays |= more.all_overlays;
            flags.overlays.extend(more.overlays);
        }
        last = Some(Instant::now());

        let save = {
            let mut state = lock(&core.state);
            let save = state
                .config_changed
                .is_some_and(|at| at.elapsed() >= SAVE_DELAY);
            let pulse_over = state.pulse.is_some_and(|(_, _, at)| at.elapsed() >= PULSE);
            if pulse_over && let Some((file_id, _, _)) = state.pulse.take() {
                flags.overlays.insert(file_id);
            }
            save
        };
        if save {
            core.persist_config();
        }
        // Keep waking while something time-based is pending.
        let timed = {
            let state = lock(&core.state);
            state.config_changed.is_some() || state.pulse.is_some()
        };
        check_displays(&core, &mut memo, &mut flags);

        let snapshot = if flags.hud || flags.all_overlays || !flags.overlays.is_empty() {
            core.host.snapshot().ok()
        } else {
            None
        };
        // Panes that came or went: their settings, layers and HUD.
        if let Some(snapshot) = &snapshot {
            let live: BTreeSet<PaneId> = snapshot.panes.iter().map(|p| p.pane).collect();
            if live != memo.panes {
                core.retain_panes(&live);
                memo.displays.retain(|pane, _| live.contains(pane));
                memo.panes = live;
                flags.all_overlays = true;
                flags.hud = true;
            }
        }
        if flags.all_overlays || !flags.overlays.is_empty() {
            let files: Vec<FileId> = if flags.all_overlays {
                core.file_ids()
            } else {
                flags.overlays.iter().copied().collect()
            };
            let live: HashSet<FileId> = core.file_ids().into_iter().collect();
            memo.files.retain(|file_id, _| live.contains(file_id));
            for file_id in files {
                if live.contains(&file_id) {
                    refresh_file(&core, &mut memo, file_id, snapshot.as_ref());
                }
            }
        }
        if flags.hud {
            refresh_hud(&core, &mut memo, snapshot.as_ref());
        }
        if flags.status || flags.sidebar {
            refresh_status(&core, &mut memo);
        }
        if flags.sidebar {
            refresh_sidebar(&core, &mut memo);
        }
        // Statistics of runs that are gone (closed slides, a cleared cache).
        if !memo.stats.is_empty() {
            let results = lock(&core.results);
            memo.stats.retain(|key, _| results.holds(key));
        }
        if timed {
            core.dirty.mark(|_| {});
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

/// The host sends no event when a pane's display adjustments change: about
/// once a second, re-read them for the panes showing reconstructions and
/// for the focused pane (the sidebar's note).
fn check_displays<B: Backend>(core: &Core<B>, memo: &mut Memo, flags: &mut DirtyFlags) {
    if memo.displays_unsupported
        || memo
            .displays_checked
            .is_some_and(|at| at.elapsed() < DISPLAY_CHECK)
    {
        return;
    }
    memo.displays_checked = Some(Instant::now());
    let mut panes: Vec<(PaneId, Option<FileId>)> = memo
        .files
        .iter()
        .flat_map(|(file_id, overlay)| {
            overlay
                .mosaics
                .iter()
                .filter(|(_, shown)| {
                    shown
                        .as_ref()
                        .is_some_and(|s| s.mode == Mode::Reconstruction)
                })
                .map(move |(pane, _)| (*pane, Some(*file_id)))
        })
        .collect();
    let focused = {
        let state = lock(&core.state);
        state.sidebar.as_ref().and(state.focused_pane)
    };
    if let Some(pane) = focused
        && !panes.iter().any(|(p, _)| *p == pane)
    {
        panes.push((pane, None));
    }
    for (pane, file_id) in panes {
        match core.host.display_adjustments(Some(pane)) {
            Ok(display) => {
                if memo.displays.get(&pane) == Some(&display) {
                    continue;
                }
                memo.displays.insert(pane, display);
                if let Some(file_id) = file_id {
                    flags.overlays.insert(file_id);
                }
                flags.sidebar = true;
            }
            Err(error) if error.is_unsupported() => {
                memo.displays_unsupported = true;
                flags.sidebar = true;
                return;
            }
            Err(_) => {}
        }
    }
}

fn refresh_sidebar<B: Backend>(core: &Arc<Core<B>>, memo: &mut Memo) {
    let Some(surface) = lock(&core.state).sidebar.clone() else {
        memo.sidebar_id = None;
        return;
    };
    if memo.sidebar_id.as_deref() != Some(surface.id()) {
        memo.sidebar_id = Some(surface.id().to_string());
        memo.sidebar_sent.clear();
    }
    let properties = crate::sidebar::properties(core, memo);
    let changed: Map<String, Value> = properties
        .into_iter()
        .filter(|(name, value)| memo.sidebar_sent.get(name) != Some(value))
        .collect();
    if changed.is_empty() {
        return;
    }
    if surface.set_many(changed.clone()).is_ok() {
        memo.sidebar_sent.extend(changed);
    }
}

fn refresh_status<B: Backend>(core: &Arc<Core<B>>, memo: &mut Memo) {
    let text = {
        let state = lock(&core.state);
        match (&state.job, &state.engine.building) {
            (Some(job), _) if job.progress.total > 0 => Some(format!(
                "VAE {} / {} tiles",
                job.progress.done, job.progress.total
            )),
            (Some(job), _) => Some(format!("VAE: {}", job.phase)),
            (None, Some(building)) => Some(format!("VAE: {building}")),
            _ => None,
        }
    };
    if text != memo.status {
        core.host.set_status(text.as_deref());
        memo.status = text;
    }
}

/// Each pane's HUD buttons show that pane's overlay.
fn refresh_hud<B: Backend>(core: &Arc<Core<B>>, memo: &mut Memo, snapshot: Option<&HostSnapshot>) {
    let Some(snapshot) = snapshot else {
        return;
    };
    let desired: Vec<(PaneId, PaneOverlay)> = {
        let state = lock(&core.state);
        snapshot
            .panes
            .iter()
            .filter_map(|pane| {
                let file_id = pane.file_id.filter(|id| state.files.contains_key(id))?;
                Some((pane.pane, state.overlay(file_id, Some(pane.pane))))
            })
            .collect()
    };
    memo.hud
        .retain(|pane, _| snapshot.panes.iter().any(|p| p.pane == *pane));
    for (pane, overlay) in desired {
        if memo.hud.get(&pane) == Some(&overlay) {
            continue;
        }
        let a = core.host.update_hud_button(ButtonUpdate {
            button_id: HUD_GRID.into(),
            pane: Some(pane),
            active: Some(overlay.grid),
            ..Default::default()
        });
        let b = core.host.update_hud_button(ButtonUpdate {
            button_id: HUD_VIEW.into(),
            pane: Some(pane),
            active: Some(overlay.mode != Mode::Original),
            menu: Some(view_menu(overlay.mode)),
            ..Default::default()
        });
        if a.is_ok() && b.is_ok() {
            memo.hud.insert(pane, overlay);
        }
    }
}

/// A pane showing the slide, and what it shows.
struct Target {
    pane: PaneId,
    view: Option<ViewInfo>,
    overlay: PaneOverlay,
}

/// Shows a layer in the panes `shown` accepts and hides it in the other
/// `panes`. Returns false when the host cannot hide layers per pane.
fn show_only(
    host: &Host,
    hidden: &mut HashMap<String, BTreeSet<PaneId>>,
    layer: &str,
    panes: &[PaneId],
    shown: impl Fn(PaneId) -> bool,
) -> bool {
    let set = hidden.entry(layer.to_string()).or_default();
    for &pane in panes {
        let hide = !shown(pane);
        if set.contains(&pane) == hide {
            continue;
        }
        match host.layer(layer).set_visible_in_pane(pane, !hide) {
            Ok(()) => {
                if hide {
                    set.insert(pane);
                } else {
                    set.remove(&pane);
                }
            }
            Err(error) if error.is_unsupported() => return false,
            Err(_) => {}
        }
    }
    true
}

/// Brings one slide's layers up to date.
fn refresh_file<B: Backend>(
    core: &Arc<Core<B>>,
    memo: &mut Memo,
    file_id: FileId,
    snapshot: Option<&HostSnapshot>,
) {
    let (targets, ctx, mark, busy, per_pane) = {
        let state = lock(&core.state);
        let Some(file) = state.files.get(&file_id) else {
            return;
        };
        let per_pane = state.pane_layers;
        let busy = state
            .job
            .as_ref()
            .is_some_and(|j| j.request.file_id == file_id);
        let ctx = run_context(&state, file_id).ok();
        let mark = match (state.pulse, state.hover) {
            (Some((f, coord, _)), _) if f == file_id => Some(coord),
            (_, Some((f, coord))) if f == file_id => Some(coord),
            _ => None,
        };
        // The host's snapshot is authoritative about which panes exist; our
        // tracked panes fill in until it reports them.
        let panes: Vec<(PaneId, Option<ViewInfo>)> = match snapshot {
            Some(snapshot) => snapshot
                .panes
                .iter()
                .filter(|pane| pane.file_id == Some(file_id))
                .map(|pane| (pane.pane, pane.view))
                .collect(),
            None => state
                .panes
                .iter()
                .filter(|(_, pane)| pane.file_id == Some(file_id))
                .map(|(id, pane)| (*id, pane.view))
                .collect(),
        };
        let targets: Vec<Target> = panes
            .into_iter()
            .map(|(pane, view)| Target {
                pane,
                view: view.filter(|v| v.file_id == file_id),
                overlay: file.overlay(Some(pane), per_pane),
            })
            .collect();
        (targets, ctx, mark, busy, per_pane)
    };
    let host = &core.host;
    let overlay = memo.files.entry(file_id).or_default();
    if !overlay.layers {
        let heat = LayerSpec::new(heat_layer(file_id), file_id)
            .named("VAE error map")
            .at_z(-90);
        let marks = LayerSpec::new(mark_layer(file_id), file_id)
            .named("VAE highlight")
            .at_z(-55)
            .with_style(Style {
                stroke: Rgba(0xFF, 0xC1, 0x07, 0xFF),
                stroke_width: 2.5,
                fill: Rgba(0xFF, 0xC1, 0x07, 0x22),
                ..Style::default()
            });
        if host.create_layer(heat).is_err() || host.create_layer(marks).is_err() {
            // The file is not published yet; retry on the next change.
            return;
        }
        overlay.layers = true;
    }

    // Stored results of a slide with an overlay load from the disk cache
    // once (any pane, not only the active one).
    if targets.iter().any(|t| t.overlay.mode != Mode::Original)
        && let Some(ctx) = &ctx
    {
        core.ensure_loaded(ctx);
    }
    let stats = ctx.as_ref().and_then(|ctx| memo.stats(core, &ctx.key));
    // Reconstructions are shown as their pane shows the slide.
    let tones: HashMap<PaneId, Option<DisplayTone>> = targets
        .iter()
        .filter(|t| t.overlay.mode == Mode::Reconstruction)
        .map(|t| {
            let tone = memo
                .display(core, t.pane)
                .map(|d| DisplayTone::new(&d))
                .filter(|tone| !tone.is_identity());
            (t.pane, tone)
        })
        .collect();
    let overlay = memo.files.entry(file_id).or_default();
    let panes: Vec<PaneId> = targets.iter().map(|t| t.pane).collect();
    let mut separate = true;

    // The error map, shown in the panes that chose it.
    let heat = host.layer(heat_layer(file_id));
    let heat_panes: HashSet<PaneId> = targets
        .iter()
        .filter(|t| t.overlay.mode == Mode::ErrorMap)
        .map(|t| t.pane)
        .collect();
    match &ctx {
        Some(ctx) if !heat_panes.is_empty() => {
            if per_pane {
                separate &= show_only(
                    host,
                    &mut overlay.hidden,
                    &heat_layer(file_id),
                    &panes,
                    |pane| heat_panes.contains(&pane),
                );
            }
            update_heat(
                core,
                overlay,
                &heat,
                ctx,
                &stats.clone().unwrap_or_default(),
                busy,
            );
        }
        _ => {
            if overlay.heat.take().is_some() {
                let _ = heat.clear();
            }
        }
    }

    // Mosaics: one layer per pane, shown only there.
    let wanted: Vec<&Target> = targets
        .iter()
        .filter(|t| ctx.is_some() && t.overlay.mode.needs_pixels() && t.view.is_some())
        .collect();
    let gone: Vec<PaneId> = overlay
        .mosaics
        .keys()
        .copied()
        .filter(|pane| !wanted.iter().any(|t| t.pane == *pane))
        .collect();
    for pane in gone {
        let id = mosaic_layer(file_id, pane);
        let _ = host.layer(id.as_str()).remove_layer();
        overlay.mosaics.remove(&pane);
        overlay.hidden.remove(&id);
    }
    if let Some(ctx) = &ctx {
        let revision = stats.as_ref().map(|s| s.revision).unwrap_or(0);
        let error_hi = stats.as_ref().map(|s| s.pixel_range()).unwrap_or(0.1);
        let throttled = busy
            && overlay
                .mosaic_uploaded
                .is_some_and(|at| at.elapsed() < RUNNING_UPLOAD_INTERVAL);
        for target in wanted {
            let Some(view) = target.view else {
                continue;
            };
            let mode = target.overlay.mode;
            let id = mosaic_layer(file_id, target.pane);
            if let std::collections::hash_map::Entry::Vacant(entry) =
                overlay.mosaics.entry(target.pane)
            {
                let spec = LayerSpec::new(id.clone(), file_id)
                    .named("VAE overlay")
                    .at_z(-90);
                if host.create_layer(spec).is_err() {
                    continue;
                }
                entry.insert(None);
                overlay.hidden.remove(&id);
            }
            if per_pane {
                separate &= show_only(host, &mut overlay.hidden, &id, &panes, |pane| {
                    pane == target.pane
                });
            }
            let Some(frame) = MosaicFrame::for_view(&view, &ctx.grid) else {
                continue;
            };
            let tone = tones.get(&target.pane).copied().flatten();
            let current = overlay.mosaics.get(&target.pane).and_then(|s| s.as_ref());
            let same = current.is_some_and(|s| {
                *s.run == *ctx.key.run && s.mode == mode && s.tone == tone && s.frame.covers(&frame)
            });
            let fresh = same && current.is_some_and(|s| s.revision == revision);
            if fresh || (same && throttled) {
                continue;
            }
            let image = build_mosaic(core, ctx, &frame, mode, error_hi, tone.as_ref());
            if host
                .layer(id.as_str())
                .put_raster("mosaic", frame.rect, image, true)
                .is_ok()
            {
                overlay.mosaics.insert(
                    target.pane,
                    Some(MosaicShown {
                        run: Arc::clone(&ctx.key.run),
                        revision,
                        mode,
                        frame,
                        tone,
                    }),
                );
                overlay.mosaic_uploaded = Some(Instant::now());
            }
        }
    }

    // The tile grid: whole-slide polylines, hidden when too dense.
    let grid_panes: HashSet<PaneId> = targets
        .iter()
        .filter(|t| t.overlay.grid)
        .map(|t| t.pane)
        .collect();
    let grid_id = grid_layer(file_id);
    match ctx.as_ref().filter(|_| !grid_panes.is_empty()) {
        Some(ctx) => {
            let grid = ctx.grid;
            let descriptor = format!(
                "{}x{}@{:.6}x{:.6}",
                grid.cols,
                grid.rows,
                grid.step_x(),
                grid.step_y()
            );
            let min_zoom = Some(6.0 / grid.step_min());
            if !overlay.grid_layer {
                if host.create_layer(grid_spec(file_id, min_zoom)).is_ok() {
                    overlay.grid_layer = true;
                    overlay.grid = None;
                    overlay.hidden.remove(&grid_id);
                }
            } else if overlay
                .grid
                .as_deref()
                .is_some_and(|lines| lines != descriptor)
            {
                // Other lines (another level or tile size) get too dense at
                // another zoom.
                if let Err(error) = host.layer(grid_id.as_str()).set_min_zoom(min_zoom)
                    && error.is_unsupported()
                    && host.create_layer(grid_spec(file_id, min_zoom)).is_ok()
                {
                    // Older hosts: a new layer.
                    overlay.grid = None;
                    overlay.hidden.remove(&grid_id);
                }
            }
            if overlay.grid_layer {
                if per_pane {
                    separate &= show_only(host, &mut overlay.hidden, &grid_id, &panes, |pane| {
                        grid_panes.contains(&pane)
                    });
                }
                if overlay.grid.as_deref() != Some(descriptor.as_str())
                    && host
                        .layer(grid_id.as_str())
                        .replace_all(grid_lines(&grid))
                        .is_ok()
                {
                    overlay.grid = Some(descriptor);
                }
            }
        }
        None => {
            if overlay.grid_layer {
                let _ = host.layer(grid_id.as_str()).remove_layer();
                overlay.grid_layer = false;
                overlay.grid = None;
                overlay.hidden.remove(&grid_id);
            }
        }
    }

    // Hover / jump highlight (every pane).
    if overlay.mark != mark {
        let marks = host.layer(mark_layer(file_id));
        let result = match (mark, &ctx) {
            (Some(coord), Some(ctx)) => {
                let rect = ctx.grid.valid_footprint(coord);
                marks.replace_all(vec![OverlayObject::new(
                    "region",
                    Shape::Rectangle {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                    },
                )])
            }
            _ => marks.clear(),
        };
        if result.is_ok() {
            overlay.mark = mark;
        }
    }

    if !separate {
        core.without_pane_layers();
    }
}

fn grid_spec(file_id: FileId, min_zoom: Option<f64>) -> LayerSpec {
    LayerSpec {
        min_zoom,
        ..LayerSpec::new(grid_layer(file_id), file_id)
            .named("VAE tile grid")
            .at_z(-85)
            .with_style(Style {
                stroke: Rgba(0xFF, 0xFF, 0xFF, 0x70),
                stroke_width: 1.0,
                ..Style::default()
            })
    }
}

fn grid_lines(grid: &TileGrid) -> Vec<OverlayObject> {
    let (step_x, step_y) = (grid.step_x(), grid.step_y());
    let (w, h) = (grid.width0 as f64, grid.height0 as f64);
    let mut lines = Vec::with_capacity((grid.cols + grid.rows + 2) as usize);
    for col in 0..=grid.cols {
        let x = (f64::from(col) * step_x).min(w);
        lines.push(OverlayObject::new(
            format!("v{col}"),
            Shape::Polyline {
                points: vec![[x, 0.0], [x, h]],
            },
        ));
    }
    for row in 0..=grid.rows {
        let y = (f64::from(row) * step_y).min(h);
        lines.push(OverlayObject::new(
            format!("h{row}"),
            Shape::Polyline {
                points: vec![[0.0, y], [w, y]],
            },
        ));
    }
    lines
}

/// Whether two colour ranges are within a tenth of the newer's span.
fn close(shown: (f32, f32), exact: (f32, f32)) -> bool {
    let span = (exact.1 - exact.0).abs().max(1e-9);
    (shown.0 - exact.0).abs() <= 0.1 * span && (shown.1 - exact.1).abs() <= 0.1 * span
}

pub fn heat_chunk_id((col, row): (u32, u32)) -> String {
    format!("heat-{col}-{row}")
}

/// Sends the heat map, chunk by chunk: only chunks whose pixels changed
/// since they were last sent go to the host.
fn update_heat<B: Backend>(
    core: &Core<B>,
    overlay: &mut FileOverlay,
    layer: &Layer,
    ctx: &RunContext,
    stats: &RunStats,
    busy: bool,
) {
    let exact = stats.heat_range();
    let shown = overlay.heat.as_ref().filter(|h| *h.run == *ctx.key.run);
    // While a job adds tiles, the colour scale moves only when it changed
    // noticeably, so that only the chunks with new tiles are sent again;
    // the job's end sets it exactly.
    let range = match shown {
        Some(h) if busy && close(h.range, exact) => h.range,
        _ => exact,
    };
    if shown.is_some_and(|h| h.revision == stats.revision && h.range == range) {
        return;
    }
    if busy
        && shown.is_some()
        && overlay
            .heat_uploaded
            .is_some_and(|at| at.elapsed() < RUNNING_UPLOAD_INTERVAL)
    {
        return;
    }
    let tiles: Vec<(TileCoord, TileState)> = lock(&core.results)
        .run(&ctx.key)
        .map(|run| run.iter().collect())
        .unwrap_or_default();
    let map = HeatMap::new(&ctx.grid, &tiles, range.0, range.1);
    let layout = map.layout();
    let mut chunks = match overlay.heat.take() {
        Some(old) if *old.run == *ctx.key.run && old.layout == layout => old.chunks,
        // Another run or shape: start from an empty layer.
        Some(_) => {
            let _ = layer.clear();
            HashMap::new()
        }
        None => HashMap::new(),
    };
    let mut complete = true;
    let mut emptied = Vec::new();
    for chunk in map.chunks(HEAT_CHUNK) {
        let key = (chunk.col, chunk.row);
        match chunk.content {
            None => {
                if chunks.remove(&key).is_some() {
                    emptied.push(key);
                }
            }
            Some((image, hash)) => {
                if chunks.get(&key) == Some(&hash) {
                    continue;
                }
                if layer
                    .put_raster(heat_chunk_id(key), chunk.rect, image, false)
                    .is_ok()
                {
                    chunks.insert(key, hash);
                } else {
                    chunks.remove(&key);
                    complete = false;
                }
            }
        }
    }
    if !emptied.is_empty()
        && layer
            .remove(emptied.iter().copied().map(heat_chunk_id).collect())
            .is_err()
    {
        // Try again next time.
        for key in emptied {
            chunks.insert(key, 0);
        }
        complete = false;
    }
    overlay.heat = Some(HeatShown {
        run: Arc::clone(&ctx.key.run),
        // Revisions start at 1: 0 makes the next refresh retry.
        revision: if complete { stats.revision } else { 0 },
        range,
        layout,
        chunks,
    });
    overlay.heat_uploaded = Some(Instant::now());
}

fn build_mosaic<B: Backend>(
    core: &Core<B>,
    ctx: &RunContext,
    frame: &MosaicFrame,
    mode: Mode,
    error_hi: f32,
    tone: Option<&DisplayTone>,
) -> Image {
    let range = ctx.grid.range_in(&frame.rect);
    let tiles: Vec<MosaicTile> = {
        let mut results = lock(&core.results);
        let states: Vec<(TileCoord, Option<TileState>)> = range
            .iter()
            .map(|coord| {
                (
                    coord,
                    results.run(&ctx.key).and_then(|run| run.state(coord)),
                )
            })
            .collect();
        states
            .into_iter()
            .map(|(coord, state)| MosaicTile {
                coord,
                payload: state
                    .filter(|s| matches!(s, TileState::Analysed { .. }))
                    .and_then(|_| results.payload(&ctx.key, coord)),
                state,
            })
            .collect()
    };
    render::mosaic(&ctx.grid, frame, &tiles, mode, error_hi, tone)
}

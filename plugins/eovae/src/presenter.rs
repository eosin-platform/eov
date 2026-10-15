//! The presenter thread: everything the user sees, pushed at a bounded
//! rate.
//!
//! Workers and event handlers only set dirty flags; this thread wakes at
//! most a few times per second, compares with what it last sent and pushes
//! the differences: sidebar properties, overlay layers and rasters, HUD
//! button states, the status-bar text and the saved settings (VAE-13). It
//! never holds the plugin's locks while calling the host.

use crate::engine::Backend;
use crate::plugin::{
    Core, FileState, HUD_GRID, HUD_VIEW, RunContext, grid_layer, lock, mark_layer, raster_layer,
    run_context, view_menu,
};
use crate::render::{self, Mode, MosaicFrame, MosaicTile};
use crate::results::{Counts, ResultKey};
use crate::stats::{self, Bin, Summary};
use crate::tiling::TileCoord;
use eov_plugin_api::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// Shortest time between two refreshes.
const MIN_INTERVAL: Duration = Duration::from_millis(200);
/// Settings are saved this long after the last change.
const SAVE_DELAY: Duration = Duration::from_millis(700);
/// How long a region stays highlighted after a jump.
pub const PULSE: Duration = Duration::from_millis(1600);

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

#[derive(Default)]
struct FileOverlay {
    layers: bool,
    heat: Option<(Arc<str>, u64, (f32, f32))>,
    mosaics: HashMap<PaneId, (Arc<str>, u64, Mode, MosaicFrame)>,
    grid: Option<String>,
    mark: Option<TileCoord>,
}

#[derive(Default)]
pub struct Memo {
    sidebar_id: Option<String>,
    pub sidebar_sent: Map<String, Value>,
    stats: HashMap<ResultKey, Arc<RunStats>>,
    files: HashMap<FileId, FileOverlay>,
    hud: HashMap<PaneId, (bool, Mode)>,
    status: Option<String>,
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

        let snapshot = if flags.hud || flags.all_overlays || !flags.overlays.is_empty() {
            core.host.snapshot().ok()
        } else {
            None
        };
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
        if timed {
            core.dirty.mark(|_| {});
            std::thread::sleep(Duration::from_millis(100));
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

fn refresh_hud<B: Backend>(core: &Arc<Core<B>>, memo: &mut Memo, snapshot: Option<&HostSnapshot>) {
    let Some(snapshot) = snapshot else {
        return;
    };
    let desired: Vec<(PaneId, (bool, Mode))> = {
        let state = lock(&core.state);
        snapshot
            .panes
            .iter()
            .filter_map(|pane| {
                let file = state.files.get(&pane.file_id?)?;
                Some((pane.pane, (file.grid, file.mode)))
            })
            .collect()
    };
    let live: HashSet<PaneId> = snapshot.panes.iter().map(|p| p.pane).collect();
    memo.hud.retain(|pane, _| live.contains(pane));
    for (pane, (grid, mode)) in desired {
        if memo.hud.get(&pane) == Some(&(grid, mode)) {
            continue;
        }
        let a = core.host.update_hud_button(ButtonUpdate {
            button_id: HUD_GRID.into(),
            pane: Some(pane),
            active: Some(grid),
            ..Default::default()
        });
        let b = core.host.update_hud_button(ButtonUpdate {
            button_id: HUD_VIEW.into(),
            pane: Some(pane),
            active: Some(mode != Mode::Original),
            menu: Some(view_menu(mode)),
            ..Default::default()
        });
        if a.is_ok() && b.is_ok() {
            memo.hud.insert(pane, (grid, mode));
        }
    }
}

/// Brings one slide's layers up to date.
fn refresh_file<B: Backend>(
    core: &Arc<Core<B>>,
    memo: &mut Memo,
    file_id: FileId,
    snapshot: Option<&HostSnapshot>,
) {
    let (mode, grid_on, ctx, mark, views) = {
        let state = lock(&core.state);
        let Some(FileState { mode, grid, .. }) = state.files.get(&file_id) else {
            return;
        };
        let ctx = run_context(&state, file_id).ok();
        let mark = match (state.pulse, state.hover) {
            (Some((f, coord, _)), _) if f == file_id => Some(coord),
            (_, Some((f, coord))) if f == file_id => Some(coord),
            _ => None,
        };
        // Pane views: the host's snapshot is authoritative about which panes
        // exist; our tracked views fill in until it reports them.
        let views: Vec<ViewInfo> = match snapshot {
            Some(snapshot) => snapshot
                .panes
                .iter()
                .filter_map(|pane| pane.view)
                .filter(|view| view.file_id == file_id)
                .collect(),
            None => state
                .panes
                .values()
                .filter_map(|pane| pane.view)
                .filter(|view| view.file_id == file_id)
                .collect(),
        };
        (*mode, *grid, ctx, mark, views)
    };
    let overlay = memo.files.entry(file_id).or_default();
    let host = &core.host;
    if !overlay.layers {
        let raster = LayerSpec::new(raster_layer(file_id), file_id)
            .named("VAE overlay")
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
        if host.create_layer(raster).is_err() || host.create_layer(marks).is_err() {
            // The file is not published yet; retry on the next change.
            return;
        }
        overlay.layers = true;
    }
    let raster = host.layer(raster_layer(file_id));

    // Rasters. Stored results of a slide with an overlay load from the disk
    // cache once (any pane, not only the active one).
    if mode != Mode::Original
        && let Some(ctx) = &ctx
    {
        core.ensure_loaded(ctx);
    }
    let stats = ctx.as_ref().and_then(|ctx| memo.stats(core, &ctx.key));
    let overlay = memo.files.entry(file_id).or_default();
    match (mode, &ctx) {
        (Mode::ErrorMap, Some(ctx)) => {
            remove_mosaics(&raster, overlay, |_| true);
            let stats = stats.clone().unwrap_or_default();
            let range = stats.heat_range();
            let revision = stats.revision;
            let current = overlay.heat.as_ref();
            let stale = current.is_none_or(|(run, rev, old_range)| {
                **run != *ctx.key.run || *rev != revision || *old_range != range
            });
            if stale {
                let tiles: Vec<(TileCoord, crate::results::TileState)> = lock(&core.results)
                    .run(&ctx.key)
                    .map(|run| run.iter().collect())
                    .unwrap_or_default();
                let (image, rect) = render::heat_raster(&ctx.grid, &tiles, range.0, range.1);
                if raster.put_raster("heat", rect, image, false).is_ok() {
                    overlay.heat = Some((Arc::clone(&ctx.key.run), revision, range));
                }
            }
        }
        (mode, Some(ctx)) if mode.needs_pixels() => {
            if overlay.heat.take().is_some() {
                let _ = raster.remove(vec!["heat".to_string()]);
            }
            let panes: HashSet<PaneId> = views.iter().map(|v| v.pane).collect();
            remove_mosaics(&raster, overlay, |pane| !panes.contains(&pane));
            let revision = stats.as_ref().map(|s| s.revision).unwrap_or(0);
            let error_hi = stats.as_ref().map(|s| s.pixel_range()).unwrap_or(0.1);
            for view in &views {
                let Some(frame) = MosaicFrame::for_view(view, &ctx.grid) else {
                    continue;
                };
                let fresh = overlay
                    .mosaics
                    .get(&view.pane)
                    .is_some_and(|(run, rev, m, old)| {
                        **run == *ctx.key.run
                            && *rev == revision
                            && *m == mode
                            && old.covers(&frame)
                    });
                if fresh {
                    continue;
                }
                let image = build_mosaic(core, ctx, &frame, mode, error_hi);
                if raster
                    .put_raster(format!("mosaic-{}", view.pane), frame.rect, image, true)
                    .is_ok()
                {
                    overlay
                        .mosaics
                        .insert(view.pane, (Arc::clone(&ctx.key.run), revision, mode, frame));
                }
            }
        }
        _ => {
            if overlay.heat.take().is_some() {
                let _ = raster.remove(vec!["heat".to_string()]);
            }
            remove_mosaics(&raster, overlay, |_| true);
        }
    }

    // The tile grid: whole-slide polylines, hidden when too dense.
    let wanted_grid = ctx.as_ref().filter(|_| grid_on).map(|ctx| {
        (
            format!(
                "{}x{}@{:.6}",
                ctx.grid.cols,
                ctx.grid.rows,
                ctx.grid.step0()
            ),
            ctx.grid,
        )
    });
    match wanted_grid {
        Some((descriptor, grid)) if overlay.grid.as_deref() != Some(descriptor.as_str()) => {
            let spec = LayerSpec {
                min_zoom: Some(6.0 / grid.step0()),
                ..LayerSpec::new(grid_layer(file_id), file_id)
                    .named("VAE tile grid")
                    .at_z(-85)
                    .with_style(Style {
                        stroke: Rgba(0xFF, 0xFF, 0xFF, 0x70),
                        stroke_width: 1.0,
                        ..Style::default()
                    })
            };
            let step = grid.step0();
            let (w, h) = (grid.width0 as f64, grid.height0 as f64);
            let mut lines = Vec::with_capacity((grid.cols + grid.rows + 2) as usize);
            for col in 0..=grid.cols {
                let x = (f64::from(col) * step).min(w);
                lines.push(OverlayObject::new(
                    format!("v{col}"),
                    Shape::Polyline {
                        points: vec![[x, 0.0], [x, h]],
                    },
                ));
            }
            for row in 0..=grid.rows {
                let y = (f64::from(row) * step).min(h);
                lines.push(OverlayObject::new(
                    format!("h{row}"),
                    Shape::Polyline {
                        points: vec![[0.0, y], [w, y]],
                    },
                ));
            }
            if host.create_layer(spec).is_ok()
                && host.layer(grid_layer(file_id)).replace_all(lines).is_ok()
            {
                overlay.grid = Some(descriptor);
            }
        }
        None if overlay.grid.is_some() => {
            let _ = host.layer(grid_layer(file_id)).remove_layer();
            overlay.grid = None;
        }
        _ => {}
    }

    // Hover / jump highlight.
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
}

fn remove_mosaics(raster: &Layer, overlay: &mut FileOverlay, remove: impl Fn(PaneId) -> bool) {
    let gone: Vec<PaneId> = overlay
        .mosaics
        .keys()
        .copied()
        .filter(|pane| remove(*pane))
        .collect();
    if gone.is_empty() {
        return;
    }
    let ids = gone.iter().map(|pane| format!("mosaic-{pane}")).collect();
    let _ = raster.remove(ids);
    for pane in gone {
        overlay.mosaics.remove(&pane);
    }
}

fn build_mosaic<B: Backend>(
    core: &Core<B>,
    ctx: &RunContext,
    frame: &MosaicFrame,
    mode: Mode,
    error_hi: f32,
) -> Image {
    let range = ctx.grid.range_in(&frame.rect);
    let tiles: Vec<MosaicTile> = {
        let mut results = lock(&core.results);
        let states: Vec<(TileCoord, Option<crate::results::TileState>)> = range
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
                    .filter(|s| matches!(s, crate::results::TileState::Analysed { .. }))
                    .and_then(|_| results.payload(&ctx.key, coord)),
                state,
            })
            .collect()
    };
    render::mosaic(&ctx.grid, frame, &tiles, mode, error_hi)
}

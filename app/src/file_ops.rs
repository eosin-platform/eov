//! File operations for opening and managing WSI files
//!
//! This module contains functions for opening files and generating thumbnails.

use crate::state::{
    AppState, MinimapThumbnail, NewOpenFile, PaneId, SeriesEntry, SeriesNavigationMode,
    SeriesThumbnail, TabActivation,
};
use crate::tile_loader::{TileLoader, calculate_wanted_tiles};
use crate::ui_update::{update_recent_files, update_tabs};
use crate::{PaneRenderCacheEntry, PaneUiModels, PaneViewData, request_render_loop};
use eov_common::{
    ThumbnailOptions, ThumbnailRegion, TileCache, TileManager, ViewportState, WsiFile,
};
use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use slint::{ComponentHandle, SharedString, Timer, VecModel};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use tracing::{error, info, warn};

/// The slide formats eov opens (also in the file dialog, the dataset
/// extractor and the operating system's file associations).
const WSI_EXTENSIONS: &[&str] = eov_common::dataset::SUPPORTED_SLIDE_EXTENSIONS;

pub struct OpenFileUiContext<'a> {
    pub pane_render_cache: &'a mut Vec<PaneRenderCacheEntry>,
    pub pane_ui_models: &'a mut Vec<PaneUiModels>,
    pub pane_view_model: &'a Rc<VecModel<PaneViewData>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenFileMode {
    ReuseExistingTab,
    ForceNewTab,
}

/// Where a slide of a launch goes (SH-7). Slides open in parallel and
/// finish in any order; their tabs still end up in the order they were
/// given, the first one selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Placement {
    /// The launch ([`new_batch`]).
    pub(crate) batch: u64,
    /// Position among the launch's slides in the same pane.
    pub(crate) position: usize,
    /// Select the tab (the first slide of its pane).
    pub(crate) select: bool,
    /// Focus its pane too (the first slide of the launch).
    pub(crate) focus: bool,
}

impl Placement {
    fn activation(self) -> TabActivation {
        match (self.select, self.focus) {
            (true, true) => TabActivation::Focus,
            (true, false) => TabActivation::Select,
            (false, _) => TabActivation::Background,
        }
    }
}

/// Launches whose tabs are remembered for ordering.
const KEPT_BATCHES: u64 = 16;

/// Starts a launch: slides opened with its id keep their order.
pub(crate) fn new_batch() -> u64 {
    let id = NEXT_BATCH.with(|next| {
        let id = next.get();
        next.set(id + 1);
        id
    });
    BATCH_TABS.with(|batches| {
        let mut batches = batches.borrow_mut();
        batches.retain(|&batch, _| batch + KEPT_BATCHES > id);
        batches.insert(id, Vec::new());
    });
    id
}

/// Moves a launch's new tab between the launch's tabs given before and
/// after it, and remembers it.
fn place_in_launch_order(state: &mut AppState, pane: PaneId, tab_id: i32, placement: Placement) {
    BATCH_TABS.with(|batches| {
        let mut batches = batches.borrow_mut();
        let placed = batches.entry(placement.batch).or_default();
        let position_of = |tab: i32| {
            placed
                .iter()
                .find(|(placed_tab, _)| *placed_tab == tab)
                .map(|(_, position)| *position)
        };
        let tabs = state.tabs_for_pane(pane);
        let after_earlier = tabs
            .iter()
            .rposition(|&tab| position_of(tab).is_some_and(|p| p < placement.position))
            .map(|index| index + 1);
        let before_later = tabs
            .iter()
            .position(|&tab| position_of(tab).is_some_and(|p| p > placement.position));
        if let Some(slot) = after_earlier.or(before_later) {
            state.reorder_tab(pane, tab_id, slot as i32);
        }
        placed.push((tab_id, placement.position));
    });
}

pub fn is_supported_wsi_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            WSI_EXTENSIONS
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(ext))
        })
        .unwrap_or(false)
}

fn enumerate_series_entries(folder_path: &Path) -> Vec<SeriesEntry> {
    let mut paths: Vec<PathBuf> = fs::read_dir(folder_path)
        .ok()
        .into_iter()
        .flat_map(|read_dir| read_dir.filter_map(Result::ok))
        .map(|entry| entry.path())
        // Hidden files and folders (".git", ".cache") are not slides.
        .filter(|path| {
            !path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with('.'))
        })
        .filter(|path| path.is_dir() || (path.is_file() && is_supported_wsi_path(path)))
        .map(|path| crate::paths::canonical(&path))
        .collect();
    paths.sort();
    // The instances of one DICOM series are one slide, which OpenSlide
    // opens from any of them: list the series once, by its first file.
    let mut series_index: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut instances: Vec<usize> = Vec::new();
    let mut entries: Vec<SeriesEntry> = Vec::new();
    for path in paths {
        let is_directory = path.is_dir();
        let is_dicom = !is_directory
            && path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("dcm"));
        if is_dicom
            && let Some(uid) = eov_common::dataset::dicom_series_uid_within(&path, 256 * 1024)
        {
            if let Some(&index) = series_index.get(&uid) {
                instances[index] += 1;
                continue;
            }
            series_index.insert(uid, entries.len());
        }
        let filename = path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());
        let modified = fs::metadata(&path).and_then(|meta| meta.modified()).ok();
        instances.push(1);
        entries.push(SeriesEntry {
            metadata_tooltip: if is_directory {
                format!("{}\nFolder", display_name(&path))
            } else {
                "Loading metadata…".to_string()
            },
            path,
            filename,
            is_directory,
            modified,
            objective_label: None,
            stain_label: None,
            thumbnail: None,
            thumbnail_loading: !is_directory,
        });
    }
    for (entry, count) in entries.iter_mut().zip(instances) {
        if count > 1 {
            entry.filename = format!("{} (DICOM series, {count} files)", entry.filename);
        }
    }
    entries
}

fn thumbnail_cache_dir() -> PathBuf {
    crate::paths::thumbnail_cache_dir()
}

fn thumbnail_cache_key(path: &Path) -> Option<String> {
    let normalized_path = dunce::canonicalize(path).ok()?;
    let fingerprint = eov_common::file_id::compute_fingerprint(&normalized_path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(eov_common::file_id::hex_digest(&fingerprint).as_bytes());
    hasher.update(normalized_path.to_string_lossy().as_bytes());
    Some(format!("{:x}", hasher.finalize()))
}

fn load_cached_thumbnail(path: &Path) -> Option<SeriesThumbnail> {
    let cache_key = thumbnail_cache_key(path)?;
    let cache_path = thumbnail_cache_dir().join(format!("{cache_key}.png"));
    let bytes = fs::read(cache_path).ok()?;
    let image = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .ok()?
        .to_rgba8();
    let (width, height) = image.dimensions();
    Some(SeriesThumbnail {
        rgba: image.into_raw(),
        width,
        height,
    })
}

fn persist_thumbnail(path: &Path, thumbnail: &SeriesThumbnail) {
    let Some(cache_key) = thumbnail_cache_key(path) else {
        return;
    };
    let cache_dir = thumbnail_cache_dir();
    if fs::create_dir_all(&cache_dir).is_err() {
        return;
    }
    let cache_path = cache_dir.join(format!("{cache_key}.png"));
    let Some(image) =
        image::RgbaImage::from_raw(thumbnail.width, thumbnail.height, thumbnail.rgba.clone())
    else {
        return;
    };
    let mut encoded = Vec::new();
    if image
        .write_to(&mut Cursor::new(&mut encoded), image::ImageFormat::Png)
        .is_ok()
    {
        let _ = fs::write(cache_path, encoded);
    }
}

fn build_series_metadata_tooltip(path: &Path, wsi: Option<&WsiFile>) -> String {
    let header = display_name(path);
    let Some(wsi) = wsi else {
        return format!("{}\nThe slide could not be read", header);
    };

    let properties = wsi.properties();
    let objective = properties
        .objective_power
        .map(|value| format!("{}x", eov_common::format_decimal(value)))
        .unwrap_or_else(|| "N/A".to_string());
    let mpp = properties
        .mpp_x
        .zip(properties.mpp_y)
        .map(|(x, y)| {
            format!(
                "{} × {} µm/px",
                eov_common::format_decimal(x),
                eov_common::format_decimal(y)
            )
        })
        .unwrap_or_else(|| "N/A".to_string());
    let vendor = properties.vendor.as_deref().unwrap_or("Unknown");
    let stain = properties.stain.as_deref().unwrap_or("N/A");

    format!(
        "{}\n{} × {} px, {} levels\nVendor: {}\nObjective: {}\nPixel size: {}\nStain: {}",
        header,
        eov_common::format_u64(properties.width),
        eov_common::format_u64(properties.height),
        properties.levels.len(),
        vendor,
        objective,
        mpp,
        stain,
    )
}

fn build_series_overlay_labels(wsi: Option<&WsiFile>) -> (Option<String>, Option<String>) {
    let Some(wsi) = wsi else {
        return (None, None);
    };

    let properties = wsi.properties();
    let objective_label = properties
        .objective_power
        .map(|value| format!("{}x", eov_common::format_decimal(value)));
    let stain_label = properties.stain.clone();
    (objective_label, stain_label)
}

/// A series-bar thumbnail with bounded memory and I/O (WSI-3).
fn series_thumbnail(wsi: &WsiFile) -> Option<SeriesThumbnail> {
    let options = ThumbnailOptions {
        region: ThumbnailRegion::ContentBounds,
        ..ThumbnailOptions::new(SERIES_THUMBNAIL_SIZE)
    };
    let thumbnail = wsi.thumbnail_with(&options).ok()?;
    Some(SeriesThumbnail {
        width: thumbnail.image.width as u32,
        height: thumbnail.image.height as u32,
        rgba: thumbnail.image.pixels,
    })
}

fn spawn_series_thumbnail_worker(
    ui_weak: slint::Weak<crate::AppWindow>,
    state: Arc<RwLock<AppState>>,
    revision: u64,
    paths: Vec<PathBuf>,
) {
    std::thread::spawn(move || {
        for path in paths {
            // Stop once the user moved to another folder (UI-14).
            if state.read().series_revision != revision {
                return;
            }
            let cached_thumbnail = load_cached_thumbnail(&path);
            let had_cached_thumbnail = cached_thumbnail.is_some();
            if let Some(cached_thumbnail) = cached_thumbnail.clone() {
                let metadata_tooltip = format!("{}\nLoading metadata…", display_name(&path));
                let state_handle = Arc::clone(&state);
                let ui_weak = ui_weak.clone();
                let update_path = path.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    let Some(ui) = ui_weak.upgrade() else {
                        return;
                    };

                    let updated = {
                        let mut state = state_handle.write();
                        state.update_series_entry_thumbnail(
                            revision,
                            &update_path,
                            metadata_tooltip,
                            None,
                            None,
                            Some(cached_thumbnail),
                        )
                    };

                    if updated {
                        let state = state_handle.read();
                        crate::update_tabs(&ui, &state);
                    }
                });
            }

            let opened = WsiFile::open(&path).ok();
            let metadata_tooltip = build_series_metadata_tooltip(&path, opened.as_ref());
            let (objective_label, stain_label) = build_series_overlay_labels(opened.as_ref());
            let thumbnail = cached_thumbnail.or_else(|| opened.as_ref().and_then(series_thumbnail));

            if !had_cached_thumbnail && let Some(thumbnail) = thumbnail.as_ref() {
                persist_thumbnail(&path, thumbnail);
            }

            let state_handle = Arc::clone(&state);
            let ui_weak = ui_weak.clone();
            let update_path = path.clone();
            let _ = slint::invoke_from_event_loop(move || {
                let Some(ui) = ui_weak.upgrade() else {
                    return;
                };

                let updated = {
                    let mut state = state_handle.write();
                    state.update_series_entry_thumbnail(
                        revision,
                        &update_path,
                        metadata_tooltip,
                        objective_label,
                        stain_label,
                        thumbnail,
                    )
                };

                if updated {
                    let state = state_handle.read();
                    crate::update_tabs(&ui, &state);
                }
            });
        }
    });
}

/// Lists `folder_path` on a worker and installs the entries if the series
/// is still at `revision` (UI-14: never a synchronous scan on the UI
/// thread). With `open_first`, the first slide is then opened.
pub fn load_series_entries_async(
    ui_weak: slint::Weak<crate::AppWindow>,
    state: Arc<RwLock<AppState>>,
    folder_path: PathBuf,
    revision: u64,
    open_first: bool,
) {
    std::thread::spawn(move || {
        let entries = enumerate_series_entries(&folder_path);
        let state_handle = Arc::clone(&state);
        let update_folder = folder_path.clone();
        let _ = slint::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let first_slide = entries
                .iter()
                .find(|entry| !entry.is_directory)
                .map(|entry| entry.path.clone());
            let updated = {
                let mut state = state_handle.write();
                state.replace_opened_series_entries(revision, &update_folder, entries)
            };
            if !updated {
                return;
            }
            {
                let state = state_handle.read();
                crate::update_tabs(&ui, &state);
            }
            ensure_series_thumbnails(&ui, &state_handle);
            if open_first {
                match first_slide {
                    Some(path) => {
                        crate::host_ui::with_host(|host| {
                            crate::open_file(
                                &ui,
                                &state_handle,
                                &host.tile_cache,
                                &host.render_timer,
                                path,
                            )
                        });
                    }
                    None => ui.set_status_text(SharedString::from(format!(
                        "Browsing {}",
                        update_folder.display()
                    ))),
                }
            }
        });
    });
}

/// Starts thumbnailing the series entries once, while the series bar is
/// visible (each thumbnail opens a slide).
pub fn ensure_series_thumbnails(ui: &crate::AppWindow, state: &Arc<RwLock<AppState>>) {
    let work = {
        let mut state = state.write();
        let visible = state.bottom_panel_visible
            && state.bottom_panel_kind == crate::state::BottomPanelKind::Series;
        let revision = state.series_revision;
        match state.opened_series.as_mut() {
            Some(series) if visible && !series.thumbnails_requested => {
                series.thumbnails_requested = true;
                let paths: Vec<PathBuf> = series
                    .entries
                    .iter()
                    .filter(|entry| !entry.is_directory)
                    .map(|entry| entry.path.clone())
                    .collect();
                (!paths.is_empty()).then_some((revision, paths))
            }
            _ => None,
        }
    };
    if let Some((revision, paths)) = work {
        spawn_series_thumbnail_worker(ui.as_weak(), Arc::clone(state), revision, paths);
    }
}

/// Shows `folder_path` in the series bar; its entries load in the
/// background.
pub fn set_series_from_folder(
    ui: &crate::AppWindow,
    state: &Arc<RwLock<AppState>>,
    folder_path: PathBuf,
    navigation: SeriesNavigationMode,
    open_first: bool,
) {
    let normalized_folder = crate::paths::canonical(&folder_path);
    if navigation != SeriesNavigationMode::Initialize
        && state
            .read()
            .current_series_path()
            .is_some_and(|current| current == normalized_folder.as_path())
    {
        if open_first {
            let first = state.read().opened_series.as_ref().and_then(|series| {
                series
                    .entries
                    .iter()
                    .find(|entry| !entry.is_directory)
                    .map(|entry| entry.path.clone())
            });
            if let Some(path) = first {
                crate::host_ui::with_host(|host| {
                    crate::open_file(ui, state, &host.tile_cache, &host.render_timer, path)
                });
            }
        }
        return;
    }

    let revision = {
        let mut state = state.write();
        state.set_opened_series(normalized_folder.clone(), Vec::new(), navigation)
    };
    {
        let state = state.read();
        crate::update_tabs(ui, &state);
    }
    load_series_entries_async(
        ui.as_weak(),
        Arc::clone(state),
        normalized_folder,
        revision,
        open_first,
    );
}

pub fn set_series_from_file_parent(
    ui: &crate::AppWindow,
    state: &Arc<RwLock<AppState>>,
    path: &Path,
) {
    let Some(parent) = path.parent() else {
        return;
    };
    let navigation = if state
        .read()
        .current_series_path()
        .is_some_and(|current| current == parent)
    {
        SeriesNavigationMode::Replace
    } else {
        SeriesNavigationMode::Push
    };
    set_series_from_folder(ui, state, parent.to_path_buf(), navigation, false);
}

/// Open a file and add it to the application state
#[allow(clippy::too_many_arguments)]
pub fn open_file(
    ui: &crate::AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    path: PathBuf,
    mode: OpenFileMode,
    placement: Option<Placement>,
    ui_context: OpenFileUiContext<'_>,
) {
    let OpenFileUiContext {
        pane_render_cache,
        pane_ui_models,
        pane_view_model,
    } = ui_context;

    let normalized_path = crate::paths::canonical(&path);

    let existing_tab = {
        let state_guard = state.read();
        state_guard.find_tab_by_path(&normalized_path)
    };

    if let Some((pane, tab_id)) = existing_tab {
        let mut status_text = format!("Focused {}", normalized_path.display());
        let view = INITIAL_VIEWS.with(|views| views.borrow_mut().remove(&normalized_path));

        let warnings = {
            let mut state_guard = state.write();
            match (placement, mode) {
                // Part of a launch: only the first slide of a pane is
                // selected.
                (Some(placement), _) => {
                    if placement.select && placement.focus {
                        state_guard.activate_tab_in_pane(pane, tab_id);
                    } else if placement.select {
                        state_guard.show_tab_in_pane(pane, tab_id);
                    }
                }
                (None, OpenFileMode::ReuseExistingTab) => {
                    state_guard.activate_tab_in_pane(pane, tab_id);
                }
                (None, OpenFileMode::ForceNewTab) => {
                    let target_pane = if state_guard.split_enabled {
                        state_guard.focused_pane
                    } else {
                        PaneId::PRIMARY
                    };
                    let duplicated_id = state_guard.duplicate_tab_to_pane(tab_id, target_pane);
                    state_guard.activate_tab_in_pane(target_pane, duplicated_id);
                    status_text = format!("Opened new tab for {}", normalized_path.display());
                }
            }
            state_guard.add_to_recent(&normalized_path);
            match view {
                Some(PendingView::Requested(request)) => {
                    show_view_on_open_tab(&mut state_guard, &normalized_path, &request)
                }
                _ => Vec::new(),
            }
        };

        {
            let state_guard = state.read();
            update_tabs(
                ui,
                &state_guard,
                pane_render_cache,
                pane_ui_models,
                pane_view_model,
            );
            update_recent_files(ui, &state_guard);
        }
        // Everything below may take the state lock itself; the read guard
        // above must be released first (parking_lot locks are not
        // re-entrant).
        ui.set_is_loading(false);
        ui.set_status_text(SharedString::from(status_text));
        report_view_warnings(&normalized_path, &warnings);
        request_render_loop(render_timer, &ui.as_weak(), state, tile_cache);
        resolve_open_waiters(&normalized_path, opened_file_id(state, &normalized_path));
        return;
    }

    // Panes can be inserted or closed while the slide opens: remember the
    // target by its stable id.
    let target_pane = {
        let state_guard = state.read();
        let pane = if state_guard.split_enabled {
            state_guard.focused_pane
        } else {
            PaneId::PRIMARY
        };
        state_guard.pane_uid(pane)
    };

    // A slide that is still opening gets one more pane rather than a
    // second handle.
    let already_opening = PENDING_OPENS.with(|pending| {
        let mut pending = pending.borrow_mut();
        let Some(open) = pending
            .values_mut()
            .find(|open| open.path == normalized_path)
        else {
            return false;
        };
        if open.target_pane != target_pane && !open.extra_panes.contains(&target_pane) {
            open.extra_panes.push(target_pane);
        }
        true
    });
    if already_opening {
        return;
    }

    let request = NEXT_OPEN_ID.with(|next| {
        let id = next.get();
        next.set(id + 1);
        id
    });
    PENDING_OPENS.with(|pending| {
        pending.borrow_mut().insert(
            request,
            PendingOpen {
                path: normalized_path.clone(),
                target_pane,
                extra_panes: Vec::new(),
                placement,
                state: Arc::clone(state),
                tile_cache: Arc::clone(tile_cache),
                render_timer: Rc::clone(render_timer),
            },
        )
    });
    ui.set_is_loading(true);
    ui.set_status_text(SharedString::from(format!(
        "Opening {}…",
        display_name(&normalized_path)
    )));
    // Shows the loading indicator in the pane the slide opens in (with the
    // caller's pane models: `crate::update_tabs` would borrow them again).
    update_tabs(
        ui,
        &state.read(),
        pane_render_cache,
        pane_ui_models,
        pane_view_model,
    );

    // Opening can take seconds (network shares, MRXS, DICOM): do it, and
    // the thumbnail, on a worker and install the result on the UI thread
    // (SH-11, WSI-3).
    let ui_weak = ui.as_weak();
    let worker_path = normalized_path.clone();
    let spawned = std::thread::Builder::new()
        .name("eov-open-slide".to_string())
        .spawn(move || {
            let result = open_slide_blocking(&worker_path).map_err(|e| e.to_string());
            let _ = slint::invoke_from_event_loop(move || finish_open(ui_weak, request, result));
        });
    if let Err(err) = spawned {
        // From the event loop: the caller still holds the pane models,
        // which finishing (an error refreshes the tabs) needs.
        let ui_weak = ui.as_weak();
        let error = err.to_string();
        slint::Timer::single_shot(std::time::Duration::ZERO, move || {
            finish_open(ui_weak, request, Err(error))
        });
    }
}

/// Whether a slide is being opened into the pane with stable id `uid`;
/// one opened without a target pane goes to the `focused` one.
pub(crate) fn opening_in_pane(uid: u32, focused: bool) -> bool {
    PENDING_OPENS.with(|pending| {
        pending
            .borrow()
            .values()
            .any(|open| open.target_pane.map_or(focused, |target| target == uid))
    })
}

/// Longest side of minimap thumbnails, in pixels.
const MINIMAP_THUMBNAIL_SIZE: u32 = 256;

/// Longest side of series-bar thumbnails, in pixels.
const SERIES_THUMBNAIL_SIZE: u32 = 180;

/// A slide being opened on a worker thread.
struct PendingOpen {
    /// Canonical path.
    path: PathBuf,
    /// Stable id of the pane that shows the slide.
    target_pane: Option<u32>,
    /// Panes (stable ids) that asked for the same slide while it opened.
    extra_panes: Vec<Option<u32>>,
    /// Its place in a launch.
    placement: Option<Placement>,
    state: Arc<RwLock<AppState>>,
    tile_cache: Arc<TileCache>,
    render_timer: Rc<Timer>,
}

/// What the open worker produces.
struct OpenedSlide {
    wsi: WsiFile,
    thumbnail: Option<MinimapThumbnail>,
}

/// A view waiting for its slide to open.
enum PendingView {
    /// Exactly where the slide was (a restored session).
    Saved(crate::session::SavedView),
    /// `--goto`/`--zoom` or a link, placed once the slide's size, pixel
    /// size and objective are known.
    Requested(crate::launch::ViewRequest),
}

/// Called on the UI thread with a slide's file id once it has opened, or
/// with the reason it could not be opened.
pub(crate) type OpenWaiter = Box<dyn FnOnce(Result<i32, String>)>;

thread_local! {
    /// Waiting for slides (canonical paths) to open: scripts' `open`.
    static OPEN_WAITERS: RefCell<Vec<(PathBuf, OpenWaiter)>> = const { RefCell::new(Vec::new()) };
    /// Opens in flight, by request (UI thread only).
    static PENDING_OPENS: RefCell<HashMap<u64, PendingOpen>> = RefCell::new(HashMap::new());
    static NEXT_OPEN_ID: Cell<u64> = const { Cell::new(1) };
    /// Views to show when these slides (canonical paths) open.
    static INITIAL_VIEWS: RefCell<HashMap<PathBuf, PendingView>> =
        RefCell::new(HashMap::new());
    static NEXT_BATCH: Cell<u64> = const { Cell::new(1) };
    /// Tabs of recent launches with their positions, by launch.
    static BATCH_TABS: RefCell<std::collections::BTreeMap<u64, Vec<(i32, usize)>>> =
        const { RefCell::new(std::collections::BTreeMap::new()) };
}

/// Calls `waiter` once the slide at `path` has opened, or failed to: the
/// next [`open_file`] of it, or the open in flight (UI thread).
pub(crate) fn when_open(path: &Path, waiter: OpenWaiter) {
    let path = crate::paths::canonical(path);
    OPEN_WAITERS.with(|waiters| waiters.borrow_mut().push((path, waiter)));
}

/// Tells those waiting for `path` (canonical) how its open went. Call with
/// no state lock held: waiters may take it.
fn resolve_open_waiters(path: &Path, result: Result<i32, String>) {
    let waiting: Vec<OpenWaiter> = OPEN_WAITERS.with(|waiters| {
        let mut waiters = waiters.borrow_mut();
        let (matched, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut *waiters)
            .into_iter()
            .partition(|(waited, _)| waited == path);
        *waiters = rest;
        matched.into_iter().map(|(_, waiter)| waiter).collect()
    });
    for waiter in waiting {
        waiter(result.clone());
    }
}

/// The file id of the slide open at `path` (canonical), for its waiters.
fn opened_file_id(state: &Arc<RwLock<AppState>>, path: &Path) -> Result<i32, String> {
    let state = state.read();
    state
        .find_tab_by_path(path)
        .map(|(_, tab_id)| state.resolve_tab_file_id(tab_id))
        .ok_or_else(|| "the slide was closed".to_string())
}

/// Opens `path` at `view` instead of fitting it to the pane, the next time
/// it is opened.
pub(crate) fn set_initial_view(path: &Path, view: crate::session::SavedView) {
    let path = crate::paths::canonical(path);
    INITIAL_VIEWS.with(|views| views.borrow_mut().insert(path, PendingView::Saved(view)));
}

/// Shows `path` at `request` when it opens next, or right away when it is
/// open already (`--goto`, `--zoom`, links).
pub(crate) fn request_view(path: &Path, request: crate::launch::ViewRequest) {
    if request.is_empty() {
        return;
    }
    let path = crate::paths::canonical(path);
    INITIAL_VIEWS.with(|views| {
        views
            .borrow_mut()
            .insert(path, PendingView::Requested(request))
    });
}

/// Moves the view of an open slide to `request`, where the slide is the
/// selected tab of its pane. Returns what could not be applied.
fn show_view_on_open_tab(
    state: &mut AppState,
    path: &Path,
    request: &crate::launch::ViewRequest,
) -> Vec<String> {
    let Some((pane, tab_id)) = state.find_tab_by_path(path) else {
        return Vec::new();
    };
    if state.active_tab_id_for_pane(pane) != Some(tab_id) {
        return Vec::new();
    }
    let file_id = state.resolve_tab_file_id(tab_id);
    let Some(file) = state.get_file_mut(file_id) else {
        return Vec::new();
    };
    let scale = crate::launch::SlideScale::of_open_file(file);
    let Some(pane_state) = file.pane_state_mut(pane) else {
        return Vec::new();
    };
    let viewport = &mut pane_state.viewport;
    let current = (
        viewport.viewport.center.x,
        viewport.viewport.center.y,
        viewport.viewport.zoom,
    );
    let view = crate::launch::resolve_view(request, &scale, current, viewport.fit_view());
    viewport.smooth_set_center_zoom(view.center_x, view.center_y, view.zoom);
    state.request_render();
    view.warnings
}

/// Tells the user what of a requested view could not be applied.
fn report_view_warnings(path: &Path, warnings: &[String]) {
    if warnings.is_empty() {
        return;
    }
    crate::host_ui::with_host(|host| {
        host.notify(
            eov_plugin_api::protocol::NotifyLevel::Warning,
            &format!("Could not show {} exactly as asked", display_name(path)),
            Some(&warnings.join("\n")),
            None,
        )
    });
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// The minimap thumbnail of a slide (worker thread: it reads the slide).
/// Multichannel slides show their current channel display.
pub(crate) fn minimap_thumbnail(wsi: &WsiFile) -> Option<MinimapThumbnail> {
    let options = ThumbnailOptions {
        region: ThumbnailRegion::ContentBounds,
        ..ThumbnailOptions::new(MINIMAP_THUMBNAIL_SIZE)
    };
    match wsi.thumbnail_with(&options) {
        Ok(thumbnail) => Some(MinimapThumbnail {
            width: thumbnail.image.width as u32,
            height: thumbnail.image.height as u32,
            rgba: thumbnail.image.pixels,
            region: thumbnail.region,
        }),
        Err(err) => {
            warn!(
                "No minimap thumbnail for {}: {err}",
                wsi.properties().path.display()
            );
            None
        }
    }
}

/// Opens a slide and builds its minimap thumbnail (worker thread). A
/// multichannel slide gets the channel display saved for it first, so it
/// never shows the defaults for a moment.
fn open_slide_blocking(path: &Path) -> eov_common::Result<OpenedSlide> {
    let wsi = WsiFile::open(path)?;
    crate::channels::restore_saved_display(&wsi);
    let thumbnail = minimap_thumbnail(&wsi);
    Ok(OpenedSlide { wsi, thumbnail })
}

/// Installs an opened slide (UI thread).
fn finish_open(
    ui_weak: slint::Weak<crate::AppWindow>,
    request: u64,
    result: Result<OpenedSlide, String>,
) {
    let Some(pending) = PENDING_OPENS.with(|pending| pending.borrow_mut().remove(&request)) else {
        return;
    };
    let Some(ui) = ui_weak.upgrade() else {
        resolve_open_waiters(&pending.path, Err("the window is closed".to_string()));
        return;
    };
    let still_opening = PENDING_OPENS.with(|pending| !pending.borrow().is_empty());
    ui.set_is_loading(still_opening);

    let PendingOpen {
        path,
        target_pane,
        extra_panes,
        placement,
        state,
        tile_cache,
        render_timer,
    } = pending;
    let name = display_name(&path);
    let opened = match result {
        Ok(opened) => opened,
        Err(err) => {
            error!("Failed to open {}: {err}", path.display());
            // A view asked for this slide must not apply to a later open.
            INITIAL_VIEWS.with(|views| views.borrow_mut().remove(&path));
            // A pane made for this slide (an edge drop) must not stay empty,
            // nor the window when nothing opened (a launch whose slides all
            // failed).
            let removed = {
                let mut state_guard = state.write();
                let removed = target_pane.is_some_and(|uid| state_guard.remove_empty_pane(uid));
                let nothing_shown = state_guard.panes.iter().all(|pane| pane.tabs.is_empty());
                if nothing_shown && !still_opening {
                    state_guard.create_home_tab();
                }
                removed || nothing_shown
            };
            {
                let state_guard = state.read();
                if removed {
                    ui.set_split_enabled(state_guard.split_enabled);
                    ui.set_focused_pane(state_guard.focused_pane.as_index());
                }
                // Also takes the loading indicator away.
                crate::update_tabs(&ui, &state_guard);
            }
            ui.set_status_text(SharedString::from(format!("Could not open {name}: {err}")));
            crate::host_ui::with_host(|host| {
                host.notify(
                    eov_plugin_api::protocol::NotifyLevel::Error,
                    &format!("Could not open {name}"),
                    Some(&err),
                    None,
                )
            });
            resolve_open_waiters(&path, Err(err));
            return;
        }
    };

    // Opened meanwhile under another spelling of the path.
    let existing_tab = state.read().find_tab_by_path(&path);
    let (level_count, warnings) = if let Some((pane, tab_id)) = existing_tab {
        let mut state_guard = state.write();
        state_guard.activate_tab_in_pane(pane, tab_id);
        state_guard.add_to_recent(&path);
        (None, Vec::new())
    } else {
        let (levels, warnings) = install_slide(
            &ui,
            &state,
            &tile_cache,
            path.clone(),
            target_pane,
            placement,
            opened,
        );
        (Some(levels), warnings)
    };

    {
        let mut state_guard = state.write();
        if let Some((_, tab_id)) = state_guard.find_tab_by_path(&path) {
            for uid in extra_panes.into_iter().flatten() {
                if Some(uid) != target_pane
                    && let Some(pane) = state_guard.pane_by_uid(uid)
                {
                    let duplicated = state_guard.duplicate_tab_to_pane(tab_id, pane);
                    state_guard.activate_tab_in_pane(pane, duplicated);
                }
            }
        }
    }

    {
        let state_guard = state.read();
        crate::update_tabs(&ui, &state_guard);
        update_recent_files(&ui, &state_guard);
    }
    let status = match level_count {
        Some(levels) => {
            info!(
                "Opened {} with {levels} levels",
                crate::logging::file_label(&path)
            );
            tracing::debug!("Opened {}", path.display());
            format!("Opened {name} ({levels} levels)")
        }
        None => format!("Focused {name}"),
    };
    ui.set_status_text(SharedString::from(status));
    report_view_warnings(&path, &warnings);
    request_render_loop(&render_timer, &ui.as_weak(), &state, &tile_cache);
    resolve_open_waiters(&path, opened_file_id(&state, &path));
}

/// Adds an opened slide to the state and starts loading its tiles.
/// Returns its number of levels and what of a requested view could not be
/// applied.
fn install_slide(
    ui: &crate::AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    path: PathBuf,
    target_pane: Option<u32>,
    placement: Option<Placement>,
    opened: OpenedSlide,
) -> (u32, Vec<String>) {
    let OpenedSlide { wsi, thumbnail } = opened;
    let mut state_guard = state.write();

    // The file id tags the TileCoords, keeping CPU and GPU tile caches
    // per file.
    let file_id = state_guard.allocate_file_id();
    let target_pane = target_pane
        .and_then(|uid| state_guard.pane_by_uid(uid))
        .unwrap_or(state_guard.focused_pane);
    let home_tab_to_close = state_guard
        .active_tab_id_for_pane(target_pane)
        .filter(|&active_id| state_guard.is_home_tab(active_id));

    // Viewport size of one pane (defaults until the window is laid out).
    let pane_count = state_guard.panes.len().max(1) as f64;
    let pane_gap = crate::state::PANE_GAP;
    let ui_width =
        ((ui.get_content_area_width() as f64) - pane_gap * (pane_count - 1.0)) / pane_count;
    let ui_height = ui.get_content_area_height() as f64 - 35.0;
    let viewport_width = if ui_width > 0.0 { ui_width } else { 1024.0 };
    let viewport_height = if ui_height > 0.0 { ui_height } else { 768.0 };

    let props = wsi.properties();
    // Fit and pan limits follow the scanned area of sparse slides.
    let mut viewport = ViewportState::with_content_bounds(
        viewport_width,
        viewport_height,
        props.width as f64,
        props.height as f64,
        Some(props.content_bounds()),
    );
    let mut warnings = Vec::new();
    match INITIAL_VIEWS.with(|views| views.borrow_mut().remove(&path)) {
        Some(PendingView::Saved(view)) if view.zoom.is_finite() && view.zoom > 0.0 => {
            viewport.set_orientation(view.orientation);
            viewport.set_center_zoom(view.center_x, view.center_y, view.zoom);
        }
        Some(PendingView::Requested(request)) => {
            let fit = viewport.fit_view();
            let scale = crate::launch::SlideScale::of_properties(props);
            let view = crate::launch::resolve_view(&request, &scale, fit, fit);
            viewport.set_center_zoom(view.center_x, view.center_y, view.zoom);
            warnings = view.warnings;
        }
        _ => {}
    }

    // Rendering uses a view with synthetic coarse levels for slides whose
    // pyramid is too shallow to draw at fit zoom; it shares the OpenSlide
    // handle of `wsi`, which metadata displays keep using.
    let tile_manager = Arc::new(
        TileManager::new(wsi.with_synthetic_levels(), file_id)
            .with_tile_cache(Arc::clone(tile_cache)),
    );
    tile_manager.pin_fallback_levels();
    let tile_loader = Arc::new(TileLoader::new(
        Arc::clone(&tile_manager),
        Arc::clone(tile_cache),
    ));

    // Start loading before the first render.
    let bounds = viewport.viewport.bounds();
    let best_level = tile_manager
        .wsi()
        .best_level_for_downsample(viewport.viewport.effective_downsample());
    tile_loader.set_wanted_tiles(calculate_wanted_tiles(
        &tile_manager,
        best_level,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
        1,
    ));

    let level_count = wsi.level_count();
    state_guard.add_file(NewOpenFile {
        id: file_id,
        path,
        wsi,
        tile_manager,
        tile_loader,
        viewport,
        thumbnail,
        target_pane,
        activation: placement.map_or(TabActivation::Focus, Placement::activation),
    });
    if let Some(placement) = placement
        && let Some(pane) = state_guard.pane_of_tab(file_id)
    {
        place_in_launch_order(&mut state_guard, pane, file_id, placement);
    }
    // A pane that showed its Home tab now shows the slide, whatever the
    // activation (see `AppState::add_file`).
    if let Some(home_tab_id) = home_tab_to_close {
        state_guard.close_home_tab(home_tab_id);
    }
    (level_count, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_dicom(path: &Path, uid: &str) {
        let mut bytes = vec![0u8; 128];
        bytes.extend_from_slice(b"DICM");
        let mut value = uid.as_bytes().to_vec();
        if value.len() % 2 == 1 {
            value.push(0);
        }
        bytes.extend_from_slice(&[0x20, 0x00, 0x0E, 0x00, b'U', b'I']);
        bytes.extend_from_slice(&(value.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&value);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn dicom_instances_of_one_series_are_listed_once() {
        let dir = tempfile::tempdir().unwrap();
        for (name, uid) in [("b.dcm", "1.2.3"), ("a.dcm", "1.2.3"), ("c.dcm", "1.2.44")] {
            write_dicom(&dir.path().join(name), uid);
        }
        fs::write(dir.path().join("slide10.svs"), b"").unwrap();
        fs::write(dir.path().join("slide2.svs"), b"").unwrap();
        let entries = enumerate_series_entries(dir.path());
        let names: Vec<&str> = entries.iter().map(|e| e.filename.as_str()).collect();
        assert_eq!(names.len(), 4, "{names:?}");
        assert!(
            names.contains(&"a.dcm (DICOM series, 2 files)"),
            "{names:?}"
        );
        assert!(names.contains(&"c.dcm"), "{names:?}");
        assert!(entries.iter().all(|e| e.modified.is_some()));
    }

    /// Opening a slide shows it loading in its pane without borrowing the
    /// pane models a second time (`open_file` runs inside them).
    #[test]
    fn opening_a_slide_shows_it_loading() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("a.tif");
            eov_common::test_support::write_tiled_tiff(&path, 512, 512, 256, |_, _| {
                [200, 200, 200]
            })
            .unwrap();
            let ui = crate::AppWindow::new().unwrap();
            let state = Arc::new(RwLock::new(AppState::new()));
            let tile_cache = Arc::new(TileCache::with_limits(8, 8 << 20));
            let render_timer = Rc::new(Timer::default());
            crate::open_file(&ui, &state, &tile_cache, &render_timer, path);
            assert!(ui.get_is_loading());
        });
    }
}

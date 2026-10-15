//! File operations for opening and managing WSI files
//!
//! This module contains functions for opening files and generating thumbnails.

use crate::state::{
    AppState, MinimapThumbnail, NewOpenFile, PaneId, SeriesEntry, SeriesNavigationMode,
    SeriesThumbnail,
};
use crate::tile_loader::{TileLoader, calculate_wanted_tiles};
use crate::ui_update::{update_recent_files, update_tabs};
use crate::{PaneRenderCacheEntry, PaneUiModels, PaneViewData, request_render_loop};
use eov_common::{
    ThumbnailOptions, ThumbnailRegion, TileCache, TileManager, ViewportState, WsiFile,
};
use natord::compare_ignore_case;
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

const WSI_EXTENSIONS: &[&str] = &[
    "svs", "tif", "dcm", "ndpi", "vms", "vmu", "scn", "mrxs", "tiff", "svslide", "bif", "czi",
];

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
    let mut entries = fs::read_dir(folder_path)
        .ok()
        .into_iter()
        .flat_map(|read_dir| read_dir.filter_map(Result::ok))
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() || (path.is_file() && is_supported_wsi_path(path)))
        .map(|path| {
            let normalized_path = fs::canonicalize(&path).unwrap_or(path.clone());
            let filename = normalized_path
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| normalized_path.display().to_string());
            let is_directory = normalized_path.is_dir();
            SeriesEntry {
                path: normalized_path,
                filename,
                is_directory,
                metadata_tooltip: if is_directory {
                    format!("{}\nFolder", path.display())
                } else {
                    "Loading metadata...".to_string()
                },
                objective_label: None,
                stain_label: None,
                thumbnail: None,
                thumbnail_loading: !is_directory,
            }
        })
        .collect::<Vec<_>>();

    entries.sort_by(|left, right| compare_ignore_case(&left.filename, &right.filename));
    entries
}

fn thumbnail_cache_dir() -> PathBuf {
    crate::paths::thumbnail_cache_dir()
}

fn thumbnail_cache_key(path: &Path) -> Option<String> {
    let normalized_path = fs::canonicalize(path).ok()?;
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
    let header = path.display().to_string();
    let Some(wsi) = wsi else {
        return format!("{}\nFailed to load metadata", header);
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
                "{} x {} um/px",
                eov_common::format_decimal(x),
                eov_common::format_decimal(y)
            )
        })
        .unwrap_or_else(|| "N/A".to_string());
    let vendor = properties.vendor.as_deref().unwrap_or("Unknown");
    let stain = properties.stain.as_deref().unwrap_or("N/A");

    format!(
        "{}\n{} x {} px\n{} levels\nVendor: {}\nObjective: {}\nMPP: {}\nStain: {}",
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
                let metadata_tooltip = format!("{}\nLoading metadata...", path.display());
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
    let normalized_folder = fs::canonicalize(&folder_path).unwrap_or(folder_path);
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
pub fn open_file(
    ui: &crate::AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    path: PathBuf,
    mode: OpenFileMode,
    ui_context: OpenFileUiContext<'_>,
) {
    let OpenFileUiContext {
        pane_render_cache,
        pane_ui_models,
        pane_view_model,
    } = ui_context;

    let normalized_path = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());

    let existing_tab = {
        let state_guard = state.read();
        state_guard.find_tab_by_path(&normalized_path)
    };

    if let Some((pane, tab_id)) = existing_tab {
        let mut status_text = format!("Focused {}", normalized_path.display());

        {
            let mut state_guard = state.write();
            match mode {
                OpenFileMode::ReuseExistingTab => {
                    state_guard.activate_tab_in_pane(pane, tab_id);
                }
                OpenFileMode::ForceNewTab => {
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
        }

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
        request_render_loop(render_timer, &ui.as_weak(), state, tile_cache);
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
        finish_open(ui.as_weak(), request, Err(err.to_string()));
    }
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
    state: Arc<RwLock<AppState>>,
    tile_cache: Arc<TileCache>,
    render_timer: Rc<Timer>,
}

/// What the open worker produces.
struct OpenedSlide {
    wsi: WsiFile,
    thumbnail: Option<MinimapThumbnail>,
}

thread_local! {
    /// Opens in flight, by request (UI thread only).
    static PENDING_OPENS: RefCell<HashMap<u64, PendingOpen>> = RefCell::new(HashMap::new());
    static NEXT_OPEN_ID: Cell<u64> = const { Cell::new(1) };
    /// Views to show when these slides (canonical paths) open, e.g. from
    /// a restored session.
    static INITIAL_VIEWS: RefCell<HashMap<PathBuf, crate::session::SavedView>> =
        RefCell::new(HashMap::new());
}

/// Opens `path` at `view` instead of fitting it to the pane, the next time
/// it is opened.
pub(crate) fn set_initial_view(path: &Path, view: crate::session::SavedView) {
    let path = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    INITIAL_VIEWS.with(|views| views.borrow_mut().insert(path, view));
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Opens a slide and builds its minimap thumbnail (worker thread).
fn open_slide_blocking(path: &Path) -> eov_common::Result<OpenedSlide> {
    let wsi = WsiFile::open(path)?;
    let options = ThumbnailOptions {
        region: ThumbnailRegion::ContentBounds,
        ..ThumbnailOptions::new(MINIMAP_THUMBNAIL_SIZE)
    };
    let thumbnail = match wsi.thumbnail_with(&options) {
        Ok(thumbnail) => Some(MinimapThumbnail {
            width: thumbnail.image.width as u32,
            height: thumbnail.image.height as u32,
            rgba: thumbnail.image.pixels,
            region: thumbnail.region,
        }),
        Err(err) => {
            warn!("No minimap thumbnail for {}: {err}", path.display());
            None
        }
    };
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
        return;
    };
    let still_opening = PENDING_OPENS.with(|pending| !pending.borrow().is_empty());
    ui.set_is_loading(still_opening);

    let PendingOpen {
        path,
        target_pane,
        extra_panes,
        state,
        tile_cache,
        render_timer,
    } = pending;
    let name = display_name(&path);
    let opened = match result {
        Ok(opened) => opened,
        Err(err) => {
            error!("Failed to open {}: {err}", path.display());
            // A pane made for this slide (an edge drop) must not stay empty.
            let removed = target_pane.is_some_and(|uid| state.write().remove_empty_pane(uid));
            if removed {
                let state_guard = state.read();
                ui.set_split_enabled(state_guard.split_enabled);
                ui.set_focused_pane(state_guard.focused_pane.as_index());
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
            return;
        }
    };

    // Opened meanwhile under another spelling of the path.
    let existing_tab = state.read().find_tab_by_path(&path);
    let level_count = if let Some((pane, tab_id)) = existing_tab {
        let mut state_guard = state.write();
        state_guard.activate_tab_in_pane(pane, tab_id);
        state_guard.add_to_recent(&path);
        None
    } else {
        Some(install_slide(
            &ui,
            &state,
            &tile_cache,
            path.clone(),
            target_pane,
            opened,
        ))
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
    request_render_loop(&render_timer, &ui.as_weak(), &state, &tile_cache);
}

/// Adds an opened slide to the state and starts loading its tiles.
/// Returns its number of levels.
fn install_slide(
    ui: &crate::AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    path: PathBuf,
    target_pane: Option<u32>,
    opened: OpenedSlide,
) -> u32 {
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
    let pane_gap = 6.0;
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
    if let Some(view) = INITIAL_VIEWS.with(|views| views.borrow_mut().remove(&path))
        && view.zoom.is_finite()
        && view.zoom > 0.0
    {
        viewport.set_orientation(view.orientation);
        viewport.set_center_zoom(view.center_x, view.center_y, view.zoom);
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
    });
    if let Some(home_tab_id) = home_tab_to_close {
        state_guard.close_home_tab(home_tab_id);
    }
    level_count
}

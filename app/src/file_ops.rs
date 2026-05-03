//! File operations for opening and managing WSI files
//!
//! This module contains functions for opening files and generating thumbnails.

use crate::state::{
    AppState, NewOpenFile, PaneId, SeriesEntry, SeriesNavigationMode, SeriesThumbnail,
};
use crate::tile_loader::{TileLoader, calculate_wanted_tiles};
use crate::ui_update::{update_recent_files, update_tabs};
use crate::{PaneRenderCacheEntry, PaneUiModels, PaneViewData, request_render_loop};
use common::{TileCache, TileManager, ViewportState, WsiFile};
use natord::compare_ignore_case;
use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use slint::{ComponentHandle, SharedString, Timer, VecModel};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use tracing::{error, info};

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
                thumbnail: None,
                thumbnail_loading: !is_directory,
            }
        })
        .collect::<Vec<_>>();

    entries.sort_by(|left, right| compare_ignore_case(&left.filename, &right.filename));
    entries
}

fn thumbnail_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("eov")
        .join("series_thumbnails")
}

fn thumbnail_cache_key(path: &Path) -> Option<String> {
    let normalized_path = fs::canonicalize(path).ok()?;
    let fingerprint = common::file_id::compute_fingerprint(&normalized_path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(common::file_id::hex_digest(&fingerprint).as_bytes());
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
        .map(|value| format!("{}x", common::format_decimal(value)))
        .unwrap_or_else(|| "N/A".to_string());
    let mpp = properties
        .mpp_x
        .zip(properties.mpp_y)
        .map(|(x, y)| {
            format!(
                "{} x {} um/px",
                common::format_decimal(x),
                common::format_decimal(y)
            )
        })
        .unwrap_or_else(|| "N/A".to_string());
    let vendor = properties.vendor.as_deref().unwrap_or("Unknown");

    format!(
        "{}\n{} x {} px\n{} levels\nVendor: {}\nObjective: {}\nMPP: {}",
        header,
        common::format_u64(properties.width),
        common::format_u64(properties.height),
        properties.levels.len(),
        vendor,
        objective,
        mpp,
    )
}

fn generate_thumbnail_with_dimensions(wsi: &WsiFile, max_size: u32) -> Option<SeriesThumbnail> {
    let level = wsi.level_count().saturating_sub(1);
    let level_info = wsi.level(level)?;

    let aspect = level_info.width as f64 / level_info.height as f64;
    let (thumb_w, thumb_h) = if aspect > 1.0 {
        (max_size, (max_size as f64 / aspect).max(1.0) as u32)
    } else {
        ((max_size as f64 * aspect).max(1.0) as u32, max_size)
    };

    let data = wsi
        .read_region(
            0,
            0,
            level,
            level_info.width as u32,
            level_info.height as u32,
        )
        .ok()?;

    if level_info.width <= max_size as u64 && level_info.height <= max_size as u64 {
        return Some(SeriesThumbnail {
            rgba: data,
            width: level_info.width as u32,
            height: level_info.height as u32,
        });
    }

    let image =
        image::RgbaImage::from_raw(level_info.width as u32, level_info.height as u32, data)?;
    let resized = image::imageops::resize(
        &image,
        thumb_w,
        thumb_h,
        image::imageops::FilterType::Triangle,
    );
    Some(SeriesThumbnail {
        rgba: resized.into_raw(),
        width: thumb_w,
        height: thumb_h,
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
            let thumbnail = cached_thumbnail.or_else(|| {
                opened
                    .as_ref()
                    .and_then(|wsi| generate_thumbnail_with_dimensions(wsi, 180))
            });

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

pub fn load_series_entries_async(
    ui_weak: slint::Weak<crate::AppWindow>,
    state: Arc<RwLock<AppState>>,
    folder_path: PathBuf,
    revision: u64,
) {
    std::thread::spawn(move || {
        let entries = enumerate_series_entries(&folder_path);
        let paths = entries
            .iter()
            .filter(|entry| !entry.is_directory)
            .map(|entry| entry.path.clone())
            .collect::<Vec<_>>();

        let state_handle = Arc::clone(&state);
        let update_folder = folder_path.clone();
        let _ = slint::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };

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

            if !paths.is_empty() {
                spawn_series_thumbnail_worker(
                    ui.as_weak(),
                    Arc::clone(&state_handle),
                    revision,
                    paths,
                );
            }
        });
    });
}

pub fn set_series_from_folder(
    ui: &crate::AppWindow,
    state: &Arc<RwLock<AppState>>,
    folder_path: PathBuf,
    navigation: SeriesNavigationMode,
) -> Vec<PathBuf> {
    let normalized_folder = fs::canonicalize(&folder_path).unwrap_or(folder_path);
    if navigation != SeriesNavigationMode::Initialize
        && state
            .read()
            .current_series_path()
            .is_some_and(|current| current == normalized_folder.as_path())
    {
        return state
            .read()
            .opened_series
            .as_ref()
            .map(|series| {
                series
                    .entries
                    .iter()
                    .filter(|entry| !entry.is_directory)
                    .map(|entry| entry.path.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
    }

    let entries = enumerate_series_entries(&normalized_folder);
    let paths = entries
        .iter()
        .filter(|entry| !entry.is_directory)
        .map(|entry| entry.path.clone())
        .collect::<Vec<_>>();
    let revision = {
        let mut state = state.write();
        state.set_opened_series(normalized_folder, entries, navigation)
    };

    {
        let state = state.read();
        crate::update_tabs(ui, &state);
    }

    if !paths.is_empty() {
        spawn_series_thumbnail_worker(ui.as_weak(), Arc::clone(state), revision, paths.clone());
    }

    paths
}

pub fn set_series_from_file_parent(
    ui: &crate::AppWindow,
    state: &Arc<RwLock<AppState>>,
    path: &Path,
) -> Vec<PathBuf> {
    path.parent()
        .map(|parent| {
            let navigation = if state
                .read()
                .current_series_path()
                .is_some_and(|current| current == parent)
            {
                SeriesNavigationMode::Replace
            } else {
                SeriesNavigationMode::Push
            };
            set_series_from_folder(ui, state, parent.to_path_buf(), navigation)
        })
        .unwrap_or_default()
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

        let state_guard = state.read();
        update_tabs(
            ui,
            &state_guard,
            pane_render_cache,
            pane_ui_models,
            pane_view_model,
        );
        update_recent_files(ui, &state_guard);
        let _ = crate::plugin_host::refresh_active_sidebar();
        ui.set_is_loading(false);
        ui.set_status_text(SharedString::from(status_text));
        request_render_loop(render_timer, &ui.as_weak(), state, tile_cache);
        return;
    }

    ui.set_is_loading(true);
    ui.set_status_text(SharedString::from(format!("Opening {}...", path.display())));

    match WsiFile::open(&path) {
        Ok(wsi) => {
            let id = {
                let mut state_guard = state.write();

                // Allocate the file ID up front so TileManager can tag its
                // TileCoords, ensuring CPU and GPU tile caches are per-file.
                let file_id = state_guard.allocate_file_id();

                let target_pane = if state_guard.split_enabled {
                    state_guard.focused_pane
                } else {
                    PaneId::PRIMARY
                };

                let home_tab_to_close = state_guard
                    .active_tab_id_for_pane(target_pane)
                    .filter(|&active_id| state_guard.is_home_tab(active_id));

                // Get viewport size from the focused pane (use reasonable defaults if not yet laid out)
                let pane_count = state_guard.panes.len().max(1) as f64;
                let pane_gap = 6.0;
                let ui_width = ((ui.get_content_area_width() as f64)
                    - pane_gap * (pane_count - 1.0))
                    / pane_count;
                let ui_height = ui.get_content_area_height() as f64 - 35.0;
                let viewport_width = if ui_width > 0.0 { ui_width } else { 1024.0 };
                let viewport_height = if ui_height > 0.0 { ui_height } else { 768.0 };

                let props = wsi.properties();
                let viewport = ViewportState::new(
                    viewport_width,
                    viewport_height,
                    props.width as f64,
                    props.height as f64,
                );

                // Give the tile manager its own OpenSlide handle so tile loading
                // never contends with the UI's metadata handle.
                let tile_manager_wsi = match wsi.reopen() {
                    Ok(tile_manager_wsi) => tile_manager_wsi,
                    Err(err) => {
                        error!("Failed to open dedicated tile-manager handle: {}", err);
                        ui.set_is_loading(false);
                        ui.set_status_text(SharedString::from(format!("Error: {}", err)));
                        return;
                    }
                };
                let tile_manager = Arc::new(TileManager::new(tile_manager_wsi, file_id));

                // Create background tile loader (tiles are loaded on-demand)
                let tile_loader = Arc::new(TileLoader::new(
                    Arc::clone(&tile_manager),
                    Arc::clone(tile_cache),
                ));

                // Start loading tiles immediately using the initial viewport bounds
                // This ensures tiles begin loading before the first render
                let bounds = viewport.viewport.bounds();
                let best_level =
                    wsi.best_level_for_downsample(viewport.viewport.effective_downsample());
                let initial_tiles = calculate_wanted_tiles(
                    &tile_manager,
                    best_level,
                    bounds.left,
                    bounds.top,
                    bounds.right,
                    bounds.bottom,
                    1,
                );
                tile_loader.set_wanted_tiles(initial_tiles);

                // Generate small thumbnail for minimap (lazy - only reads what's needed)
                let thumbnail = generate_thumbnail(&wsi, 150);

                let opened_file_id = state_guard.add_file(NewOpenFile {
                    id: file_id,
                    path: path.clone(),
                    wsi,
                    tile_manager,
                    tile_loader,
                    viewport,
                    thumbnail,
                });

                if let Some(home_tab_id) = home_tab_to_close {
                    state_guard.close_home_tab(home_tab_id);
                }

                opened_file_id
            };

            let level_count = {
                let state_guard = state.read();
                update_tabs(
                    ui,
                    &state_guard,
                    pane_render_cache,
                    pane_ui_models,
                    pane_view_model,
                );
                update_recent_files(ui, &state_guard);
                state_guard
                    .get_file(id)
                    .map(|f| f.wsi.level_count())
                    .unwrap_or(0)
            };

            let _ = crate::plugin_host::refresh_active_sidebar();

            ui.set_is_loading(false);
            ui.set_status_text(SharedString::from(format!(
                "Opened {} ({} levels)",
                path.file_name().unwrap_or_default().to_string_lossy(),
                level_count
            )));

            info!("Successfully opened file with {} levels", level_count);

            request_render_loop(render_timer, &ui.as_weak(), state, tile_cache);
        }
        Err(e) => {
            error!("Failed to open file: {}", e);
            ui.set_is_loading(false);
            ui.set_status_text(SharedString::from(format!("Error: {}", e)));
        }
    }
}

/// Generate a thumbnail image from the lowest resolution level of the WSI
pub fn generate_thumbnail(wsi: &WsiFile, max_size: u32) -> Option<Vec<u8>> {
    generate_thumbnail_with_dimensions(wsi, max_size).map(|thumbnail| thumbnail.rgba)
}

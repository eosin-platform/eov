// Prevent console window in addition to Slint window in Windows release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod state;
mod render;
mod tile_loader;

use anyhow::Result;
use common::{TileCache, TileManager, ViewportState, WsiFile};
use parking_lot::RwLock;
use rfd::FileDialog;
use slint::{Image, Rgba8Pixel, SharedPixelBuffer, SharedString, Timer, TimerMode, VecModel};
use state::{AppState, OpenFile, PaneId};
use tile_loader::{TileLoader, calculate_wanted_tiles};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, trace, warn};

// Debug macro - set to no-op for release, enable for debugging
#[allow(unused_macros)]
macro_rules! dbg_print {
    ($($arg:tt)*) => {{
        // Uncomment for verbose debugging:
        // eprintln!($($arg)*);
        // let _ = std::io::stderr().flush();
    }};
}

slint::include_modules!();

/// Frame rate for viewport updates
const TARGET_FPS: f64 = 30.0;
const FRAME_DURATION_MS: u64 = (1000.0 / TARGET_FPS) as u64;

fn main() -> Result<()> {
    // Initialize logging - RUST_LOG env controls level
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env(),
        )
        .init();

    info!("Starting EosMol WSI Viewer");

    // Parse command-line arguments
    let args: Vec<String> = std::env::args().skip(1).collect();
    let debug_mode = args.iter().any(|a| a == "--debug" || a == "-d");
    let files_to_open: Vec<PathBuf> = args
        .into_iter()
        .filter(|a| a != "--debug" && a != "-d")
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .collect();
    
    if debug_mode {
        info!("Debug mode enabled - FPS overlay will be shown");
    }
    
    if !files_to_open.is_empty() {
        info!("Opening {} file(s) from command line", files_to_open.len());
    }

    // Create application state
    let state = Arc::new(RwLock::new(AppState::new(debug_mode)));
    let tile_cache = Arc::new(TileCache::new());

    // Create UI
    let ui = AppWindow::new()?;
    let ui_weak = ui.as_weak();

    // Set up callbacks
    setup_callbacks(&ui, Arc::clone(&state), Arc::clone(&tile_cache));
    
    // Set debug mode on UI
    ui.set_debug_mode(debug_mode);

    // Open files from command line
    for path in files_to_open {
        open_file(&ui, &state, &tile_cache, path);
    }
    
    dbg_print!("[MAIN] Files opened, starting render timer");

    // Set up render timer
    let render_timer = Timer::default();
    {
        let ui_weak = ui_weak.clone();
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        
        render_timer.start(
            TimerMode::Repeated,
            Duration::from_millis(FRAME_DURATION_MS),
            move || {
                if let Some(ui) = ui_weak.upgrade() {
                    update_and_render(&ui, &state, &tile_cache);
                }
            },
        );
    }
    
    dbg_print!("[MAIN] Timer started, running UI");

    // Run application
    ui.run()?;

    info!("Application shutting down");
    Ok(())
}

fn setup_callbacks(ui: &AppWindow, state: Arc<RwLock<AppState>>, tile_cache: Arc<TileCache>) {
    let ui_weak = ui.as_weak();
    
    // Open file callback
    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let ui_weak = ui_weak.clone();
        
        ui.on_open_file_requested(move || {
            let ui = ui_weak.upgrade().unwrap();
            
            // Show file dialog
            let dialog = FileDialog::new()
                .add_filter("WSI Files", &["svs", "tif", "tiff", "ndpi", "vms", "vmu", "scn", "mrxs", "bif"])
                .add_filter("All Files", &["*"]);
            
            if let Some(path) = dialog.pick_file() {
                open_file(&ui, &state, &tile_cache, path);
            }
        });
    }

    // Tab activated callback
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        
        ui.on_tab_activated(move |id| {
            let mut state = state.write();
            state.activate_file(id);
            
            if let Some(ui) = ui_weak.upgrade() {
                update_tabs(&ui, &state);
            }
        });
    }

    // Tab close callback
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        
        ui.on_tab_close_requested(move |id| {
            let mut state = state.write();
            state.close_file(id);
            
            if let Some(ui) = ui_weak.upgrade() {
                update_tabs(&ui, &state);
            }
        });
    }

    // Close other tabs
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        
        ui.on_close_other_tabs(move |keep_id| {
            let mut state = state.write();
            let ids_to_close: Vec<i32> = state.open_files
                .iter()
                .map(|f| f.id)
                .filter(|&id| id != keep_id)
                .collect();
            
            for id in ids_to_close {
                state.close_file(id);
            }
            
            if let Some(ui) = ui_weak.upgrade() {
                update_tabs(&ui, &state);
            }
        });
    }

    // Close tabs to the right
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        
        ui.on_close_tabs_to_right(move |from_id| {
            let mut state = state.write();
            let mut found = false;
            let ids_to_close: Vec<i32> = state.open_files
                .iter()
                .filter_map(|f| {
                    if f.id == from_id {
                        found = true;
                        None
                    } else if found {
                        Some(f.id)
                    } else {
                        None
                    }
                })
                .collect();
            
            for id in ids_to_close {
                state.close_file(id);
            }
            
            if let Some(ui) = ui_weak.upgrade() {
                update_tabs(&ui, &state);
            }
        });
    }

    // Close all tabs
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        
        ui.on_close_all_tabs(move || {
            let mut state = state.write();
            state.open_files.clear();
            state.active_file_id = None;
            
            if let Some(ui) = ui_weak.upgrade() {
                update_tabs(&ui, &state);
            }
        });
    }

    // Open containing folder
    {
        let state = Arc::clone(&state);
        
        ui.on_open_containing_folder(move |id| {
            let state = state.read();
            if let Some(file) = state.get_file(id) {
                if let Some(parent) = file.path.parent() {
                    if let Err(e) = open::that(parent) {
                        error!("Failed to open folder: {}", e);
                    }
                }
            }
        });
    }

    // Copy path
    {
        let state = Arc::clone(&state);
        
        ui.on_copy_path(move |id| {
            let state = state.read();
            if let Some(file) = state.get_file(id) {
                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                    let _ = clipboard.set_text(file.path.display().to_string());
                }
            }
        });
    }

    // Split right - toggle split view
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        
        ui.on_split_right(move |_id| {
            let mut state = state.write();
            if state.split_enabled {
                // Close split
                state.disable_split();
                info!("Split view disabled");
            } else {
                // Enable split
                state.enable_split();
                info!("Split view enabled");
            }
            
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_split_enabled(state.split_enabled);
                ui.set_focused_pane(match state.focused_pane {
                    PaneId::Primary => 0,
                    PaneId::Secondary => 1,
                });
            }
        });
    }

    // Pane focused callback
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        
        ui.on_pane_focused(move |pane| {
            let mut state = state.write();
            state.focused_pane = if pane == 0 { PaneId::Primary } else { PaneId::Secondary };
            
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_focused_pane(pane);
            }
        });
    }

    // Split position changed callback
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        
        ui.on_split_position_changed(move |pos| {
            let mut state = state.write();
            state.split_position = pos;
            
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_split_position(pos);
            }
        });
    }

    // Viewport pan
    {
        let state = Arc::clone(&state);
        
        ui.on_viewport_pan(move |dx, dy| {
            let mut state = state.write();
            if let Some(viewport) = state.active_viewport_mut() {
                viewport.viewport.pan(dx as f64, dy as f64);
            }
        });
    }

    // Viewport start pan
    {
        let state = Arc::clone(&state);
        
        ui.on_viewport_start_pan(move |x, y| {
            let mut state = state.write();
            if let Some(viewport) = state.active_viewport_mut() {
                viewport.start_drag(x as f64, y as f64);
            }
        });
    }

    // Viewport end pan
    {
        let state = Arc::clone(&state);
        
        ui.on_viewport_end_pan(move || {
            let mut state = state.write();
            if let Some(viewport) = state.active_viewport_mut() {
                viewport.end_drag();
            }
        });
    }

    // Viewport zoom
    {
        let state = Arc::clone(&state);
        
        ui.on_viewport_zoom(move |factor, x, y| {
            let mut state = state.write();
            if let Some(viewport) = state.active_viewport_mut() {
                viewport.zoom_at(factor as f64, x as f64, y as f64);
            }
        });
    }

    // Fit to view callback
    {
        let state = Arc::clone(&state);
        
        ui.on_viewport_fit_to_view(move || {
            let mut state = state.write();
            if let Some(viewport) = state.active_viewport_mut() {
                viewport.fit_to_view();
            }
        });
    }

    // Minimap navigation callback
    {
        let state = Arc::clone(&state);
        
        ui.on_minimap_navigate(move |nx, ny| {
            info!("Minimap navigate called: nx={}, ny={}", nx, ny);
            let mut state = state.write();
            let active_id = state.active_file_id;
            let focused_pane = state.focused_pane;
            let split_enabled = state.split_enabled;
            
            if let Some(file) = state.open_files.iter_mut().find(|f| Some(f.id) == active_id) {
                // Clamp normalized coordinates to 0-1 range
                let nx = (nx as f64).clamp(0.0, 1.0);
                let ny = (ny as f64).clamp(0.0, 1.0);
                
                // Get the viewport to update based on focused pane
                let viewport_state = if split_enabled && focused_pane == PaneId::Secondary {
                    file.secondary_viewport.as_mut()
                } else {
                    Some(&mut file.viewport)
                };
                
                if let Some(vs) = viewport_state {
                    // Stop any existing movement (important!)
                    vs.stop();
                    
                    // Convert normalized (0-1) coordinates to image coordinates
                    // nx, ny represent where the viewport CENTER should be
                    let vp = &mut vs.viewport;
                    let new_x = nx * vp.image_width;
                    let new_y = ny * vp.image_height;
                    info!("Setting viewport center: ({}, {}) -> ({}, {})", vp.center.x, vp.center.y, new_x, new_y);
                    vp.center.x = new_x;
                    vp.center.y = new_y;
                }
            }
        });
    }

    // File dropped callback
    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let ui_weak = ui.as_weak();
        
        ui.on_file_dropped(move |path_str| {
            // Parse dropped path - may include file:// prefix or be a path list
            let path_string = path_str.to_string();
            let paths: Vec<&str> = path_string
                .lines()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            
            for path_str in paths {
                let path = if path_str.starts_with("file://") {
                    PathBuf::from(path_str.strip_prefix("file://").unwrap_or(path_str))
                } else {
                    PathBuf::from(path_str)
                };
                
                if path.exists() {
                    if let Some(ui) = ui_weak.upgrade() {
                        open_file(&ui, &state, &tile_cache, path);
                    }
                }
            }
        });
    }
    
    // Tool selected callback
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        
        ui.on_tool_selected(move |tool_type| {
            let mut state = state.write();
            let tool = match tool_type {
                ToolType::Navigate => state::Tool::Navigate,
                ToolType::RegionOfInterest => state::Tool::RegionOfInterest,
                ToolType::MeasureDistance => state::Tool::MeasureDistance,
            };
            state.set_tool(tool);
            
            if let Some(ui) = ui_weak.upgrade() {
                update_tool_state(&ui, &state);
            }
        });
    }
    
    // Tool mouse events
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        
        ui.on_viewport_tool_mouse_down(move |x, y| {
            let mut state = state.write();
            handle_tool_mouse_down(&mut state, x as f64, y as f64);
            
            if let Some(ui) = ui_weak.upgrade() {
                update_tool_overlays(&ui, &state);
            }
        });
    }
    
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        
        ui.on_viewport_tool_mouse_move(move |x, y| {
            let mut state = state.write();
            handle_tool_mouse_move(&mut state, x as f64, y as f64);
            
            if let Some(ui) = ui_weak.upgrade() {
                update_tool_overlays(&ui, &state);
            }
        });
    }
    
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        
        ui.on_viewport_tool_mouse_up(move |x, y| {
            let mut state = state.write();
            handle_tool_mouse_up(&mut state, x as f64, y as f64);
            
            if let Some(ui) = ui_weak.upgrade() {
                update_tool_overlays(&ui, &state);
            }
        });
    }
    
    // Cancel tool callback
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        
        ui.on_cancel_tool(move || {
            let mut state = state.write();
            state.cancel_tool();
            
            if let Some(ui) = ui_weak.upgrade() {
                update_tool_state(&ui, &state);
                update_tool_overlays(&ui, &state);
            }
        });
    }
}

fn open_file(ui: &AppWindow, state: &Arc<RwLock<AppState>>, tile_cache: &Arc<TileCache>, path: PathBuf) {
    dbg_print!("[OPEN] Opening file: {}", path.display());
    ui.set_is_loading(true);
    ui.set_status_text(SharedString::from(format!("Opening {}...", path.display())));

    match WsiFile::open(&path) {
        Ok(wsi) => {
            dbg_print!("[OPEN] File opened successfully");
            let id = {
                let mut state_guard = state.write();
                
                // Get viewport size from UI
                let viewport_width = 1024.0; // Default, will be updated on first render
                let viewport_height = 768.0;
                
                let props = wsi.properties();
                let viewport = ViewportState::new(
                    viewport_width,
                    viewport_height,
                    props.width as f64,
                    props.height as f64,
                );
                
                // Create tile manager and wrap in Arc for shared use
                let tile_manager = Arc::new(TileManager::new(wsi.clone()));
                
                // Create background tile loader (tiles are loaded on-demand)
                let tile_loader = TileLoader::new(Arc::clone(&tile_manager), Arc::clone(tile_cache));
                
                // Generate small thumbnail for minimap (lazy - only reads what's needed)
                let thumbnail = generate_thumbnail(&wsi, 150);
                
                state_guard.add_file(path.clone(), wsi, tile_manager, tile_loader, viewport, thumbnail)
            };
            
            let state_guard = state.read();
            update_tabs(ui, &state_guard);
            
            ui.set_is_loading(false);
            ui.set_status_text(SharedString::from(format!(
                "Opened {} ({} levels)",
                path.file_name().unwrap_or_default().to_string_lossy(),
                state_guard.get_file(id).map(|f| f.wsi.level_count()).unwrap_or(0)
            )));
            
            info!("Successfully opened file with {} levels", 
                state_guard.get_file(id).map(|f| f.wsi.level_count()).unwrap_or(0));
        }
        Err(e) => {
            error!("Failed to open file: {}", e);
            ui.set_is_loading(false);
            ui.set_status_text(SharedString::from(format!("Error: {}", e)));
        }
    }
}

fn generate_thumbnail(wsi: &WsiFile, max_size: u32) -> Option<Vec<u8>> {
    // Use the lowest resolution level for thumbnail
    let level = wsi.level_count().saturating_sub(1);
    let level_info = wsi.level(level)?;
    
    // Calculate thumbnail dimensions maintaining aspect ratio
    let aspect = level_info.width as f64 / level_info.height as f64;
    let (thumb_w, thumb_h) = if aspect > 1.0 {
        (max_size, (max_size as f64 / aspect).max(1.0) as u32)
    } else {
        ((max_size as f64 * aspect).max(1.0) as u32, max_size)
    };
    
    // If level is small enough, read it directly
    if level_info.width <= max_size as u64 * 2 && level_info.height <= max_size as u64 * 2 {
        match wsi.read_region(0, 0, level, level_info.width as u32, level_info.height as u32) {
            Ok(data) => {
                if level_info.width <= max_size as u64 && level_info.height <= max_size as u64 {
                    return Some(data);
                }
                // Resize to thumbnail size
                if let Some(img) = image::RgbaImage::from_raw(
                    level_info.width as u32,
                    level_info.height as u32,
                    data,
                ) {
                    let resized = image::imageops::resize(
                        &img, thumb_w, thumb_h,
                        image::imageops::FilterType::Triangle,
                    );
                    return Some(resized.into_raw());
                }
            }
            Err(e) => {
                warn!("Failed to generate thumbnail: {}", e);
            }
        }
    }
    
    // Level is too large - read the full level anyway (it's still the smallest level)
    // and resize down to thumbnail size
    match wsi.read_region(0, 0, level, level_info.width as u32, level_info.height as u32) {
        Ok(data) => {
            if let Some(img) = image::RgbaImage::from_raw(
                level_info.width as u32, 
                level_info.height as u32, 
                data
            ) {
                let resized = image::imageops::resize(
                    &img, thumb_w, thumb_h,
                    image::imageops::FilterType::Triangle,
                );
                Some(resized.into_raw())
            } else {
                None
            }
        }
        Err(e) => {
            warn!("Failed to generate thumbnail: {}", e);
            None
        }
    }
}

fn update_tabs(ui: &AppWindow, state: &AppState) {
    let tabs: Vec<TabData> = state.open_files
        .iter()
        .map(|f| TabData {
            id: f.id,
            title: SharedString::from(f.filename.clone()),
            path: SharedString::from(f.path.display().to_string()),
            is_modified: false,
            is_active: Some(f.id) == state.active_file_id,
        })
        .collect();
    
    let model = Rc::new(VecModel::from(tabs));
    ui.set_tabs(model.into());
    ui.set_active_tab_id(state.active_file_id.unwrap_or(-1));
}

fn update_and_render(ui: &AppWindow, state: &Arc<RwLock<AppState>>, tile_cache: &Arc<TileCache>) {
    // Debug: track where we are
    dbg_print!("[UPDATE] start");
    
    // Check 1: Can we acquire state lock?
    let mut state = state.write();
    dbg_print!("[UPDATE] got lock");
    
    // Update FPS counter
    state.update_fps();
    let fps = state.current_fps;
    ui.set_fps(fps);
    
    let Some(file_id) = state.active_file_id else {
        return;
    };
    
    let split_enabled = state.split_enabled;
    let split_position = state.split_position;
    
    // Capture tool state for ROI calculation (before borrowing file mutably)
    let tool_state = state.tool_state;
    let candidate_point = state.candidate_point;
    let current_tool = state.current_tool;
    
    // Update ant offset for marching ants animation (0.5 pixels per frame at 60fps = 30 pixels/sec)
    state.ant_offset = (state.ant_offset + 0.5) % 16.0;
    let ant_offset = state.ant_offset;
    
    let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
        return;
    };

    // Get viewport dimensions from window
    let window_width = ui.get_viewport_width() as f64;
    let window_height = ui.get_viewport_height() as f64;
    
    // Calculate primary viewport size
    let primary_width = if split_enabled {
        (window_width * split_position as f64 - 3.0).max(100.0)
    } else {
        window_width.max(100.0)
    };
    let primary_height = window_height.max(100.0);
    
    // Update primary viewport physics
    let _needs_redraw = file.viewport.update();
    file.viewport.set_size(primary_width, primary_height);
    
    // Update primary viewport info
    let vp = &file.viewport.viewport;
    ui.set_viewport_info(ViewportInfo {
        center_x: vp.center.x as f32,
        center_y: vp.center.y as f32,
        zoom: vp.zoom as f32,
        image_width: vp.image_width as f32,
        image_height: vp.image_height as f32,
        level: file.wsi.best_level_for_downsample(vp.effective_downsample()) as i32,
    });
    
    // Update primary minimap
    let rect = vp.minimap_rect();
    ui.set_minimap_rect(MinimapRect {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
    });
    
    // Set thumbnail for minimap (ONCE only - avoid per-frame buffer allocation)
    if !file.thumbnail_set {
        trace!("Setting thumbnail");
        if let Some(ref thumb_data) = file.thumbnail {
            let level = file.wsi.level_count().saturating_sub(1);
            if let Some(level_info) = file.wsi.level(level) {
                let aspect = level_info.width as f64 / level_info.height as f64;
                let (w, h) = if aspect > 1.0 {
                    (150u32, (150.0 / aspect) as u32)
                } else {
                    ((150.0 * aspect) as u32, 150u32)
                };
                
                if let Some(buffer) = create_image_buffer(thumb_data, w, h) {
                    ui.set_minimap_thumbnail(Image::from_rgba8(buffer));
                    file.thumbnail_set = true;
                }
            }
        }
        trace!("Thumbnail set");
    }
    
    // Render primary viewport
    dbg_print!("[UPDATE] calling render");
    render_viewport_to_buffer(ui, file, tile_cache, true);
    dbg_print!("[UPDATE] render done");
    
    // Update ROI overlay (must be done every frame for proper tracking)
    // Re-borrow viewport since render_viewport_to_buffer takes &mut
    let vp = &file.viewport.viewport;
    let bounds = vp.bounds();
    
    // Check for in-progress ROI (during drag) first, then committed ROI
    let roi_to_display = if current_tool == state::Tool::RegionOfInterest {
        if let state::ToolInteractionState::Dragging(start) = tool_state {
            if let Some(end) = candidate_point {
                // Calculate in-progress ROI from drag points
                Some(state::RegionOfInterest::from_points(start, end))
            } else {
                file.roi
            }
        } else {
            file.roi
        }
    } else {
        file.roi
    };
    
    if let Some(roi) = roi_to_display {
        let screen_x = (roi.x - bounds.left) * vp.zoom;
        let screen_y = (roi.y - bounds.top) * vp.zoom;
        let screen_w = roi.width * vp.zoom;
        let screen_h = roi.height * vp.zoom;
        
        ui.set_roi_rect(ROIRect {
            x: screen_x as f32,
            y: screen_y as f32,
            width: screen_w as f32,
            height: screen_h as f32,
            visible: true,
            ant_offset,
        });
    } else {
        ui.set_roi_rect(ROIRect {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            visible: false,
            ant_offset,
        });
    }
    
    // Handle secondary viewport if split is enabled
    if split_enabled {
        if let Some(ref mut secondary) = file.secondary_viewport {
            // Update secondary viewport physics
            let secondary_width = (window_width * (1.0 - split_position as f64) - 3.0).max(100.0);
            let secondary_height = window_height.max(100.0);
            
            secondary.update();
            secondary.set_size(secondary_width, secondary_height);
            
            // Update secondary viewport info
            let vp2 = &secondary.viewport;
            ui.set_secondary_viewport_info(ViewportInfo {
                center_x: vp2.center.x as f32,
                center_y: vp2.center.y as f32,
                zoom: vp2.zoom as f32,
                image_width: vp2.image_width as f32,
                image_height: vp2.image_height as f32,
                level: file.wsi.best_level_for_downsample(vp2.effective_downsample()) as i32,
            });
            
            // Update secondary minimap
            let rect2 = vp2.minimap_rect();
            ui.set_secondary_minimap_rect(MinimapRect {
                x: rect2.x,
                y: rect2.y,
                width: rect2.width,
                height: rect2.height,
            });
            
            // Render secondary viewport
            render_secondary_viewport(ui, file, tile_cache);
        }
    }
}

fn render_viewport_to_buffer(ui: &AppWindow, file: &mut OpenFile, tile_cache: &Arc<TileCache>, _is_primary: bool) {
    dbg_print!("[RENDER] start");
    let vp = &file.viewport.viewport;
    
    // Copy viewport values for dirty tracking
    let vp_zoom = vp.zoom;
    let vp_center_x = vp.center.x;
    let vp_center_y = vp.center.y;
    let vp_width = vp.width;
    let vp_height = vp.height;
    
    trace!("render: viewport {}x{} zoom={}", vp_width, vp_height, vp_zoom);
    
    // Is this the first frame? (must render at least once)
    let is_first_frame = file.frame_count == 0;
    
    // Check if viewport changed since last render
    let viewport_changed = 
        (file.last_render_zoom - vp_zoom).abs() > 0.0001 ||
        (file.last_render_center_x - vp_center_x).abs() > 0.1 ||
        (file.last_render_center_y - vp_center_y).abs() > 0.1 ||
        (file.last_render_width - vp_width).abs() > 0.1 ||
        (file.last_render_height - vp_height).abs() > 0.1;
    
    // Determine best level for current zoom
    let level = file.wsi.best_level_for_downsample(vp.effective_downsample());
    let tile_size = file.tile_manager.tile_size();
    
    trace!("render: level={} tile_size={}", level, tile_size);
    
    // Get visible tiles using viewport bounds
    let bounds = vp.bounds();
    trace!("render: bounds left={} top={} right={} bottom={}", bounds.left, bounds.top, bounds.right, bounds.bottom);
    
    let visible_tiles = file.tile_manager.visible_tiles(
        level,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
    );
    
    trace!("render: visible_tiles count={}", visible_tiles.len());
    
    // Limit visible tiles to a reasonable maximum (typical viewport needs ~200-500 tiles when zoomed in)
    let visible_tiles: Vec<_> = visible_tiles.into_iter().take(500).collect();
    
    // Count how many tiles are in cache
    let cached_count = visible_tiles.iter().filter(|c| tile_cache.contains(c)).count() as u32;
    let new_tiles_loaded = cached_count > file.tiles_loaded_since_render;
    
    trace!("render: cached={} new_tiles={}", cached_count, new_tiles_loaded);
    
    // Update wanted tiles for background loading (always do this)
    let wanted = calculate_wanted_tiles(
        &file.tile_manager,
        level,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
    );
    
    trace!("render: wanted tiles={}", wanted.len());
    
    file.tile_loader.set_wanted_tiles(wanted);
    
    trace!("render: set_wanted_tiles done");
    
    // Throttle tile update renders to max 4x per second (250ms between renders)
    let time_since_render = file.last_render_time.elapsed();
    let throttle_ok = time_since_render.as_millis() >= 250;
    
    // Skip rendering if nothing changed (but always render first frame and viewport changes)
    if !is_first_frame && !viewport_changed && (!new_tiles_loaded || !throttle_ok) {
        trace!("render: skipping (no changes)");
        return;
    }
    
    dbg_print!("[RENDER] proceeding with render");
    
    // Update tracking state
    file.frame_count += 1;
    file.last_render_time = std::time::Instant::now();
    file.last_render_zoom = vp_zoom;
    file.last_render_center_x = vp_center_x;
    file.last_render_center_y = vp_center_y;
    
    dbg_print!("[RENDER] tracking state updated");
    file.last_render_width = vp_width;
    file.last_render_height = vp_height;
    file.tiles_loaded_since_render = cached_count;
    
    // Use persistent buffer for rendering
    let render_width = vp_width as u32;
    let render_height = (vp_height - 24.0).max(1.0) as u32; // Account for status bar, ensure > 0
    
    dbg_print!("[RENDER] buffer {}x{}", render_width, render_height);
    
    if render_width == 0 || render_height == 0 {
        return;
    }
    
    let buffer_size = (render_width * render_height * 4) as usize;
    dbg_print!("[RENDER] buffer_size={}", buffer_size);
    
    // Resize buffer if needed (only reallocates if capacity insufficient)
    if file.render_buffer.len() != buffer_size {
        dbg_print!("[RENDER] resizing buffer to {}", buffer_size);
        file.render_buffer.resize(buffer_size, 0);
    }
    
    dbg_print!("[RENDER] clearing buffer len={}", file.render_buffer.len());
    
    // Fast clear to dark background (30, 30, 30, 255) using chunks
    let pattern = [30u8, 30, 30, 255];
    for chunk in file.render_buffer.chunks_exact_mut(4) {
        chunk.copy_from_slice(&pattern);
    }
    
    dbg_print!("[RENDER] buffer cleared");
    
    let buffer = &mut file.render_buffer[..];
    dbg_print!("[RENDER] got buffer slice");
    
    let level_info = match file.wsi.level(level) {
        Some(info) => info,
        None => return,
    };
    dbg_print!("[RENDER] got level_info: downsample={}", level_info.downsample);
    
    // First pass: render lower-resolution fallback tiles
    // These provide immediate visual feedback while high-res tiles load
    // We render ALL cached low-res tiles that overlap the viewport, then high-res tiles overdraw
    let mut _fallbacks_rendered = 0;
    let level_count = file.wsi.level_count();
    
    // Iterate from lowest resolution (highest level index) to one above current
    // This ensures higher quality fallbacks draw on top of lower quality ones
    for fallback_level in (0..level_count).rev() {
        if fallback_level <= level {
            continue; // Skip current and higher resolution levels
        }
        
        let fallback_level_info = match file.wsi.level(fallback_level) {
            Some(info) => info,
            None => continue,
        };
        
        // Get visible tiles at this fallback level
        let fallback_tiles = file.tile_manager.visible_tiles(
            fallback_level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
        );
        
        for fb_coord in fallback_tiles.iter().take(100) {
            let Some(fallback_tile) = tile_cache.get(fb_coord) else {
                continue;
            };
            
            // Calculate screen position for this fallback tile
            // Each fallback tile covers (tile_size * fallback_downsample) pixels in image space
            let fb_tile_x = fb_coord.x as f64 * tile_size as f64;
            let fb_tile_y = fb_coord.y as f64 * tile_size as f64;
            
            // Convert to image coordinates (level 0) and then to screen
            let fb_image_x = fb_tile_x * fallback_level_info.downsample;
            let fb_image_y = fb_tile_y * fallback_level_info.downsample;
            let fb_image_w = fallback_tile.width as f64 * fallback_level_info.downsample;
            let fb_image_h = fallback_tile.height as f64 * fallback_level_info.downsample;
            
            let screen_x = ((fb_image_x - bounds.left) * vp.zoom).floor() as i32;
            let screen_y = ((fb_image_y - bounds.top) * vp.zoom).floor() as i32;
            let screen_w = (fb_image_w * vp.zoom).ceil() as i32;
            let screen_h = (fb_image_h * vp.zoom).ceil() as i32;
            
            if screen_w <= 0 || screen_h <= 0 {
                continue;
            }
            
            // Render the entire fallback tile at its correct screen position
            blit_tile(
                buffer,
                render_width,
                render_height,
                &fallback_tile.data,
                fallback_tile.width,
                fallback_tile.height,
                screen_x,
                screen_y,
                screen_w,
                screen_h,
            );
            _fallbacks_rendered += 1;
        }
    }
    
    dbg_print!("[RENDER] first pass: rendered {} fallbacks", _fallbacks_rendered);
    
    dbg_print!("[RENDER] starting second pass (high-res tiles), count={}", visible_tiles.len());
    
    // Second pass: render high-res tiles that are available (these overdraw fallbacks)
    let mut _tiles_blitted = 0;
    for (_i, coord) in visible_tiles.iter().enumerate() {
        dbg_print!("[RENDER] tile {} of {}: {:?}", _i, visible_tiles.len(), coord);
        let Some(tile_data) = tile_cache.get(coord) else {
            dbg_print!("[RENDER] tile {:?} not in cache, skip", coord);
            continue;
        };
        dbg_print!("[RENDER] tile {:?} in cache, blitting", coord);
        
        let tile_x = coord.x as f64 * tile_size as f64;
        let tile_y = coord.y as f64 * tile_size as f64;
        
        let screen_x = ((tile_x * level_info.downsample - bounds.left) * vp.zoom).floor() as i32;
        let screen_y = ((tile_y * level_info.downsample - bounds.top) * vp.zoom).floor() as i32;
        let screen_x_end = (((tile_x + tile_data.width as f64) * level_info.downsample - bounds.left) * vp.zoom).ceil() as i32;
        let screen_y_end = (((tile_y + tile_data.height as f64) * level_info.downsample - bounds.top) * vp.zoom).ceil() as i32;
        let screen_w = screen_x_end - screen_x;
        let screen_h = screen_y_end - screen_y;
        
        dbg_print!("[RENDER] blit tile {:?}: screen=({},{}) size={}x{}", coord, screen_x, screen_y, screen_w, screen_h);
        
        blit_tile(
            buffer,
            render_width,
            render_height,
            &tile_data.data,
            tile_data.width,
            tile_data.height,
            screen_x,
            screen_y,
            screen_w,
            screen_h,
        );
        _tiles_blitted += 1;
        dbg_print!("[RENDER] blit done, total={}", _tiles_blitted);
    }
    
    dbg_print!("[RENDER] blitted {} tiles total", _tiles_blitted);
    
    dbg_print!("[RENDER] creating image buffer");
    
    // Create image from buffer and update UI
    if let Some(pixel_buffer) = create_image_buffer(buffer, render_width, render_height) {
        dbg_print!("[RENDER] updating UI");
        ui.set_viewport_content(Image::from_rgba8(pixel_buffer));
    }
    
    dbg_print!("[RENDER] done");
}

fn render_secondary_viewport(ui: &AppWindow, file: &mut OpenFile, tile_cache: &Arc<TileCache>) {
    let Some(ref secondary) = file.secondary_viewport else {
        return;
    };
    
    let vp = &secondary.viewport;
    
    // Determine best level for current zoom
    let level = file.wsi.best_level_for_downsample(vp.effective_downsample());
    let tile_size = file.tile_manager.tile_size();
    
    // Get visible tiles using viewport bounds
    let bounds = vp.bounds();
    let visible_tiles = file.tile_manager.visible_tiles(
        level,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
    );
    
    // Limit visible tiles to a reasonable maximum
    let visible_tiles: Vec<_> = visible_tiles.into_iter().take(500).collect();
    
    // Update wanted tiles for background loading (secondary viewport)
    let wanted = calculate_wanted_tiles(
        &file.tile_manager,
        level,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
    );
    file.tile_loader.set_wanted_tiles(wanted);
    
    // Create a buffer for rendering
    let render_width = vp.width as u32;
    let render_height = vp.height.max(1.0) as u32;
    
    if render_width == 0 || render_height == 0 {
        return;
    }
    
    let mut buffer = vec![30u8; (render_width * render_height * 4) as usize]; // Dark background
    
    // Set alpha channel
    for i in (3..buffer.len()).step_by(4) {
        buffer[i] = 255;
    }
    
    let level_info = match file.wsi.level(level) {
        Some(info) => info,
        None => return,
    };
    
    let level_count = file.wsi.level_count();
    
    // First pass: render fallback tiles (whole tiles at lower resolutions)
    // Iterate from lowest resolution to highest, so better quality draws on top
    for fallback_level in (0..level_count).rev() {
        if fallback_level <= level {
            continue; // Skip current and higher resolution levels
        }
        
        let fallback_level_info = match file.wsi.level(fallback_level) {
            Some(info) => info,
            None => continue,
        };
        
        // Get visible tiles at this fallback level
        let fallback_tiles = file.tile_manager.visible_tiles(
            fallback_level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
        );
        
        for fb_coord in fallback_tiles.iter().take(100) {
            let Some(fallback_tile) = tile_cache.get(fb_coord) else {
                continue;
            };
            
            // Calculate screen position for this fallback tile
            let fb_tile_x = fb_coord.x as f64 * tile_size as f64;
            let fb_tile_y = fb_coord.y as f64 * tile_size as f64;
            
            // Convert to image coordinates (level 0) and then to screen
            let fb_image_x = fb_tile_x * fallback_level_info.downsample;
            let fb_image_y = fb_tile_y * fallback_level_info.downsample;
            let fb_image_w = fallback_tile.width as f64 * fallback_level_info.downsample;
            let fb_image_h = fallback_tile.height as f64 * fallback_level_info.downsample;
            
            let screen_x = ((fb_image_x - bounds.left) * vp.zoom).floor() as i32;
            let screen_y = ((fb_image_y - bounds.top) * vp.zoom).floor() as i32;
            let screen_w = (fb_image_w * vp.zoom).ceil() as i32;
            let screen_h = (fb_image_h * vp.zoom).ceil() as i32;
            
            if screen_w <= 0 || screen_h <= 0 {
                continue;
            }
            
            blit_tile(
                &mut buffer,
                render_width,
                render_height,
                &fallback_tile.data,
                fallback_tile.width,
                fallback_tile.height,
                screen_x,
                screen_y,
                screen_w,
                screen_h,
            );
        }
    }
    
    // Second pass: render high-res tiles that are available
    for coord in &visible_tiles {
        let Some(tile_data) = tile_cache.get(coord) else {
            continue;
        };
        
        let tile_x = coord.x as f64 * tile_size as f64;
        let tile_y = coord.y as f64 * tile_size as f64;
        
        let screen_x = ((tile_x * level_info.downsample - bounds.left) * vp.zoom).floor() as i32;
        let screen_y = ((tile_y * level_info.downsample - bounds.top) * vp.zoom).floor() as i32;
        let screen_x_end = (((tile_x + tile_data.width as f64) * level_info.downsample - bounds.left) * vp.zoom).ceil() as i32;
        let screen_y_end = (((tile_y + tile_data.height as f64) * level_info.downsample - bounds.top) * vp.zoom).ceil() as i32;
        let screen_w = screen_x_end - screen_x;
        let screen_h = screen_y_end - screen_y;
        
        blit_tile(
            &mut buffer,
            render_width,
            render_height,
            &tile_data.data,
            tile_data.width,
            tile_data.height,
            screen_x,
            screen_y,
            screen_w,
            screen_h,
        );
    }
    
    // Create image from buffer
    if let Some(pixel_buffer) = create_image_buffer(&buffer, render_width, render_height) {
        ui.set_secondary_viewport_content(Image::from_rgba8(pixel_buffer));
    }
}

fn blit_tile(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    src: &[u8],
    src_width: u32,
    src_height: u32,
    dest_x: i32,
    dest_y: i32,
    scaled_width: i32,
    scaled_height: i32,
) {
    if scaled_width <= 0 || scaled_height <= 0 {
        return;
    }
    
    // Early return if tile is entirely off-screen
    if dest_x + scaled_width <= 0 || dest_y + scaled_height <= 0 {
        return;
    }
    if dest_x >= dest_width as i32 || dest_y >= dest_height as i32 {
        return;
    }
    
    let scale_x = src_width as f64 / scaled_width as f64;
    let scale_y = src_height as f64 / scaled_height as f64;
    
    // Calculate visible region (clamp to destination bounds)
    let start_x = dest_x.max(0) as u32;
    let start_y = dest_y.max(0) as u32;
    let end_x = (dest_x + scaled_width).min(dest_width as i32) as u32;
    let end_y = (dest_y + scaled_height).min(dest_height as i32) as u32;
    
    for y in start_y..end_y {
        for x in start_x..end_x {
            // Calculate source position with sub-pixel precision
            let src_x_f = (x as i32 - dest_x) as f64 * scale_x;
            let src_y_f = (y as i32 - dest_y) as f64 * scale_y;
            
            // Bilinear interpolation
            let x0 = src_x_f.floor() as i32;
            let y0 = src_y_f.floor() as i32;
            let x1 = (x0 + 1).min(src_width as i32 - 1);
            let y1 = (y0 + 1).min(src_height as i32 - 1);
            let x0 = x0.max(0).min(src_width as i32 - 1) as u32;
            let y0 = y0.max(0).min(src_height as i32 - 1) as u32;
            let x1 = x1.max(0) as u32;
            let y1 = y1.max(0) as u32;
            
            // Fractional parts for interpolation weights
            let fx = (src_x_f - src_x_f.floor()).max(0.0).min(1.0);
            let fy = (src_y_f - src_y_f.floor()).max(0.0).min(1.0);
            
            // Get the four neighboring pixels
            let idx00 = ((y0 * src_width + x0) * 4) as usize;
            let idx10 = ((y0 * src_width + x1) * 4) as usize;
            let idx01 = ((y1 * src_width + x0) * 4) as usize;
            let idx11 = ((y1 * src_width + x1) * 4) as usize;
            
            let dest_idx = ((y * dest_width + x) * 4) as usize;
            
            if idx11 + 3 < src.len() && dest_idx + 3 < dest.len() {
                // Bilinear interpolation for each channel
                for c in 0..4 {
                    let p00 = src[idx00 + c] as f64;
                    let p10 = src[idx10 + c] as f64;
                    let p01 = src[idx01 + c] as f64;
                    let p11 = src[idx11 + c] as f64;
                    
                    // Interpolate horizontally, then vertically
                    let top = p00 * (1.0 - fx) + p10 * fx;
                    let bottom = p01 * (1.0 - fx) + p11 * fx;
                    let value = top * (1.0 - fy) + bottom * fy;
                    
                    dest[dest_idx + c] = value.round() as u8;
                }
            }
        }
    }
}

fn create_image_buffer(data: &[u8], width: u32, height: u32) -> Option<SharedPixelBuffer<Rgba8Pixel>> {
    let expected_len = (width * height * 4) as usize;
    if data.len() < expected_len {
        return None;
    }
    
    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(width, height);
    buffer.make_mut_bytes().copy_from_slice(&data[..expected_len]);
    Some(buffer)
}

// ============ Tool handling functions ============

fn update_tool_state(ui: &AppWindow, state: &AppState) {
    let tool_type = match state.current_tool {
        state::Tool::Navigate => ToolType::Navigate,
        state::Tool::RegionOfInterest => ToolType::RegionOfInterest,
        state::Tool::MeasureDistance => ToolType::MeasureDistance,
    };
    ui.set_current_tool(tool_type);
}

fn update_tool_overlays(ui: &AppWindow, state: &AppState) {
    let Some(file_id) = state.active_file_id else {
        return;
    };
    
    let Some(file) = state.open_files.iter().find(|f| f.id == file_id) else {
        return;
    };
    
    // Get the active viewport for coordinate conversion
    let viewport_state = if state.split_enabled && state.focused_pane == PaneId::Secondary {
        file.secondary_viewport.as_ref().unwrap_or(&file.viewport)
    } else {
        &file.viewport
    };
    let vp = &viewport_state.viewport;
    
    // Update ROI overlay
    let ant_offset = state.ant_offset;
    if let Some(roi) = &file.roi {
        let bounds = vp.bounds();
        let screen_x = (roi.x - bounds.left) * vp.zoom;
        let screen_y = (roi.y - bounds.top) * vp.zoom;
        let screen_w = roi.width * vp.zoom;
        let screen_h = roi.height * vp.zoom;
        
        ui.set_roi_rect(ROIRect {
            x: screen_x as f32,
            y: screen_y as f32,
            width: screen_w as f32,
            height: screen_h as f32,
            visible: true,
            ant_offset,
        });
    } else {
        ui.set_roi_rect(ROIRect {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            visible: false,
            ant_offset,
        });
    }
    
    // Update candidate measurement
    if let (state::ToolInteractionState::Dragging(start) | state::ToolInteractionState::FirstPointPlaced(start), 
            Some(end)) = (&state.tool_state, &state.candidate_point) 
    {
        if state.current_tool == state::Tool::MeasureDistance || state.current_tool == state::Tool::RegionOfInterest {
            let bounds = vp.bounds();
            let x1 = (start.x - bounds.left) * vp.zoom;
            let y1 = (start.y - bounds.top) * vp.zoom;
            let x2 = (end.x - bounds.left) * vp.zoom;
            let y2 = (end.y - bounds.top) * vp.zoom;
            
            // For ROI, update the roi_rect
            if state.current_tool == state::Tool::RegionOfInterest {
                let roi = state::RegionOfInterest::from_points(*start, *end);
                let screen_x = (roi.x - bounds.left) * vp.zoom;
                let screen_y = (roi.y - bounds.top) * vp.zoom;
                let screen_w = roi.width * vp.zoom;
                let screen_h = roi.height * vp.zoom;
                
                ui.set_roi_rect(ROIRect {
                    x: screen_x as f32,
                    y: screen_y as f32,
                    width: screen_w as f32,
                    height: screen_h as f32,
                    visible: true,
                    ant_offset,
                });
            } else {
                // For measurement, update candidate line
                ui.set_candidate_measurement(MeasurementLine {
                    x1: x1 as f32,
                    y1: y1 as f32,
                    x2: x2 as f32,
                    y2: y2 as f32,
                    distance_um: 0.0, // TODO: Calculate actual distance in microns
                    visible: true,
                });
            }
        }
    } else {
        ui.set_candidate_measurement(MeasurementLine {
            x1: 0.0,
            y1: 0.0,
            x2: 0.0,
            y2: 0.0,
            distance_um: 0.0,
            visible: false,
        });
    }
}

fn handle_tool_mouse_down(state: &mut AppState, screen_x: f64, screen_y: f64) {
    let Some(file_id) = state.active_file_id else {
        return;
    };
    
    let focused_pane = state.focused_pane;
    let split_enabled = state.split_enabled;
    
    let Some(file) = state.open_files.iter().find(|f| f.id == file_id) else {
        return;
    };
    
    // Get the active viewport for coordinate conversion
    let viewport_state = if split_enabled && focused_pane == PaneId::Secondary {
        file.secondary_viewport.as_ref().unwrap_or(&file.viewport)
    } else {
        &file.viewport
    };
    
    // Convert screen coordinates to image coordinates
    let image_point = viewport_state.screen_to_image(screen_x, screen_y);
    let point = state::ImagePoint { x: image_point.0, y: image_point.1 };
    
    match state.current_tool {
        state::Tool::Navigate => {
            // Should not happen - Navigate uses LMB for panning
        }
        state::Tool::RegionOfInterest | state::Tool::MeasureDistance => {
            state.tool_state = state::ToolInteractionState::Dragging(point);
            state.candidate_point = Some(point);
        }
    }
}

fn handle_tool_mouse_move(state: &mut AppState, screen_x: f64, screen_y: f64) {
    let Some(file_id) = state.active_file_id else {
        return;
    };
    
    let focused_pane = state.focused_pane;
    let split_enabled = state.split_enabled;
    
    let Some(file) = state.open_files.iter().find(|f| f.id == file_id) else {
        return;
    };
    
    // Get the active viewport for coordinate conversion
    let viewport_state = if split_enabled && focused_pane == PaneId::Secondary {
        file.secondary_viewport.as_ref().unwrap_or(&file.viewport)
    } else {
        &file.viewport
    };
    
    // Convert screen coordinates to image coordinates
    let image_point = viewport_state.screen_to_image(screen_x, screen_y);
    let point = state::ImagePoint { x: image_point.0, y: image_point.1 };
    
    // Update candidate point during drag
    if matches!(state.tool_state, state::ToolInteractionState::Dragging(_)) {
        state.candidate_point = Some(point);
    }
}

fn handle_tool_mouse_up(state: &mut AppState, screen_x: f64, screen_y: f64) {
    let Some(file_id) = state.active_file_id else {
        return;
    };
    
    let focused_pane = state.focused_pane;
    let split_enabled = state.split_enabled;
    let current_tool = state.current_tool;
    
    let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
        return;
    };
    
    // Get the active viewport for coordinate conversion
    let viewport_state = if split_enabled && focused_pane == PaneId::Secondary {
        file.secondary_viewport.as_ref().unwrap_or(&file.viewport)
    } else {
        &file.viewport
    };
    
    // Convert screen coordinates to image coordinates
    let image_point = viewport_state.screen_to_image(screen_x, screen_y);
    let end_point = state::ImagePoint { x: image_point.0, y: image_point.1 };
    
    if let state::ToolInteractionState::Dragging(start) = state.tool_state {
        match current_tool {
            state::Tool::Navigate => {}
            state::Tool::RegionOfInterest => {
                // Create ROI from the two points
                let roi = state::RegionOfInterest::from_points(start, end_point);
                if roi.is_valid() {
                    file.roi = Some(roi);
                }
            }
            state::Tool::MeasureDistance => {
                // Create measurement from the two points
                let measurement = state::Measurement {
                    start,
                    end: end_point,
                };
                file.measurements.push(measurement);
            }
        }
    }
    
    // Reset tool state
    state.tool_state = state::ToolInteractionState::Idle;
    state.candidate_point = None;
}
use crate::config;
use crate::state::{self, AppState, HudSettings, PaneId};
use crate::{
    AppWindow, ExportFilteringMode as SlintExportFilteringMode, ExportFormat as SlintExportFormat,
    ExportPreviewInfo as SlintExportPreviewInfo, ExportSettings as SlintExportSettings,
    FilteringMode as SlintFilteringMode, MeasurementUnit as SlintMeasurementUnit, RenderMode,
    StainNormalization as SlintStainNormalization, ToolType, build_recent_menu_items,
    capture_pane_clipboard_image, copy_image_to_clipboard, copy_text_to_clipboard,
    crop_image_to_viewport_bounds, handle_tool_mouse_down, handle_tool_mouse_move,
    handle_tool_mouse_up, insert_pane_ui_state, open_file, pane_from_index, refresh_tab_ui,
    request_render_loop, slider_value_to_zoom, update_filtering_mode, update_render_backend,
    update_tabs, update_tool_overlays, update_tool_state,
};
use common::viewport::ZOOM_FACTOR;
use common::{FilteringMode, MeasurementUnit, RenderBackend, StainNormalization, TileCache};
use parking_lot::RwLock;
use rfd::FileDialog;
use slint::{ComponentHandle, SharedString, Timer, VecModel};
use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

const ACTION_ZOOM_FACTOR: f64 = ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR;
#[cfg(target_os = "macos")]
const MACOS_TOOLBAR_HEIGHT: f64 = 40.0;

fn active_hud_mut(state: &mut AppState) -> Option<&mut HudSettings> {
    let pane = state.focused_pane;
    let file_id = state.active_file_id_for_pane(pane)?;
    let file = state.get_file_mut(file_id)?;
    let pane_state = file.pane_state_mut(pane)?;
    Some(&mut pane_state.hud)
}

fn measurement_unit_from_slint(unit: SlintMeasurementUnit) -> MeasurementUnit {
    match unit {
        SlintMeasurementUnit::Um => MeasurementUnit::Um,
        SlintMeasurementUnit::Mm => MeasurementUnit::Mm,
        SlintMeasurementUnit::Inches => MeasurementUnit::Inches,
    }
}

fn stain_normalization_from_slint(sn: SlintStainNormalization) -> StainNormalization {
    match sn {
        SlintStainNormalization::None => StainNormalization::None,
        SlintStainNormalization::Macenko => StainNormalization::Macenko,
        SlintStainNormalization::Vahadane => StainNormalization::Vahadane,
    }
}

/// Query the cursor position in window-local physical pixels via X11.
#[cfg(target_os = "linux")]
fn query_dnd_cursor_physical(ui: &AppWindow) -> Option<(f64, f64)> {
    use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
    use slint::winit_030::WinitWindowAccessor;

    ui.window().with_winit_window(|w| {
        let wh = w.window_handle().ok()?;
        let dh = w.display_handle().ok()?;

        match (wh.as_ref(), dh.as_ref()) {
            (RawWindowHandle::Xlib(wh), RawDisplayHandle::Xlib(dh)) => {
                let display = dh.display?.as_ptr();
                let window = wh.window;
                let xlib = x11_dl::xlib::Xlib::open().ok()?;
                unsafe {
                    let mut root = 0;
                    let mut child = 0;
                    let mut root_x = 0;
                    let mut root_y = 0;
                    let mut win_x = 0;
                    let mut win_y = 0;
                    let mut mask = 0;

                    let result = (xlib.XQueryPointer)(
                        display as *mut _,
                        window,
                        &mut root,
                        &mut child,
                        &mut root_x,
                        &mut root_y,
                        &mut win_x,
                        &mut win_y,
                        &mut mask,
                    );

                    if result != 0 {
                        Some((win_x as f64, win_y as f64))
                    } else {
                        None
                    }
                }
            }
            _ => None,
        }
    })?
}

/// Query the cursor position in window-local physical pixels via Win32 API.
#[cfg(target_os = "windows")]
fn query_dnd_cursor_physical(ui: &AppWindow) -> Option<(f64, f64)> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use slint::winit_030::WinitWindowAccessor;
    use windows_sys::Win32::Foundation::POINT;
    use windows_sys::Win32::Graphics::Gdi::ScreenToClient;
    use windows_sys::Win32::UI::WindowsAndMessaging::GetCursorPos;

    ui.window().with_winit_window(|w| {
        let wh = w.window_handle().ok()?;
        match wh.as_ref() {
            RawWindowHandle::Win32(wh) => {
                let hwnd = wh.hwnd.get() as isize;
                unsafe {
                    let mut pt = POINT { x: 0, y: 0 };
                    if GetCursorPos(&mut pt) == 0 {
                        return None;
                    }
                    if ScreenToClient(hwnd, &mut pt) == 0 {
                        return None;
                    }
                    Some((pt.x as f64, pt.y as f64))
                }
            }
            _ => None,
        }
    })?
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn query_dnd_cursor_physical(_ui: &AppWindow) -> Option<(f64, f64)> {
    None
}

/// Query cursor position during DnD and return logical coordinates.
fn query_dnd_cursor_logical(ui: &AppWindow) -> Option<(f32, f32)> {
    let (px, py) = query_dnd_cursor_physical(ui)?;
    let scale = ui.window().scale_factor() as f64;
    Some(((px / scale) as f32, (py / scale) as f32))
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct NSPoint {
    x: f64,
    y: f64,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct NSSize {
    width: f64,
    height: f64,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct NSRect {
    origin: NSPoint,
    size: NSSize,
}

#[cfg(target_os = "macos")]
unsafe impl objc::Encode for NSPoint {
    fn encode() -> objc::Encoding {
        unsafe { objc::Encoding::from_str("{CGPoint=dd}") }
    }
}

#[cfg(target_os = "macos")]
unsafe impl objc::Encode for NSSize {
    fn encode() -> objc::Encoding {
        unsafe { objc::Encoding::from_str("{CGSize=dd}") }
    }
}

#[cfg(target_os = "macos")]
unsafe impl objc::Encode for NSRect {
    fn encode() -> objc::Encoding {
        unsafe { objc::Encoding::from_str("{CGRect={CGPoint=dd}{CGSize=dd}}") }
    }
}

#[cfg(target_os = "macos")]
#[allow(unexpected_cfgs)]
fn align_macos_window_controls(slint_window: &slint::Window) {
    use objc::runtime::Object;
    use objc::{sel, sel_impl};
    use slint::winit_030::WinitWindowAccessor;
    use slint::winit_030::winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

    slint_window.with_winit_window(|w| {
        let Ok(window_handle) = w.window_handle() else {
            return;
        };

        let RawWindowHandle::AppKit(handle) = window_handle.as_ref() else {
            return;
        };

        let ns_view = handle.ns_view.as_ptr().cast::<Object>();

        unsafe {
            let ns_window: *mut Object = objc::msg_send![ns_view, window];
            if ns_window.is_null() {
                return;
            }

            for button_kind in [0usize, 1, 2] {
                let button: *mut Object =
                    objc::msg_send![ns_window, standardWindowButton: button_kind];
                if button.is_null() {
                    continue;
                }

                let frame: NSRect = objc::msg_send![button, frame];
                let origin = NSPoint {
                    x: frame.origin.x,
                    y: (MACOS_TOOLBAR_HEIGHT - frame.size.height) / 2.0,
                };
                let _: () = objc::msg_send![button, setFrameOrigin: origin];
            }
        }
    });
}

fn frame_active_viewport(state: &mut AppState) -> bool {
    let pane = state.focused_pane;
    let roi = state
        .active_file_id_for_pane(pane)
        .and_then(|file_id| state.get_file(file_id))
        .and_then(|file| file.roi.filter(|roi| roi.pane == pane));

    let Some(viewport) = state.active_viewport_mut() else {
        return false;
    };

    if let Some(roi) = roi {
        viewport.smooth_frame_rect(roi.x, roi.y, roi.width, roi.height);
    } else {
        viewport.smooth_fit_to_view();
    }

    true
}

fn zoom_active_viewport(state: &mut AppState, factor: f64) -> bool {
    let Some(viewport) = state.active_viewport_mut() else {
        return false;
    };

    let center_x = viewport.viewport.width / 2.0;
    let center_y = viewport.viewport.height / 2.0;
    viewport.zoom_at_discrete(factor, center_x, center_y);
    true
}

fn show_toast(ui: &AppWindow, toast_timer: &Rc<Timer>, message: &str) {
    ui.set_toast_text(SharedString::from(message));
    ui.set_toast_visible(true);
    let ui_weak = ui.as_weak();
    toast_timer.start(
        slint::TimerMode::SingleShot,
        Duration::from_millis(2200),
        move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_toast_visible(false);
            }
        },
    );
}

fn toggle_minimap_visibility(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    let show_minimap = {
        let mut state = state.write();
        state.toggle_minimap();
        state.show_minimap
    };
    ui.set_show_minimap(show_minimap);
}

fn toggle_metadata_visibility(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    let show_metadata = {
        let mut state = state.write();
        state.toggle_metadata();
        state.show_metadata
    };
    ui.set_show_metadata(show_metadata);
}

/// Convert Slint export settings to common crate's `ExportSettings`.
fn slint_to_export_settings(s: &SlintExportSettings) -> common::ExportSettings {
    let filtering_mode = match s.filtering_mode {
        SlintExportFilteringMode::Bilinear => FilteringMode::Bilinear,
        SlintExportFilteringMode::Trilinear => FilteringMode::Trilinear,
        SlintExportFilteringMode::Lanczos => FilteringMode::Lanczos3,
    };
    let stain_normalization = match s.stain_normalization {
        SlintStainNormalization::None => StainNormalization::None,
        SlintStainNormalization::Macenko => StainNormalization::Macenko,
        SlintStainNormalization::Vahadane => StainNormalization::Vahadane,
    };
    common::ExportSettings {
        dpi: s.dpi.max(1) as u32,
        filtering_mode,
        stain_normalization,
        sharpness: s.sharpness,
        gamma: s.gamma,
        brightness: s.brightness,
        contrast: s.contrast,
        background_rgba: [255, 255, 255, 255],
    }
}

fn slint_color_to_overlay(color: slint::Color, opacity_pct: f32) -> common::overlay::OverlayColor {
    let a = (color.alpha() as f32 * opacity_pct / 100.0)
        .round()
        .clamp(0.0, 255.0) as u8;
    common::overlay::OverlayColor::new(color.red(), color.green(), color.blue(), a)
}

fn slint_stroke_to_overlay(
    style: crate::StrokeStyle,
    dash_length: f32,
    dash_gap: f32,
    dot_spacing: f32,
) -> common::overlay::StrokeStyle {
    match style {
        crate::StrokeStyle::Solid => common::overlay::StrokeStyle::Solid,
        crate::StrokeStyle::Dashed => common::overlay::StrokeStyle::Dashed {
            length: dash_length,
            gap: dash_gap,
        },
        crate::StrokeStyle::Dotted => common::overlay::StrokeStyle::Dotted {
            spacing: dot_spacing,
        },
    }
}

fn slint_cap_to_overlay(cap: crate::CapStyle) -> common::overlay::CapStyle {
    match cap {
        crate::CapStyle::Round => common::overlay::CapStyle::Round,
        crate::CapStyle::Square => common::overlay::CapStyle::Square,
        crate::CapStyle::Flat => common::overlay::CapStyle::Flat,
    }
}

/// Draw measurement overlays on an export image buffer.
fn draw_measurement_overlays(
    image_data: &mut common::RgbaImageData,
    file: &state::OpenFile,
    pane: PaneId,
    export_vp: &common::Viewport,
    settings: &SlintExportSettings,
    dpi_scale: f32,
    font: Option<&common::overlay::FontArc>,
) {
    if !settings.show_measurement || !settings.has_measurement {
        return;
    }
    let color = slint_color_to_overlay(settings.measurement_color, settings.measurement_opacity);
    let stroke = slint_stroke_to_overlay(
        settings.measurement_stroke_style,
        settings.measurement_dash_length * dpi_scale,
        settings.measurement_dash_gap * dpi_scale,
        settings.measurement_dot_spacing * dpi_scale,
    );
    let cap = slint_cap_to_overlay(settings.measurement_cap_style);
    let thickness = settings.measurement_thickness * dpi_scale;
    let w = image_data.width as u32;
    let h = image_data.height as u32;
    let font_size_px = settings.measurement_font_size * dpi_scale;

    // Compute mpp for distance labels (same as tools.rs)
    let mpp = file
        .wsi
        .properties()
        .mpp_x
        .zip(file.wsi.properties().mpp_y)
        .map(|(mx, my)| (mx + my) / 2.0)
        .unwrap_or(0.0);

    for m in &file.measurements {
        if m.pane != pane {
            continue;
        }
        let p1 = export_vp.image_to_screen(m.start.x, m.start.y);
        let p2 = export_vp.image_to_screen(m.end.x, m.end.y);

        // Draw line
        common::overlay::draw_line(
            &mut image_data.pixels,
            w,
            h,
            p1.x as f32,
            p1.y as f32,
            p2.x as f32,
            p2.y as f32,
            color,
            thickness,
            stroke,
            cap,
        );

        // Endpoint circles (white border + filled color, matching viewport style)
        let endpoint_r = (thickness + 1.0) * 0.5 + 1.0 * dpi_scale;
        let white = common::overlay::OverlayColor::new(255, 255, 255, color.a);
        for p in [p1, p2] {
            common::overlay::draw_filled_circle(
                &mut image_data.pixels,
                w,
                h,
                p.x as f32,
                p.y as f32,
                endpoint_r + 1.5 * dpi_scale,
                white,
            );
            common::overlay::draw_filled_circle(
                &mut image_data.pixels,
                w,
                h,
                p.x as f32,
                p.y as f32,
                endpoint_r,
                color,
            );
        }

        // Distance label pill (centered between endpoints, above the line)
        if let Some(font) = font {
            let distance_um = m.distance() * mpp;
            let label = if distance_um > 0.0 {
                common::overlay::format_measurement_label(distance_um)
            } else {
                // Fallback: distance in screen pixels
                let dx = p2.x - p1.x;
                let dy = p2.y - p1.y;
                let px_dist = (dx * dx + dy * dy).sqrt();
                format!("{:.1} px", (px_dist * 10.0).round() / 10.0)
            };

            let mid_x = (p1.x + p2.x) as f32 / 2.0;
            let mid_y = (p1.y + p2.y) as f32 / 2.0;
            common::overlay::draw_measurement_label(
                &mut image_data.pixels,
                w,
                h,
                mid_x,
                mid_y,
                &label,
                font,
                font_size_px,
                dpi_scale,
            );
        }
    }
}

/// Draw ROI overlays on an export image buffer.
fn draw_roi_overlays(
    image_data: &mut common::RgbaImageData,
    file: &state::OpenFile,
    export_vp: &common::Viewport,
    settings: &SlintExportSettings,
    dpi_scale: f32,
) {
    let Some(roi) = &file.roi else { return };
    let w = image_data.width as u32;
    let h = image_data.height as u32;
    let tl = export_vp.image_to_screen(roi.x, roi.y);
    let br = export_vp.image_to_screen(roi.x + roi.width, roi.y + roi.height);
    let rx = tl.x as f32;
    let ry = tl.y as f32;
    let rw = (br.x - tl.x) as f32;
    let rh = (br.y - tl.y) as f32;

    // Outside overlay (drawn first, beneath the outline)
    if settings.roi_outside_overlay {
        let outside_color =
            slint_color_to_overlay(settings.roi_outside_color, settings.roi_outside_opacity);
        common::overlay::fill_outside_rect(
            &mut image_data.pixels,
            w,
            h,
            rx,
            ry,
            rw,
            rh,
            outside_color,
        );
    }

    // ROI outline
    if settings.show_roi_outline && settings.has_roi {
        let color = slint_color_to_overlay(settings.roi_color, settings.roi_opacity);
        let stroke = slint_stroke_to_overlay(
            settings.roi_stroke_style,
            settings.roi_dash_length * dpi_scale,
            settings.roi_dash_gap * dpi_scale,
            settings.roi_dot_spacing * dpi_scale,
        );
        let cap = slint_cap_to_overlay(settings.roi_cap_style);
        let thickness = settings.roi_thickness * dpi_scale;
        common::overlay::draw_rect_outline(
            &mut image_data.pixels,
            w,
            h,
            rx,
            ry,
            rw,
            rh,
            color,
            thickness,
            stroke,
            cap,
        );
    }
}

/// Render an export preview/final image using the common crate's export renderer.
///
/// If `override_dpi` is `Some(dpi)`, renders at that DPI instead of the
/// settings' DPI.  Used for preview (always 96 DPI) vs. final export.
fn render_export_image(
    state: &AppState,
    tile_cache: &Arc<TileCache>,
    pane: PaneId,
    slint_settings: &SlintExportSettings,
    override_dpi: Option<u32>,
    overlay_font: Option<&common::overlay::FontArc>,
) -> Option<common::RgbaImageData> {
    let file_id = state.active_file_id_for_pane(pane)?;
    let file = state.get_file(file_id)?;
    let pane_state = file.pane_state(pane)?;
    let viewport = &pane_state.viewport.viewport;
    let mut settings = slint_to_export_settings(slint_settings);
    let render_dpi = override_dpi.unwrap_or(settings.dpi);
    settings.dpi = render_dpi;
    let mut img =
        common::export::render_export(&file.tile_manager, tile_cache, viewport, &settings)?;

    // Draw overlays
    let export_vp = common::export::export_viewport(viewport, render_dpi);
    let dpi_scale = render_dpi as f32 / 96.0;
    draw_measurement_overlays(
        &mut img,
        file,
        pane,
        &export_vp,
        slint_settings,
        dpi_scale,
        overlay_font,
    );
    draw_roi_overlays(&mut img, file, &export_vp, slint_settings, dpi_scale);
    Some(img)
}

/// Build default export settings based on the current viewport state.
fn build_default_export_settings(state: &AppState, pane: PaneId) -> SlintExportSettings {
    let file_id = state.active_file_id_for_pane(pane);
    let hud = file_id
        .and_then(|id| state.get_file(id))
        .and_then(|f| f.pane_state(pane))
        .map(|ps| &ps.hud)
        .cloned()
        .unwrap_or_default();

    let has_measurement = file_id
        .and_then(|id| state.get_file(id))
        .is_some_and(|f| !f.measurements.is_empty());

    let has_roi = file_id
        .and_then(|id| state.get_file(id))
        .is_some_and(|f| f.roi.is_some());

    let stain_norm = match hud.stain_normalization {
        StainNormalization::None => SlintStainNormalization::None,
        StainNormalization::Macenko => SlintStainNormalization::Macenko,
        StainNormalization::Vahadane => SlintStainNormalization::Vahadane,
    };

    SlintExportSettings {
        dpi: 150,
        filtering_mode: SlintExportFilteringMode::Trilinear,
        sharpness: hud.sharpness,
        gamma: hud.gamma,
        brightness: hud.brightness,
        contrast: hud.contrast,
        stain_normalization: stain_norm,
        show_measurement: has_measurement,
        has_measurement,
        measurement_color: slint::Color::from_argb_u8(255, 46, 204, 113),
        measurement_opacity: 100.0,
        measurement_thickness: 2.0,
        measurement_stroke_style: crate::StrokeStyle::Solid,
        measurement_cap_style: crate::CapStyle::Round,
        measurement_dash_length: 8.0,
        measurement_dash_gap: 4.0,
        measurement_dot_spacing: 4.0,
        measurement_font_size: 12.0,
        show_roi_outline: has_roi,
        has_roi,
        roi_color: slint::Color::from_argb_u8(255, 241, 196, 15),
        roi_opacity: 100.0,
        roi_thickness: 2.0,
        roi_stroke_style: crate::StrokeStyle::Solid,
        roi_cap_style: crate::CapStyle::Round,
        roi_dash_length: 8.0,
        roi_dash_gap: 4.0,
        roi_dot_spacing: 4.0,
        roi_outside_overlay: false,
        roi_outside_color: slint::Color::from_argb_u8(255, 0, 0, 0),
        roi_outside_opacity: 50.0,
        format: SlintExportFormat::Png,
        jpeg_quality: 85,
    }
}

/// Open the export dialog, populating it with defaults derived from the current viewport.
/// If `cached` has a value, reuse those settings (except has_measurement/has_roi which
/// are derived from the current file state).
fn open_export_dialog(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    pane: PaneId,
    cached: &Rc<RefCell<Option<SlintExportSettings>>>,
    overlay_font: Option<&common::overlay::FontArc>,
) {
    let settings = {
        let state = state.read();
        let mut s = cached
            .borrow()
            .clone()
            .unwrap_or_else(|| build_default_export_settings(&state, pane));

        // Always refresh has_measurement / has_roi from current file state
        let file_id = state.active_file_id_for_pane(pane);
        s.has_measurement = file_id
            .and_then(|id| state.get_file(id))
            .is_some_and(|f| !f.measurements.is_empty());
        s.has_roi = file_id
            .and_then(|id| state.get_file(id))
            .is_some_and(|f| f.roi.is_some());
        // Disable overlay toggles if the feature no longer exists
        if !s.has_measurement {
            s.show_measurement = false;
        }
        if !s.has_roi {
            s.show_roi_outline = false;
            s.roi_outside_overlay = false;
        }
        s
    };
    ui.set_export_settings(settings.clone());

    // Render preview at 96 DPI (thumbnail-sized, independent of settings DPI)
    let (preview_image, vp_width, vp_height) = {
        let state = state.read();
        let (vp_w, vp_h) = state
            .active_file_id_for_pane(pane)
            .and_then(|id| state.get_file(id))
            .and_then(|f| f.pane_state(pane))
            .map(|ps| (ps.viewport.viewport.width, ps.viewport.viewport.height))
            .unwrap_or((0.0, 0.0));

        let img = render_export_image(&state, tile_cache, pane, &settings, Some(96), overlay_font);
        let slint_img = img
            .and_then(|img| {
                let w = img.width as u32;
                let h = img.height as u32;
                crate::blitter::create_image_buffer(&img.pixels, w, h).map(slint::Image::from_rgba8)
            })
            .or_else(|| {
                capture_pane_clipboard_image(pane).and_then(|d| {
                    crate::blitter::create_image_buffer(&d.pixels, d.width as u32, d.height as u32)
                        .map(slint::Image::from_rgba8)
                })
            })
            .unwrap_or_default();

        (slint_img, vp_w, vp_h)
    };

    let preview_info = SlintExportPreviewInfo {
        preview_image,
        estimated_size_mb: 0.0, // computed reactively in Slint
        width_px: 0,            // computed reactively in Slint
        height_px: 0,           // computed reactively in Slint
        viewport_width: vp_width as f32,
        viewport_height: vp_height as f32,
    };
    ui.set_export_preview_info(preview_info);
    ui.set_export_dialog_visible(true);
}

fn apply_filtering_mode(
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    ui_weak: &slint::Weak<AppWindow>,
    mode: FilteringMode,
) {
    {
        let mut state = state.write();
        state.select_filtering_mode(mode);
        if let Err(err) = config::save_filtering_mode(state.filtering_mode) {
            warn!("Failed to save filtering mode config: {}", err);
        }
    }
    if let Some(ui) = ui_weak.upgrade() {
        let state = state.read();
        update_filtering_mode(&ui, &state);
    }
    if let Some(ui) = ui_weak.upgrade() {
        request_render_loop(render_timer, &ui.as_weak(), state, tile_cache);
    }
}

pub fn setup_callbacks(
    ui: &AppWindow,
    state: Arc<RwLock<AppState>>,
    tile_cache: Arc<TileCache>,
    render_timer: Rc<Timer>,
) {
    let ui_weak = ui.as_weak();
    let clipboard = Rc::new(RefCell::new(None));
    let toast_timer = Rc::new(Timer::default());
    let cached_export_settings: Rc<RefCell<Option<SlintExportSettings>>> =
        Rc::new(RefCell::new(None));
    let overlay_font: Option<common::overlay::FontArc> = common::overlay::load_system_font();

    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_open_file_requested(move || {
            let ui = ui_weak.upgrade().unwrap();

            let dialog = FileDialog::new()
                .add_filter(
                    "WSI Files",
                    &[
                        ".svs", ".tif", ".dcm", ".ndpi", ".vms", ".vmu", ".scn", ".mrxs", ".tiff",
                        ".svslide", ".bif", ".czi",
                    ],
                )
                .add_filter("All Files", &["*"]);

            if let Some(path) = dialog.pick_file() {
                open_file(&ui, &state, &tile_cache, &render_timer, path);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_new_tab_requested(move |pane| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    state.set_focused_pane(pane_from_index(pane));
                    state.create_home_tab();
                }
                let state = state_handle.read();
                update_tabs(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_open_recent_file(move |path_str| {
            let path = PathBuf::from(path_str.as_str());
            if path.exists()
                && let Some(ui) = ui_weak.upgrade()
            {
                open_file(&ui, &state, &tile_cache, &render_timer, path);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_open_recent_menu_requested(move |x, y| {
            if let Some(ui) = ui_weak.upgrade() {
                let items = {
                    let state = state_handle.read();
                    build_recent_menu_items(&state)
                };
                ui.set_context_menu_items(Rc::new(VecModel::from(items)).into());
                ui.set_context_menu_tab_id(-1);
                ui.set_drag_source_pane(-1);
                ui.set_context_menu_x(x);
                ui.set_context_menu_y(y);
                ui.set_context_menu_visible(true);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();
        let clipboard = Rc::clone(&clipboard);
        let toast_timer = Rc::clone(&toast_timer);
        let cached_export = Rc::clone(&cached_export_settings);
        let overlay_font_clone = overlay_font.clone();

        ui.on_context_menu_command(move |id| {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };

            ui.set_context_menu_visible(false);

            let command = id.to_string();
            if let Some(path_str) = command.strip_prefix("recent-file:") {
                let path = PathBuf::from(path_str);
                if path.exists() {
                    open_file(&ui, &state_handle, &tile_cache, &render_timer, path);
                }
                return;
            }

            let pane = pane_from_index(ui.get_drag_source_pane());
            let tab_id = ui.get_context_menu_tab_id();

            match command.as_str() {
                "viewport-copy-image" => {
                    if let Some(image) = capture_pane_clipboard_image(pane) {
                        let viewport = {
                            let state = state_handle.read();
                            state
                                .active_file_id_for_pane(pane)
                                .and_then(|file_id| state.get_file(file_id))
                                .and_then(|file| file.pane_state(pane))
                                .map(|pane_state| pane_state.viewport.viewport.clone())
                        };
                        let image = if let Some(viewport) = viewport {
                            crop_image_to_viewport_bounds(image, &viewport)
                        } else {
                            image
                        };
                        if copy_image_to_clipboard(&clipboard, image) {
                            show_toast(&ui, &toast_timer, "Viewport image copied to clipboard.");
                        } else {
                            ui.set_status_text(SharedString::from(
                                "Failed to copy viewport image to clipboard",
                            ));
                        }
                    } else {
                        ui.set_status_text(SharedString::from(
                            "Viewport image is not available yet",
                        ));
                    }
                }
                "viewport-export-image" => {
                    open_export_dialog(
                        &ui,
                        &state_handle,
                        &tile_cache,
                        pane,
                        &cached_export,
                        overlay_font_clone.as_ref(),
                    );
                }
                "close" => {
                    {
                        let mut state = state_handle.write();
                        state.set_focused_pane(pane);
                        state.close_tab_in_pane(pane, tab_id);
                    }
                    let state = state_handle.read();
                    refresh_tab_ui(&ui, &state);
                    request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
                }
                "close-others" => {
                    {
                        let mut state = state_handle.write();
                        let ids_to_close: Vec<i32> = state
                            .tabs_for_pane(pane)
                            .iter()
                            .copied()
                            .filter(|&id| id != tab_id)
                            .collect();
                        for id in ids_to_close {
                            state.set_focused_pane(pane);
                            state.close_tab_in_pane(pane, id);
                        }
                    }
                    let state = state_handle.read();
                    refresh_tab_ui(&ui, &state);
                    request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
                }
                "close-right" => {
                    {
                        let mut state = state_handle.write();
                        let all_tabs: Vec<i32> = state.tabs_for_pane(pane).to_vec();
                        let mut found = false;
                        let ids_to_close: Vec<i32> = all_tabs
                            .iter()
                            .filter_map(|&id| {
                                if id == tab_id {
                                    found = true;
                                    None
                                } else if found {
                                    Some(id)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        for id in ids_to_close {
                            state.set_focused_pane(pane);
                            state.close_tab_in_pane(pane, id);
                        }
                    }
                    let state = state_handle.read();
                    refresh_tab_ui(&ui, &state);
                    request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
                }
                "close-all" => {
                    {
                        let mut state = state_handle.write();
                        state.close_all_tabs();
                    }
                    let state = state_handle.read();
                    refresh_tab_ui(&ui, &state);
                    request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
                }
                "open-folder" => {
                    let state = state_handle.read();
                    if let Some(file) = state.get_file(tab_id)
                        && let Some(parent) = file.path.parent()
                        && let Err(err) = open::that(parent)
                    {
                        error!("Failed to open folder: {}", err);
                    }
                }
                "copy-path" => {
                    let state = state_handle.read();
                    if let Some(file) = state.get_file(tab_id) {
                        copy_text_to_clipboard(&clipboard, file.path.display().to_string());
                    }
                }
                "split-right" => {
                    let (split_enabled, source_pane, new_pane) = {
                        let mut state = state_handle.write();
                        let source_pane = pane;
                        let new_pane = state.insert_pane(source_pane.0 + 1);
                        state.duplicate_tab_to_pane(tab_id, new_pane);
                        state.set_focused_pane(new_pane);
                        state.request_render();
                        (state.split_enabled, source_pane, new_pane)
                    };
                    insert_pane_ui_state(new_pane, Some(source_pane));
                    ui.set_split_enabled(split_enabled);
                    let state = state_handle.read();
                    ui.set_focused_pane(state.focused_pane.as_index());
                    update_tabs(&ui, &state);
                    request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
                }
                _ => {}
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_render_backend_selected(move |mode| {
            let backend = match mode {
                RenderMode::Cpu => RenderBackend::Cpu,
                RenderMode::Gpu => RenderBackend::Gpu,
            };

            // If switching to CPU while Lanczos is active, show confirmation dialog
            let has_lanczos = {
                let state = state_handle.read();
                matches!(state.filtering_mode, FilteringMode::Lanczos3)
            };
            if backend == RenderBackend::Cpu && has_lanczos {
                if let Some(ui) = ui_weak.upgrade() {
                    ui.set_lanczos_confirm_from_backend_switch(true);
                    ui.set_lanczos_confirm_visible(true);
                }
                return;
            }

            let gpu_fallback = {
                let mut state = state_handle.write();
                state.select_render_backend(backend);
                if let Err(err) = config::save_render_backend(state.render_backend) {
                    warn!("Failed to save render backend config: {}", err);
                }
                backend == RenderBackend::Gpu && state.render_backend != RenderBackend::Gpu
            };

            if let Some(ui) = ui_weak.upgrade() {
                let state = state_handle.read();
                update_render_backend(&ui, &state);
                if gpu_fallback {
                    ui.set_status_text(SharedString::from(
                        "GPU renderer unavailable; using CPU renderer",
                    ));
                }
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Filtering mode selection with Lanczos confirmation dialog (CPU only)
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_filtering_mode_selected(move |slint_mode| {
            let mode = match slint_mode {
                SlintFilteringMode::Bilinear => FilteringMode::Bilinear,
                SlintFilteringMode::Trilinear => FilteringMode::Trilinear,
                SlintFilteringMode::Lanczos3 => FilteringMode::Lanczos3,
            };

            let (current_mode, is_cpu) = {
                let state = state_handle.read();
                (
                    state.filtering_mode,
                    state.render_backend == RenderBackend::Cpu,
                )
            };
            let is_lanczos = matches!(mode, FilteringMode::Lanczos3);
            let was_lanczos = matches!(current_mode, FilteringMode::Lanczos3);

            // If switching into Lanczos on CPU, show in-window confirmation dialog
            if is_lanczos && !was_lanczos && is_cpu {
                if let Some(ui) = ui_weak.upgrade() {
                    ui.set_lanczos_confirm_visible(true);
                }
            } else {
                apply_filtering_mode(&state_handle, &tile_cache, &render_timer, &ui_weak, mode);
            }
        });
    }

    // Lanczos confirmation dialog accepted — keep Lanczos on CPU
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_lanczos_confirm_accepted(move || {
            let from_backend_switch = ui_weak
                .upgrade()
                .map(|ui| ui.get_lanczos_confirm_from_backend_switch())
                .unwrap_or(false);

            if from_backend_switch {
                // Switch to CPU and keep Lanczos
                {
                    let mut state = state_handle.write();
                    state.select_render_backend(RenderBackend::Cpu);
                    if let Err(err) = config::save_render_backend(state.render_backend) {
                        warn!("Failed to save render backend config: {}", err);
                    }
                }
                if let Some(ui) = ui_weak.upgrade() {
                    let state = state_handle.read();
                    update_render_backend(&ui, &state);
                }
                if let Some(ui) = ui_weak.upgrade() {
                    request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
                }
            } else {
                apply_filtering_mode(
                    &state_handle,
                    &tile_cache,
                    &render_timer,
                    &ui_weak,
                    FilteringMode::Lanczos3,
                );
            }
        });
    }

    // Lanczos confirmation dialog — use trilinear instead
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_lanczos_confirm_use_trilinear(move || {
            let from_backend_switch = ui_weak
                .upgrade()
                .map(|ui| ui.get_lanczos_confirm_from_backend_switch())
                .unwrap_or(false);

            if from_backend_switch {
                // Switch to CPU and change filtering to Trilinear
                {
                    let mut state = state_handle.write();
                    state.select_render_backend(RenderBackend::Cpu);
                    if let Err(err) = config::save_render_backend(state.render_backend) {
                        warn!("Failed to save render backend config: {}", err);
                    }
                }
                if let Some(ui) = ui_weak.upgrade() {
                    let state = state_handle.read();
                    update_render_backend(&ui, &state);
                }
            }
            apply_filtering_mode(
                &state_handle,
                &tile_cache,
                &render_timer,
                &ui_weak,
                FilteringMode::Trilinear,
            );
        });
    }

    // Lanczos confirmation dialog cancelled
    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_lanczos_confirm_cancelled(move || {
            let from_backend_switch = ui_weak
                .upgrade()
                .map(|ui| ui.get_lanczos_confirm_from_backend_switch())
                .unwrap_or(false);

            if let Some(ui) = ui_weak.upgrade() {
                if from_backend_switch {
                    // Revert backend combobox (stay on GPU)
                    let state = state_handle.read();
                    update_render_backend(&ui, &state);
                } else {
                    // Revert filtering combobox
                    let state = state_handle.read();
                    update_filtering_mode(&ui, &state);
                }
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_frame_requested(move || {
            {
                let mut state = state_handle.write();
                if frame_active_viewport(&mut state) {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_zoom_in_requested(move || {
            {
                let mut state = state_handle.write();
                if zoom_active_viewport(&mut state, ACTION_ZOOM_FACTOR) {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_zoom_out_requested(move || {
            {
                let mut state = state_handle.write();
                if zoom_active_viewport(&mut state, 1.0 / ACTION_ZOOM_FACTOR) {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_toggle_minimap_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                toggle_minimap_visibility(&ui, &state_handle);
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_toggle_metadata_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                toggle_metadata_visibility(&ui, &state_handle);
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_request_render(move || {
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_tab_activated(move |pane, id| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    state.activate_tab_in_pane(pane_from_index(pane), id);
                }
                let state = state_handle.read();
                update_tabs(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_tab_close_requested(move |pane, id| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let pane_id = pane_from_index(pane);
                    state.set_focused_pane(pane_id);
                    state.close_tab_in_pane(pane_id, id);
                }
                let state = state_handle.read();
                refresh_tab_ui(&ui, &state);
                ui.invoke_focus_keyboard();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_close_other_tabs(move |pane, keep_id| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let pane_id = pane_from_index(pane);
                    let ids_to_close: Vec<i32> = state
                        .tabs_for_pane(pane_id)
                        .iter()
                        .copied()
                        .filter(|&id| id != keep_id)
                        .collect();
                    for id in ids_to_close {
                        state.set_focused_pane(pane_id);
                        state.close_tab_in_pane(pane_id, id);
                    }
                }
                let state = state_handle.read();
                refresh_tab_ui(&ui, &state);
                ui.invoke_focus_keyboard();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_close_tabs_to_right(move |pane, from_id| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let pane_id = pane_from_index(pane);
                    let all_tabs: Vec<i32> = state.tabs_for_pane(pane_id).to_vec();
                    let mut found = false;
                    let ids_to_close: Vec<i32> = all_tabs
                        .iter()
                        .filter_map(|&id| {
                            if id == from_id {
                                found = true;
                                None
                            } else if found {
                                Some(id)
                            } else {
                                None
                            }
                        })
                        .collect();
                    for id in ids_to_close {
                        state.set_focused_pane(pane_id);
                        state.close_tab_in_pane(pane_id, id);
                    }
                }
                let state = state_handle.read();
                refresh_tab_ui(&ui, &state);
                ui.invoke_focus_keyboard();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_close_all_tabs(move || {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    state.close_all_tabs();
                }
                let state = state_handle.read();
                refresh_tab_ui(&ui, &state);
                ui.invoke_focus_keyboard();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state = Arc::clone(&state);

        ui.on_open_containing_folder(move |id| {
            let state = state.read();
            if let Some(file) = state.get_file(id)
                && let Some(parent) = file.path.parent()
                && let Err(e) = open::that(parent)
            {
                error!("Failed to open folder: {}", e);
            }
        });
    }

    {
        let state = Arc::clone(&state);
        let clipboard = Rc::clone(&clipboard);

        ui.on_copy_path(move |id| {
            let state = state.read();
            if let Some(file) = state.get_file(id) {
                copy_text_to_clipboard(&clipboard, file.path.display().to_string());
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_split_right(move |pane, id| {
            if let Some(ui) = ui_weak.upgrade() {
                let (split_enabled, focused_pane, source_pane) = {
                    let mut state = state_handle.write();
                    let source_pane = pane_from_index(pane);
                    let new_pane = state.insert_pane(source_pane.0 + 1);
                    state.duplicate_tab_to_pane(id, new_pane);
                    state.set_focused_pane(new_pane);
                    state.request_render();
                    (state.split_enabled, state.focused_pane, source_pane)
                };
                insert_pane_ui_state(focused_pane, Some(source_pane));
                ui.set_split_enabled(split_enabled);
                ui.set_focused_pane(focused_pane.as_index());
                let state = state_handle.read();
                update_tabs(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_pane_focused(move |pane| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                state.request_render();
            }

            if let Some(ui) = ui_weak.upgrade() {
                ui.set_focused_pane(pane);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_split_position_changed(move |pos| {
            {
                let mut state = state_handle.write();
                state.split_position = pos;
                state.request_render();
            }

            if let Some(ui) = ui_weak.upgrade() {
                ui.set_split_position(pos);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_pan(move |x, y| {
            {
                let mut state = state_handle.write();
                let mut changed = false;
                if let Some(viewport) = state.active_viewport_mut() {
                    viewport.drag_to(x as f64, y as f64);
                    changed = true;
                }
                if changed {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_start_pan(move |x, y| {
            {
                let mut state = state_handle.write();
                let mut changed = false;
                if let Some(viewport) = state.active_viewport_mut() {
                    viewport.start_drag(x as f64, y as f64);
                    changed = true;
                }
                if changed {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_end_pan(move || {
            {
                let mut state = state_handle.write();
                let mut changed = false;
                if let Some(viewport) = state.active_viewport_mut() {
                    viewport.end_drag();
                    changed = true;
                }
                if changed {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_zoom(move |factor, x, y| {
            {
                let mut state = state_handle.write();
                let mut changed = false;
                if let Some(viewport) = state.active_viewport_mut() {
                    viewport.zoom_at(factor as f64, x as f64, y as f64);
                    changed = true;
                }
                if changed {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_fit_to_view(move || {
            {
                let mut state = state_handle.write();
                if frame_active_viewport(&mut state) {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_minimap_navigate(move |nx, ny| {
            info!("Minimap navigate called: nx={}, ny={}", nx, ny);
            {
                let mut state = state_handle.write();
                let mut changed = false;

                if let Some(vs) = state.active_viewport_mut() {
                    let nx = (nx as f64).clamp(0.0, 1.0);
                    let ny = (ny as f64).clamp(0.0, 1.0);
                    vs.stop();
                    let vp = &mut vs.viewport;
                    let new_x = nx * vp.image_width;
                    let new_y = ny * vp.image_height;
                    info!(
                        "Setting viewport center: ({}, {}) -> ({}, {})",
                        vp.center.x, vp.center.y, new_x, new_y
                    );
                    vp.center.x = new_x;
                    vp.center.y = new_y;
                    changed = true;
                }
                if changed {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui.as_weak();

        ui.on_file_dropped(move |path_str| {
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

                if path.exists()
                    && let Some(ui) = ui_weak.upgrade()
                {
                    open_file(&ui, &state, &tile_cache, &render_timer, path);
                }
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_os_file_drop(move |pane, side, path_str| {
            let path = PathBuf::from(path_str.as_str());
            if !path.exists() {
                return;
            }
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };

            let target_pane = pane_from_index(pane);

            if side == -1 {
                // Drop in center: open as tab in the target pane
                {
                    let mut state = state_handle.write();
                    state.set_focused_pane(target_pane);
                }
                ui.set_focused_pane(pane);
                open_file(&ui, &state_handle, &tile_cache, &render_timer, path);
            } else {
                // Drop on edge: create a new pane split
                let insert_index = if side == 0 {
                    target_pane.0
                } else {
                    target_pane.0 + 1
                };
                let (new_pane, source_pane) = {
                    let mut state = state_handle.write();
                    let source = target_pane;
                    let new_pane = state.insert_pane(insert_index);
                    state.set_focused_pane(new_pane);
                    state.request_render();
                    (new_pane, source)
                };
                insert_pane_ui_state(new_pane, Some(source_pane));
                {
                    let state = state_handle.read();
                    ui.set_split_enabled(state.split_enabled);
                    ui.set_focused_pane(state.focused_pane.as_index());
                    update_tabs(&ui, &state);
                }
                open_file(&ui, &state_handle, &tile_cache, &render_timer, path);
            }
            request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_tool_selected(move |tool_type| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let tool = match tool_type {
                        ToolType::Navigate => state::Tool::Navigate,
                        ToolType::RegionOfInterest => state::Tool::RegionOfInterest,
                        ToolType::MeasureDistance => state::Tool::MeasureDistance,
                    };
                    state.set_tool(tool);
                }
                let state = state_handle.read();
                update_tool_state(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_tool_mouse_down(move |x, y| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    handle_tool_mouse_down(&mut state, x as f64, y as f64);
                }
                let state = state_handle.read();
                update_tool_overlays(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_tool_mouse_move(move |x, y| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    handle_tool_mouse_move(&mut state, x as f64, y as f64);
                }
                let state = state_handle.read();
                update_tool_overlays(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_tool_mouse_up(move |x, y| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    handle_tool_mouse_up(&mut state, x as f64, y as f64);
                }
                let state = state_handle.read();
                update_tool_overlays(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_cancel_tool(move || {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    state.cancel_tool();
                }
                let state = state_handle.read();
                update_tool_state(&ui, &state);
                update_tool_overlays(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_move_tab_to_pane(move |id, from, to| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let target_pane = pane_from_index(to);
                    state.move_tab_between_panes(id, pane_from_index(from), target_pane);
                    state.set_focused_pane(target_pane);
                }
                let state = state_handle.read();
                refresh_tab_ui(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_split_tab_by_drop(move |id, to| {
            if let Some(ui) = ui_weak.upgrade() {
                let (source_pane_for_cache, target_pane_for_cache) = {
                    let mut state = state_handle.write();
                    let source_pane = state
                        .panes
                        .iter()
                        .enumerate()
                        .find(|(_, pane_state)| pane_state.tabs.contains(&id))
                        .map(|(index, _)| PaneId(index))
                        .unwrap_or(PaneId::PRIMARY);
                    state.split_tab_to_new_pane(id, source_pane, to.max(0) as usize);
                    (source_pane, state.focused_pane)
                };
                insert_pane_ui_state(target_pane_for_cache, Some(source_pane_for_cache));
                let state = state_handle.read();
                ui.set_split_enabled(state.split_enabled);
                ui.set_focused_pane(state.focused_pane.as_index());
                update_tabs(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_reorder_tab(move |pane, id, new_index| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    state.reorder_tab(pane_from_index(pane), id, new_index);
                }
                let state = state_handle.read();
                update_tabs(&ui, &state);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_zoom_slider_changed(move |pane, value| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                let mut changed = false;
                if let Some(viewport) = state.active_viewport_mut() {
                    viewport.zoom_to(slider_value_to_zoom(value));
                    changed = true;
                }
                if changed {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_close_active_tab(move || {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let pane = state.focused_pane;
                    if let Some(active_id) = state.active_tab_id_for_pane(pane) {
                        state.close_tab_in_pane(pane, active_id);
                    }
                }
                let state = state_handle.read();
                refresh_tab_ui(&ui, &state);
                ui.invoke_focus_keyboard();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // HUD callbacks
    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_hud_toggle_scale_bar(move |pane| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.show_scale_bar = !hud.show_scale_bar;
                }
            }
            if let Some(ui) = ui_weak.upgrade() {
                let state = state_handle.read();
                update_tabs(&ui, &state);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_hud_toggle_dropdown(move |pane| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.hud_dropdown_open = !hud.hud_dropdown_open;
                }
            }
            if let Some(ui) = ui_weak.upgrade() {
                let state = state_handle.read();
                update_tabs(&ui, &state);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_hud_gamma_changed(move |pane, value| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.gamma = value;
                }
                state.request_render();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_hud_sharpness_changed(move |pane, value| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.sharpness = value;
                }
                state.request_render();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_hud_brightness_changed(move |pane, value| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.brightness = value;
                }
                state.request_render();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_hud_contrast_changed(move |pane, value| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.contrast = value;
                }
                state.request_render();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_hud_measurement_unit_changed(move |pane, unit| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.measurement_unit = measurement_unit_from_slint(unit);
                }
            }
            if let Some(ui) = ui_weak.upgrade() {
                let state = state_handle.read();
                update_tabs(&ui, &state);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_hud_stain_normalization_changed(move |pane, sn| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.stain_normalization = stain_normalization_from_slint(sn);
                }
                state.request_render();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_hud_reset_adjustments(move |pane| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.reset_adjustments();
                }
                state.request_render();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_hud_zoom_input(move |pane, value| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(viewport) = state.active_viewport_mut() {
                    viewport.zoom_to(value as f64);
                }
                state.request_render();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // -- Export dialog callbacks --
    let export_debounce_timer = Rc::new(Timer::default());
    {
        let ui_weak = ui_weak.clone();
        let debounce_timer = Rc::clone(&export_debounce_timer);
        let cached = Rc::clone(&cached_export_settings);

        ui.on_export_dialog_settings_changed(move |settings| {
            // Persist to cache for reuse across dialog opens
            *cached.borrow_mut() = Some(settings);
            let ui_weak = ui_weak.clone();
            debounce_timer.start(
                slint::TimerMode::SingleShot,
                Duration::from_millis(300),
                move || {
                    if let Some(ui) = ui_weak.upgrade() {
                        ui.invoke_export_dialog_request_preview();
                    }
                },
            );
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache_clone = Arc::clone(&tile_cache);
        let ui_weak = ui_weak.clone();
        let overlay_font_preview = overlay_font.clone();

        ui.on_export_dialog_request_preview(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let settings = ui.get_export_settings();
                let state = state_handle.read();
                let pane = state.focused_pane;

                // Render at 96 DPI for preview (thumbnail-sized)
                let preview_image = if let Some(img) = render_export_image(
                    &state,
                    &tile_cache_clone,
                    pane,
                    &settings,
                    Some(96),
                    overlay_font_preview.as_ref(),
                ) {
                    let w = img.width as u32;
                    let h = img.height as u32;
                    crate::blitter::create_image_buffer(&img.pixels, w, h)
                        .map(slint::Image::from_rgba8)
                        .unwrap_or_default()
                } else {
                    ui.get_export_preview_info().preview_image
                };

                let mut info = ui.get_export_preview_info();
                info.preview_image = preview_image;
                ui.set_export_preview_info(info);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache_clone = Arc::clone(&tile_cache);
        let ui_weak = ui_weak.clone();
        let toast_timer = Rc::clone(&toast_timer);
        let overlay_font_export = overlay_font.clone();

        ui.on_export_dialog_export_requested(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let settings = ui.get_export_settings();
            let extension = if settings.format == SlintExportFormat::Png {
                "png"
            } else {
                "jpg"
            };
            let filter_name = if settings.format == SlintExportFormat::Png {
                "PNG Image"
            } else {
                "JPEG Image"
            };

            let dialog = FileDialog::new()
                .add_filter(filter_name, &[extension])
                .set_file_name(format!("export.{extension}"));

            if let Some(path) = dialog.save_file() {
                let state = state_handle.read();
                let pane = state.focused_pane;

                let image_data = render_export_image(
                    &state,
                    &tile_cache_clone,
                    pane,
                    &settings,
                    None,
                    overlay_font_export.as_ref(),
                );
                drop(state);

                let Some(image_data) = image_data else {
                    ui.set_status_text(SharedString::from("No viewport image available to export"));
                    return;
                };

                let width = image_data.width as u32;
                let height = image_data.height as u32;

                let result = if settings.format == SlintExportFormat::Png {
                    image::save_buffer(
                        &path,
                        &image_data.pixels,
                        width,
                        height,
                        image::ExtendedColorType::Rgba8,
                    )
                } else {
                    // Convert RGBA to RGB for JPEG
                    let rgb: Vec<u8> = image_data
                        .pixels
                        .chunks_exact(4)
                        .flat_map(|px| [px[0], px[1], px[2]])
                        .collect();
                    image::save_buffer(&path, &rgb, width, height, image::ExtendedColorType::Rgb8)
                };

                match result {
                    Ok(()) => {
                        ui.set_export_dialog_visible(false);
                        show_toast(
                            &ui,
                            &toast_timer,
                            &format!("Image exported to {}", path.display()),
                        );
                    }
                    Err(e) => {
                        error!("Failed to export image: {e}");
                        ui.set_status_text(SharedString::from(format!("Export failed: {e}")));
                    }
                }
            }
        });
    }

    {
        let ui_weak = ui_weak.clone();
        let cached = Rc::clone(&cached_export_settings);

        ui.on_export_dialog_cancel_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                // Save current settings for reuse
                *cached.borrow_mut() = Some(ui.get_export_settings());
                ui.set_export_dialog_visible(false);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        let cached = Rc::clone(&cached_export_settings);

        ui.on_export_dialog_reset_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let state = state_handle.read();
                let pane = state.focused_pane;
                let settings = build_default_export_settings(&state, pane);
                *cached.borrow_mut() = None; // clear cache on reset
                ui.set_export_settings(settings);
            }
        });
    }

    {
        let ui_weak = ui_weak.clone();

        ui.on_window_minimize(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.window().set_minimized(true);
            }
        });
    }

    {
        let ui_weak = ui_weak.clone();

        ui.on_window_maximize(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let maximized = ui.window().is_maximized();
                ui.window().set_maximized(!maximized);
            }
        });
    }

    {
        let ui_weak = ui_weak.clone();

        ui.on_window_close(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let _ = ui.hide();
                let _ = slint::quit_event_loop();
            }
        });
    }

    #[cfg(target_os = "macos")]
    {
        let ui_weak = ui_weak.clone();
        Timer::single_shot(Duration::from_millis(0), move || {
            if let Some(ui) = ui_weak.upgrade() {
                align_macos_window_controls(ui.window());
            }
        });
    }

    {
        use slint::winit_030::{EventResult, WinitWindowAccessor, winit};

        let state_handle = Arc::clone(&state);
        let last_cursor = Rc::new(Cell::new((0.0f64, 0.0f64)));
        let modifiers = Rc::new(Cell::new(winit::keyboard::ModifiersState::default()));
        let drag_press: Rc<Cell<Option<(f64, f64)>>> = Rc::new(Cell::new(None));
        let last_press_time = Rc::new(Cell::new(std::time::Instant::now()));
        let dbl_count = Rc::new(Cell::new(0u32));
        let os_file_hovering = Rc::new(Cell::new(false));
        let dnd_poll_timer = Rc::new(Timer::default());

        ui.window().on_winit_window_event({
            let last_cursor = Rc::clone(&last_cursor);
            let modifiers = Rc::clone(&modifiers);
            let drag_press = Rc::clone(&drag_press);
            let last_press_time = Rc::clone(&last_press_time);
            let dbl_count = Rc::clone(&dbl_count);
            let os_file_hovering = Rc::clone(&os_file_hovering);
            let dnd_poll_timer = Rc::clone(&dnd_poll_timer);
            let ui_weak = ui_weak.clone();
            let state_handle = Arc::clone(&state_handle);
            let tile_cache = Arc::clone(&tile_cache);
            let render_timer = Rc::clone(&render_timer);

            move |slint_window, event| match event {
                winit::event::WindowEvent::Resized(_physical_size) => {
                    if let Some(ui) = ui_weak.upgrade() {
                        request_render_loop(
                            &render_timer,
                            &ui.as_weak(),
                            &state_handle,
                            &tile_cache,
                        );
                    }
                    EventResult::Propagate
                }
                winit::event::WindowEvent::ScaleFactorChanged { .. } => {
                    if let Some(ui) = ui_weak.upgrade() {
                        request_render_loop(
                            &render_timer,
                            &ui.as_weak(),
                            &state_handle,
                            &tile_cache,
                        );
                    }
                    EventResult::Propagate
                }
                winit::event::WindowEvent::ModifiersChanged(next_modifiers) => {
                    modifiers.set(next_modifiers.state());
                    EventResult::Propagate
                }
                winit::event::WindowEvent::KeyboardInput { event, .. } => {
                    let modifier_state = modifiers.get();
                    let plain_shortcut = event.state == winit::event::ElementState::Pressed
                        && !event.repeat
                        && !modifier_state.control_key()
                        && !modifier_state.alt_key()
                        && !modifier_state.super_key();
                    let close_app_shortcut = event.state == winit::event::ElementState::Pressed
                        && !event.repeat
                        && modifier_state.control_key()
                        && modifier_state.shift_key()
                        && matches!(
                            event.physical_key,
                            winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyW)
                        );

                    if close_app_shortcut {
                        if let Some(ui) = ui_weak.upgrade() {
                            let _ = ui.hide();
                            let _ = slint::quit_event_loop();
                        }
                        return EventResult::PreventDefault;
                    }

                    let handled = if plain_shortcut {
                        use winit::keyboard::{Key, KeyCode, PhysicalKey};

                        let shortcut = match (&event.logical_key, &event.physical_key) {
                            (Key::Character(text), _) if text.eq_ignore_ascii_case("f") => {
                                Some("frame")
                            }
                            (Key::Character(text), _) if text.eq_ignore_ascii_case("m") => {
                                Some("toggle-minimap")
                            }
                            (Key::Character(text), _) if text == "+" => Some("zoom-in"),
                            (Key::Character(text), _) if text == "-" => Some("zoom-out"),
                            (_, PhysicalKey::Code(KeyCode::NumpadAdd)) => Some("zoom-in"),
                            (_, PhysicalKey::Code(KeyCode::NumpadSubtract)) => Some("zoom-out"),
                            _ => None,
                        };

                        if let Some(shortcut) = shortcut {
                            if let Some(ui) = ui_weak.upgrade() {
                                match shortcut {
                                    "frame" => {
                                        let mut state = state_handle.write();
                                        if frame_active_viewport(&mut state) {
                                            state.request_render();
                                        }
                                    }
                                    "zoom-in" => {
                                        let mut state = state_handle.write();
                                        if zoom_active_viewport(&mut state, ACTION_ZOOM_FACTOR) {
                                            state.request_render();
                                        }
                                    }
                                    "zoom-out" => {
                                        let mut state = state_handle.write();
                                        if zoom_active_viewport(
                                            &mut state,
                                            1.0 / ACTION_ZOOM_FACTOR,
                                        ) {
                                            state.request_render();
                                        }
                                    }
                                    "toggle-minimap" => {
                                        toggle_minimap_visibility(&ui, &state_handle);
                                    }
                                    _ => {}
                                }
                                request_render_loop(
                                    &render_timer,
                                    &ui.as_weak(),
                                    &state_handle,
                                    &tile_cache,
                                );
                            }
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if handled {
                        return EventResult::PreventDefault;
                    }

                    EventResult::Propagate
                }
                winit::event::WindowEvent::CursorMoved { position, .. } => {
                    last_cursor.set((position.x, position.y));

                    if let Some((px, py)) = drag_press.get() {
                        let dx = position.x - px;
                        let dy = position.y - py;
                        if dx * dx + dy * dy > 25.0 {
                            drag_press.set(None);

                            let scale = slint_window.scale_factor();
                            let lx = (px / scale as f64) as f32;
                            let ly = (py / scale as f64) as f32;
                            let _ = slint_window.try_dispatch_event(
                                slint::platform::WindowEvent::PointerReleased {
                                    position: slint::LogicalPosition::new(lx, ly),
                                    button: slint::platform::PointerEventButton::Left,
                                },
                            );

                            slint_window.with_winit_window(|w| {
                                let _ = w.drag_window();
                            });
                            return EventResult::PreventDefault;
                        }
                    }
                    EventResult::Propagate
                }
                winit::event::WindowEvent::MouseInput {
                    state: winit::event::ElementState::Pressed,
                    button: winit::event::MouseButton::Left,
                    ..
                } => {
                    let (cx, cy) = last_cursor.get();
                    let scale = slint_window.scale_factor() as f64;
                    let ly = cy / scale;
                    let lx = cx / scale;

                    let toolbar_height = 40.0;
                    let win_w = slint_window.size().width as f64 / scale;
                    let toolbar_action_width = if let Some(ui) = ui_weak.upgrade() {
                        ui.get_toolbar_action_width() as f64
                    } else {
                        0.0
                    };
                    let buttons_right_start = if cfg!(target_os = "macos") {
                        win_w
                    } else {
                        win_w - 138.0
                    };

                    if ly < toolbar_height && lx >= toolbar_action_width && lx < buttons_right_start
                    {
                        drag_press.set(Some((cx, cy)));

                        let now = std::time::Instant::now();
                        if now.duration_since(last_press_time.get()).as_millis() < 400 {
                            dbl_count.set(dbl_count.get() + 1);
                        } else {
                            dbl_count.set(1);
                        }
                        last_press_time.set(now);

                        if dbl_count.get() >= 2 {
                            dbl_count.set(0);
                            drag_press.set(None);
                            let maximized = slint_window.is_maximized();
                            slint_window.set_maximized(!maximized);
                            return EventResult::PreventDefault;
                        }
                    }
                    EventResult::Propagate
                }
                winit::event::WindowEvent::MouseInput {
                    state: winit::event::ElementState::Released,
                    button: winit::event::MouseButton::Left,
                    ..
                } => {
                    drag_press.set(None);
                    EventResult::Propagate
                }
                winit::event::WindowEvent::HoveredFile(_path) => {
                    os_file_hovering.set(true);
                    // Start a polling timer to track cursor position during DnD.
                    // On X11, winit doesn't emit CursorMoved during external DnD,
                    // so we query the pointer position directly from X11.
                    {
                        let ui_weak = ui_weak.clone();
                        let dnd_poll_timer_inner = Rc::clone(&dnd_poll_timer);
                        let os_file_hovering = Rc::clone(&os_file_hovering);
                        dnd_poll_timer.start(
                            slint::TimerMode::Repeated,
                            Duration::from_millis(16),
                            move || {
                                if !os_file_hovering.get() {
                                    dnd_poll_timer_inner.stop();
                                    return;
                                }
                                let Some(ui) = ui_weak.upgrade() else {
                                    dnd_poll_timer_inner.stop();
                                    return;
                                };
                                if let Some((lx, ly)) = query_dnd_cursor_logical(&ui) {
                                    ui.invoke_update_os_file_hover(lx, ly);
                                }
                            },
                        );
                    }
                    EventResult::Propagate
                }
                winit::event::WindowEvent::DroppedFile(path) => {
                    os_file_hovering.set(false);
                    dnd_poll_timer.stop();
                    if let Some(ui) = ui_weak.upgrade() {
                        // Query final cursor position for accurate drop zone
                        if let Some((lx, ly)) = query_dnd_cursor_logical(&ui) {
                            ui.invoke_update_os_file_hover(lx, ly);
                        }
                        let pane = ui.get_os_file_hover_pane();
                        let side = ui.get_os_file_hover_side();
                        ui.invoke_clear_os_file_hover();
                        let path_str = path.to_string_lossy().to_string();
                        ui.invoke_os_file_drop(pane, side, SharedString::from(path_str));
                    }
                    EventResult::Propagate
                }
                winit::event::WindowEvent::HoveredFileCancelled => {
                    os_file_hovering.set(false);
                    dnd_poll_timer.stop();
                    if let Some(ui) = ui_weak.upgrade() {
                        ui.invoke_clear_os_file_hover();
                    }
                    EventResult::Propagate
                }
                _ => EventResult::Propagate,
            }
        });
    }
}

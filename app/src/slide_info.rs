//! The Slide Information panel (AUDIT §14.1): the label, macro and overview
//! images and every property of a slide, with a summary line (format, size,
//! levels and, for multichannel slides, the channels, which the Channels
//! panel adjusts).
//!
//! The label (and the macro photo, which usually shows it) carry patient
//! identifiers: a persisted "hide label" toggle blanks them for teaching
//! and screen sharing. Images are decoded on a worker, bounded by the WSI
//! core's limit for associated images.

use crate::state::{AppState, PaneId};
use crate::{AppWindow, PropertyRow};
use eov_common::{RgbaImageData, WsiFile};
use eov_plugin_api::protocol::NotifyLevel;
use parking_lot::RwLock;
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

/// Longest side of the images shown, in pixels.
const IMAGE_MAX_SIDE: u32 = 640;

#[derive(Default)]
struct Panel {
    /// Every property of the slide shown, in key order.
    rows: Vec<(String, String)>,
    /// Indices into `rows` that match the filter.
    shown: Vec<usize>,
    /// Bumped per open, so late images for another slide are dropped.
    generation: u64,
    /// The slide shown.
    file_id: Option<i32>,
}

thread_local! {
    static PANEL: RefCell<Panel> = RefCell::new(Panel::default());
}

/// The slide the panel shows (or showed last).
pub(crate) fn shown_file_id() -> Option<i32> {
    PANEL.with(|panel| panel.borrow().file_id)
}

/// Opens the panel for the slide behind tab `tab_id`.
pub(crate) fn open_for_tab(ui: &AppWindow, state: &Arc<RwLock<AppState>>, tab_id: i32) {
    let file_id = state.read().resolve_tab_file_id(tab_id);
    open(ui, state, file_id);
}

/// Opens the panel for the slide shown in `pane`.
pub(crate) fn open_for_pane(ui: &AppWindow, state: &Arc<RwLock<AppState>>, pane: PaneId) {
    let Some(file_id) = state.read().active_file_id_for_pane(pane) else {
        return;
    };
    open(ui, state, file_id);
}

fn open(ui: &AppWindow, state: &Arc<RwLock<AppState>>, file_id: i32) {
    let (wsi, title, rows) = {
        let state = state.read();
        let Some(file) = state.get_file(file_id) else {
            return;
        };
        let rows: Vec<(String, String)> = file
            .wsi
            .property_map()
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        (file.wsi.clone(), file.filename.clone(), rows)
    };
    let generation = PANEL.with(|panel| {
        let mut panel = panel.borrow_mut();
        panel.generation += 1;
        panel.shown = (0..rows.len()).collect();
        panel.rows = rows;
        panel.file_id = Some(file_id);
        panel.generation
    });
    show_pixel_size(ui, &state.read(), file_id);
    ui.set_slide_info_summary(SharedString::from(summary(&wsi)));
    ui.set_slide_info_multichannel(wsi.is_multichannel());
    ui.set_slide_info_pixel_size_input(SharedString::new());
    ui.set_slide_info_title(SharedString::from(title));
    ui.set_slide_info_filter(SharedString::new());
    ui.set_slide_info_hide_label(crate::config::load_hide_slide_label());
    for (name, setter) in [
        (
            "label",
            AppWindow::set_slide_info_has_label as fn(&AppWindow, bool),
        ),
        ("macro", AppWindow::set_slide_info_has_macro),
        ("thumbnail", AppWindow::set_slide_info_has_overview),
    ] {
        setter(ui, wsi.associated_image(name).is_some());
    }
    ui.set_slide_info_label(slint::Image::default());
    ui.set_slide_info_macro(slint::Image::default());
    ui.set_slide_info_overview(slint::Image::default());
    apply_rows(ui);
    ui.set_slide_info_visible(true);
    load_images(ui, wsi, generation);
}

/// Channel names listed in the summary before "and N more".
const SUMMARY_CHANNELS: usize = 8;

/// The slide in one line: "OME-TIFF · 24000 × 18000 px · 5 levels ·
/// 12 channels (uint16): DAPI, CD45, …".
fn summary(wsi: &WsiFile) -> String {
    let props = wsi.real_properties();
    let format = match props.vendor.as_deref() {
        Some(eov_common::wsi::OME_TIFF_VENDOR) => "OME-TIFF".to_string(),
        Some(vendor) => vendor.to_string(),
        None => "Slide".to_string(),
    };
    let levels = props.levels.len();
    let mut text = format!(
        "{format} · {} × {} px · {levels} {}",
        props.width,
        props.height,
        if levels == 1 { "level" } else { "levels" }
    );
    use crate::shortcuts::Action;
    let planes = [
        (
            wsi.focal_planes(),
            wsi.focal_plane(),
            "focal planes",
            [Action::NextFocalPlane, Action::PreviousFocalPlane],
        ),
        (
            wsi.time_points(),
            wsi.time_point(),
            "time points",
            [Action::NextTimePoint, Action::PreviousTimePoint],
        ),
    ];
    for (count, current, what, actions) in planes {
        if count <= 1 {
            continue;
        }
        let shortcuts = crate::shortcuts::current();
        let keys = actions
            .map(|action| shortcuts.label(action))
            .into_iter()
            .filter(|key| !key.is_empty())
            .collect::<Vec<_>>()
            .join(" / ");
        text.push_str(&format!(
            " · {count} {what} (showing {}{})",
            current + 1,
            if keys.is_empty() {
                String::new()
            } else {
                format!("; {keys}")
            }
        ));
    }
    if let Some(set) = wsi.channels() {
        let names: Vec<&str> = set
            .channels
            .iter()
            .take(SUMMARY_CHANNELS)
            .map(|c| c.name.as_str())
            .collect();
        let more = set.len().saturating_sub(SUMMARY_CHANNELS);
        text.push_str(&format!(
            " · {} {} ({}): {}{}",
            set.len(),
            if set.len() == 1 {
                "channel"
            } else {
                "channels"
            },
            set.sample_type.name(),
            names.join(", "),
            if more > 0 {
                format!(" and {more} more")
            } else {
                String::new()
            }
        ));
    }
    text
}

/// Opens the Channels panel for the slide shown here.
pub(crate) fn open_channels(host: &crate::host_ui::UiHost) {
    if let Some(file_id) = PANEL.with(|panel| panel.borrow().file_id) {
        crate::channels::open(host, file_id);
    }
}

fn load_images(ui: &AppWindow, wsi: WsiFile, generation: u64) {
    let ui_weak = ui.as_weak();
    let _ = std::thread::Builder::new()
        .name("eov-slide-info".to_string())
        .spawn(move || {
            for name in ["label", "macro", "thumbnail"] {
                if wsi.associated_image(name).is_none() {
                    continue;
                }
                let image = match wsi.read_associated_image(name) {
                    Ok(image) => shrink(image),
                    Err(error) => {
                        tracing::warn!("Could not read the {name} image: {error}");
                        continue;
                    }
                };
                let ui_weak = ui_weak.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    let current = PANEL.with(|panel| panel.borrow().generation);
                    let Some(ui) = ui_weak.upgrade() else {
                        return;
                    };
                    if current != generation {
                        return;
                    }
                    let Some(buffer) = crate::blitter::create_image_buffer(
                        &image.pixels,
                        image.width as u32,
                        image.height as u32,
                    ) else {
                        return;
                    };
                    let image = slint::Image::from_rgba8(buffer);
                    match name {
                        "label" => ui.set_slide_info_label(image),
                        "macro" => ui.set_slide_info_macro(image),
                        _ => ui.set_slide_info_overview(image),
                    }
                });
            }
        });
}

/// Scales an image down to at most [`IMAGE_MAX_SIDE`].
fn shrink(image: RgbaImageData) -> RgbaImageData {
    let (width, height) = (image.width as u32, image.height as u32);
    if width.max(height) <= IMAGE_MAX_SIDE {
        return image;
    }
    let Some(buffer) = image::RgbaImage::from_raw(width, height, image.pixels) else {
        return RgbaImageData::default();
    };
    let scale = f64::from(IMAGE_MAX_SIDE) / f64::from(width.max(height));
    let (w, h) = (
        ((f64::from(width) * scale).round() as u32).max(1),
        ((f64::from(height) * scale).round() as u32).max(1),
    );
    let resized = image::imageops::resize(&buffer, w, h, image::imageops::FilterType::Triangle);
    RgbaImageData {
        width: w as usize,
        height: h as usize,
        pixels: resized.into_raw(),
    }
}

fn apply_rows(ui: &AppWindow) {
    let (rows, total) = PANEL.with(|panel| {
        let panel = panel.borrow();
        let rows: Vec<PropertyRow> = panel
            .shown
            .iter()
            .map(|&index| {
                let (key, value) = &panel.rows[index];
                PropertyRow {
                    key: SharedString::from(key.as_str()),
                    value: SharedString::from(value.as_str()),
                }
            })
            .collect();
        (rows, panel.rows.len())
    });
    ui.set_slide_info_rows(ModelRc::from(Rc::new(VecModel::from(rows))));
    ui.set_slide_info_total_rows(total as i32);
}

/// Indices of the rows whose key or value contains `filter` (ignoring case).
fn matching(rows: &[(String, String)], filter: &str) -> Vec<usize> {
    let needle = filter.trim().to_lowercase();
    rows.iter()
        .enumerate()
        .filter(|(_, (key, value))| {
            needle.is_empty()
                || key.to_lowercase().contains(&needle)
                || value.to_lowercase().contains(&needle)
        })
        .map(|(index, _)| index)
        .collect()
}

pub(crate) fn filter_edited(ui: &AppWindow, filter: &str) {
    PANEL.with(|panel| {
        let mut panel = panel.borrow_mut();
        panel.shown = matching(&panel.rows, filter);
    });
    apply_rows(ui);
}

/// Copies the shown properties as tab-separated `key value` lines.
pub(crate) fn copy_all() {
    let (text, count) = PANEL.with(|panel| {
        let panel = panel.borrow();
        let text: String = panel
            .shown
            .iter()
            .map(|&index| {
                let (key, value) = &panel.rows[index];
                format!("{key}\t{value}\n")
            })
            .collect();
        (text, panel.shown.len())
    });
    crate::clipboard::copy_text(text);
    crate::host_ui::with_host(|host| {
        host.notify(
            NotifyLevel::Success,
            &format!("Copied {count} properties"),
            None,
            None,
        )
    });
}

pub(crate) fn copy_row(ui: &AppWindow, index: i32) {
    let row = PANEL.with(|panel| {
        let panel = panel.borrow();
        usize::try_from(index)
            .ok()
            .and_then(|index| panel.shown.get(index))
            .map(|&index| panel.rows[index].clone())
    });
    if let Some((key, value)) = row {
        crate::clipboard::copy_text(value);
        ui.set_status_text(SharedString::from(format!("Copied {key}")));
    }
}

/// Describes the pixel size that applies to `file_id` and its source.
fn show_pixel_size(ui: &AppWindow, state: &AppState, file_id: i32) {
    let Some(file) = state.get_file(file_id) else {
        return;
    };
    let props = file.wsi.properties();
    let text = match (file.calibration_override, props.mpp_x, props.mpp_y) {
        (Some(um), _, _) => format!("{} µm/px (set by you)", eov_common::format_decimal(um)),
        (None, Some(x), Some(y)) => format!(
            "{} × {} µm/px (from the file)",
            eov_common::format_decimal(x),
            eov_common::format_decimal(y)
        ),
        (None, Some(v), None) | (None, None, Some(v)) => format!(
            "{} µm/px (one axis in the file)",
            eov_common::format_decimal(v)
        ),
        (None, None, None) => "not recorded — measurements are in pixels".to_string(),
    };
    ui.set_slide_info_pixel_size(SharedString::from(text));
    ui.set_slide_info_pixel_size_manual(file.calibration_override.is_some());
}

/// Sets (`Some`) or clears the manual pixel size of the slide shown, here
/// and for future sessions (by slide identity).
pub(crate) fn set_pixel_size(ui: &AppWindow, state: &Arc<RwLock<AppState>>, text: Option<&str>) {
    let Some(file_id) = PANEL.with(|panel| panel.borrow().file_id) else {
        return;
    };
    let um = match text {
        Some(text) => match crate::calibration_store::parse_um_per_px(text) {
            Some(um) => Some(um),
            None => {
                ui.set_status_text(SharedString::from(
                    "Enter the pixel size in µm per pixel, e.g. 0.25",
                ));
                return;
            }
        },
        None => None,
    };
    let (wsi, name) = {
        let state = state.read();
        let Some(file) = state.get_file(file_id) else {
            return;
        };
        (file.wsi.clone(), file.filename.clone())
    };
    // Usually already computed (and cached) when the slide was opened.
    let saved = match wsi.identity() {
        Ok(identity) => crate::calibration_store::store(identity.key(), um, &name),
        Err(error) => Err(error.to_string()),
    };
    if let Err(error) = &saved {
        tracing::warn!("Could not save the pixel size: {error}");
    }
    {
        let mut state = state.write();
        if let Some(file) = state.get_file_mut(file_id) {
            file.calibration_override = um;
        }
        state.request_render();
    }
    show_pixel_size(ui, &state.read(), file_id);
    ui.set_slide_info_pixel_size_input(SharedString::new());
    crate::host_ui::with_host(|host| {
        host.publish_now();
        host.request_render();
        match (saved, um) {
            (Ok(()), Some(um)) => host.notify(
                NotifyLevel::Info,
                &format!("Pixel size set to {} µm/px", eov_common::format_decimal(um)),
                Some("Measurements, the scale bar and exports use it; it is remembered for this slide."),
                None,
            ),
            (Ok(()), None) => host.status("Using the pixel size recorded in the file"),
            (Err(error), _) => host.notify(
                NotifyLevel::Warning,
                "The pixel size applies to this session only",
                Some(&error),
                None,
            ),
        }
    });
}

pub(crate) fn set_hide_label(ui: &AppWindow, hidden: bool) {
    ui.set_slide_info_hide_label(hidden);
    if let Err(error) = crate::config::save_hide_slide_label(hidden) {
        tracing::warn!("Could not save the label setting: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_filter_matches_keys_and_values_ignoring_case() {
        let rows = vec![
            ("aperio.AppMag".to_string(), "40".to_string()),
            ("openslide.vendor".to_string(), "aperio".to_string()),
            (
                "tiff.Software".to_string(),
                "Aperio Image Library".to_string(),
            ),
        ];
        assert_eq!(matching(&rows, ""), [0, 1, 2]);
        assert_eq!(matching(&rows, "VENDOR"), [1]);
        assert_eq!(matching(&rows, "library"), [2]);
        assert!(matching(&rows, "zzz").is_empty());
    }

    #[test]
    fn the_summary_names_the_format_size_and_channels() {
        use eov_common::test_support::{OmeTiffSpec, write_ome_tiff};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.ome.tif");
        let names: Vec<String> = (0..10).map(|i| format!("M{i}")).collect();
        let channels: Vec<(&str, [u8; 3])> = names
            .iter()
            .map(|n| (n.as_str(), [255, 255, 255]))
            .collect();
        write_ome_tiff(
            &path,
            &OmeTiffSpec::channels(64, 48, 32, 2, &channels),
            |c, x, _| f64::from(x) + c as f64,
        )
        .unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        assert_eq!(
            summary(&wsi),
            "OME-TIFF · 64 × 48 px · 2 levels · 10 channels (uint16): M0, M1, M2, M3, M4, M5, \
             M6, M7 and 2 more"
        );
        let rgb = dir.path().join("plain.tif");
        eov_common::test_support::write_tiled_tiff(&rgb, 64, 64, 32, |_, _| [1, 2, 3]).unwrap();
        assert_eq!(
            summary(&WsiFile::open(&rgb).unwrap()),
            "generic-tiff · 64 × 64 px · 1 level"
        );
    }

    #[test]
    fn multichannel_slides_open_the_channels_panel_from_here() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let setup = crate::test_support::HostWithSlide::new();
            let pane = setup.state.read().focused_pane;
            open_for_pane(&setup.ui, &setup.state, pane);
            assert!(setup.ui.get_slide_info_visible());
            assert!(setup.ui.get_slide_info_multichannel());
            assert!(
                setup
                    .ui
                    .get_slide_info_summary()
                    .contains("3 channels (uint16): DAPI, CD45, CD8")
            );
            setup.ui.invoke_slide_info_channels_requested();
            assert!(setup.ui.get_channels_panel_visible());
            assert!(!setup.ui.get_slide_info_visible());
        });
    }

    #[test]
    fn large_images_are_shrunk_keeping_the_aspect() {
        let image = RgbaImageData {
            width: 2000,
            height: 1000,
            pixels: vec![128; 2000 * 1000 * 4],
        };
        let small = shrink(image);
        assert_eq!((small.width, small.height), (640, 320));
    }
}

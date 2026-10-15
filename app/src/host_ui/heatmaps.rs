//! Heat maps from other tools over a slide (AUDIT §14.2): a model's
//! per-patch probabilities, an attention map or any image that covers the
//! slide, read from PNG, JPEG or TIFF ([`eov_common::heatmap`]), placed over
//! the whole slide, its scanned region, or by downsample or pixel size,
//! coloured through a colour map and drawn as a host layer under the
//! annotations. What was shown is remembered per slide identity in
//! `<data>/heatmaps.toml` and shown again when the slide opens.
//!
//! Decoding and colouring run on workers; a newer change makes an older
//! result stale. Opacity and visibility are properties of the layer and
//! need no new colouring.

use super::{NotifyLevel, UiHost, with_host};
use crate::HeatMapPanel;
use crate::overlay::store::{Owner, StoredRaster};
use crate::{AppWindow, ChoiceItem};
use eov_common::heatmap::{self, ColorMap, HeatMap, HeatStyle, Placement, SlideRect};
use eov_plugin_api::protocol::{FileId, LayerSpec, LayerUpdate, Rect};
use serde::{Deserialize, Serialize};
use slint::{ComponentHandle, Image, ModelRc, Rgba8Pixel, SharedPixelBuffer, VecModel};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

/// Under the region of interest and every annotation.
const HEATMAP_Z: i32 = -60;

const LAYER_PREFIX: &str = "host-heatmap-";

fn layer_id(file_id: i32) -> String {
    format!("{LAYER_PREFIX}{file_id}")
}

/// Whether a host layer is a heat map (exports draw it like the view).
pub(crate) fn is_heatmap_layer(layer_id: &str) -> bool {
    layer_id.starts_with(LAYER_PREFIX)
}

/// A heat map shown over an open file.
struct Shown {
    path: PathBuf,
    map: Arc<HeatMap>,
    placement: Placement,
    /// What the user typed for a downsample or pixel size.
    scale_text: String,
    color_map: ColorMap,
    threshold: f32,
    opacity: f32,
    smooth: bool,
    visible: bool,
    /// Where it lies, or why it cannot be placed.
    placed: Result<SlideRect, String>,
    /// Raised by every change that needs new colouring.
    generation: u64,
}

thread_local! {
    static SHOWN: RefCell<HashMap<i32, Shown>> = RefCell::new(HashMap::new());
    /// The file the panel shows.
    static PANEL_FILE: Cell<Option<i32>> = const { Cell::new(None) };
    /// One counter for every map, so a result for a replaced map is never
    /// taken for its successor's.
    static GENERATION: Cell<u64> = const { Cell::new(0) };
}

fn next_generation() -> u64 {
    GENERATION.with(|generation| {
        generation.set(generation.get() + 1);
        generation.get()
    })
}

// -- persistence ---------------------------------------------------------------

/// What is remembered for a slide.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Saved {
    path: PathBuf,
    placement: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scale: Option<f64>,
    color_map: String,
    threshold: f32,
    opacity: f32,
    smooth: bool,
    visible: bool,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Store {
    #[serde(default)]
    slide: BTreeMap<String, Saved>,
}

fn store_path() -> PathBuf {
    crate::paths::data_dir().join("heatmaps.toml")
}

fn read_store(path: &Path) -> Store {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| toml::from_str(&text).ok())
        .unwrap_or_default()
}

fn write_store(path: &Path, store: &Store) -> Result<(), String> {
    let text = toml::to_string(store).map_err(|error| error.to_string())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let temp = path.with_extension("toml.tmp");
    std::fs::write(&temp, text).map_err(|error| error.to_string())?;
    std::fs::rename(&temp, path).map_err(|error| error.to_string())
}

fn saved_for(key: &str) -> Option<Saved> {
    read_store(&store_path()).slide.get(key).cloned()
}

fn remember(key: &str, saved: Option<Saved>) {
    let path = store_path();
    let mut store = read_store(&path);
    match saved {
        Some(saved) => store.slide.insert(key.to_string(), saved),
        None => store.slide.remove(key),
    };
    if let Err(error) = write_store(&path, &store) {
        tracing::warn!("Could not save the heat maps: {error}");
    }
}

fn placement_from(id: &str, scale: Option<f64>) -> Placement {
    match (id, scale) {
        ("fit-bounds", _) => Placement::FitBounds,
        ("downsample", Some(value)) => Placement::Downsample(value),
        ("micrometres", Some(value)) => Placement::Micrometres(value),
        _ => Placement::FitSlide,
    }
}

fn parse_scale(text: &str) -> Option<f64> {
    text.trim()
        .replace(',', ".")
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && *value > 0.0)
}

impl Shown {
    fn saved(&self) -> Saved {
        Saved {
            path: self.path.clone(),
            placement: self.placement.id().to_string(),
            scale: match self.placement {
                Placement::Downsample(value) | Placement::Micrometres(value) => Some(value),
                _ => None,
            },
            color_map: self.color_map.id().to_string(),
            threshold: self.threshold,
            opacity: self.opacity,
            smooth: self.smooth,
            visible: self.visible,
        }
    }
}

// -- the host side ---------------------------------------------------------------

impl UiHost {
    /// Viewport menu: Heat Map… for the slide the menu was opened on
    /// (choosing an image first when none is shown).
    pub(super) fn open_heatmap_panel_for(&self, file_id: i32) {
        if SHOWN.with(|shown| shown.borrow().contains_key(&file_id)) {
            PANEL_FILE.with(|panel| panel.set(Some(file_id)));
            self.refresh_heatmap_panel();
            return;
        }
        self.choose_heatmap(file_id);
    }

    fn choose_heatmap(&self, file_id: i32) {
        let Some(path) = rfd::FileDialog::new()
            .set_title("Heat map")
            .add_filter("Images", &["png", "jpg", "jpeg", "tif", "tiff"])
            .pick_file()
        else {
            return;
        };
        let previous = SHOWN.with(|shown| shown.borrow().get(&file_id).map(Shown::saved));
        let settings = Saved {
            path: path.clone(),
            ..previous.unwrap_or_else(|| Saved {
                path: path.clone(),
                placement: "fit-slide".to_string(),
                scale: None,
                color_map: ColorMap::default().id().to_string(),
                threshold: 0.0,
                opacity: HeatStyle::default().opacity,
                smooth: false,
                visible: true,
            })
        };
        self.load_heatmap(file_id, settings, true);
    }

    /// The slide's identity is known: shows the heat map remembered for it.
    pub(crate) fn restore_heatmap(&self, file_id: i32, key: &str) {
        let Some(saved) = saved_for(key) else {
            return;
        };
        if !saved.path.is_file() {
            self.notify(
                NotifyLevel::Info,
                "The heat map shown on this slide before was not found",
                Some(&saved.path.display().to_string()),
                None,
            );
            return;
        }
        self.load_heatmap(file_id, saved, false);
    }

    /// Reads `settings.path` on a worker and shows it over `file_id`.
    fn load_heatmap(&self, file_id: i32, settings: Saved, open_panel: bool) {
        let job = self.start_host_job("Reading the heat map", None);
        let spawned = std::thread::Builder::new()
            .name("eov-heatmap-read".to_string())
            .spawn({
                let job = job.clone();
                let settings = settings.clone();
                move || {
                    let decoded = heatmap::decode(&settings.path);
                    let _ = slint::invoke_from_event_loop(move || {
                        with_host(|host| {
                            host.finish_host_job(&job);
                            host.heatmap_read(file_id, settings, decoded, open_panel);
                        });
                    });
                }
            });
        if let Err(error) = spawned {
            self.finish_host_job(&job);
            self.heatmap_read(file_id, settings, Err(error.to_string()), open_panel);
        }
    }

    fn heatmap_read(
        &self,
        file_id: i32,
        settings: Saved,
        decoded: Result<HeatMap, String>,
        open_panel: bool,
    ) {
        if self.state.read().get_file(file_id).is_none() {
            return;
        }
        let map = match decoded {
            Ok(map) => map,
            Err(error) => {
                let name = settings
                    .path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                self.notify(
                    NotifyLevel::Error,
                    &format!("Could not show {name} as a heat map"),
                    Some(&error),
                    None,
                );
                return;
            }
        };
        let placement = placement_from(&settings.placement, settings.scale);
        let shown = Shown {
            path: settings.path.clone(),
            map: Arc::new(map),
            placement,
            scale_text: settings.scale.map(trim_number).unwrap_or_default(),
            color_map: ColorMap::from_id(&settings.color_map).unwrap_or_default(),
            threshold: settings.threshold.clamp(0.0, 1.0),
            opacity: settings.opacity.clamp(0.0, 1.0),
            smooth: settings.smooth,
            visible: settings.visible,
            placed: Err(String::new()),
            generation: 0,
        };
        SHOWN.with(|all| all.borrow_mut().insert(file_id, shown));
        self.recolour_heatmap(file_id);
        if open_panel {
            PANEL_FILE.with(|panel| panel.set(Some(file_id)));
        }
        self.refresh_heatmap_panel();
        self.save_heatmap(file_id);
    }

    /// Where the map of `file_id` lies, from the slide's size, scanned
    /// region and pixel size.
    fn heatmap_place(
        &self,
        file_id: i32,
        map: &HeatMap,
        placement: Placement,
    ) -> Result<SlideRect, String> {
        let state = self.state.read();
        let file = state
            .get_file(file_id)
            .ok_or_else(|| "the slide is closed".to_string())?;
        let props = file.wsi.properties();
        let bounds = props.bounds.map(|b| SlideRect {
            x: b.x as f64,
            y: b.y as f64,
            width: b.width as f64,
            height: b.height as f64,
        });
        let mpp = super::calibration(file).spacing().map(|(x, y, _)| (x, y));
        heatmap::placement_rect(
            map.width,
            map.height,
            placement,
            (props.width, props.height),
            bounds,
            mpp,
        )
    }

    /// Places and colours the map of `file_id` again (on a worker).
    fn recolour_heatmap(&self, file_id: i32) {
        let job = SHOWN.with(|all| {
            let mut all = all.borrow_mut();
            let shown = all.get_mut(&file_id)?;
            shown.generation = next_generation();
            Some((
                Arc::clone(&shown.map),
                shown.placement,
                shown.color_map,
                shown.threshold,
                shown.smooth,
                shown.generation,
            ))
        });
        let Some((map, placement, color_map, threshold, smooth, generation)) = job else {
            return;
        };
        let placed = self.heatmap_place(file_id, &map, placement);
        SHOWN.with(|all| {
            if let Some(shown) = all.borrow_mut().get_mut(&file_id) {
                shown.placed = placed.clone();
            }
        });
        let layer = layer_id(file_id);
        let Ok(rect) = placed else {
            // Nothing sensible to draw; the panel says why.
            let _ = self.store.clear_layer(&Owner::Host, &layer);
            self.request_render();
            return;
        };
        let style = HeatStyle {
            color_map,
            opacity: 1.0,
            threshold,
        };
        let spawned = std::thread::Builder::new()
            .name("eov-heatmap-colour".to_string())
            .spawn(move || {
                let tiles = colour_tiles(&map, &rect, &style, smooth);
                let _ = slint::invoke_from_event_loop(move || {
                    with_host(|host| host.heatmap_coloured(file_id, generation, tiles));
                });
            });
        if let Err(error) = spawned {
            tracing::error!("Cannot colour the heat map: {error}");
        }
    }

    fn heatmap_coloured(&self, file_id: i32, generation: u64, tiles: Vec<StoredRaster>) {
        let current = SHOWN.with(|all| {
            all.borrow()
                .get(&file_id)
                .filter(|shown| shown.generation == generation)
                .map(|shown| (shown.opacity, shown.visible))
        });
        let Some((opacity, visible)) = current else {
            return;
        };
        let layer = layer_id(file_id);
        if self.store.layer(&Owner::Host, &layer).is_none() {
            let spec = LayerSpec::new(layer.clone(), FileId::from(file_id))
                .named("Heat map")
                .at_z(HEATMAP_Z);
            if let Err(error) = self.store.create_layer(Owner::Host, spec) {
                tracing::error!("Cannot show the heat map: {error}");
                return;
            }
        }
        let _ = self.store.clear_layer(&Owner::Host, &layer);
        for tile in tiles {
            if let Err(error) = self.store.put_raster(&Owner::Host, &layer, tile) {
                tracing::error!("Cannot show the heat map: {error}");
            }
        }
        self.apply_heatmap_layer(file_id, opacity, visible);
    }

    fn apply_heatmap_layer(&self, file_id: i32, opacity: f32, visible: bool) {
        let _ = self.store.update_layer(
            &Owner::Host,
            LayerUpdate {
                layer_id: layer_id(file_id),
                opacity: Some(opacity),
                visible: Some(visible),
                ..LayerUpdate::default()
            },
        );
        self.request_render();
    }

    fn save_heatmap(&self, file_id: i32) {
        let Some(key) = self.identity_key(FileId::from(file_id)) else {
            return;
        };
        let saved = SHOWN.with(|all| all.borrow().get(&file_id).map(Shown::saved));
        remember(&key, saved);
    }

    /// The file closed: its layer went with it.
    pub(crate) fn heatmap_file_closed(&self, file_id: i32) {
        SHOWN.with(|all| all.borrow_mut().remove(&file_id));
        if PANEL_FILE.with(Cell::get) == Some(file_id) {
            PANEL_FILE.with(|panel| panel.set(None));
            self.refresh_heatmap_panel();
        }
    }

    // -- the panel -----------------------------------------------------------------

    fn refresh_heatmap_panel(&self) {
        let Some(ui) = self.ui() else {
            return;
        };
        let panel = ui.global::<HeatMapPanel>();
        let file_id = PANEL_FILE.with(Cell::get);
        let filled = file_id.and_then(|file_id| {
            let slide = self.state.read().get_file(file_id).map(|file| {
                (
                    file.filename.clone(),
                    file.wsi.properties().bounds.is_some(),
                )
            })?;
            SHOWN.with(|all| {
                let all = all.borrow();
                let shown = all.get(&file_id)?;
                fill_panel(&ui, &slide.0, slide.1, shown);
                Some(())
            })
        });
        panel.set_visible(filled.is_some());
    }

    pub(super) fn heatmap_placement_chosen(&self, id: &str) {
        self.edit_heatmap(true, |shown| {
            let scale = parse_scale(&shown.scale_text);
            shown.placement = match id {
                "downsample" | "micrometres" => placement_from(id, scale.or(Some(1.0))),
                other => placement_from(other, None),
            };
            if matches!(id, "downsample" | "micrometres") && scale.is_none() {
                shown.scale_text = "1".to_string();
            }
        });
    }

    pub(super) fn heatmap_scale_edited(&self, text: &str) {
        let value = parse_scale(text);
        // A number being typed keeps the map where it is until it is valid.
        self.edit_heatmap(value.is_some(), |shown| {
            shown.scale_text = text.to_string();
            match (shown.placement, value) {
                (Placement::Downsample(_), Some(value)) => {
                    shown.placement = Placement::Downsample(value)
                }
                (Placement::Micrometres(_), Some(value)) => {
                    shown.placement = Placement::Micrometres(value)
                }
                (Placement::Downsample(_) | Placement::Micrometres(_), None) => {
                    shown.placed = Err("Type a positive number.".to_string());
                }
                _ => {}
            }
        });
    }

    pub(super) fn heatmap_color_map_chosen(&self, id: &str) {
        if let Some(color_map) = ColorMap::from_id(id) {
            self.edit_heatmap(true, |shown| shown.color_map = color_map);
        }
    }

    /// Opacity, threshold, smoothing or visibility changed in the panel.
    pub(super) fn heatmap_style_changed(&self) {
        let Some(ui) = self.ui() else {
            return;
        };
        let panel = ui.global::<HeatMapPanel>();
        let (opacity, threshold, smooth, visible) = (
            panel.get_opacity().clamp(0.0, 1.0),
            panel.get_threshold().clamp(0.0, 1.0),
            panel.get_smooth(),
            panel.get_shown(),
        );
        let mut recolour = false;
        self.edit_heatmap(false, |shown| {
            recolour = shown.smooth != smooth || (shown.threshold - threshold).abs() > 1e-6;
            shown.opacity = opacity;
            shown.threshold = threshold;
            shown.smooth = smooth;
            shown.visible = visible;
        });
        if let Some(file_id) = PANEL_FILE.with(Cell::get) {
            if recolour {
                self.recolour_heatmap(file_id);
            } else {
                self.apply_heatmap_layer(file_id, opacity, visible);
            }
        }
    }

    /// Applies `edit` to the panel's map, then colours it again when
    /// `recolour`, saves it and refreshes the panel.
    fn edit_heatmap(&self, recolour: bool, edit: impl FnOnce(&mut Shown)) {
        let Some(file_id) = PANEL_FILE.with(Cell::get) else {
            return;
        };
        let edited = SHOWN.with(|all| {
            let mut all = all.borrow_mut();
            let shown = all.get_mut(&file_id)?;
            edit(shown);
            Some(())
        });
        if edited.is_none() {
            return;
        }
        if recolour {
            self.recolour_heatmap(file_id);
        }
        self.save_heatmap(file_id);
        self.refresh_heatmap_panel();
    }

    pub(super) fn heatmap_replace(&self) {
        if let Some(file_id) = PANEL_FILE.with(Cell::get) {
            self.choose_heatmap(file_id);
        }
    }

    pub(super) fn heatmap_remove(&self) {
        let Some(file_id) = PANEL_FILE.with(Cell::get) else {
            return;
        };
        SHOWN.with(|all| all.borrow_mut().remove(&file_id));
        self.store.remove_layer(&Owner::Host, &layer_id(file_id));
        self.save_heatmap(file_id);
        PANEL_FILE.with(|panel| panel.set(None));
        self.refresh_heatmap_panel();
        self.request_render();
    }

    pub(super) fn heatmap_close(&self) {
        PANEL_FILE.with(|panel| panel.set(None));
        self.refresh_heatmap_panel();
    }

    /// Whether a heat map is shown over `file_id` (for the menu's wording).
    pub(super) fn has_heatmap(&self, file_id: i32) -> bool {
        SHOWN.with(|all| all.borrow().contains_key(&file_id))
    }
}

/// The map's tiles, coloured, as rasters over `rect` (worker thread).
fn colour_tiles(
    map: &HeatMap,
    rect: &SlideRect,
    style: &HeatStyle,
    smooth: bool,
) -> Vec<StoredRaster> {
    heatmap::tiles(map.width, map.height)
        .into_iter()
        .map(|tile| {
            let (x0, y0, width, height) = tile;
            let r = heatmap::tile_rect(rect, map.width, map.height, tile);
            StoredRaster {
                object_id: format!("tile-{x0}-{y0}"),
                rect: Rect {
                    x: r.x,
                    y: r.y,
                    width: r.width,
                    height: r.height,
                },
                width,
                height,
                premultiplied: Arc::new(heatmap::render_block(map, x0, y0, width, height, style)),
                smooth,
                seq: 0,
            }
        })
        .collect()
}

fn trim_number(value: f64) -> String {
    let text = format!("{value:.4}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn choices(items: &[(&str, &str)]) -> ModelRc<ChoiceItem> {
    ModelRc::from(Rc::new(VecModel::from(
        items
            .iter()
            .map(|(id, label)| ChoiceItem {
                id: (*id).into(),
                label: (*label).into(),
            })
            .collect::<Vec<_>>(),
    )))
}

fn legend_image(color_map: ColorMap) -> Image {
    const WIDTH: u32 = 256;
    let pixels = heatmap::legend(color_map, WIDTH);
    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(WIDTH, 1);
    buffer.make_mut_bytes().copy_from_slice(&pixels);
    Image::from_rgba8(buffer)
}

fn describe_rect(rect: &SlideRect, slide_wide: bool) -> String {
    let whole = if slide_wide { " (the whole slide)" } else { "" };
    format!(
        "Covers {} × {} level-0 pixels from ({}, {}){whole}.",
        rect.width.round(),
        rect.height.round(),
        rect.x.round(),
        rect.y.round()
    )
}

fn fill_panel(ui: &AppWindow, slide_name: &str, has_bounds: bool, shown: &Shown) {
    let panel = ui.global::<HeatMapPanel>();
    panel.set_title(format!("Heat Map · {slide_name}").into());
    let name = shown
        .path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    panel.set_summary(
        format!(
            "{name} · {} × {} px · {}",
            shown.map.width, shown.map.height, shown.map.scaling
        )
        .into(),
    );
    let mut placements = vec![("fit-slide", "Fit the slide")];
    if has_bounds {
        placements.push(("fit-bounds", "Fit the scanned region"));
    }
    placements.push(("downsample", "Downsample"));
    placements.push(("micrometres", "µm per pixel"));
    panel.set_placements(choices(&placements));
    panel.set_placement(shown.placement.id().into());
    panel.set_scale_label(
        match shown.placement {
            Placement::Downsample(_) => "Level-0 pixels per map pixel",
            Placement::Micrometres(_) => "Micrometres per map pixel",
            _ => "",
        }
        .into(),
    );
    panel.set_scale_text(shown.scale_text.clone().into());
    let (note, ok) = match &shown.placed {
        Ok(rect) => (
            describe_rect(rect, shown.placement == Placement::FitSlide),
            true,
        ),
        Err(problem) => (problem.clone(), false),
    };
    panel.set_placement_note(note.into());
    panel.set_placement_ok(ok);
    panel.set_scalar(shown.map.is_scalar());
    let maps: Vec<(&str, &str)> = ColorMap::ALL
        .iter()
        .map(|map| (map.id(), map.label()))
        .collect();
    panel.set_color_maps(choices(&maps));
    panel.set_color_map(shown.color_map.id().into());
    panel.set_legend(legend_image(shown.color_map));
    panel.set_opacity(shown.opacity);
    panel.set_threshold(shown.threshold);
    panel.set_smooth(shown.smooth);
    panel.set_shown(shown.visible);
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_common::heatmap::HeatValues;

    /// Colours the map of `file_id` at once, as the worker would.
    fn colour_now(host: &UiHost, file_id: i32) {
        let (map, rect, style, smooth, generation) = SHOWN.with(|all| {
            let all = all.borrow();
            let shown = &all[&file_id];
            (
                Arc::clone(&shown.map),
                shown.placed.clone().expect("placed"),
                HeatStyle {
                    color_map: shown.color_map,
                    opacity: 1.0,
                    threshold: shown.threshold,
                },
                shown.smooth,
                shown.generation,
            )
        });
        host.heatmap_coloured(
            file_id,
            generation,
            colour_tiles(&map, &rect, &style, smooth),
        );
    }

    fn raster_rects(host: &UiHost, file_id: i32) -> Vec<Rect> {
        host.store
            .layer(&Owner::Host, &layer_id(file_id))
            .map(|layer| layer.rasters().map(|raster| raster.rect).collect())
            .unwrap_or_default()
    }

    #[test]
    fn the_panel_places_colours_restyles_and_removes_a_map() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let setup = crate::test_support::HostWithSlide::new();
            let (host, file_id) = (&setup.host, setup.file_id);
            let map = HeatMap {
                width: 4,
                height: 3,
                values: HeatValues::Scalar((0..12).map(|i| i as f32 / 11.0).collect()),
                scaling: "8-bit values".to_string(),
            };
            let settings = Saved {
                path: PathBuf::from("/maps/attention.png"),
                placement: "fit-slide".to_string(),
                scale: None,
                color_map: "viridis".to_string(),
                threshold: 0.0,
                opacity: 0.5,
                smooth: false,
                visible: true,
            };
            host.heatmap_read(file_id, settings, Ok(map), true);
            let panel = setup.ui.global::<HeatMapPanel>();
            assert!(panel.get_visible());
            assert!(panel.get_summary().contains("attention.png · 4 × 3 px"));
            assert!(panel.get_scalar());
            assert!(
                panel.get_placement_note().contains("512 × 384"),
                "{}",
                panel.get_placement_note()
            );
            colour_now(host, file_id);
            let whole = Rect {
                x: 0.0,
                y: 0.0,
                width: 512.0,
                height: 384.0,
            };
            assert_eq!(raster_rects(host, file_id), [whole]);
            let layer = host.store.layer(&Owner::Host, &layer_id(file_id)).unwrap();
            assert!((layer.spec.opacity - 0.5).abs() < 1e-6);
            assert_eq!(layer.spec.z, HEATMAP_Z);

            // One map pixel per 64 level-0 pixels.
            host.heatmap_placement_chosen("downsample");
            host.heatmap_scale_edited("64");
            assert!(panel.get_placement_note().contains("256 × 192"));
            colour_now(host, file_id);
            assert_eq!(raster_rects(host, file_id)[0].width, 256.0);
            // A number being typed leaves the map where it is.
            host.heatmap_scale_edited("-");
            assert!(!panel.get_placement_ok());
            assert_eq!(raster_rects(host, file_id)[0].width, 256.0);

            // Opacity and visibility change the layer, not the colours.
            let generation = SHOWN.with(|all| all.borrow()[&file_id].generation);
            panel.set_opacity(0.8);
            panel.set_shown(false);
            host.heatmap_style_changed();
            let layer = host.store.layer(&Owner::Host, &layer_id(file_id)).unwrap();
            assert!((layer.spec.opacity - 0.8).abs() < 1e-6 && !layer.spec.visible);
            assert_eq!(
                SHOWN.with(|all| all.borrow()[&file_id].generation),
                generation
            );
            // A threshold needs new colours; a stale result is dropped.
            panel.set_threshold(0.5);
            host.heatmap_style_changed();
            host.heatmap_coloured(file_id, generation, Vec::new());
            assert_eq!(
                raster_rects(host, file_id).len(),
                1,
                "the stale result was ignored"
            );

            assert!(host.has_heatmap(file_id));
            host.heatmap_remove();
            assert!(host.store.layer(&Owner::Host, &layer_id(file_id)).is_none());
            assert!(!panel.get_visible());
            assert!(!host.has_heatmap(file_id));
        });
    }

    #[test]
    fn settings_round_trip_through_the_store() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("heatmaps.toml");
        let mut store = Store::default();
        let saved = Saved {
            path: PathBuf::from("/maps/attention.png"),
            placement: "downsample".to_string(),
            scale: Some(256.0),
            color_map: "inferno".to_string(),
            threshold: 0.1,
            opacity: 0.6,
            smooth: true,
            visible: false,
        };
        store.slide.insert("slide-key".to_string(), saved.clone());
        write_store(&path, &store).unwrap();
        assert_eq!(read_store(&path).slide["slide-key"], saved);
        assert_eq!(
            placement_from(&saved.placement, saved.scale),
            Placement::Downsample(256.0)
        );
        assert_eq!(placement_from("micrometres", None), Placement::FitSlide);
        assert_eq!(parse_scale(" 0,5 "), Some(0.5));
        assert_eq!(parse_scale("-2"), None);
    }
}

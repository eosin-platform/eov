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
use eov_plugin_api::i18n::{self, isolate, tr};
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

/// The placement of a newly chosen map: [`Placement::FromFile`] when the
/// file records one, else over the whole slide.
const AUTO_PLACEMENT: &str = "auto";

/// Why a map has no place: kept in English (as eov_common gives its
/// reasons) and translated when the panel shows it, see
/// [`placement_problem_text`].
const SLIDE_CLOSED: &str = "the slide is closed";
const TYPE_A_NUMBER: &str = "Type a positive number.";

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
    /// The scale field's text: what the user typed for a downsample or
    /// pixel size, or `scale` as eov wrote it.
    scale_text: String,
    /// The last number the scale field held. It is kept while the
    /// placement does not use one (choosing Downsample or µm per pixel
    /// again finds it), and it is what the field shows after a change of
    /// language: text is only ever read in the language it was written in
    /// ("4.032" is four thousand and thirty-two in German).
    scale: Option<f64>,
    /// Whether `scale_text` is `scale` written out (not text being typed
    /// that is not a number).
    scale_text_is_scale: bool,
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
    threshold: f64,
    opacity: f64,
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

/// A slider value as people would write it (0.3, not 0.30000001…).
fn rounded(value: f32) -> f64 {
    (f64::from(value) * 1000.0).round() / 1000.0
}

fn placement_from(id: &str, scale: Option<f64>) -> Placement {
    match (id, scale) {
        ("fit-bounds", _) => Placement::FitBounds,
        ("from-file", _) => Placement::FromFile,
        ("downsample", Some(value)) => Placement::Downsample(value),
        ("micrometres", Some(value)) => Placement::Micrometres(value),
        _ => Placement::FitSlide,
    }
}

fn parse_scale(text: &str) -> Option<f64> {
    i18n::parse_decimal(text).filter(|value| *value > 0.0)
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
            threshold: rounded(self.threshold),
            opacity: rounded(self.opacity),
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
            .set_title(tr!("Heat map"))
            .add_filter(tr!("Images"), &["png", "jpg", "jpeg", "tif", "tiff"])
            .pick_file()
        else {
            return;
        };
        let previous = SHOWN.with(|shown| shown.borrow().get(&file_id).map(Shown::saved));
        let settings = Saved {
            path: path.clone(),
            ..previous.unwrap_or_else(|| Saved {
                path: path.clone(),
                // Where the file says it lies, else over the slide.
                placement: AUTO_PLACEMENT.to_string(),
                scale: None,
                color_map: ColorMap::default().id().to_string(),
                threshold: 0.0,
                opacity: f64::from(HeatStyle::default().opacity),
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
                &tr!("The heat map shown on this slide before was not found"),
                Some(&saved.path.display().to_string()),
                None,
            );
            return;
        }
        self.load_heatmap(file_id, saved, false);
    }

    /// Reads `settings.path` on a worker and shows it over `file_id`.
    fn load_heatmap(&self, file_id: i32, settings: Saved, open_panel: bool) {
        let job = self.start_host_job(&tr!("Reading the heat map"), None);
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
                    &tr!("Could not show {} as a heat map", isolate(&name)),
                    Some(&error),
                    None,
                );
                return;
            }
        };
        let placement = if settings.placement == AUTO_PLACEMENT {
            if map.geometry.is_some() {
                Placement::FromFile
            } else {
                Placement::FitSlide
            }
        } else {
            placement_from(&settings.placement, settings.scale)
        };
        // The number saved, when it is one the field takes.
        let scale = settings
            .scale
            .filter(|value| value.is_finite() && *value > 0.0);
        let shown = Shown {
            path: settings.path.clone(),
            map: Arc::new(map),
            placement,
            scale_text: settings.scale.map(trim_number).unwrap_or_default(),
            scale,
            scale_text_is_scale: scale == settings.scale,
            color_map: ColorMap::from_id(&settings.color_map).unwrap_or_default(),
            threshold: settings.threshold.clamp(0.0, 1.0) as f32,
            opacity: settings.opacity.clamp(0.0, 1.0) as f32,
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
            .ok_or_else(|| SLIDE_CLOSED.to_string())?;
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
            map.geometry,
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

    pub(super) fn refresh_heatmap_panel(&self) {
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
            shown.placement = match id {
                "downsample" | "micrometres" => {
                    // The field's number (the text is not read again: it
                    // was read as it was typed), else 1.
                    let value = shown.scale.unwrap_or(1.0);
                    if shown.scale.is_none() || !shown.scale_text_is_scale {
                        shown.scale_text = trim_number(value);
                        shown.scale_text_is_scale = true;
                    }
                    shown.scale = Some(value);
                    placement_from(id, Some(value))
                }
                other => placement_from(other, None),
            };
        });
    }

    pub(super) fn heatmap_scale_edited(&self, text: &str) {
        let value = parse_scale(text);
        // A number being typed keeps the map where it is until it is valid.
        self.edit_heatmap(value.is_some(), |shown| {
            shown.scale_text = text.to_string();
            shown.scale_text_is_scale = value.is_some();
            if value.is_some() {
                shown.scale = value;
            }
            match (shown.placement, value) {
                (Placement::Downsample(_), Some(value)) => {
                    shown.placement = Placement::Downsample(value)
                }
                (Placement::Micrometres(_), Some(value)) => {
                    shown.placement = Placement::Micrometres(value)
                }
                (Placement::Downsample(_) | Placement::Micrometres(_), None) => {
                    shown.placed = Err(TYPE_A_NUMBER.to_string());
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

    /// The interface changed language: the scale of every map (also of
    /// slides whose panel is closed) is written again with the new decimal
    /// separator, from its number. Text that is not a number is left as it
    /// was typed. Nothing is placed or saved again: the numbers are the
    /// same.
    pub(super) fn heatmap_language_changed(&self) {
        SHOWN.with(|all| {
            for shown in all.borrow_mut().values_mut() {
                if shown.scale_text_is_scale
                    && let Some(value) = shown.scale
                {
                    shown.scale_text = trim_number(value);
                }
            }
        });
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

/// A number for the scale field, with the language's decimal separator
/// (written again when the language changes, see
/// [`UiHost::heatmap_language_changed`]).
fn trim_number(value: f64) -> String {
    let text = format!("{value:.4}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    let separator = i18n::current().decimal_separator;
    if separator == '.' {
        text.to_string()
    } else {
        text.replacen('.', &separator.to_string(), 1)
    }
}

/// Why a map cannot be placed, in the language of the interface: the
/// reasons eov_common gives (in English) and this module's own. `None`
/// for a reason this list does not know.
fn known_placement_problem(problem: &str) -> Option<String> {
    Some(match problem {
        "the file does not say where it lies" => tr!("the file does not say where it lies"),
        "the slide does not report a scanned region" => {
            tr!("the slide does not report a scanned region")
        }
        "the downsample must be a positive number" => {
            tr!("the downsample must be a positive number")
        }
        "the pixel size must be a positive number" => {
            tr!("the pixel size must be a positive number")
        }
        "the slide has no pixel size; set it in Slide Information, or place the map another way" => {
            tr!(
                "the slide has no pixel size; set it in Slide Information, or place the map another way"
            )
        }
        "the heat map would cover nothing" => tr!("the heat map would cover nothing"),
        SLIDE_CLOSED => tr!("the slide is closed"),
        TYPE_A_NUMBER => tr!("Type a positive number."),
        _ => return None,
    })
}

fn placement_problem_text(problem: &str) -> String {
    known_placement_problem(problem).unwrap_or_else(|| problem.to_string())
}

/// How the map's values are shown, in the language of the interface
/// (eov_common describes it in English, with `.` before the decimals of
/// its numbers). `None` for a description this list does not know.
fn known_scaling(scaling: &str) -> Option<String> {
    let number = |text: &str| isolate(crate::ui_update::localized_number(text));
    Some(match scaling {
        "8-bit values, 0–255 shown as 0–1" => tr!("8-bit values, 0–255 shown as 0–1"),
        "16-bit values, 0–65535 shown as 0–1" => tr!("16-bit values, 0–65535 shown as 0–1"),
        "grey pixels, 0–255 shown as 0–1" => tr!("grey pixels, 0–255 shown as 0–1"),
        "colour image, drawn as it is" => tr!("colour image, drawn as it is"),
        // Translators: how a heat map's values are shown: they all lie
        // between 0 and 1 and are shown as they are.
        "values 0–1" => tr!("values 0–1"),
        // Translators: how a heat map's values are shown: it holds no
        // number at all (every pixel is empty).
        "no finite values" => tr!("no finite values"),
        other => {
            if let Some((min, max)) = other
                .strip_prefix("values ")
                .and_then(|rest| rest.strip_suffix(" shown as 0–1"))
                .and_then(|range| range.split_once(" to "))
            {
                tr!(
                    "values {min} to {max} shown as 0–1",
                    min = number(min),
                    max = number(max)
                )
            } else {
                let value = other.strip_prefix("every value is ")?;
                tr!("every value is {}", number(value))
            }
        }
    })
}

fn scaling_text(scaling: &str) -> String {
    known_scaling(scaling).unwrap_or_else(|| scaling.to_string())
}

/// A colour map's name in the language of the interface (the others are
/// names of their own).
fn color_map_label(map: ColorMap) -> String {
    match map {
        ColorMap::Gray => tr!("Colour map" => "Grey"),
        other => other.label().to_string(),
    }
}

fn choices(items: &[(&str, String)]) -> ModelRc<ChoiceItem> {
    ModelRc::from(Rc::new(VecModel::from(
        items
            .iter()
            .map(|(id, label)| ChoiceItem {
                id: (*id).into(),
                label: label.into(),
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
    let (width, height, x, y) = (
        rect.width.round(),
        rect.height.round(),
        rect.x.round(),
        rect.y.round(),
    );
    if slide_wide {
        tr!(
            "Covers {width} × {height} level-0 pixels from ({x}, {y}) (the whole slide).",
            width = width,
            height = height,
            x = x,
            y = y
        )
    } else {
        tr!(
            "Covers {width} × {height} level-0 pixels from ({x}, {y}).",
            width = width,
            height = height,
            x = x,
            y = y
        )
    }
}

fn fill_panel(ui: &AppWindow, slide_name: &str, has_bounds: bool, shown: &Shown) {
    let panel = ui.global::<HeatMapPanel>();
    panel.set_title(tr!("Heat Map · {}", isolate(slide_name)).into());
    let name = shown
        .path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    panel.set_summary(
        // Translators: the heat map's file name, its size in pixels and
        // how its values are shown.
        tr!(
            "{name} · {width} × {height} px · {scaling}",
            name = isolate(&name),
            width = shown.map.width,
            height = shown.map.height,
            scaling = scaling_text(&shown.map.scaling)
        )
        .into(),
    );
    let mut placements = Vec::new();
    if shown.map.geometry.is_some() {
        placements.push(("from-file", tr!("As the file says")));
    }
    placements.push(("fit-slide", tr!("Fit the slide")));
    if has_bounds {
        placements.push(("fit-bounds", tr!("Fit the scanned region")));
    }
    placements.push(("downsample", tr!("Heat map placement" => "Downsample")));
    placements.push(("micrometres", tr!("µm per pixel")));
    panel.set_placements(choices(&placements));
    panel.set_placement(shown.placement.id().into());
    panel.set_scale_label(
        match shown.placement {
            Placement::Downsample(_) => tr!("Level-0 pixels per map pixel"),
            Placement::Micrometres(_) => tr!("Micrometres per map pixel"),
            _ => String::new(),
        }
        .into(),
    );
    panel.set_scale_text(shown.scale_text.clone().into());
    let (note, ok) = match &shown.placed {
        Ok(rect) => (
            describe_rect(rect, shown.placement == Placement::FitSlide),
            true,
        ),
        Err(problem) => (placement_problem_text(problem), false),
    };
    panel.set_placement_note(note.into());
    panel.set_placement_ok(ok);
    panel.set_scalar(shown.map.is_scalar());
    let maps: Vec<(&str, String)> = ColorMap::ALL
        .iter()
        .map(|map| (map.id(), color_map_label(*map)))
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
                geometry: None,
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

    /// A change of language writes the scale again in the new language's
    /// notation, from its number. Text written in one language is never
    /// read in another ("4.032" is four thousand and thirty-two in German),
    /// so a placement chosen after the change keeps the number, and the
    /// number saved stays the same.
    #[test]
    fn the_scale_keeps_its_number_when_the_language_changes() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            /// Back to English when the test ends, also when it fails.
            struct English<'a>(&'a AppWindow);
            impl Drop for English<'_> {
                fn drop(&mut self) {
                    crate::i18n::switch_to(self.0, i18n::source());
                }
            }
            let setup = crate::test_support::HostWithSlide::new();
            let _english = English(&setup.ui);
            let (host, file_id) = (&setup.host, setup.file_id);
            let map = HeatMap {
                width: 4,
                height: 3,
                values: HeatValues::Scalar(vec![0.5; 12]),
                scaling: "values 0–1".to_string(),
                geometry: None,
            };
            let settings = Saved {
                path: PathBuf::from("/maps/attention.png"),
                // 0.252 µm/px × 16.
                placement: "micrometres".to_string(),
                scale: Some(4.032),
                color_map: "viridis".to_string(),
                threshold: 0.0,
                opacity: 0.5,
                smooth: false,
                visible: true,
            };
            host.heatmap_read(file_id, settings, Ok(map), true);
            let panel = setup.ui.global::<HeatMapPanel>();
            // The placement, and the scale that would be saved.
            let shown = || {
                SHOWN.with(|all| {
                    let all = all.borrow();
                    (all[&file_id].placement, all[&file_id].saved().scale)
                })
            };
            assert_eq!(panel.get_scale_text(), "4.032");

            let german = i18n::by_tag("de").unwrap();
            crate::i18n::switch_to(&setup.ui, german);
            assert_eq!(panel.get_scale_text(), "4,032");
            // The chip already chosen, then another placement and back.
            host.heatmap_placement_chosen("micrometres");
            assert_eq!(shown(), (Placement::Micrometres(4.032), Some(4.032)));
            host.heatmap_placement_chosen("fit-slide");
            assert_eq!(shown(), (Placement::FitSlide, None));
            host.heatmap_placement_chosen("micrometres");
            assert_eq!(shown(), (Placement::Micrometres(4.032), Some(4.032)));
            assert_eq!(panel.get_scale_text(), "4,032");

            // Typed in German, then read in English.
            host.heatmap_placement_chosen("downsample");
            host.heatmap_scale_edited("2,125");
            assert_eq!(shown(), (Placement::Downsample(2.125), Some(2.125)));
            crate::i18n::switch_to(&setup.ui, i18n::source());
            assert_eq!(panel.get_scale_text(), "2.125");
            host.heatmap_placement_chosen("downsample");
            assert_eq!(shown(), (Placement::Downsample(2.125), Some(2.125)));
            host.heatmap_placement_chosen("micrometres");
            assert_eq!(shown(), (Placement::Micrometres(2.125), Some(2.125)));

            // Text that is not a number stays as it was typed; a placement
            // chosen then takes the last number the field held.
            host.heatmap_scale_edited("-");
            crate::i18n::switch_to(&setup.ui, german);
            assert_eq!(panel.get_scale_text(), "-");
            host.heatmap_placement_chosen("downsample");
            assert_eq!(shown(), (Placement::Downsample(2.125), Some(2.125)));
            assert_eq!(panel.get_scale_text(), "2,125");
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

    /// The panel translates every reason eov_common gives for a map it
    /// cannot place (they are matched by their English text).
    #[test]
    fn every_placement_problem_is_translatable() {
        let place = |placement, mpp, file| {
            heatmap::placement_rect(4, 3, placement, (400, 300), None, mpp, file)
                .expect_err("cannot be placed")
        };
        let problems = [
            place(Placement::FromFile, None, None),
            place(Placement::FitBounds, None, None),
            place(Placement::Downsample(-1.0), None, None),
            place(Placement::Micrometres(0.0), None, None),
            place(Placement::Micrometres(1.0), None, None),
            heatmap::placement_rect(4, 3, Placement::FitSlide, (0, 0), None, None, None)
                .expect_err("covers nothing"),
            SLIDE_CLOSED.to_string(),
            TYPE_A_NUMBER.to_string(),
        ];
        for problem in problems {
            assert!(known_placement_problem(&problem).is_some(), "{problem}");
        }
    }

    /// A single-strip little-endian TIFF of 32-bit float samples, one row.
    fn float_tiff(values: &[f32]) -> Vec<u8> {
        let width = values.len() as u32;
        // (tag, type: 3 SHORT or 4 LONG, value), in the order of the tags.
        let data_offset = 8 + 2 + 10 * 12 + 4;
        let entries: [(u16, u16, u32); 10] = [
            (256, 4, width),
            (257, 4, 1),
            (258, 3, 32),
            (259, 3, 1),
            (262, 3, 1),
            (273, 4, data_offset),
            (277, 3, 1),
            (278, 4, 1),
            (279, 4, width * 4),
            // SampleFormat: IEEE floating point.
            (339, 3, 3),
        ];
        let mut bytes = b"II*\0".to_vec();
        bytes.extend(8u32.to_le_bytes());
        bytes.extend((entries.len() as u16).to_le_bytes());
        for (tag, kind, value) in entries {
            bytes.extend(tag.to_le_bytes());
            bytes.extend(kind.to_le_bytes());
            bytes.extend(1u32.to_le_bytes());
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(0u32.to_le_bytes());
        assert_eq!(bytes.len(), data_offset as usize);
        bytes.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        bytes
    }

    fn png(color: image::ColorType, width: u32, data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        image::write_buffer_with_format(
            &mut std::io::Cursor::new(&mut bytes),
            data,
            width,
            1,
            color,
            image::ImageFormat::Png,
        )
        .unwrap();
        bytes
    }

    /// The panel translates every description eov_common gives of how a
    /// map's values are shown (they are matched by their English text).
    #[test]
    fn every_scaling_is_translatable() {
        let files = [
            png(image::ColorType::L8, 2, &[0, 255]),
            png(image::ColorType::L16, 2, &[0, 0, 255, 255]),
            png(image::ColorType::Rgb8, 1, &[51, 51, 51]),
            png(image::ColorType::Rgba8, 1, &[255, 0, 0, 128]),
            float_tiff(&[0.25, 0.75]),
            float_tiff(&[-1.0, 3.0]),
            float_tiff(&[5.0, 5.0]),
            float_tiff(&[f32::NAN, f32::NAN]),
        ];
        let mut seen = std::collections::BTreeSet::new();
        for file in files {
            let map = heatmap::decode_bytes(&file).unwrap();
            assert!(known_scaling(&map.scaling).is_some(), "{}", map.scaling);
            seen.insert(map.scaling);
        }
        assert_eq!(seen.len(), 8, "each kind of map is described: {seen:?}");
        // In English, the descriptions are eov_common's.
        assert_eq!(
            scaling_text("values -1 to 3.5 shown as 0–1"),
            "values -1 to 3.5 shown as 0–1"
        );
        assert_eq!(scaling_text("something new"), "something new");
    }

    /// The numbers of a description are written with the decimal
    /// separator of the interface's language.
    #[test]
    fn scaling_numbers_follow_the_interface_s_language() {
        struct English;
        impl Drop for English {
            fn drop(&mut self) {
                i18n::set_catalog(i18n::source(), None);
            }
        }
        let _english = English;
        let german = i18n::by_tag("de").unwrap();
        let messages: [(Option<&str>, &str, &[&str]); 3] = [
            (None, "values 0–1", &["Werte 0–1"]),
            (
                None,
                "values {min} to {max} shown as 0–1",
                &["Werte {min} bis {max}, als 0–1 gezeigt"],
            ),
            (None, "every value is {}", &["jeder Wert ist {}"]),
        ];
        i18n::set_catalog(
            german,
            Some(i18n::Catalog::from_messages(german, "", messages).unwrap()),
        );
        assert_eq!(scaling_text("values 0–1"), "Werte 0–1");
        assert_eq!(
            scaling_text("values -0.5 to 2.25 shown as 0–1"),
            "Werte -0,5 bis 2,25, als 0–1 gezeigt"
        );
        assert_eq!(scaling_text("every value is 1.5"), "jeder Wert ist 1,5");
    }
}

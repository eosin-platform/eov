//! The retained overlay store: every vector and raster object drawn over the
//! slides, from the host (ROI, measurements) and from plugins.
//!
//! - Layers belong to one owner (the host or a plugin) and one open file.
//! - Each layer keeps an R-tree of object bounds for culling and
//!   hit-testing, so drawing and picking cost scales with what is visible.
//! - Layers are copy-on-write (`Arc`): the render worker snapshots them
//!   without holding the store lock while it rasterises, and writers (plugin
//!   threads, the UI thread) never wait for a frame.
//! - Draw order is deterministic: layers by `(z, creation order)`, objects by
//!   insertion order (an upsert keeps an object's position).

use super::geometry::{self, Point};
use eov_plugin_api::protocol::{
    FileId, LayerSpec, LayerUpdate, OverlayObject, Rect, Shape, SizeUnit, Style,
};
use parking_lot::RwLock;
use rstar::{AABB, RTree, RTreeObject};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Upper bound on objects per layer.
pub const MAX_OBJECTS_PER_LAYER: usize = 2_000_000;
/// Upper bound on vertices per object.
pub const MAX_VERTICES_PER_OBJECT: usize = 100_000;
/// Upper bound on layers per owner.
pub const MAX_LAYERS_PER_OWNER: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Owner {
    Host,
    Plugin(Arc<str>),
}

impl Owner {
    pub fn plugin(id: &str) -> Self {
        Owner::Plugin(Arc::from(id))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerKey {
    pub owner: Owner,
    pub id: String,
}

impl LayerKey {
    pub fn new(owner: Owner, id: impl Into<String>) -> Self {
        Self {
            owner,
            id: id.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ObjectKey {
    pub layer: LayerKey,
    pub object_id: String,
}

/// Host-only rendering behaviour that plugins cannot request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostDecoration {
    /// Darken everything outside this (rectangle) object.
    pub dim_outside: bool,
    /// Two-tone dashed border (black and white) for maximum contrast.
    pub contrast_border: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StoredObject {
    pub object: OverlayObject,
    pub bounds: Rect,
    pub seq: u64,
    pub decoration: HostDecoration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StoredRaster {
    pub object_id: String,
    pub rect: Rect,
    pub width: u32,
    pub height: u32,
    /// Premultiplied RGBA8 (ready for the rasteriser).
    pub premultiplied: Arc<Vec<u8>>,
    pub smooth: bool,
    pub seq: u64,
}

#[derive(Clone, Debug, PartialEq)]
struct IndexEntry {
    seq: u64,
    envelope: AABB<[f64; 2]>,
}

impl RTreeObject for IndexEntry {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.envelope
    }
}

fn envelope_of(rect: &Rect) -> AABB<[f64; 2]> {
    AABB::from_corners(
        [rect.x, rect.y],
        [rect.x + rect.width, rect.y + rect.height],
    )
}

/// One layer. Cheap to clone into render snapshots via `Arc`.
#[derive(Clone, Debug)]
pub struct Layer {
    pub key: LayerKey,
    pub spec: LayerSpec,
    /// Creation order, the tie-breaker for equal `z`.
    pub order: u64,
    objects: HashMap<u64, Arc<StoredObject>>,
    ids: HashMap<String, u64>,
    index: RTree<IndexEntry>,
    rasters: BTreeMap<String, Arc<StoredRaster>>,
    selection: BTreeSet<String>,
    /// Largest screen-space size of any object's stroke or point marker, used
    /// to pad culling queries.
    max_screen_extent: f64,
    /// Largest image-space size (µm/level-0 units), already in level-0 px.
    max_image_extent: f64,
    /// Panes (stable ids) the owner hid this layer in.
    pub hidden_panes: BTreeSet<u32>,
}

impl Layer {
    fn new(key: LayerKey, spec: LayerSpec, order: u64) -> Self {
        let mut layer = Self {
            key,
            spec,
            order,
            objects: HashMap::new(),
            ids: HashMap::new(),
            index: RTree::new(),
            rasters: BTreeMap::new(),
            selection: BTreeSet::new(),
            max_screen_extent: 0.0,
            max_image_extent: 0.0,
            hidden_panes: BTreeSet::new(),
        };
        let style = layer.spec.style.clone();
        layer.note_style(&style);
        if let Some(style) = layer.spec.selected_style.clone() {
            layer.note_style(&style);
        }
        if let Some(style) = layer.spec.hover_style.clone() {
            layer.note_style(&style);
        }
        layer
    }

    /// Whether the layer is shown in the pane with stable id `pane`.
    pub fn visible_in(&self, pane: u32) -> bool {
        self.spec.visible && !self.hidden_panes.contains(&pane)
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty() && self.rasters.is_empty()
    }

    pub fn object(&self, object_id: &str) -> Option<&Arc<StoredObject>> {
        self.ids
            .get(object_id)
            .and_then(|seq| self.objects.get(seq))
    }

    #[cfg(test)]
    pub fn rasters(&self) -> impl Iterator<Item = &Arc<StoredRaster>> {
        self.rasters.values()
    }

    pub fn is_selected(&self, object_id: &str) -> bool {
        self.selection.contains(object_id)
    }

    pub fn selection(&self) -> impl Iterator<Item = &String> {
        self.selection.iter()
    }

    /// Whether objects of this layer may be reshaped by host gestures.
    pub fn object_is_editable(&self, object: &StoredObject) -> bool {
        object.object.editable.unwrap_or(self.spec.editable)
    }

    /// Padding (level-0 px) to add to culling and picking queries at `zoom`.
    pub fn query_margin(&self, zoom: f64) -> f64 {
        let zoom = zoom.max(1e-9);
        self.max_screen_extent / zoom + self.max_image_extent
    }

    /// Objects whose bounds intersect `rect` (padded by the layer's marker
    /// sizes at `zoom`), in draw order.
    pub fn visible_objects(&self, rect: &Rect, zoom: f64) -> Vec<Arc<StoredObject>> {
        let query = geometry::expand_rect(rect, self.query_margin(zoom));
        let mut seqs: Vec<u64> = self
            .index
            .locate_in_envelope_intersecting(envelope_of(&query))
            .map(|entry| entry.seq)
            .collect();
        seqs.sort_unstable();
        seqs.iter()
            .filter_map(|seq| self.objects.get(seq).cloned())
            .collect()
    }

    pub fn visible_rasters(&self, rect: &Rect) -> Vec<Arc<StoredRaster>> {
        let mut rasters: Vec<_> = self
            .rasters
            .values()
            .filter(|r| geometry::rect_intersects(&r.rect, rect))
            .cloned()
            .collect();
        rasters.sort_by_key(|r| r.seq);
        rasters
    }

    /// Notes the extents a style can add beyond an object's bounds.
    fn note_style(&mut self, style: &Style) {
        let extent = f64::from(style.stroke_width.max(0.0)) / 2.0
            + f64::from(style.point_radius.max(0.0))
            + 2.0;
        match style.size_unit {
            SizeUnit::Screen => self.max_screen_extent = self.max_screen_extent.max(extent),
            // Micrometre sizes are converted with an assumed spacing ≥
            // 0.05 µm/px, which over-pads rather than clips.
            SizeUnit::Micrometre => {
                self.max_image_extent = self.max_image_extent.max(extent / 0.05)
            }
            SizeUnit::Image => self.max_image_extent = self.max_image_extent.max(extent),
        }
    }

    fn insert_object(
        &mut self,
        object: OverlayObject,
        seq_source: &AtomicU64,
    ) -> Result<(), String> {
        validate_object(&object)?;
        if let Some(style) = object.style.clone() {
            self.note_style(&style);
        }
        let bounds = object.shape.bounds();
        let seq = match self.ids.get(&object.object_id).copied() {
            Some(seq) => {
                if let Some(old) = self.objects.get(&seq) {
                    self.index.remove(&IndexEntry {
                        seq,
                        envelope: envelope_of(&old.bounds),
                    });
                }
                seq
            }
            None => {
                if self.objects.len() >= MAX_OBJECTS_PER_LAYER {
                    return Err(format!(
                        "layer '{}' is full ({MAX_OBJECTS_PER_LAYER} objects)",
                        self.key.id
                    ));
                }
                let seq = seq_source.fetch_add(1, Ordering::Relaxed);
                self.ids.insert(object.object_id.clone(), seq);
                seq
            }
        };
        self.index.insert(IndexEntry {
            seq,
            envelope: envelope_of(&bounds),
        });
        self.objects.insert(
            seq,
            Arc::new(StoredObject {
                object,
                bounds,
                seq,
                decoration: HostDecoration::default(),
            }),
        );
        Ok(())
    }

    fn remove_object(&mut self, object_id: &str) -> bool {
        let Some(seq) = self.ids.remove(object_id) else {
            return self.rasters.remove(object_id).is_some();
        };
        if let Some(old) = self.objects.remove(&seq) {
            self.index.remove(&IndexEntry {
                seq,
                envelope: envelope_of(&old.bounds),
            });
        }
        self.selection.remove(object_id);
        true
    }

    fn clear(&mut self) {
        self.objects.clear();
        self.ids.clear();
        self.index = RTree::new();
        self.rasters.clear();
        self.selection.clear();
    }

    /// Replaces all vector objects at once, bulk-loading the index. Rasters
    /// are removed too unless `keep_rasters`.
    fn replace_all(
        &mut self,
        objects: Vec<OverlayObject>,
        seq_source: &AtomicU64,
        keep_rasters: bool,
    ) -> Result<(), String> {
        if objects.len() > MAX_OBJECTS_PER_LAYER {
            return Err(format!(
                "too many objects for layer '{}' ({} > {MAX_OBJECTS_PER_LAYER})",
                self.key.id,
                objects.len()
            ));
        }
        for object in &objects {
            validate_object(object)?;
        }
        let previous_selection = std::mem::take(&mut self.selection);
        let rasters = std::mem::take(&mut self.rasters);
        self.clear();
        if keep_rasters {
            self.rasters = rasters;
        }
        let mut entries = Vec::with_capacity(objects.len());
        for object in objects {
            if let Some(style) = object.style.clone() {
                self.note_style(&style);
            }
            if let Some(old_seq) = self.ids.get(&object.object_id).copied() {
                // Duplicate id in one batch: the later object wins.
                self.objects.remove(&old_seq);
                entries.retain(|e: &IndexEntry| e.seq != old_seq);
            }
            let seq = seq_source.fetch_add(1, Ordering::Relaxed);
            let bounds = object.shape.bounds();
            entries.push(IndexEntry {
                seq,
                envelope: envelope_of(&bounds),
            });
            self.ids.insert(object.object_id.clone(), seq);
            self.objects.insert(
                seq,
                Arc::new(StoredObject {
                    object,
                    bounds,
                    seq,
                    decoration: HostDecoration::default(),
                }),
            );
        }
        self.index = RTree::bulk_load(entries);
        self.selection = previous_selection
            .into_iter()
            .filter(|id| self.ids.contains_key(id))
            .collect();
        Ok(())
    }

    fn set_decoration(&mut self, object_id: &str, decoration: HostDecoration) {
        if let Some(seq) = self.ids.get(object_id)
            && let Some(object) = self.objects.get_mut(seq)
        {
            Arc::make_mut(object).decoration = decoration;
        }
    }
}

fn validate_object(object: &OverlayObject) -> Result<(), String> {
    if object.object_id.is_empty() || object.object_id.len() > 256 {
        return Err("object ids must be 1–256 bytes".to_string());
    }
    if !object.shape.is_valid() {
        return Err(format!(
            "object '{}' has an invalid shape (non-finite coordinates or too few vertices)",
            object.object_id
        ));
    }
    let vertices = match &object.shape {
        Shape::Polyline { points } => points.len(),
        Shape::Polygon { exterior, holes } => {
            exterior.len() + holes.iter().map(Vec::len).sum::<usize>()
        }
        _ => 1,
    };
    if vertices > MAX_VERTICES_PER_OBJECT {
        return Err(format!(
            "object '{}' has {vertices} vertices (limit {MAX_VERTICES_PER_OBJECT})",
            object.object_id
        ));
    }
    if let Some(style) = &object.style {
        validate_style(style)?;
    }
    Ok(())
}

fn validate_style(style: &Style) -> Result<(), String> {
    let sizes = [style.stroke_width, style.point_radius, style.font_size];
    if sizes
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0 || *v > 10_000.0)
    {
        return Err("style sizes must be finite, non-negative and ≤ 10000".to_string());
    }
    Ok(())
}

fn validate_spec(spec: &LayerSpec) -> Result<(), String> {
    if spec.layer_id.is_empty() || spec.layer_id.len() > 128 {
        return Err("layer ids must be 1–128 bytes".to_string());
    }
    if !spec.opacity.is_finite() || !(0.0..=1.0).contains(&spec.opacity) {
        return Err("layer opacity must be within 0–1".to_string());
    }
    if let Some(min_zoom) = spec.min_zoom
        && !(min_zoom.is_finite() && min_zoom >= 0.0)
    {
        return Err("min_zoom must be finite and non-negative".to_string());
    }
    validate_style(&spec.style)?;
    for style in [&spec.selected_style, &spec.hover_style]
        .into_iter()
        .flatten()
    {
        validate_style(style)?;
    }
    Ok(())
}

/// What a pick found, ranked by [`OverlayStore::hit_test`].
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub key: ObjectKey,
    /// The pointer is on the outline (or marker) rather than an interior.
    pub on_outline: bool,
    /// Screen distance to the outline, in logical pixels.
    pub distance_px: f64,
    /// Area of the object in level-0 px² (0 for open shapes).
    pub area: f64,
    pub z: i32,
    pub layer_order: u64,
    pub seq: u64,
}

/// Picking parameters.
#[derive(Clone, Copy, Debug)]
pub struct PickParams {
    pub file_id: FileId,
    /// The pane picked in (stable id): layers hidden there are skipped.
    pub pane: Option<u32>,
    pub point: Point,
    /// Logical screen pixels per level-0 pixel.
    pub zoom: f64,
    /// Average micrometres per pixel, for µm-sized markers.
    pub mpp: Option<f64>,
    /// Outline tolerance in logical pixels.
    pub tolerance_px: f64,
}

/// A consistent view of the layers of one file for rendering.
#[derive(Clone, Debug, Default)]
pub struct FileLayers {
    pub layers: Vec<Arc<Layer>>,
    pub revision: u64,
}

#[derive(Default)]
struct Inner {
    layers: BTreeMap<LayerKey, Arc<Layer>>,
}

/// The overlay store shared by the UI thread, plugin threads and renderers.
#[derive(Default)]
pub struct OverlayStore {
    inner: RwLock<Inner>,
    revision: AtomicU64,
    seq: AtomicU64,
    layer_order: AtomicU64,
}

impl OverlayStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Increases on every change.
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    fn bump(&self) {
        self.revision.fetch_add(1, Ordering::AcqRel);
    }

    /// Creates or replaces a layer (replacing removes its objects).
    pub fn create_layer(&self, owner: Owner, spec: LayerSpec) -> Result<(), String> {
        validate_spec(&spec)?;
        let key = LayerKey::new(owner.clone(), spec.layer_id.clone());
        let mut inner = self.inner.write();
        if !inner.layers.contains_key(&key)
            && inner.layers.keys().filter(|k| k.owner == owner).count() >= MAX_LAYERS_PER_OWNER
        {
            return Err(format!("too many layers (limit {MAX_LAYERS_PER_OWNER})"));
        }
        let order = self.layer_order.fetch_add(1, Ordering::Relaxed);
        inner
            .layers
            .insert(key.clone(), Arc::new(Layer::new(key, spec, order)));
        drop(inner);
        self.bump();
        Ok(())
    }

    fn with_layer<R>(
        &self,
        owner: &Owner,
        layer_id: &str,
        f: impl FnOnce(&mut Layer, &AtomicU64) -> Result<R, String>,
    ) -> Result<R, String> {
        let key = LayerKey::new(owner.clone(), layer_id);
        let mut inner = self.inner.write();
        let layer = inner
            .layers
            .get_mut(&key)
            .ok_or_else(|| format!("unknown layer '{layer_id}'"))?;
        let result = f(Arc::make_mut(layer), &self.seq)?;
        drop(inner);
        self.bump();
        Ok(result)
    }

    pub fn update_layer(&self, owner: &Owner, update: LayerUpdate) -> Result<(), String> {
        let layer_id = update.layer_id.clone();
        self.with_layer(owner, &layer_id, |layer, _| {
            let mut spec = layer.spec.clone();
            if let Some(name) = update.name {
                spec.name = name;
            }
            if let Some(z) = update.z {
                spec.z = z;
            }
            if let Some(visible) = update.visible {
                spec.visible = visible;
            }
            if let Some(opacity) = update.opacity {
                spec.opacity = opacity;
            }
            if let Some(interactive) = update.interactive {
                spec.interactive = interactive;
            }
            if let Some(editable) = update.editable {
                spec.editable = editable;
            }
            if let Some(style) = update.style {
                spec.style = style;
            }
            if update.selected_style.is_some() {
                spec.selected_style = update.selected_style;
            }
            if update.hover_style.is_some() {
                spec.hover_style = update.hover_style;
            }
            validate_spec(&spec)?;
            let style = spec.style.clone();
            layer.note_style(&style);
            for style in [spec.selected_style.clone(), spec.hover_style.clone()]
                .into_iter()
                .flatten()
            {
                layer.note_style(&style);
            }
            layer.spec = spec;
            Ok(())
        })
    }

    pub fn remove_layer(&self, owner: &Owner, layer_id: &str) -> bool {
        let removed = self
            .inner
            .write()
            .layers
            .remove(&LayerKey::new(owner.clone(), layer_id))
            .is_some();
        if removed {
            self.bump();
        }
        removed
    }

    pub fn upsert(
        &self,
        owner: &Owner,
        layer_id: &str,
        objects: Vec<OverlayObject>,
        replace_all: bool,
    ) -> Result<(), String> {
        self.with_layer(owner, layer_id, |layer, seq| {
            if replace_all {
                layer.replace_all(objects, seq, false)
            } else if layer.objects.is_empty() && objects.len() > 64 {
                // Fast path: bulk-load the index of an empty layer.
                layer.replace_all(objects, seq, true)
            } else {
                for object in objects {
                    layer.insert_object(object, seq)?;
                }
                Ok(())
            }
        })
    }

    /// Inserts or replaces one host object with decorations.
    pub fn upsert_decorated(
        &self,
        layer_id: &str,
        object: OverlayObject,
        decoration: HostDecoration,
    ) -> Result<(), String> {
        let object_id = object.object_id.clone();
        self.with_layer(&Owner::Host, layer_id, |layer, seq| {
            layer.insert_object(object, seq)?;
            layer.set_decoration(&object_id, decoration);
            Ok(())
        })
    }

    pub fn remove_objects(
        &self,
        owner: &Owner,
        layer_id: &str,
        ids: &[String],
    ) -> Result<(), String> {
        self.with_layer(owner, layer_id, |layer, _| {
            for id in ids {
                layer.remove_object(id);
            }
            Ok(())
        })
    }

    pub fn clear_layer(&self, owner: &Owner, layer_id: &str) -> Result<(), String> {
        self.with_layer(owner, layer_id, |layer, _| {
            layer.clear();
            Ok(())
        })
    }

    /// Replaces the selection of a layer; unknown ids are ignored. Returns
    /// whether it changed.
    pub fn select(&self, owner: &Owner, layer_id: &str, ids: &[String]) -> Result<bool, String> {
        self.with_layer(owner, layer_id, |layer, _| {
            let selection: BTreeSet<String> = ids
                .iter()
                .filter(|id| layer.ids.contains_key(id.as_str()))
                .cloned()
                .collect();
            let changed = selection != layer.selection;
            layer.selection = selection;
            Ok(changed)
        })
    }

    /// Clears the selection in every layer of `owner` (or of everyone).
    /// Returns the owners whose selection changed.
    pub fn clear_selection(&self, owner: Option<&Owner>) -> Vec<Owner> {
        let mut inner = self.inner.write();
        let mut changed = BTreeSet::new();
        for (key, layer) in inner.layers.iter_mut() {
            if owner.is_some_and(|o| o != &key.owner) || layer.selection.is_empty() {
                continue;
            }
            Arc::make_mut(layer).selection.clear();
            changed.insert(key.owner.clone());
        }
        drop(inner);
        if !changed.is_empty() {
            self.bump();
        }
        changed.into_iter().collect()
    }

    /// The selected objects of `owner`, across its layers.
    pub fn selection_of(&self, owner: &Owner) -> Vec<ObjectKey> {
        self.inner
            .read()
            .layers
            .iter()
            .filter(|(key, _)| &key.owner == owner)
            .flat_map(|(key, layer)| {
                layer.selection.iter().map(|object_id| ObjectKey {
                    layer: key.clone(),
                    object_id: object_id.clone(),
                })
            })
            .collect()
    }

    /// Keys of every layer with a non-empty selection.
    pub fn selected_layer_keys(&self) -> Vec<LayerKey> {
        self.inner
            .read()
            .layers
            .iter()
            .filter(|(_, layer)| !layer.selection.is_empty())
            .map(|(key, _)| key.clone())
            .collect()
    }

    pub fn put_raster(
        &self,
        owner: &Owner,
        layer_id: &str,
        mut raster: StoredRaster,
    ) -> Result<(), String> {
        let r = raster.rect;
        if ![r.x, r.y, r.width, r.height].iter().all(|v| v.is_finite())
            || r.width <= 0.0
            || r.height <= 0.0
        {
            return Err("raster rectangles must be finite with a positive size".to_string());
        }
        self.with_layer(owner, layer_id, |layer, seq| {
            if layer.ids.contains_key(&raster.object_id) {
                return Err(format!(
                    "'{}' is already a vector object in this layer",
                    raster.object_id
                ));
            }
            raster.seq = layer
                .rasters
                .get(&raster.object_id)
                .map(|old| old.seq)
                .unwrap_or_else(|| seq.fetch_add(1, Ordering::Relaxed));
            layer
                .rasters
                .insert(raster.object_id.clone(), Arc::new(raster));
            Ok(())
        })
    }

    /// Hides the layer below `min_zoom`.
    pub fn set_min_zoom(
        &self,
        owner: &Owner,
        layer_id: &str,
        min_zoom: Option<f64>,
    ) -> Result<(), String> {
        if min_zoom.is_some_and(|zoom| !(zoom.is_finite() && zoom > 0.0)) {
            return Err("min_zoom must be a positive number".to_string());
        }
        self.with_layer(owner, layer_id, |layer, _| {
            layer.spec.min_zoom = min_zoom;
            Ok(())
        })
    }

    /// Shows or hides the layer in one pane.
    pub fn set_pane_visibility(
        &self,
        owner: &Owner,
        layer_id: &str,
        pane: u32,
        visible: bool,
    ) -> Result<(), String> {
        self.with_layer(owner, layer_id, |layer, _| {
            if visible {
                layer.hidden_panes.remove(&pane);
            } else {
                layer.hidden_panes.insert(pane);
            }
            Ok(())
        })
    }

    /// Overwrites a `width` x `height` block of a raster's premultiplied
    /// pixels at `(x, y)`.
    #[allow(clippy::too_many_arguments)]
    pub fn patch_raster(
        &self,
        owner: &Owner,
        layer_id: &str,
        object_id: &str,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        premultiplied: &[u8],
    ) -> Result<(), String> {
        self.with_layer(owner, layer_id, |layer, _| {
            let raster = layer
                .rasters
                .get_mut(object_id)
                .ok_or_else(|| format!("unknown raster '{object_id}'"))?;
            let fits = u64::from(x) + u64::from(width) <= u64::from(raster.width)
                && u64::from(y) + u64::from(height) <= u64::from(raster.height);
            if !fits || premultiplied.len() != width as usize * height as usize * 4 {
                return Err("the patch must fit inside the raster".to_string());
            }
            let raster = Arc::make_mut(raster);
            let pixels = Arc::make_mut(&mut raster.premultiplied);
            let stride = raster.width as usize * 4;
            let row_bytes = width as usize * 4;
            for row in 0..height as usize {
                let dst = (y as usize + row) * stride + x as usize * 4;
                pixels[dst..dst + row_bytes]
                    .copy_from_slice(&premultiplied[row * row_bytes..(row + 1) * row_bytes]);
            }
            Ok(())
        })
    }

    /// Removes every layer shown on `file_id` (the file was closed).
    pub fn remove_file(&self, file_id: FileId) {
        let mut inner = self.inner.write();
        let before = inner.layers.len();
        inner
            .layers
            .retain(|_, layer| layer.spec.file_id != file_id);
        let changed = inner.layers.len() != before;
        drop(inner);
        if changed {
            self.bump();
        }
    }

    /// Removes every layer of `owner` (plugin unloaded or quarantined).
    pub fn remove_owner(&self, owner: &Owner) {
        let mut inner = self.inner.write();
        let before = inner.layers.len();
        inner.layers.retain(|key, _| &key.owner != owner);
        let changed = inner.layers.len() != before;
        drop(inner);
        if changed {
            self.bump();
        }
    }

    pub fn layer(&self, owner: &Owner, layer_id: &str) -> Option<Arc<Layer>> {
        self.inner
            .read()
            .layers
            .get(&LayerKey::new(owner.clone(), layer_id))
            .cloned()
    }

    pub fn object(&self, key: &ObjectKey) -> Option<Arc<StoredObject>> {
        self.inner
            .read()
            .layers
            .get(&key.layer)
            .and_then(|layer| layer.object(&key.object_id).cloned())
    }

    /// The layers of `file_id` in draw order (bottom first).
    pub fn file_layers(&self, file_id: FileId) -> FileLayers {
        let inner = self.inner.read();
        let revision = self.revision();
        let mut layers: Vec<Arc<Layer>> = inner
            .layers
            .values()
            .filter(|layer| layer.spec.file_id == file_id)
            .cloned()
            .collect();
        drop(inner);
        layers.sort_by_key(|layer| (layer.spec.z, layer.order));
        FileLayers { layers, revision }
    }

    /// Objects under `params.point`, best candidate first.
    ///
    /// Only visible layers are searched, and `filter` decides which layers
    /// take part (e.g. interactive layers of one plugin). The ranking is:
    /// topmost layer first; within a layer, outline hits before interior
    /// hits, smaller interiors before larger ones, later objects first.
    pub fn hit_test(&self, params: PickParams, filter: impl Fn(&Layer) -> bool) -> Vec<Hit> {
        let file_layers = self.file_layers(params.file_id);
        let zoom = params.zoom.max(1e-9);
        let mut hits = Vec::new();
        for layer in &file_layers.layers {
            if !layer.spec.visible
                || params.pane.is_some_and(|pane| !layer.visible_in(pane))
                || layer.spec.min_zoom.is_some_and(|min| zoom < min)
                || !filter(layer)
            {
                continue;
            }
            let probe = Rect {
                x: params.point[0],
                y: params.point[1],
                width: 0.0,
                height: 0.0,
            };
            let pad = params.tolerance_px / zoom;
            let probe = geometry::expand_rect(&probe, pad);
            for object in layer.visible_objects(&probe, zoom) {
                let style = object.object.style.as_ref().unwrap_or(&layer.spec.style);
                if let Some((on_outline, distance_px)) =
                    pick_object(&object.object.shape, style, params, zoom)
                {
                    hits.push(Hit {
                        key: ObjectKey {
                            layer: layer.key.clone(),
                            object_id: object.object.object_id.clone(),
                        },
                        on_outline,
                        distance_px,
                        area: geometry::shape_area(&object.object.shape),
                        z: layer.spec.z,
                        layer_order: layer.order,
                        seq: object.seq,
                    });
                }
            }
        }
        hits.sort_by(|a, b| {
            (b.z, b.layer_order)
                .cmp(&(a.z, a.layer_order))
                .then_with(|| b.on_outline.cmp(&a.on_outline))
                .then_with(|| {
                    if a.on_outline {
                        a.distance_px.total_cmp(&b.distance_px)
                    } else {
                        a.area.total_cmp(&b.area)
                    }
                })
                .then_with(|| b.seq.cmp(&a.seq))
        });
        hits
    }
}

/// Converts a style size to logical screen pixels.
pub fn size_to_screen(value: f32, unit: SizeUnit, zoom: f64, mpp: Option<f64>) -> f64 {
    let value = f64::from(value);
    match unit {
        SizeUnit::Screen => value,
        SizeUnit::Image => value * zoom,
        SizeUnit::Micrometre => match mpp {
            Some(mpp) if mpp > 0.0 => value / mpp * zoom,
            _ => value,
        },
    }
}

/// Tests one shape; returns `(on_outline, outline distance in px)`.
fn pick_object(shape: &Shape, style: &Style, params: PickParams, zoom: f64) -> Option<(bool, f64)> {
    let p = params.point;
    let stroke_px = size_to_screen(style.stroke_width, style.size_unit, zoom, params.mpp);
    let tolerance = params.tolerance_px + stroke_px / 2.0;
    let to_px = |d: f64| d * zoom;
    match shape {
        Shape::Point { x, y } => {
            let radius = size_to_screen(style.point_radius, style.size_unit, zoom, params.mpp);
            let d = to_px(geometry::distance(p, [*x, *y]));
            (d <= radius + tolerance).then_some((true, (d - radius).max(0.0)))
        }
        Shape::Text { x, y, text } => {
            // Approximate the text box at the style's font size.
            let font = f64::from(style.font_size.max(1.0));
            let width_px = font * 0.6 * text.chars().count() as f64;
            let (dx, dy) = (to_px(p[0] - x), to_px(p[1] - y));
            (dx >= -tolerance
                && dx <= width_px + tolerance
                && dy >= -tolerance
                && dy <= font * 1.3 + tolerance)
                .then_some((true, 0.0))
        }
        Shape::Polyline { points } => {
            let d = points
                .windows(2)
                .map(|w| geometry::distance_to_segment(p, w[0], w[1]))
                .fold(f64::INFINITY, f64::min);
            let d = to_px(d);
            (d <= tolerance).then_some((true, d))
        }
        Shape::Polygon { exterior, holes } => {
            let mut d = f64::INFINITY;
            for ring in std::iter::once(exterior).chain(holes.iter()) {
                for i in 0..ring.len() {
                    let a = ring[i];
                    let b = ring[(i + 1) % ring.len()];
                    d = d.min(geometry::distance_to_segment(p, a, b));
                }
            }
            let d = to_px(d);
            if d <= tolerance {
                Some((true, d))
            } else if geometry::point_in_polygon(p, exterior, holes) {
                Some((false, d))
            } else {
                None
            }
        }
        Shape::Rectangle {
            x,
            y,
            width,
            height,
        } => {
            let ring = [
                [*x, *y],
                [x + width, *y],
                [x + width, y + height],
                [*x, y + height],
            ];
            pick_object(
                &Shape::Polygon {
                    exterior: ring.to_vec(),
                    holes: Vec::new(),
                },
                style,
                params,
                zoom,
            )
        }
        Shape::Ellipse { cx, cy, rx, ry } => pick_object(
            &Shape::Polygon {
                exterior: geometry::ellipse_ring(*cx, *cy, *rx, *ry, 72),
                holes: Vec::new(),
            },
            style,
            params,
            zoom,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plugin() -> Owner {
        Owner::plugin("cells")
    }

    fn point(id: &str, x: f64, y: f64) -> OverlayObject {
        OverlayObject::new(id, Shape::Point { x, y })
    }

    fn square(id: &str, x: f64, y: f64, size: f64) -> OverlayObject {
        OverlayObject::new(
            id,
            Shape::Polygon {
                exterior: vec![[x, y], [x + size, y], [x + size, y + size], [x, y + size]],
                holes: vec![],
            },
        )
    }

    fn pick(file_id: FileId, x: f64, y: f64) -> PickParams {
        PickParams {
            pane: None,
            file_id,
            point: [x, y],
            zoom: 1.0,
            mpp: None,
            tolerance_px: 4.0,
        }
    }

    #[test]
    fn layers_can_be_hidden_in_one_pane() {
        let store = OverlayStore::new();
        store
            .create_layer(plugin(), LayerSpec::new("marks", 1))
            .unwrap();
        store
            .upsert(&plugin(), "marks", vec![square("a", 0.0, 0.0, 10.0)], false)
            .unwrap();
        let in_pane = |pane| PickParams {
            pane: Some(pane),
            ..pick(1, 5.0, 5.0)
        };
        assert_eq!(store.hit_test(in_pane(7), |_| true).len(), 1);
        store
            .set_pane_visibility(&plugin(), "marks", 7, false)
            .unwrap();
        assert!(store.hit_test(in_pane(7), |_| true).is_empty());
        assert_eq!(store.hit_test(in_pane(8), |_| true).len(), 1);
        let layer = store.layer(&plugin(), "marks").unwrap();
        assert!(!layer.visible_in(7) && layer.visible_in(8));
        store
            .set_pane_visibility(&plugin(), "marks", 7, true)
            .unwrap();
        assert_eq!(store.hit_test(in_pane(7), |_| true).len(), 1);
    }

    #[test]
    fn min_zoom_can_change_without_recreating_the_layer() {
        let store = OverlayStore::new();
        store
            .create_layer(plugin(), LayerSpec::new("grid", 1))
            .unwrap();
        store.set_min_zoom(&plugin(), "grid", Some(0.5)).unwrap();
        assert_eq!(
            store.layer(&plugin(), "grid").unwrap().spec.min_zoom,
            Some(0.5)
        );
        assert!(store.set_min_zoom(&plugin(), "grid", Some(-1.0)).is_err());
        store.set_min_zoom(&plugin(), "grid", None).unwrap();
        assert_eq!(store.layer(&plugin(), "grid").unwrap().spec.min_zoom, None);
    }

    #[test]
    fn rasters_can_be_patched_in_place() {
        let store = OverlayStore::new();
        store
            .create_layer(plugin(), LayerSpec::new("heat", 1))
            .unwrap();
        store
            .put_raster(
                &plugin(),
                "heat",
                StoredRaster {
                    object_id: "map".into(),
                    rect: Rect {
                        x: 0.0,
                        y: 0.0,
                        width: 100.0,
                        height: 100.0,
                    },
                    width: 4,
                    height: 4,
                    premultiplied: Arc::new(vec![0; 4 * 4 * 4]),
                    smooth: false,
                    seq: 0,
                },
            )
            .unwrap();
        let patch = [255u8; 2 * 2 * 4];
        store
            .patch_raster(&plugin(), "heat", "map", 1, 1, 2, 2, &patch)
            .unwrap();
        let layer = store.layer(&plugin(), "heat").unwrap();
        let raster = layer.rasters().next().unwrap();
        let px = |x: usize, y: usize| raster.premultiplied[(y * 4 + x) * 4];
        assert_eq!((px(0, 0), px(1, 1), px(2, 2), px(3, 3)), (0, 255, 255, 0));
        // Patches must fit.
        assert!(
            store
                .patch_raster(&plugin(), "heat", "map", 3, 3, 2, 2, &patch)
                .is_err()
        );
        assert!(
            store
                .patch_raster(&plugin(), "heat", "nope", 0, 0, 2, 2, &patch)
                .is_err()
        );
    }

    #[test]
    fn upsert_cull_and_remove() {
        let store = OverlayStore::new();
        store
            .create_layer(plugin(), LayerSpec::new("l", 1))
            .unwrap();
        let objects: Vec<_> = (0..1000)
            .map(|i| point(&format!("p{i}"), i as f64 * 10.0, 0.0))
            .collect();
        store.upsert(&plugin(), "l", objects, false).unwrap();
        let layer = store.layer(&plugin(), "l").unwrap();
        assert_eq!(layer.len(), 1000);
        let visible = layer.visible_objects(
            &Rect {
                x: 0.0,
                y: -1.0,
                width: 95.0,
                height: 2.0,
            },
            100.0,
        );
        assert_eq!(visible.len(), 10);
        // Draw order follows insertion order.
        assert!(visible.windows(2).all(|w| w[0].seq < w[1].seq));

        store
            .remove_objects(&plugin(), "l", &["p0".to_string(), "p1".to_string()])
            .unwrap();
        assert_eq!(store.layer(&plugin(), "l").unwrap().len(), 998);
    }

    #[test]
    fn upserting_an_existing_id_keeps_its_draw_position() {
        let store = OverlayStore::new();
        store
            .create_layer(plugin(), LayerSpec::new("l", 1))
            .unwrap();
        store
            .upsert(
                &plugin(),
                "l",
                vec![point("a", 0.0, 0.0), point("b", 1.0, 0.0)],
                false,
            )
            .unwrap();
        let before = store
            .layer(&plugin(), "l")
            .unwrap()
            .object("a")
            .unwrap()
            .seq;
        store
            .upsert(&plugin(), "l", vec![point("a", 500.0, 500.0)], false)
            .unwrap();
        let layer = store.layer(&plugin(), "l").unwrap();
        let a = layer.object("a").unwrap();
        assert_eq!(a.seq, before);
        assert_eq!(a.object.shape, Shape::Point { x: 500.0, y: 500.0 });
        // The index follows the move.
        let near_new = layer.visible_objects(
            &Rect {
                x: 499.0,
                y: 499.0,
                width: 2.0,
                height: 2.0,
            },
            100.0,
        );
        assert_eq!(near_new.len(), 1);
    }

    #[test]
    fn snapshots_are_isolated_from_later_writes() {
        let store = OverlayStore::new();
        store
            .create_layer(plugin(), LayerSpec::new("l", 1))
            .unwrap();
        store
            .upsert(&plugin(), "l", vec![point("a", 0.0, 0.0)], false)
            .unwrap();
        let snapshot = store.file_layers(1);
        store
            .upsert(&plugin(), "l", vec![point("b", 0.0, 0.0)], false)
            .unwrap();
        assert_eq!(snapshot.layers[0].len(), 1);
        assert_eq!(store.file_layers(1).layers[0].len(), 2);
        assert!(store.revision() > snapshot.revision);
    }

    #[test]
    fn invalid_input_is_rejected() {
        let store = OverlayStore::new();
        assert!(
            store
                .create_layer(
                    plugin(),
                    LayerSpec {
                        opacity: 2.0,
                        ..LayerSpec::new("l", 1)
                    }
                )
                .is_err()
        );
        store
            .create_layer(plugin(), LayerSpec::new("l", 1))
            .unwrap();
        let bad = OverlayObject::new(
            "x",
            Shape::Point {
                x: f64::NAN,
                y: 0.0,
            },
        );
        assert!(store.upsert(&plugin(), "l", vec![bad], false).is_err());
        assert!(store.upsert(&plugin(), "missing", vec![], false).is_err());
        // Another owner cannot touch the layer.
        assert!(
            store
                .upsert(
                    &Owner::plugin("other"),
                    "l",
                    vec![point("a", 0.0, 0.0)],
                    false
                )
                .is_err()
        );
    }

    #[test]
    fn layers_are_ordered_by_z_then_creation() {
        let store = OverlayStore::new();
        store
            .create_layer(plugin(), LayerSpec::new("top", 1).at_z(5))
            .unwrap();
        store
            .create_layer(plugin(), LayerSpec::new("first", 1))
            .unwrap();
        store
            .create_layer(Owner::Host, LayerSpec::new("second", 1))
            .unwrap();
        store
            .create_layer(plugin(), LayerSpec::new("other-file", 2))
            .unwrap();
        let ids: Vec<_> = store
            .file_layers(1)
            .layers
            .iter()
            .map(|l| l.key.id.clone())
            .collect();
        assert_eq!(ids, ["first", "second", "top"]);
        store.remove_file(1);
        assert!(store.file_layers(1).layers.is_empty());
        assert_eq!(store.file_layers(2).layers.len(), 1);
    }

    #[test]
    fn hit_testing_prefers_the_smallest_interior_and_topmost_layer() {
        let store = OverlayStore::new();
        store
            .create_layer(plugin(), LayerSpec::new("l", 1).interactive(true))
            .unwrap();
        store
            .upsert(
                &plugin(),
                "l",
                vec![
                    square("big", 0.0, 0.0, 100.0),
                    square("small", 40.0, 40.0, 20.0),
                ],
                false,
            )
            .unwrap();
        // Inside both: the small one wins although it was drawn later... and
        // also when drawn first.
        let hits = store.hit_test(pick(1, 50.0, 50.0), |_| true);
        assert_eq!(hits[0].key.object_id, "small");
        assert_eq!(hits.len(), 2);
        // On the big square's outline: outline hits beat interiors.
        let hits = store.hit_test(pick(1, 100.0, 50.0), |_| true);
        assert_eq!(hits[0].key.object_id, "big");
        assert!(hits[0].on_outline);

        store
            .create_layer(plugin(), LayerSpec::new("above", 1).at_z(1))
            .unwrap();
        store
            .upsert(
                &plugin(),
                "above",
                vec![square("cover", 0.0, 0.0, 200.0)],
                false,
            )
            .unwrap();
        let hits = store.hit_test(pick(1, 50.0, 50.0), |_| true);
        assert_eq!(hits[0].key.object_id, "cover");
        // Filters restrict the search.
        let hits = store.hit_test(pick(1, 50.0, 50.0), |layer| layer.key.id == "l");
        assert_eq!(hits[0].key.object_id, "small");
    }

    #[test]
    fn hidden_layers_are_not_picked() {
        let store = OverlayStore::new();
        store
            .create_layer(
                plugin(),
                LayerSpec {
                    visible: false,
                    ..LayerSpec::new("l", 1)
                },
            )
            .unwrap();
        store
            .upsert(&plugin(), "l", vec![point("a", 0.0, 0.0)], false)
            .unwrap();
        assert!(store.hit_test(pick(1, 0.0, 0.0), |_| true).is_empty());
    }

    #[test]
    fn selection_follows_removals_and_replacement() {
        let store = OverlayStore::new();
        store
            .create_layer(plugin(), LayerSpec::new("l", 1))
            .unwrap();
        store
            .upsert(
                &plugin(),
                "l",
                vec![point("a", 0.0, 0.0), point("b", 1.0, 1.0)],
                false,
            )
            .unwrap();
        assert!(
            store
                .select(&plugin(), "l", &["a".into(), "b".into(), "zzz".into()])
                .unwrap()
        );
        assert_eq!(store.selection_of(&plugin()).len(), 2);
        store.remove_objects(&plugin(), "l", &["a".into()]).unwrap();
        assert_eq!(store.selection_of(&plugin()).len(), 1);
        store
            .upsert(&plugin(), "l", vec![point("c", 0.0, 0.0)], true)
            .unwrap();
        assert!(store.selection_of(&plugin()).is_empty());
        assert!(store.clear_selection(None).is_empty());
    }

    #[test]
    fn bulk_load_handles_duplicate_ids() {
        let store = OverlayStore::new();
        store
            .create_layer(plugin(), LayerSpec::new("l", 1))
            .unwrap();
        let mut objects: Vec<_> = (0..100)
            .map(|i| point(&format!("p{i}"), i as f64, 0.0))
            .collect();
        objects.push(point("p5", 1000.0, 1000.0));
        store.upsert(&plugin(), "l", objects, true).unwrap();
        let layer = store.layer(&plugin(), "l").unwrap();
        assert_eq!(layer.len(), 100);
        assert_eq!(
            layer.object("p5").unwrap().object.shape,
            Shape::Point {
                x: 1000.0,
                y: 1000.0
            }
        );
        let all = layer.visible_objects(
            &Rect {
                x: -10.0,
                y: -10.0,
                width: 2000.0,
                height: 2000.0,
            },
            1.0,
        );
        assert_eq!(all.len(), 100);
    }

    #[test]
    fn markers_sized_in_screen_pixels_are_found_at_low_zoom() {
        let store = OverlayStore::new();
        store
            .create_layer(plugin(), LayerSpec::new("l", 1))
            .unwrap();
        store
            .upsert(&plugin(), "l", vec![point("a", 1000.0, 1000.0)], false)
            .unwrap();
        // At zoom 0.01 a 5 px marker covers 500 level-0 px.
        let hits = store.hit_test(
            PickParams {
                zoom: 0.01,
                ..pick(1, 1300.0, 1000.0)
            },
            |_| true,
        );
        assert_eq!(hits.len(), 1);
    }
}

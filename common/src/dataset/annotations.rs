//! GeoJSON annotations for region-driven dataset extraction.
//!
//! Accepts what QuPath exports — a `FeatureCollection`, a single `Feature`,
//! an array of features or a bare geometry — with `Polygon` and
//! `MultiPolygon` geometries in **level-0 pixel coordinates** (x right,
//! y down, no OpenSlide bounds offset). The class of a feature is
//! `properties.classification.name` (or `.names` joined with ": "), then
//! `properties.classification` as a string, then `properties.class` /
//! `properties.label`; unclassified features get [`UNCLASSIFIED`]. Other
//! geometry types are counted and ignored. Inner rings are holes (even-odd
//! rule); where features overlap, later ones win when masks are drawn.

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::Value;

/// Class name of features without a classification.
pub const UNCLASSIFIED: &str = "Unclassified";

/// One polygon (outer ring plus holes) with its class.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnotationPolygon {
    pub class: String,
    pub rings: Vec<Vec<[f64; 2]>>,
    /// `[min_x, min_y, max_x, max_y]`.
    pub bbox: [f64; 4],
}

/// Error reading an annotation file.
#[derive(Debug, thiserror::Error)]
pub enum AnnotationError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("invalid GeoJSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid GeoJSON: {0}")]
    Format(String),
}

/// Per-class coverage of a rectangle.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Coverage {
    /// Fraction of the rectangle inside any polygon.
    pub any: f32,
    /// Fraction per class, sorted by class name.
    pub classes: Vec<(String, f32)>,
}

impl Coverage {
    /// The class covering the largest area, with its fraction.
    pub fn majority(&self) -> Option<(&str, f32)> {
        self.classes
            .iter()
            .filter(|(_, f)| *f > 0.0)
            .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
            .map(|(c, f)| (c.as_str(), *f))
    }
}

/// The polygons of one slide with a bucket index for rectangle queries.
#[derive(Debug, Clone, Default)]
pub struct AnnotationSet {
    polygons: Vec<AnnotationPolygon>,
    /// Features that were not polygons (points, lines, empty geometries).
    pub ignored_features: usize,
    cell: f64,
    buckets: std::collections::HashMap<(i64, i64), Vec<u32>>,
}

const CELL_SIZE: f64 = 2048.0;

fn class_of(properties: Option<&Value>) -> String {
    let Some(props) = properties else {
        return UNCLASSIFIED.to_string();
    };
    if let Some(class) = props.get("classification") {
        if let Some(name) = class.get("name").and_then(Value::as_str) {
            return name.to_string();
        }
        if let Some(names) = class.get("names").and_then(Value::as_array) {
            let joined: Vec<&str> = names.iter().filter_map(Value::as_str).collect();
            if !joined.is_empty() {
                return joined.join(": ");
            }
        }
        if let Some(name) = class.as_str() {
            return name.to_string();
        }
    }
    for key in ["class", "label"] {
        if let Some(name) = props.get(key).and_then(Value::as_str) {
            return name.to_string();
        }
    }
    UNCLASSIFIED.to_string()
}

fn parse_ring(value: &Value) -> Result<Vec<[f64; 2]>, AnnotationError> {
    let points = value
        .as_array()
        .ok_or_else(|| AnnotationError::Format("a ring must be an array of positions".into()))?;
    let mut ring = Vec::with_capacity(points.len());
    for p in points {
        let xy = p
            .as_array()
            .filter(|a| a.len() >= 2)
            .and_then(|a| Some([a[0].as_f64()?, a[1].as_f64()?]))
            .filter(|xy| xy[0].is_finite() && xy[1].is_finite())
            .ok_or_else(|| AnnotationError::Format(format!("invalid position {p}")))?;
        ring.push(xy);
    }
    if ring.len() > 1 && ring.first() == ring.last() {
        ring.pop();
    }
    Ok(ring)
}

fn parse_polygon(
    coords: &Value,
    class: &str,
) -> Result<Option<AnnotationPolygon>, AnnotationError> {
    let rings_json = coords
        .as_array()
        .ok_or_else(|| AnnotationError::Format("Polygon coordinates must be an array".into()))?;
    let mut rings = Vec::new();
    for ring in rings_json {
        let ring = parse_ring(ring)?;
        if ring.len() >= 3 {
            rings.push(ring);
        }
    }
    if rings.is_empty() {
        return Ok(None);
    }
    let mut bbox = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for p in &rings[0] {
        bbox = [
            bbox[0].min(p[0]),
            bbox[1].min(p[1]),
            bbox[2].max(p[0]),
            bbox[3].max(p[1]),
        ];
    }
    Ok(Some(AnnotationPolygon {
        class: class.to_string(),
        rings,
        bbox,
    }))
}

impl AnnotationSet {
    /// Parses GeoJSON text.
    pub fn from_geojson(text: &str) -> Result<Self, AnnotationError> {
        let root: Value = serde_json::from_str(text)?;
        let mut set = Self {
            cell: CELL_SIZE,
            ..Self::default()
        };
        set.add_value(&root, None)?;
        set.build_index();
        Ok(set)
    }

    /// Reads a GeoJSON file.
    pub fn load(path: &Path) -> Result<Self, AnnotationError> {
        let text = std::fs::read_to_string(path).map_err(|source| AnnotationError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_geojson(&text)
    }

    fn add_value(
        &mut self,
        value: &Value,
        properties: Option<&Value>,
    ) -> Result<(), AnnotationError> {
        if let Some(array) = value.as_array() {
            for item in array {
                self.add_value(item, None)?;
            }
            return Ok(());
        }
        match value.get("type").and_then(Value::as_str) {
            Some("FeatureCollection") => {
                for feature in value
                    .get("features")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    self.add_value(feature, None)?;
                }
            }
            Some("Feature") => match value.get("geometry") {
                Some(geometry) if !geometry.is_null() => {
                    self.add_value(geometry, value.get("properties"))?
                }
                _ => self.ignored_features += 1,
            },
            Some("GeometryCollection") => {
                for g in value
                    .get("geometries")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    self.add_value(g, properties)?;
                }
            }
            Some("Polygon") => {
                let class = class_of(properties);
                let coords = value.get("coordinates").unwrap_or(&Value::Null);
                match parse_polygon(coords, &class)? {
                    Some(p) => self.polygons.push(p),
                    None => self.ignored_features += 1,
                }
            }
            Some("MultiPolygon") => {
                let class = class_of(properties);
                let polys = value
                    .get("coordinates")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        AnnotationError::Format("MultiPolygon coordinates must be an array".into())
                    })?;
                for coords in polys {
                    match parse_polygon(coords, &class)? {
                        Some(p) => self.polygons.push(p),
                        None => self.ignored_features += 1,
                    }
                }
            }
            Some(_) => self.ignored_features += 1,
            None => {
                return Err(AnnotationError::Format(
                    "expected a FeatureCollection, Feature, geometry or array of features".into(),
                ));
            }
        }
        Ok(())
    }

    fn build_index(&mut self) {
        self.buckets.clear();
        if self.cell <= 0.0 {
            self.cell = CELL_SIZE;
        }
        for (i, p) in self.polygons.iter().enumerate() {
            let (cx0, cy0) = (
                (p.bbox[0] / self.cell).floor() as i64,
                (p.bbox[1] / self.cell).floor() as i64,
            );
            let (cx1, cy1) = (
                (p.bbox[2] / self.cell).floor() as i64,
                (p.bbox[3] / self.cell).floor() as i64,
            );
            for cy in cy0..=cy1 {
                for cx in cx0..=cx1 {
                    self.buckets.entry((cx, cy)).or_default().push(i as u32);
                }
            }
        }
    }

    /// Keeps only polygons whose class is included (empty list = all) and
    /// not excluded. Matching is case-insensitive.
    pub fn retain_classes(&mut self, include: &[String], exclude: &[String]) {
        let matches =
            |list: &[String], class: &str| list.iter().any(|c| c.eq_ignore_ascii_case(class));
        self.polygons.retain(|p| {
            (include.is_empty() || matches(include, &p.class)) && !matches(exclude, &p.class)
        });
        self.build_index();
    }

    pub fn polygons(&self) -> &[AnnotationPolygon] {
        &self.polygons
    }

    pub fn is_empty(&self) -> bool {
        self.polygons.is_empty()
    }

    /// Distinct class names, sorted.
    pub fn class_names(&self) -> BTreeSet<String> {
        self.polygons.iter().map(|p| p.class.clone()).collect()
    }

    /// Indices of polygons whose bounding box intersects the rectangle, in
    /// file order.
    fn candidates(&self, x: f64, y: f64, w: f64, h: f64) -> Vec<u32> {
        let (cx0, cy0) = (
            (x / self.cell).floor() as i64,
            (y / self.cell).floor() as i64,
        );
        let (cx1, cy1) = (
            ((x + w) / self.cell).floor() as i64,
            ((y + h) / self.cell).floor() as i64,
        );
        let mut out = Vec::new();
        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                if let Some(ids) = self.buckets.get(&(cx, cy)) {
                    out.extend(ids.iter().copied().filter(|&i| {
                        let b = self.polygons[i as usize].bbox;
                        b[0] < x + w && b[2] > x && b[1] < y + h && b[3] > y
                    }));
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Whether any polygon's bounding box touches the rectangle (cheap
    /// pre-check).
    pub fn may_intersect(&self, x: f64, y: f64, w: f64, h: f64) -> bool {
        !self.candidates(x, y, w, h).is_empty()
    }

    /// For each output pixel of an `out_w × out_h` raster over the level-0
    /// rectangle, the index (into `self.polygons`) of the last polygon that
    /// contains the pixel centre, if any.
    fn owner_raster(
        &self,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        out_w: u32,
        out_h: u32,
    ) -> Vec<Option<u32>> {
        let mut owners = vec![None; out_w as usize * out_h as usize];
        let ids = self.candidates(x, y, w, h);
        if ids.is_empty() || out_w == 0 || out_h == 0 {
            return owners;
        }
        let sx = w / out_w as f64;
        let sy = h / out_h as f64;
        let mut crossings: Vec<f64> = Vec::new();
        for &id in &ids {
            let polygon = &self.polygons[id as usize];
            let row0 = (((polygon.bbox[1] - y) / sy - 0.5).floor().max(0.0) as u32).min(out_h);
            let row1 = (((polygon.bbox[3] - y) / sy - 0.5).ceil().max(0.0) as u32 + 1).min(out_h);
            for row in row0..row1 {
                let py = y + (row as f64 + 0.5) * sy;
                crossings.clear();
                for ring in &polygon.rings {
                    for i in 0..ring.len() {
                        let a = ring[i];
                        let b = ring[(i + 1) % ring.len()];
                        if (a[1] <= py) != (b[1] <= py) {
                            crossings.push(a[0] + (py - a[1]) / (b[1] - a[1]) * (b[0] - a[0]));
                        }
                    }
                }
                crossings.sort_unstable_by(f64::total_cmp);
                for pair in crossings.as_chunks::<2>().0 {
                    // Pixel centres x + (col + 0.5)·sx inside [pair0, pair1).
                    let c0 = ((pair[0] - x) / sx - 0.5).ceil().max(0.0) as u32;
                    let c1 = (((pair[1] - x) / sx - 0.5).ceil().max(0.0) as u32).min(out_w);
                    for col in c0.min(out_w)..c1 {
                        owners[row as usize * out_w as usize + col as usize] = Some(id);
                    }
                }
            }
        }
        owners
    }

    /// Per-class coverage of a level-0 rectangle, sampled on a
    /// `samples × samples` grid of points.
    pub fn coverage(&self, x: f64, y: f64, w: f64, h: f64, samples: u32) -> Coverage {
        let samples = samples.max(1);
        let owners = self.owner_raster(x, y, w, h, samples, samples);
        let total = owners.len() as f32;
        let mut counts: std::collections::BTreeMap<&str, u32> = std::collections::BTreeMap::new();
        let mut any = 0u32;
        for owner in owners.iter().flatten() {
            any += 1;
            *counts
                .entry(self.polygons[*owner as usize].class.as_str())
                .or_default() += 1;
        }
        Coverage {
            any: any as f32 / total,
            classes: counts
                .into_iter()
                .map(|(c, n)| (c.to_string(), n as f32 / total))
                .collect(),
        }
    }

    /// Rasterises the rectangle into an `out_w × out_h` 8-bit mask:
    /// `value(class)` for pixels inside a polygon (the last one wins), 0
    /// elsewhere.
    #[allow(clippy::too_many_arguments)]
    pub fn rasterize(
        &self,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        out_w: u32,
        out_h: u32,
        value: &dyn Fn(&str) -> u8,
    ) -> Vec<u8> {
        self.owner_raster(x, y, w, h, out_w, out_h)
            .into_iter()
            .map(|owner| owner.map_or(0, |id| value(&self.polygons[id as usize].class)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUPATH: &str = r#"{
      "type": "FeatureCollection",
      "features": [
        {"type": "Feature", "id": "a",
         "geometry": {"type": "Polygon", "coordinates": [[[0,0],[100,0],[100,100],[0,100],[0,0]],
                                                         [[40,40],[60,40],[60,60],[40,60],[40,40]]]},
         "properties": {"objectType": "annotation", "classification": {"name": "Tumor", "color": [200,0,0]}}},
        {"type": "Feature",
         "geometry": {"type": "MultiPolygon", "coordinates": [[[[200,0],[300,0],[300,100],[200,100]]]]},
         "properties": {"classification": {"names": ["Immune", "CD8"]}}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [5, 5]}, "properties": {}},
        {"type": "Feature", "geometry": null}
      ]
    }"#;

    #[test]
    fn qupath_feature_collections_are_parsed() {
        let set = AnnotationSet::from_geojson(QUPATH).unwrap();
        assert_eq!(set.polygons().len(), 2);
        assert_eq!(set.ignored_features, 2);
        assert_eq!(
            set.class_names().into_iter().collect::<Vec<_>>(),
            vec!["Immune: CD8".to_string(), "Tumor".to_string()]
        );
        assert_eq!(
            set.polygons()[0].rings.len(),
            2,
            "hole kept, closing point dropped"
        );
        assert_eq!(set.polygons()[0].rings[0].len(), 4);
    }

    #[test]
    fn other_root_shapes_are_accepted() {
        let array = r#"[{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1]]]},"properties":{"label":"x"}}]"#;
        let set = AnnotationSet::from_geojson(array).unwrap();
        assert_eq!(set.polygons()[0].class, "x");
        let bare = r#"{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1]]]}"#;
        assert_eq!(
            AnnotationSet::from_geojson(bare).unwrap().polygons()[0].class,
            UNCLASSIFIED
        );
        assert!(AnnotationSet::from_geojson(r#"{"foo": 1}"#).is_err());
        assert!(
            AnnotationSet::from_geojson(r#"{"type":"Polygon","coordinates":[[[0,"a"]]]}"#).is_err()
        );
    }

    #[test]
    fn coverage_respects_holes_and_classes() {
        let set = AnnotationSet::from_geojson(QUPATH).unwrap();
        let full = set.coverage(10.0, 10.0, 20.0, 20.0, 16);
        assert_eq!(full.any, 1.0);
        assert_eq!(full.majority(), Some(("Tumor", 1.0)));
        let hole = set.coverage(42.0, 42.0, 16.0, 16.0, 16);
        assert_eq!(hole.any, 0.0, "inside the hole");
        let half = set.coverage(50.0, 0.0, 100.0, 40.0, 20);
        assert!((half.any - 0.5).abs() < 0.06, "{}", half.any);
        let between = set.coverage(150.0, 0.0, 100.0, 100.0, 20);
        assert_eq!(between.majority().map(|m| m.0), Some("Immune: CD8"));
        assert!(!set.may_intersect(1000.0, 1000.0, 10.0, 10.0));
    }

    #[test]
    fn class_filters_and_rasterisation() {
        let mut set = AnnotationSet::from_geojson(QUPATH).unwrap();
        set.retain_classes(&["tumor".into()], &[]);
        assert_eq!(set.polygons().len(), 1);
        let mask = set.rasterize(0.0, 0.0, 100.0, 100.0, 10, 10, &|_| 255);
        assert_eq!(mask[0], 255);
        assert_eq!(mask[5 * 10 + 5], 0, "hole");
        assert_eq!(mask.iter().filter(|&&v| v == 255).count(), 100 - 4);
        set.retain_classes(&[], &["Tumor".into()]);
        assert!(set.is_empty());
    }
}

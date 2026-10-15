//! The host's annotations: `roi.get`, `roi.set`, `measurements.list`,
//! `measurements.add`, `measurements.remove`, `bookmarks.list` and
//! `bookmarks.add`.
//!
//! Edits go through the same code as the ROI and Measure tools
//! ([`crate::host_ui::UiHost::apply_host_edit`]): they can be undone with
//! Ctrl+Z, are kept for the slide (`measurement_store`), redraw the
//! overlays, refresh the Measurements panel and reach plugins
//! (`RoiChanged`). Bookmarks are added as Ctrl+B adds them
//! (`bookmarks.toml`). Reads answer from the published state.

use super::params::{self, CoordinateParam, MAX_COORDINATE, ZoomParam};
use super::{CallResult, RpcError, Session};
use crate::bookmarks::{self, Bookmark};
use crate::host_ui::HostEdit;
use crate::interaction::measure::{self, Calibration, HostMeasurement, MeasureMode};
use crate::plugin_runtime::shared::Published;
use eov_common::MeasurementUnit;
use eov_plugin_api::protocol::{FileId, Rect, Shape};
use serde::{Deserialize, Deserializer};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};

/// Most measurements a slide may have for a script to add one more.
pub(crate) const MAX_MEASUREMENTS: usize = 10_000;
/// Most vertices of a measurement a script adds.
pub(crate) const MAX_POINTS: usize = 10_000;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SlideOnly {
    /// An open slide (`file_id` of `slides`); default: the focused one.
    #[serde(default)]
    slide: Option<i64>,
}

// -- the region of interest ---------------------------------------------------

pub(super) fn roi_get(session: &Session, params: SlideOnly) -> CallResult {
    let published = session.published();
    let (file_id, file) = super::published_slide(&published, params.slide)?;
    Ok(json!({ "file_id": file_id, "roi": file.roi }))
}

/// Present (`Some`), even as `null` (`Some(None)`).
fn present<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Option<Rect>>, D::Error> {
    Option::<Rect>::deserialize(deserializer).map(Some)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RoiSet {
    #[serde(default)]
    slide: Option<i64>,
    /// The region (`null` clears it)...
    #[serde(default, deserialize_with = "present")]
    roi: Option<Option<Rect>>,
    /// ...or its parts.
    #[serde(default)]
    x: Option<f64>,
    #[serde(default)]
    y: Option<f64>,
    #[serde(default)]
    width: Option<f64>,
    #[serde(default)]
    height: Option<f64>,
}

pub(super) fn roi_set(session: &Session, params: RoiSet) -> CallResult {
    let parts = [params.x, params.y, params.width, params.height];
    let roi = match (params.roi, parts) {
        (Some(roi), [None, None, None, None]) => roi.map(params::rect).transpose()?,
        (None, [Some(x), Some(y), Some(width), Some(height)]) => Some(params::rect(Rect {
            x,
            y,
            width,
            height,
        })?),
        _ => {
            return Err(RpcError::invalid(
                "give x, y, width and height, or roi (null clears it)",
            ));
        }
    };
    let slide = params.slide;
    session.on_ui(move |host| {
        let file_id = FileId::from(params::slide(&host.state.read(), slide)?);
        host.apply_host_edit(file_id, HostEdit::SetRoi(roi), true);
        // Plugins hear `RoiChanged` now, not with the next frame, and
        // reads that follow see it.
        host.publish_now();
        let roi = host
            .shared
            .published()
            .files
            .get(&file_id)
            .and_then(|file| file.roi);
        Ok(json!({ "file_id": file_id, "roi": roi }))
    })
}

// -- measurements ---------------------------------------------------------------

/// A measurement as scripts see it: the plugin API's `MeasurementInfo`
/// with the mode, the values in pixels and the label the viewer shows.
fn measurement_json(
    file_id: FileId,
    measurement: &HostMeasurement,
    calibration: &Calibration,
    unit: MeasurementUnit,
) -> Value {
    let value = measure::measure(measurement.mode, &measurement.shape, calibration);
    let closed = matches!(
        measurement.mode,
        MeasureMode::Polygon | MeasureMode::Rectangle | MeasureMode::Ellipse
    );
    json!({
        "measurement_id": measurement.id,
        "file_id": file_id,
        "mode": measurement.mode.id(),
        "shape": measurement.shape,
        "length_px": (measurement.mode != MeasureMode::Angle).then_some(value.length_px),
        "area_px2": closed.then_some(value.area_px2),
        "length_um": value.length_um,
        "area_um2": value.area_um2,
        "angle_deg": value.angle_deg,
        "approximate": value.approximate,
        "label": measure::label(measurement.mode, &value, unit),
    })
}

/// A slide's measurements, as last published.
fn measurements_of(published: &Published, file_id: FileId) -> Result<Vec<Value>, RpcError> {
    let file = published
        .files
        .get(&file_id)
        .ok_or_else(|| RpcError::not_found("the slide was closed"))?;
    let calibration = Calibration::new(file.info.mpp_x, file.info.mpp_y);
    Ok(file
        .host_measurements
        .iter()
        .map(|measurement| measurement_json(file_id, measurement, &calibration, file.unit))
        .collect())
}

pub(super) fn measurements_list(session: &Session, params: SlideOnly) -> CallResult {
    let published = session.published();
    let (file_id, _) = super::published_slide(&published, params.slide)?;
    Ok(Value::Array(measurements_of(&published, file_id)?))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MeasurementAdd {
    #[serde(default)]
    slide: Option<i64>,
    /// `line`, `polyline`, `polygon`, `rectangle`, `ellipse` or `angle`.
    mode: String,
    /// The shape, as the plugin API writes it...
    #[serde(default)]
    shape: Option<Shape>,
    /// ...or its vertices (line, polyline, polygon, angle: arm, vertex,
    /// arm)...
    #[serde(default)]
    points: Option<Vec<[f64; 2]>>,
    /// ...or its bounding box (rectangle, ellipse).
    #[serde(default)]
    x: Option<f64>,
    #[serde(default)]
    y: Option<f64>,
    #[serde(default)]
    width: Option<f64>,
    #[serde(default)]
    height: Option<f64>,
}

/// The shape of a measurement a script adds, checked against its mode.
fn measurement_shape(params: &MeasurementAdd) -> Result<(MeasureMode, Shape), RpcError> {
    let mode = MeasureMode::from_id(&params.mode).ok_or_else(|| {
        RpcError::invalid(format!(
            "{:?} is not a measurement mode (line, polyline, polygon, rectangle, ellipse, angle)",
            params.mode
        ))
    })?;
    let bounds = [params.x, params.y, params.width, params.height];
    let given = [
        params.shape.is_some(),
        params.points.is_some(),
        bounds.iter().any(Option::is_some),
    ];
    if given.iter().filter(|given| **given).count() != 1 {
        return Err(RpcError::invalid(
            "give the measurement as shape, as points, or as x, y, width and height",
        ));
    }
    let shape = if let Some(shape) = &params.shape {
        shape.clone()
    } else if let Some(points) = &params.points {
        match mode {
            MeasureMode::Polygon => Shape::Polygon {
                exterior: points.clone(),
                holes: Vec::new(),
            },
            MeasureMode::Rectangle | MeasureMode::Ellipse => {
                return Err(RpcError::invalid(format!(
                    "give a {} as x, y, width and height",
                    mode.id()
                )));
            }
            _ => Shape::Polyline {
                points: points.clone(),
            },
        }
    } else {
        let [Some(x), Some(y), Some(width), Some(height)] = bounds else {
            return Err(RpcError::invalid("give x, y, width and height"));
        };
        match mode {
            MeasureMode::Rectangle => Shape::Rectangle {
                x,
                y,
                width,
                height,
            },
            MeasureMode::Ellipse => Shape::Ellipse {
                cx: x + width / 2.0,
                cy: y + height / 2.0,
                rx: width / 2.0,
                ry: height / 2.0,
            },
            _ => {
                return Err(RpcError::invalid(format!("give a {} as points", mode.id())));
            }
        }
    };
    let vertices: Vec<&[f64; 2]> = match &shape {
        Shape::Polyline { points } => points.iter().collect(),
        Shape::Polygon { exterior, holes } => {
            exterior.iter().chain(holes.iter().flatten()).collect()
        }
        _ => Vec::new(),
    };
    if vertices.len() > MAX_POINTS {
        return Err(RpcError::too_large(format!(
            "a measurement has at most {MAX_POINTS} points"
        )));
    }
    let on_slide = |value: f64| value.is_finite() && value.abs() <= MAX_COORDINATE;
    let numbers_fit = match &shape {
        Shape::Rectangle {
            x,
            y,
            width,
            height,
        } => [x, y, width, height].iter().all(|v| on_slide(**v)) && *width > 0.0 && *height > 0.0,
        Shape::Ellipse { cx, cy, rx, ry } => {
            [cx, cy, rx, ry].iter().all(|v| on_slide(**v)) && *rx > 0.0 && *ry > 0.0
        }
        _ => vertices.iter().all(|p| on_slide(p[0]) && on_slide(p[1])),
    };
    let fits_mode = match (mode, &shape) {
        (MeasureMode::Line, Shape::Polyline { points }) => points.len() == 2,
        (MeasureMode::Angle, Shape::Polyline { points }) => points.len() == 3,
        (MeasureMode::Polyline, Shape::Polyline { points }) => points.len() >= 2,
        (MeasureMode::Polygon, Shape::Polygon { .. })
        | (MeasureMode::Rectangle, Shape::Rectangle { .. })
        | (MeasureMode::Ellipse, Shape::Ellipse { .. }) => true,
        _ => false,
    };
    if !fits_mode {
        return Err(RpcError::invalid(format!(
            "a {} measurement is {}",
            mode.id(),
            match mode {
                MeasureMode::Line => "a polyline of 2 points",
                MeasureMode::Angle => "a polyline of 3 points (arm, vertex, arm)",
                MeasureMode::Polyline => "a polyline of 2 points or more",
                MeasureMode::Polygon => "a polygon",
                MeasureMode::Rectangle => "a rectangle",
                MeasureMode::Ellipse => "an ellipse",
            }
        )));
    }
    if !numbers_fit || !shape.is_valid() {
        return Err(RpcError::invalid(
            "the shape is not on a slide (coordinates, sizes or too few points)",
        ));
    }
    Ok((mode, shape))
}

/// A new measurement id; the mode prefix is how edits know the mode
/// ([`crate::interaction::controller::measurement_mode_of`]).
fn measurement_id(mode: MeasureMode) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let count = COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    format!("{}:{nanos:x}-script{count}", mode.id())
}

pub(super) fn measurements_add(session: &Session, params: MeasurementAdd) -> CallResult {
    let (mode, shape) = measurement_shape(&params)?;
    let slide = params.slide;
    session.on_ui(move |host| {
        let file_id = {
            let state = host.state.read();
            let file_id = params::slide(&state, slide)?;
            let count = state
                .get_file(file_id)
                .map_or(0, |file| file.measurements.len());
            if count >= MAX_MEASUREMENTS {
                return Err(RpcError::too_large(format!(
                    "the slide has {count} measurements; scripts add at most \
                     {MAX_MEASUREMENTS}"
                )));
            }
            file_id
        };
        let measurement = HostMeasurement {
            id: measurement_id(mode),
            mode,
            shape,
        };
        let id = measurement.id.clone();
        let file_id = FileId::from(file_id);
        host.apply_host_edit(file_id, HostEdit::Add(measurement), true);
        host.publish_now();
        measurements_of(&host.shared.published(), file_id)?
            .into_iter()
            .find(|added| added["measurement_id"] == id.as_str())
            .ok_or_else(|| RpcError::internal("the measurement was not added"))
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MeasurementRemove {
    #[serde(default)]
    slide: Option<i64>,
    /// The measurements to delete (`measurement_id` of the list)...
    #[serde(default)]
    ids: Option<Vec<String>>,
    /// ...or all of them.
    #[serde(default)]
    all: bool,
}

pub(super) fn measurements_remove(session: &Session, params: MeasurementRemove) -> CallResult {
    let edit = match (params.ids, params.all) {
        (Some(ids), false) if !ids.is_empty() => {
            if ids.len() > MAX_MEASUREMENTS {
                return Err(RpcError::too_large(format!(
                    "at most {MAX_MEASUREMENTS} ids at once"
                )));
            }
            HostEdit::Remove(ids)
        }
        (None, true) => HostEdit::ClearMeasurements,
        _ => return Err(RpcError::invalid("give ids (a list) or \"all\": true")),
    };
    let slide = params.slide;
    session.on_ui(move |host| {
        let (file_id, removed) = {
            let state = host.state.read();
            let file_id = params::slide(&state, slide)?;
            let existing: Vec<String> = state
                .get_file(file_id)
                .map(|file| file.measurements.iter().map(|m| m.id.clone()).collect())
                .unwrap_or_default();
            let removed = match &edit {
                HostEdit::Remove(ids) => {
                    if let Some(missing) = ids.iter().find(|id| !existing.contains(id)) {
                        return Err(RpcError::not_found(format!(
                            "the slide has no measurement {missing:?} (see measurements.list)"
                        )));
                    }
                    ids.clone()
                }
                _ => existing,
            };
            (file_id, removed)
        };
        host.apply_host_edit(FileId::from(file_id), edit, true);
        host.publish_now();
        Ok(json!({ "file_id": file_id, "removed": removed }))
    })
}

// -- bookmarks ------------------------------------------------------------------

/// A bookmark as scripts see it: its place in the slide's list, the view
/// and the objective-equivalent magnification.
fn bookmark_json(index: usize, bookmark: &Bookmark, native_magnification: Option<f64>) -> Value {
    json!({
        "index": index,
        "name": bookmark.name,
        "center_x": bookmark.center_x,
        "center_y": bookmark.center_y,
        "zoom": bookmark.zoom,
        "rotation": bookmark.rotation,
        "flipped": bookmark.flipped,
        "magnification": native_magnification.map(|native| native * bookmark.zoom),
    })
}

/// The identity key bookmarks are kept under (a worker thread: it may
/// sample the slide's files).
fn identity_key(wsi: &eov_common::WsiFile) -> Result<String, RpcError> {
    wsi.identity()
        .map(|identity| identity.key().to_string())
        .map_err(|error| RpcError::failed(format!("this slide cannot be identified: {error}")))
}

pub(super) fn bookmarks_list(session: &Session, params: SlideOnly) -> CallResult {
    let published = session.published();
    let (file_id, file) = super::published_slide(&published, params.slide)?;
    let info = &file.info;
    let native = super::viewer::native_magnification(info);
    let key = match &info.identity {
        Some(key) => key.clone(),
        None => identity_key(&session.open_slide(file_id)?)?,
    };
    let list = bookmarks::list(&key);
    Ok(Value::Array(
        list.iter()
            .enumerate()
            .map(|(index, bookmark)| bookmark_json(index, bookmark, native))
            .collect(),
    ))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BookmarkAdd {
    #[serde(default)]
    slide: Option<i64>,
    /// Default: "Bookmark N".
    #[serde(default)]
    name: Option<String>,
    /// The view; what is left out is the slide's view in its pane.
    #[serde(default)]
    x: Option<CoordinateParam>,
    #[serde(default)]
    y: Option<CoordinateParam>,
    #[serde(default)]
    zoom: Option<ZoomParam>,
    #[serde(default)]
    magnification: Option<f64>,
    #[serde(default)]
    rotation: Option<f64>,
    #[serde(default)]
    flipped: Option<bool>,
}

/// A bookmark waiting for the slide's identity.
struct Pending {
    wsi: eov_common::WsiFile,
    file_name: String,
    native: Option<f64>,
    bookmark: Bookmark,
}

pub(super) fn bookmarks_add(session: &Session, params: BookmarkAdd) -> CallResult {
    let request = params::view_request(
        params.x.as_ref(),
        params.y.as_ref(),
        params.zoom.as_ref(),
        params.magnification,
    )?;
    let name = params.name.as_deref().map(str::trim).map(str::to_string);
    if let Some(name) = &name
        && name.chars().count() > bookmarks::MAX_NAME_CHARS
    {
        return Err(RpcError::invalid(format!(
            "a bookmark's name has at most {} characters",
            bookmarks::MAX_NAME_CHARS
        )));
    }
    let (slide, rotation, flipped) = (params.slide, params.rotation, params.flipped);
    let pending = session.on_ui(move |host| {
        let state = host.state.read();
        let file_id = params::slide(&state, slide)?;
        let file = state
            .get_file(file_id)
            .ok_or_else(|| RpcError::not_found("the slide was closed"))?;
        // The view in the pane showing the slide, or the slide's own.
        let viewport = params::pane_showing(&state, file_id)
            .and_then(|pane| file.pane_state(pane))
            .map_or(&file.viewport, |pane_state| &pane_state.viewport);
        let current = (
            viewport.viewport.center.x,
            viewport.viewport.center.y,
            viewport.viewport.zoom,
        );
        let scale = crate::launch::SlideScale::of_open_file(file);
        let view = crate::launch::resolve_view(&request, &scale, current, viewport.fit_view());
        if let Some(warning) = view.warnings.first() {
            return Err(RpcError::invalid(warning.clone()));
        }
        let orientation = params::orientation(rotation, flipped, viewport.viewport.orientation)?;
        Ok(Pending {
            wsi: file.wsi.clone(),
            file_name: file.filename.clone(),
            native: crate::callbacks::native_magnification(file),
            bookmark: Bookmark {
                name: name.unwrap_or_default(),
                center_x: view.center_x,
                center_y: view.center_y,
                zoom: view.zoom,
                rotation: orientation.degrees(),
                flipped: orientation.flipped(),
            },
        })
    })?;
    let key = identity_key(&pending.wsi)?;
    // On the UI thread, where the window adds bookmarks too.
    session.on_ui(move |_| {
        let Pending {
            file_name,
            native,
            mut bookmark,
            ..
        } = pending;
        let count = bookmarks::list(&key).len();
        if count >= bookmarks::MAX_PER_SLIDE {
            return Err(RpcError::too_large(format!(
                "a slide can have at most {} bookmarks",
                bookmarks::MAX_PER_SLIDE
            )));
        }
        if bookmark.name.is_empty() {
            bookmark.name = format!("Bookmark {}", count + 1);
        }
        bookmarks::add(&key, &file_name, bookmark.clone()).map_err(RpcError::failed)?;
        let index = bookmarks::list(&key)
            .iter()
            .rposition(|kept| *kept == bookmark)
            .unwrap_or(count);
        Ok(bookmark_json(index, &bookmark, native))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add(json: Value) -> MeasurementAdd {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn measurements_take_shapes_points_or_boxes() {
        let (mode, shape) =
            measurement_shape(&add(json!({"mode": "line", "points": [[0, 0], [30, 40]]}))).unwrap();
        assert_eq!(mode, MeasureMode::Line);
        assert_eq!(
            shape,
            Shape::Polyline {
                points: vec![[0.0, 0.0], [30.0, 40.0]]
            }
        );
        let (_, shape) = measurement_shape(&add(
            json!({"mode": "polygon", "points": [[0, 0], [10, 0], [10, 10]]}),
        ))
        .unwrap();
        assert!(matches!(shape, Shape::Polygon { ref holes, .. } if holes.is_empty()));
        let (_, shape) = measurement_shape(&add(
            json!({"mode": "ellipse", "x": 10, "y": 20, "width": 100, "height": 50}),
        ))
        .unwrap();
        assert_eq!(
            shape,
            Shape::Ellipse {
                cx: 60.0,
                cy: 45.0,
                rx: 50.0,
                ry: 25.0
            }
        );
        let (mode, _) = measurement_shape(&add(json!({
            "mode": "rectangle",
            "shape": {"kind": "rectangle", "x": 0, "y": 0, "width": 5, "height": 5}
        })))
        .unwrap();
        assert_eq!(mode, MeasureMode::Rectangle);
        assert!(
            measurement_shape(&add(
                json!({"mode": "angle", "points": [[10, 0], [0, 0], [0, 10]]})
            ))
            .is_ok()
        );
    }

    #[test]
    fn measurements_must_fit_their_mode() {
        for bad in [
            json!({"mode": "line", "points": [[0, 0], [1, 1], [2, 2]]}),
            json!({"mode": "angle", "points": [[0, 0], [1, 1]]}),
            json!({"mode": "polygon", "points": [[0, 0], [1, 1]]}),
            json!({"mode": "rectangle", "points": [[0, 0], [1, 1]]}),
            json!({"mode": "line", "x": 0, "y": 0, "width": 1, "height": 1}),
            json!({"mode": "rectangle", "x": 0, "y": 0, "width": 0, "height": 1}),
            json!({"mode": "rectangle", "x": 0, "y": 0, "width": 1}),
            json!({"mode": "circle", "points": [[0, 0], [1, 1]]}),
            json!({"mode": "line"}),
            json!({"mode": "line", "points": [[0, 0], [1, 1]], "x": 3}),
            json!({"mode": "line", "points": [[0, 0], [1e300, 1]]}),
            json!({"mode": "polyline", "shape": {"kind": "point", "x": 1, "y": 2}}),
        ] {
            assert!(measurement_shape(&add(bad.clone())).is_err(), "{bad}");
        }
        let many: Vec<[f64; 2]> = (0..=MAX_POINTS).map(|i| [i as f64, 0.0]).collect();
        let error =
            measurement_shape(&add(json!({"mode": "polyline", "points": many}))).unwrap_err();
        assert_eq!(error.kind, super::super::ErrorKind::TooLarge);
    }

    #[test]
    fn measurement_ids_carry_their_mode() {
        let id = measurement_id(MeasureMode::Angle);
        assert_eq!(
            crate::interaction::controller::measurement_mode_of(&id),
            MeasureMode::Angle
        );
        assert_ne!(id, measurement_id(MeasureMode::Angle));
    }

    #[test]
    fn regions_of_interest_are_given_whole_or_cleared() {
        let set: RoiSet = serde_json::from_value(json!({"roi": null})).unwrap();
        assert_eq!(set.roi, Some(None));
        let set: RoiSet =
            serde_json::from_value(json!({"x": 1, "y": 2, "width": 3, "height": 4})).unwrap();
        assert_eq!(set.roi, None);
        assert!(serde_json::from_value::<RoiSet>(json!({"roi": {"x": 1}})).is_err());
    }
}

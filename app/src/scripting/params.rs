//! What scripts pass: slides, panes, paths, coordinates and zooms, and the
//! rules paths follow.
//!
//! Scripts get no special trust. A path is used only if eov would open it
//! from an `eov://` link: absolute, and on Windows on a network server
//! only when the server is in `trusted_link_servers`
//! ([`crate::launch::link_path_refusal`]). Slides must be existing regular
//! files (never a device, a pipe or a socket, which could block the reader
//! for ever); exports go to a new file in an existing folder unless the
//! script asks to replace one.

use super::RpcError;
use crate::launch::{Coordinate, Point, ViewRequest, Zoom};
use crate::state::{AppState, PaneId};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Longest path accepted, in bytes (Windows' long paths are 32767 UTF-16
/// units).
const MAX_PATH_BYTES: usize = 32 * 1024;

/// Largest coordinate, in level-0 pixels (slides are far smaller).
pub(crate) const MAX_COORDINATE: f64 = (1u64 << 40) as f64;

/// A path as a script gives it: `path` (text), or `os_path`, the hex of
/// the platform's encoding (bytes on Unix, UTF-16 units on Windows), for a
/// path that is not valid Unicode.
pub(super) fn path_from(
    path: Option<String>,
    os_path: Option<String>,
) -> Result<Option<PathBuf>, RpcError> {
    let path = match (path, os_path) {
        (Some(_), Some(_)) => return Err(RpcError::invalid("give path or os_path, not both")),
        (None, None) => return Ok(None),
        (Some(text), None) => {
            if text.len() > MAX_PATH_BYTES {
                return Err(RpcError::too_large("the path is too long"));
            }
            PathBuf::from(text)
        }
        (None, Some(hex)) => {
            if hex.len() > 4 * MAX_PATH_BYTES {
                return Err(RpcError::too_large("the path is too long"));
            }
            crate::recent::decode_os(&hex)
                .ok_or_else(|| RpcError::invalid("os_path is not the hex of a path"))?
        }
    };
    if path.as_os_str().is_empty() {
        return Err(RpcError::invalid("the path is empty"));
    }
    if !path.is_absolute() {
        return Err(RpcError::invalid(format!(
            "paths must be absolute, not {}",
            path.display()
        )));
    }
    if let Some(reason) =
        crate::launch::link_path_refusal(&path, &crate::config::load_trusted_link_servers())
    {
        return Err(RpcError::refused(reason));
    }
    Ok(Some(path))
}

/// A slide a script may read or open: an existing regular file, as eov
/// names it ([`crate::paths::canonical`]).
pub(super) fn slide_file(path: &Path) -> Result<PathBuf, RpcError> {
    let metadata = std::fs::metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            RpcError::not_found(format!("{} does not exist", path.display()))
        } else {
            RpcError::failed(format!("cannot read {}: {error}", path.display()))
        }
    })?;
    if metadata.is_dir() {
        return Err(RpcError::invalid(format!(
            "{} is a folder; give a slide file",
            path.display()
        )));
    }
    if !metadata.is_file() {
        return Err(RpcError::refused(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    Ok(crate::paths::canonical(path))
}

/// A file `export.view` may write: in an existing folder, not a folder
/// itself, and only replacing an existing file with `overwrite`.
pub(super) fn output_file(path: &Path, overwrite: bool) -> Result<PathBuf, RpcError> {
    let no_file = || RpcError::invalid("the path names no file");
    let name = path.file_name().ok_or_else(no_file)?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(no_file)?;
    if !parent.is_dir() {
        return Err(RpcError::not_found(format!(
            "the folder {} does not exist",
            parent.display()
        )));
    }
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => {
            Err(RpcError::invalid(format!("{} is a folder", path.display())))
        }
        Ok(metadata) if !metadata.is_file() => Err(RpcError::refused(format!(
            "{} is not a regular file",
            path.display()
        ))),
        Ok(_) if !overwrite => Err(RpcError::refused(format!(
            "{} exists; pass \"overwrite\": true to replace it",
            path.display()
        ))),
        _ => Ok(crate::paths::canonical(parent).join(name)),
    }
}

/// A coordinate: a number of level-0 pixels, or text with a unit
/// (`"1200px"`, `"300um"`), as `--goto` takes.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(super) enum CoordinateParam {
    Pixels(f64),
    Text(String),
}

impl CoordinateParam {
    fn coordinate(&self) -> Result<Coordinate, RpcError> {
        let coordinate = match self {
            CoordinateParam::Pixels(value) => Coordinate::Px(*value),
            CoordinateParam::Text(text) => {
                crate::launch::parse_coordinate(text).map_err(RpcError::invalid)?
            }
        };
        let value = match coordinate {
            Coordinate::Px(value) | Coordinate::Um(value) => value,
        };
        if !value.is_finite() || value.abs() > MAX_COORDINATE {
            return Err(RpcError::invalid(format!(
                "{value} is not a coordinate on a slide"
            )));
        }
        Ok(coordinate)
    }
}

/// A zoom: a number of screen pixels per level-0 pixel, or text as
/// `--zoom` takes (`"20x"`, `"0.5"`, `"1:1"`, `"fit"`).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(super) enum ZoomParam {
    Factor(f64),
    Text(String),
}

/// The view a script asks for: a centre and a zoom, either optional.
pub(super) fn view_request(
    x: Option<&CoordinateParam>,
    y: Option<&CoordinateParam>,
    zoom: Option<&ZoomParam>,
    magnification: Option<f64>,
) -> Result<ViewRequest, RpcError> {
    let center = match (x, y) {
        (Some(x), Some(y)) => {
            let point = Point {
                x: x.coordinate()?,
                y: y.coordinate()?,
            };
            if std::mem::discriminant(&point.x) != std::mem::discriminant(&point.y) {
                return Err(RpcError::invalid("give x and y in the same unit"));
            }
            Some(point)
        }
        (None, None) => None,
        _ => return Err(RpcError::invalid("give both x and y, or neither")),
    };
    let zoom = match (zoom, magnification) {
        (Some(_), Some(_)) => {
            return Err(RpcError::invalid("give zoom or magnification, not both"));
        }
        (None, None) => None,
        (Some(ZoomParam::Factor(factor)), None) => Some(Zoom::Factor(*factor)),
        (Some(ZoomParam::Text(text)), None) => {
            Some(crate::launch::parse_zoom(text).map_err(RpcError::invalid)?)
        }
        (None, Some(magnification)) => Some(Zoom::Magnification(magnification)),
    };
    if let Some(Zoom::Factor(value) | Zoom::Magnification(value)) = zoom
        && !(value.is_finite() && value > 0.0 && value <= 1000.0)
    {
        return Err(RpcError::invalid(format!(
            "{value} is not a zoom (more than 0, at most 1000)"
        )));
    }
    Ok(ViewRequest { center, zoom })
}

/// The orientation `rotation` (degrees clockwise, a multiple of 90) and
/// `flipped` ask for, keeping what they leave out from `current`.
pub(super) fn orientation(
    rotation: Option<f64>,
    flipped: Option<bool>,
    current: eov_common::viewport::Orientation,
) -> Result<eov_common::viewport::Orientation, RpcError> {
    let degrees = rotation.unwrap_or(f64::from(current.degrees()));
    let flipped = flipped.unwrap_or(current.flipped());
    eov_common::viewport::Orientation::from_degrees(degrees, flipped)
        .ok_or_else(|| RpcError::invalid(format!("{degrees}° is not a multiple of 90°")))
}

/// The open slide `slide` (a `file_id` of `slides`), or the one the focused
/// pane shows.
pub(super) fn slide(state: &AppState, slide: Option<i64>) -> Result<i32, RpcError> {
    match slide {
        Some(id) => i32::try_from(id)
            .ok()
            .filter(|id| state.open_files.iter().any(|file| file.id == *id))
            .ok_or_else(|| {
                RpcError::not_found(format!("no open slide has the id {id} (see slides)"))
            }),
        None => state
            .active_file_id_for_pane(state.active_viewport_pane())
            .ok_or_else(|| {
                RpcError::not_found("the focused pane shows no slide; pass \"slide\" (see slides)")
            }),
    }
}

/// The pane `pane` (a `pane` of `panes`), or the focused one.
pub(super) fn pane(state: &AppState, pane: Option<u32>) -> Result<PaneId, RpcError> {
    match pane {
        Some(uid) => state
            .pane_by_uid(uid)
            .ok_or_else(|| RpcError::not_found(format!("there is no pane {uid} (see panes)"))),
        None => Ok(state.active_viewport_pane()),
    }
}

/// The pane that shows `file_id`: the focused one if it does, else the
/// first.
pub(super) fn pane_showing(state: &AppState, file_id: i32) -> Option<PaneId> {
    let focused = state.active_viewport_pane();
    if state.active_file_id_for_pane(focused) == Some(file_id) {
        return Some(focused);
    }
    (0..state.panes.len())
        .map(PaneId)
        .find(|pane| state.active_file_id_for_pane(*pane) == Some(file_id))
}

/// A level-0 rectangle with a positive size.
pub(super) fn rect(
    rect: eov_plugin_api::protocol::Rect,
) -> Result<eov_plugin_api::protocol::Rect, RpcError> {
    let values = [rect.x, rect.y, rect.width, rect.height];
    if !values
        .iter()
        .all(|value| value.is_finite() && value.abs() <= MAX_COORDINATE)
    {
        return Err(RpcError::invalid("the rectangle is not on a slide"));
    }
    if rect.width <= 0.0 || rect.height <= 0.0 {
        return Err(RpcError::invalid(
            "the rectangle must have a positive width and height",
        ));
    }
    Ok(rect)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn number(value: f64) -> CoordinateParam {
        CoordinateParam::Pixels(value)
    }

    fn text(value: &str) -> CoordinateParam {
        CoordinateParam::Text(value.to_string())
    }

    #[test]
    fn views_take_pixels_micrometres_and_zooms() {
        let view = view_request(
            Some(&number(1200.0)),
            Some(&number(800.0)),
            Some(&ZoomParam::Factor(0.5)),
            None,
        )
        .unwrap();
        assert_eq!(
            view.center,
            Some(Point {
                x: Coordinate::Px(1200.0),
                y: Coordinate::Px(800.0)
            })
        );
        assert_eq!(view.zoom, Some(Zoom::Factor(0.5)));

        let view = view_request(
            Some(&text("300um")),
            Some(&text("200um")),
            Some(&ZoomParam::Text("20x".into())),
            None,
        )
        .unwrap();
        assert_eq!(view.zoom, Some(Zoom::Magnification(20.0)));
        assert!(matches!(view.center.unwrap().x, Coordinate::Um(_)));

        let view = view_request(None, None, None, Some(10.0)).unwrap();
        assert_eq!(view.zoom, Some(Zoom::Magnification(10.0)));
        assert_eq!(
            view_request(None, None, Some(&ZoomParam::Text("fit".into())), None)
                .unwrap()
                .zoom,
            Some(Zoom::Fit)
        );

        for bad in [
            view_request(Some(&number(1.0)), None, None, None),
            view_request(Some(&number(1.0)), Some(&text("2um")), None, None),
            view_request(None, None, Some(&ZoomParam::Factor(0.0)), None),
            view_request(None, None, Some(&ZoomParam::Factor(f64::NAN)), None),
            view_request(None, None, None, Some(-3.0)),
            view_request(None, None, Some(&ZoomParam::Factor(1.0)), Some(20.0)),
            view_request(Some(&number(1e300)), Some(&number(0.0)), None, None),
            view_request(Some(&text("5mm")), Some(&text("5mm")), None, None),
        ] {
            assert_eq!(
                bad.unwrap_err().kind,
                super::super::ErrorKind::InvalidParams
            );
        }
    }

    #[test]
    fn orientations_are_quarter_turns() {
        use eov_common::viewport::Orientation;
        let current = Orientation::new(1, false);
        assert_eq!(orientation(None, None, current).unwrap(), current);
        assert_eq!(
            orientation(Some(180.0), None, current).unwrap(),
            Orientation::new(2, false)
        );
        assert_eq!(
            orientation(None, Some(true), current).unwrap(),
            Orientation::new(1, true)
        );
        assert!(orientation(Some(45.0), None, current).is_err());
    }

    #[test]
    fn paths_must_be_absolute_regular_files() {
        let dir = tempfile::tempdir().unwrap();
        let slide = dir.path().join("a.svs");
        std::fs::write(&slide, b"x").unwrap();
        let given = path_from(Some(slide.to_string_lossy().into_owned()), None)
            .unwrap()
            .unwrap();
        assert_eq!(slide_file(&given).unwrap(), crate::paths::canonical(&slide));

        assert!(path_from(None, None).unwrap().is_none());
        let relative = path_from(Some("a.svs".into()), None).unwrap_err();
        assert!(relative.message.contains("absolute"), "{relative}");
        assert!(path_from(Some(String::new()), None).is_err());
        assert!(path_from(Some("/x".into()), Some("2f78".into())).is_err());
        assert_eq!(
            path_from(Some("/".repeat(MAX_PATH_BYTES + 1)), None)
                .unwrap_err()
                .kind,
            super::super::ErrorKind::TooLarge
        );
        assert_eq!(
            slide_file(&dir.path().join("missing.svs"))
                .unwrap_err()
                .kind,
            super::super::ErrorKind::NotFound
        );
        assert_eq!(
            slide_file(dir.path()).unwrap_err().kind,
            super::super::ErrorKind::InvalidParams
        );
        #[cfg(unix)]
        {
            // A device would block or never end.
            let device = path_from(Some("/dev/null".into()), None).unwrap().unwrap();
            assert_eq!(
                slide_file(&device).unwrap_err().kind,
                super::super::ErrorKind::Refused
            );
            // Paths that are not UTF-8 come as hex.
            let odd = path_from(None, Some(crate::recent::encode_os(&slide)))
                .unwrap()
                .unwrap();
            assert_eq!(odd, slide);
        }
    }

    #[test]
    fn exports_never_replace_a_file_unasked() {
        let dir = tempfile::tempdir().unwrap();
        let new = dir.path().join("figure.png");
        assert_eq!(
            output_file(&new, false).unwrap(),
            crate::paths::canonical(dir.path()).join("figure.png")
        );
        std::fs::write(&new, b"old").unwrap();
        assert_eq!(
            output_file(&new, false).unwrap_err().kind,
            super::super::ErrorKind::Refused
        );
        assert!(output_file(&new, true).is_ok());
        assert_eq!(
            output_file(&dir.path().join("no/such/folder.png"), true)
                .unwrap_err()
                .kind,
            super::super::ErrorKind::NotFound
        );
        assert!(output_file(dir.path(), true).is_err());
    }

    #[test]
    fn rectangles_need_a_size() {
        use eov_plugin_api::protocol::Rect;
        let good = Rect {
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
        };
        assert_eq!(rect(good).unwrap(), good);
        assert!(rect(Rect { width: 0.0, ..good }).is_err());
        assert!(
            rect(Rect {
                x: f64::INFINITY,
                ..good
            })
            .is_err()
        );
    }
}

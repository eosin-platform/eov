//! What eov is asked to open: files and panes from the command line, the
//! view of the first slide, `eov://` links, and the same requests forwarded
//! by a second eov or by macOS (SH-7, SH-8, AUDIT §14.1).
//!
//! - Every positional argument is a slide or a folder, opened as a tab of
//!   the current pane; the first one is selected and the others open
//!   behind it, in order. An argument that exists on disk is taken
//!   literally, commas and all (`"Case 12, H&E.svs"`). One that does not
//!   exist but contains commas is a list of files, the syntax of eov < 1.0
//!   (`eov a.svs,b.svs`).
//! - `--pane FILES` (repeatable) opens another pane side by side, its value
//!   split by the same rule.
//! - `file://` URIs, which file managers pass, are decoded; non-UTF-8 paths
//!   are kept as they are.
//! - `--goto X,Y` (level-0 pixels, or micrometres with `um`) and `--zoom`
//!   (`20x`, a factor such as `0.5`, or `fit`) set the view of the first
//!   slide. An `eov://open?path=…&x=…&y=…&zoom=…` link carries its own.
//! - Arguments that cannot be used do not stop eov: they are reported in
//!   the window, and everything else opens.

use crate::AppWindow;
use crate::state::{AppState, PaneId};
use eov_common::TileCache;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use slint::{ComponentHandle, Timer};
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

/// A position along one axis of a slide.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Coordinate {
    /// Level-0 pixels.
    Px(f64),
    /// Micrometres from the slide's top-left corner.
    Um(f64),
}

/// How far a slide is zoomed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Zoom {
    /// The whole slide (its scanned area).
    Fit,
    /// An objective-equivalent magnification (`20x`).
    Magnification(f64),
    /// Screen pixels per level-0 pixel (`1` = full resolution).
    Factor(f64),
}

/// A point on a slide, `--goto X,Y`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct Point {
    pub(crate) x: Coordinate,
    pub(crate) y: Coordinate,
}

/// Where a slide should open.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct ViewRequest {
    /// Centred on this point (the opening view's centre otherwise).
    pub(crate) center: Option<Point>,
    /// At this zoom (the opening zoom otherwise).
    pub(crate) zoom: Option<Zoom>,
}

impl ViewRequest {
    pub(crate) fn is_empty(&self) -> bool {
        self.center.is_none() && self.zoom.is_none()
    }

    /// `self`, with what it leaves open taken from `other`.
    fn or(self, other: ViewRequest) -> ViewRequest {
        ViewRequest {
            center: self.center.or(other.center),
            zoom: self.zoom.or(other.zoom),
        }
    }
}

/// An argument that could not be used.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Problem {
    /// The argument, as given.
    pub(crate) subject: String,
    pub(crate) reason: String,
}

impl Problem {
    fn new(subject: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.subject, self.reason)
    }
}

/// What to open. Paths are absolute.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct OpenRequest {
    /// Tabs of the current pane, in order; the first is selected.
    pub(crate) tabs: Vec<PathBuf>,
    /// Panes to add side by side, each with its tabs.
    pub(crate) panes: Vec<Vec<PathBuf>>,
    /// Where slides open (`--goto`/`--zoom`, links).
    pub(crate) views: Vec<(PathBuf, ViewRequest)>,
    /// Arguments that could not be used, reported in the window.
    pub(crate) problems: Vec<Problem>,
}

impl OpenRequest {
    /// Whether there is nothing to open (problems aside).
    pub(crate) fn opens_nothing(&self) -> bool {
        self.tabs.is_empty() && self.panes.iter().all(Vec::is_empty)
    }
}

/// What a path names on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Found {
    File,
    Folder,
}

/// Looks a path up on disk.
pub(crate) fn find_on_disk(path: &Path) -> Option<Found> {
    let metadata = std::fs::metadata(path).ok()?;
    Some(if metadata.is_dir() {
        Found::Folder
    } else {
        Found::File
    })
}

// -- parsing -----------------------------------------------------------------

fn parse_number(text: &str) -> Result<f64, String> {
    let value: f64 = text
        .trim()
        .parse()
        .map_err(|_| format!("{:?} is not a number", text.trim()))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{:?} is not a finite number", text.trim()))
    }
}

/// `1200`, `1200px`, `350um` or `350µm`.
pub(crate) fn parse_coordinate(text: &str) -> Result<Coordinate, String> {
    let text = text.trim();
    let end = text
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '+' | '-' | 'e' | 'E')))
        .unwrap_or(text.len());
    let (number, unit) = text.split_at(end);
    let value = parse_number(number)?;
    if value < 0.0 {
        return Err(format!("{text:?} is negative"));
    }
    match unit.trim().to_lowercase().as_str() {
        "" | "px" => Ok(Coordinate::Px(value)),
        "um" | "\u{b5}m" | "\u{3bc}m" => Ok(Coordinate::Um(value)),
        other => Err(format!("unknown unit {other:?} in {text:?} (use px or um)")),
    }
}

/// `--goto X,Y`: level-0 pixels (`1200,800`) or micrometres
/// (`300um,200um`).
pub(crate) fn parse_point(text: &str) -> Result<Point, String> {
    let (x, y) = text.split_once(',').ok_or_else(|| {
        format!("expected X,Y (for example 1200,800 or 300um,200um), not {text:?}")
    })?;
    let point = Point {
        x: parse_coordinate(x)?,
        y: parse_coordinate(y)?,
    };
    match (point.x, point.y) {
        (Coordinate::Px(_), Coordinate::Px(_)) | (Coordinate::Um(_), Coordinate::Um(_)) => {
            Ok(point)
        }
        _ => Err(format!(
            "give both coordinates in the same unit, not {text:?}"
        )),
    }
}

/// Upper bound for zoom factors and magnifications.
const MAX_ZOOM_VALUE: f64 = 1000.0;

/// `--zoom`: `20x` (magnification), `0.5` or `1:1` (screen pixels per
/// slide pixel), or `fit`.
pub(crate) fn parse_zoom(text: &str) -> Result<Zoom, String> {
    let text = text.trim();
    if text.eq_ignore_ascii_case("fit") {
        return Ok(Zoom::Fit);
    }
    if text == "1:1" {
        return Ok(Zoom::Factor(1.0));
    }
    let (number, magnification) = match text.strip_suffix(['x', 'X', '\u{d7}']) {
        Some(number) => (number, true),
        None => (text, false),
    };
    let value = parse_number(number)?;
    if value <= 0.0 || value > MAX_ZOOM_VALUE {
        return Err(format!(
            "{text:?} is out of range (more than 0, at most {MAX_ZOOM_VALUE})"
        ));
    }
    Ok(if magnification {
        Zoom::Magnification(value)
    } else {
        Zoom::Factor(value)
    })
}

/// Percent-decodes a query component (`+` is a space, as in HTML forms).
fn percent_decode(text: &str) -> Result<Vec<u8>, String> {
    let bytes = text.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                decoded.push(b' ');
                index += 1;
            }
            b'%' => {
                let digits = bytes
                    .get(index + 1..index + 3)
                    .filter(|digits| digits.iter().all(u8::is_ascii_hexdigit))
                    .ok_or_else(|| "a % is not followed by two hex digits".to_string())?;
                let text = std::str::from_utf8(digits).expect("hex digits are ASCII");
                decoded.push(u8::from_str_radix(text, 16).expect("two hex digits"));
                index += 3;
            }
            byte => {
                decoded.push(byte);
                index += 1;
            }
        }
    }
    Ok(decoded)
}

#[cfg(unix)]
fn path_from_bytes(bytes: Vec<u8>) -> Result<PathBuf, String> {
    use std::os::unix::ffi::OsStringExt;
    Ok(PathBuf::from(OsString::from_vec(bytes)))
}

#[cfg(not(unix))]
fn path_from_bytes(bytes: Vec<u8>) -> Result<PathBuf, String> {
    String::from_utf8(bytes)
        .map(PathBuf::from)
        .map_err(|_| "the path is not valid UTF-8".to_string())
}

fn has_scheme(text: &str, scheme: &str) -> bool {
    text.len() > scheme.len()
        && text.as_bytes()[scheme.len()] == b':'
        && text[..scheme.len()].eq_ignore_ascii_case(scheme)
}

/// A `file://` URI's local path.
pub(crate) fn file_uri_path(text: &str) -> Result<PathBuf, String> {
    let url = url::Url::parse(text).map_err(|error| format!("not a valid file URI ({error})"))?;
    if url.scheme() != "file" {
        return Err("not a file URI".to_string());
    }
    url.to_file_path().map_err(|()| match url.host_str() {
        Some(host) => format!("the file is on another computer ({host})"),
        None => "not a local file".to_string(),
    })
}

/// An `eov://open?path=…&x=…&y=…&zoom=…` link's slide and view.
/// `path` is percent-encoded (`+` is a space); `x`, `y` and `zoom` take
/// the values of `--goto` and `--zoom`. Unknown parameters are ignored.
pub(crate) fn parse_link(text: &str) -> Result<(PathBuf, ViewRequest), String> {
    if !has_scheme(text, "eov") {
        return Err("not an eov link".to_string());
    }
    let rest = &text[4..];
    let rest = rest.strip_prefix("//").unwrap_or(rest);
    let rest = rest.split('#').next().unwrap_or_default();
    let (action, query) = rest.split_once('?').unwrap_or((rest, ""));
    if !action.trim_end_matches('/').eq_ignore_ascii_case("open") {
        return Err(format!(
            "unknown action {action:?} (links look like eov://open?path=…)"
        ));
    }
    let mut path = None;
    let (mut x, mut y, mut zoom) = (None, None, None);
    for pair in query.split('&').filter(|pair| !pair.is_empty()) {
        let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
        let name = String::from_utf8_lossy(&percent_decode(name)?).into_owned();
        let value = percent_decode(value)?;
        let text = || String::from_utf8_lossy(&value).into_owned();
        match name.as_str() {
            "path" => {
                path = Some(
                    if value.len() > 5 && value[..5].eq_ignore_ascii_case(b"file:") {
                        file_uri_path(&text())?
                    } else {
                        path_from_bytes(value.clone())?
                    },
                )
            }
            "x" => x = Some(parse_coordinate(&text())?),
            "y" => y = Some(parse_coordinate(&text())?),
            "zoom" => zoom = Some(parse_zoom(&text())?),
            _ => {}
        }
    }
    let path = path.ok_or_else(|| "the link names no slide (path=…)".to_string())?;
    if !path.is_absolute() {
        return Err(format!(
            "the slide's path must be absolute, not {}",
            path.display()
        ));
    }
    let center = match (x, y) {
        (Some(x), Some(y)) => {
            if std::mem::discriminant(&x) != std::mem::discriminant(&y) {
                return Err("give x and y in the same unit".to_string());
            }
            Some(Point { x, y })
        }
        (None, None) => None,
        _ => return Err("the link gives only one of x and y".to_string()),
    };
    Ok((path, ViewRequest { center, zoom }))
}

/// Why a link's path is refused before eov touches it, if it is.
///
/// On Windows, merely looking at `\\server\share\…` makes Windows sign in
/// to that server with the user's credentials, so a link on a web page
/// could leak them. Links to network paths are therefore only followed for
/// servers listed in `trusted_link_servers` in `config.toml`; mapped drive
/// letters work as they are.
#[cfg_attr(not(windows), allow(unused_variables))]
pub(crate) fn link_path_refusal(path: &Path, trusted_servers: &[String]) -> Option<String> {
    #[cfg(windows)]
    {
        use std::path::{Component, Prefix};
        let server = match path.components().next() {
            Some(Component::Prefix(prefix)) => match prefix.kind() {
                Prefix::UNC(server, _) | Prefix::VerbatimUNC(server, _) => {
                    Some(server.to_string_lossy().into_owned())
                }
                Prefix::DeviceNS(_) => return Some("links cannot open devices".to_string()),
                _ => None,
            },
            _ => None,
        };
        if let Some(server) = server
            && !trusted_servers
                .iter()
                .any(|trusted| trusted.eq_ignore_ascii_case(&server))
        {
            return Some(format!(
                "links to network folders are only followed for servers listed in \
                 trusted_link_servers in config.toml ({server} is not)"
            ));
        }
    }
    None
}

// -- command-line arguments ----------------------------------------------------

/// Splits `text` at commas, trimming spaces around each part.
fn split_commas(text: &OsStr) -> Vec<OsString> {
    text.as_encoded_bytes()
        .split(|&byte| byte == b',')
        .map(|part| part.trim_ascii())
        .filter(|part| !part.is_empty())
        // SAFETY: the parts are split at, and trimmed of, ASCII characters
        // of bytes from `as_encoded_bytes`, which the documentation of
        // `OsStr::from_encoded_bytes_unchecked` allows.
        .map(|part| unsafe { OsStr::from_encoded_bytes_unchecked(part) }.to_os_string())
        .collect()
}

fn absolute(path: &Path, cwd: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

/// One argument (a path, a `file://` URI or an `eov://` link) as paths to
/// open, plus the view of a link.
struct Classified {
    paths: Vec<PathBuf>,
    link_view: Option<ViewRequest>,
}

fn classify(
    argument: &OsStr,
    cwd: &Path,
    lookup: &dyn Fn(&Path) -> Option<Found>,
    trusted_servers: &[String],
    problems: &mut Vec<Problem>,
) -> Classified {
    let shown = argument.to_string_lossy().into_owned();
    let mut classified = Classified {
        paths: Vec::new(),
        link_view: None,
    };
    if let Some(text) = argument.to_str() {
        if has_scheme(text, "eov") {
            match parse_link(text) {
                Ok((path, view)) => match link_path_refusal(&path, trusted_servers) {
                    Some(reason) => problems.push(Problem::new(shown, reason)),
                    None => {
                        classified.paths.push(path);
                        classified.link_view = Some(view);
                    }
                },
                Err(reason) => problems.push(Problem::new(shown, reason)),
            }
            return classified;
        }
        if has_scheme(text, "file") {
            match file_uri_path(text) {
                Ok(path) => classified.paths.push(path),
                Err(reason) => problems.push(Problem::new(shown, reason)),
            }
            return classified;
        }
    }
    let literal = absolute(Path::new(argument), cwd);
    if lookup(&literal).is_some() {
        classified.paths.push(literal);
        return classified;
    }
    // Quoted `~/…` and `--pane=~/…` reach eov unexpanded.
    let expanded = absolute(&crate::paths::expand_tilde(Path::new(argument)), cwd);
    if expanded != literal && lookup(&expanded).is_some() {
        classified.paths.push(expanded);
        return classified;
    }
    let parts = split_commas(argument);
    if parts.len() > 1 || (parts.len() == 1 && parts[0].as_os_str() != argument) {
        for part in parts {
            let shown = part.to_string_lossy().into_owned();
            if part.to_str().is_some_and(|text| has_scheme(text, "file")) {
                match file_uri_path(part.to_str().unwrap_or_default()) {
                    Ok(path) => classified.paths.push(path),
                    Err(reason) => problems.push(Problem::new(shown, reason)),
                }
                continue;
            }
            let path = absolute(&crate::paths::expand_tilde(Path::new(&part)), cwd);
            if lookup(&path).is_some() {
                classified.paths.push(path);
            } else {
                problems.push(Problem::new(shown, "no such file or folder"));
            }
        }
        return classified;
    }
    problems.push(Problem::new(shown, "no such file or folder"));
    classified
}

/// Builds the request for the command line's `files` (current pane),
/// `panes` (`--pane`) and `view` (`--goto`/`--zoom`, for the first slide).
/// `lookup` says what a path names on disk.
pub(crate) fn request_from_arguments(
    files: &[OsString],
    panes: &[OsString],
    view: ViewRequest,
    cwd: &Path,
    lookup: &dyn Fn(&Path) -> Option<Found>,
    trusted_servers: &[String],
) -> OpenRequest {
    let mut request = OpenRequest::default();
    let mut link_views = Vec::new();
    for argument in files {
        let classified = classify(
            argument,
            cwd,
            lookup,
            trusted_servers,
            &mut request.problems,
        );
        if let (Some(view), Some(path)) = (classified.link_view, classified.paths.first()) {
            link_views.push((path.clone(), view));
        }
        request.tabs.extend(classified.paths);
    }
    for argument in panes {
        let classified = classify(
            argument,
            cwd,
            lookup,
            trusted_servers,
            &mut request.problems,
        );
        if let (Some(view), Some(path)) = (classified.link_view, classified.paths.first()) {
            link_views.push((path.clone(), view));
        }
        if !classified.paths.is_empty() {
            request.panes.push(classified.paths);
        }
    }
    if !view.is_empty() {
        let first_slide = request
            .tabs
            .iter()
            .chain(request.panes.iter().flatten())
            .find(|path| lookup(path) != Some(Found::Folder))
            .cloned();
        match first_slide {
            Some(path) => {
                // The command line's view wins over the link's own.
                let index = link_views.iter().position(|(link, _)| *link == path);
                let view = match index {
                    Some(index) => view.or(link_views.remove(index).1),
                    None => view,
                };
                request.views.push((path, view));
            }
            None => request.problems.push(Problem::new(
                "--goto/--zoom",
                "there is no slide to show at that place",
            )),
        }
    }
    request.views.extend(link_views);
    request
}

// -- views -------------------------------------------------------------------

/// The objective-equivalent magnification of a slide's full resolution:
/// the scanner's objective power, or about 10 µm ÷ pixel size when only
/// the pixel size is known (0.25 µm/px ≈ 40×).
pub(crate) fn native_magnification(objective: Option<f64>, mpp: Option<f64>) -> Option<f64> {
    objective
        .filter(|power| power.is_finite() && *power > 0.0)
        .or_else(|| {
            let mpp = mpp.filter(|mpp| mpp.is_finite() && *mpp > 0.0)?;
            Some(10.0 / mpp)
        })
}

/// What placing a view on a slide needs to know about it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SlideScale {
    /// Level-0 size in pixels.
    pub(crate) width: f64,
    pub(crate) height: f64,
    /// Micrometres per pixel along each axis.
    pub(crate) mpp_x: Option<f64>,
    pub(crate) mpp_y: Option<f64>,
    pub(crate) native_magnification: Option<f64>,
}

impl SlideScale {
    pub(crate) fn of_open_file(file: &crate::state::OpenFile) -> Self {
        let properties = file.wsi.properties();
        let calibration = crate::host_ui::calibration(file);
        let spacing = calibration.spacing();
        Self {
            width: properties.width as f64,
            height: properties.height as f64,
            mpp_x: spacing.map(|(x, _, _)| x),
            mpp_y: spacing.map(|(_, y, _)| y),
            native_magnification: crate::callbacks::native_magnification(file),
        }
    }

    pub(crate) fn of_properties(properties: &eov_common::WsiProperties) -> Self {
        let calibration = crate::interaction::Calibration::new(properties.mpp_x, properties.mpp_y);
        let spacing = calibration.spacing();
        Self {
            width: properties.width as f64,
            height: properties.height as f64,
            mpp_x: spacing.map(|(x, _, _)| x),
            mpp_y: spacing.map(|(_, y, _)| y),
            native_magnification: native_magnification(
                properties.objective_power,
                calibration.average(),
            ),
        }
    }
}

/// A view placed on a slide.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedView {
    pub(crate) center_x: f64,
    pub(crate) center_y: f64,
    pub(crate) zoom: f64,
    /// What could not be applied, for the user.
    pub(crate) warnings: Vec<String>,
}

fn describe_zoom(zoom: Zoom) -> String {
    match zoom {
        Zoom::Fit => "fit".to_string(),
        Zoom::Magnification(value) => format!("{}x", eov_common::format_decimal(value)),
        Zoom::Factor(value) => eov_common::format_decimal(value),
    }
}

/// Places `request` on a slide shown at `current` (centre x, centre y,
/// zoom); `fit` is the view that frames the whole slide.
pub(crate) fn resolve_view(
    request: &ViewRequest,
    slide: &SlideScale,
    current: (f64, f64, f64),
    fit: (f64, f64, f64),
) -> ResolvedView {
    let start = if request.zoom == Some(Zoom::Fit) {
        fit
    } else {
        current
    };
    let mut view = ResolvedView {
        center_x: start.0,
        center_y: start.1,
        zoom: start.2,
        warnings: Vec::new(),
    };
    if let Some(point) = request.center {
        let center = match (point.x, point.y) {
            (Coordinate::Px(x), Coordinate::Px(y)) => Some((x, y)),
            (Coordinate::Um(x), Coordinate::Um(y)) => match (slide.mpp_x, slide.mpp_y) {
                (Some(mpp_x), Some(mpp_y)) => Some((x / mpp_x, y / mpp_y)),
                _ => {
                    view.warnings.push(
                        "The slide records no pixel size, so a position in micrometres \
                         cannot be found."
                            .to_string(),
                    );
                    None
                }
            },
            _ => None,
        };
        if let Some((x, y)) = center {
            if x > slide.width || y > slide.height {
                view.warnings.push(format!(
                    "{}, {} is outside the slide ({} × {} pixels).",
                    eov_common::format_decimal(x.round()),
                    eov_common::format_decimal(y.round()),
                    eov_common::format_u64(slide.width as u64),
                    eov_common::format_u64(slide.height as u64),
                ));
            }
            view.center_x = x;
            view.center_y = y;
        }
    }
    match request.zoom {
        None | Some(Zoom::Fit) => {}
        Some(Zoom::Factor(factor)) => view.zoom = factor,
        Some(Zoom::Magnification(target)) => match slide.native_magnification {
            Some(native) => view.zoom = target / native,
            None => view.warnings.push(format!(
                "The slide records neither its objective nor its pixel size, so it cannot \
                 be shown at {}.",
                describe_zoom(Zoom::Magnification(target))
            )),
        },
    }
    view
}

// -- opening (UI thread) -----------------------------------------------------------

/// Tells the user which arguments could not be used.
pub(crate) fn report_problems(problems: &[Problem]) {
    let (title, detail) = match problems {
        [] => return,
        [problem] => (
            format!("Could not open {}", problem.subject),
            problem.reason.clone(),
        ),
        _ => (
            format!("Could not open {} of the given files", problems.len()),
            problems
                .iter()
                .map(Problem::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        ),
    };
    crate::host_ui::with_host(|host| {
        host.notify(
            eov_plugin_api::protocol::NotifyLevel::Warning,
            &title,
            Some(&detail),
            None,
        )
    });
}

/// Opens one pane's files, in order, the first one selected. Returns the
/// pane that has the focus once the first slide was handed to the opener
/// (its own pane, or the one where the slide was already open).
#[allow(clippy::too_many_arguments)]
fn open_group(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    pane_uid: Option<u32>,
    files: &[PathBuf],
    batch: u64,
    focus: bool,
    new_pane: bool,
) -> Option<u32> {
    let mut position = 0;
    let mut focused = None;
    for path in files {
        let pane = {
            let mut state = state.write();
            let pane = pane_uid
                .and_then(|uid| state.pane_by_uid(uid))
                .unwrap_or(state.focused_pane);
            state.set_focused_pane(pane);
            pane
        };
        if path.is_dir() {
            crate::callbacks::open_folder_as_series(ui, state, path.clone());
            continue;
        }
        let select = position == 0;
        let placement = crate::file_ops::Placement {
            batch,
            position,
            select,
            focus: focus && select,
        };
        position += 1;
        if new_pane {
            // A slide that is open elsewhere is shown in the new pane too.
            let existing = state.read().find_tab_by_path(path);
            if let Some((source, tab_id)) = existing
                && source != pane
            {
                let mut state = state.write();
                let duplicated = state.duplicate_tab_to_pane(tab_id, pane);
                if select {
                    state.show_tab_in_pane(pane, duplicated);
                }
                continue;
            }
        }
        crate::open_file_placed(ui, state, tile_cache, render_timer, path.clone(), placement);
        if placement.focus {
            let state = state.read();
            focused = state.pane_uid(state.focused_pane);
        }
    }
    focused
}

/// Opens `request` in the window (UI thread, no lock held).
pub(crate) fn open(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    request: OpenRequest,
) {
    let OpenRequest {
        mut tabs,
        mut panes,
        views,
        problems,
    } = request;
    panes.retain(|pane| !pane.is_empty());
    // With nothing open yet, `--pane a --pane b` means two panes, not an
    // empty one and two more.
    if tabs.is_empty() && !panes.is_empty() && state.read().open_files.is_empty() {
        tabs = panes.remove(0);
    }
    for (path, view) in views {
        crate::file_ops::request_view(&path, view);
    }

    let (current_uid, added) = {
        let mut state = state.write();
        let current = if state.split_enabled {
            state.focused_pane
        } else {
            PaneId::PRIMARY
        };
        let current_uid = state.pane_uid(current);
        let mut added = Vec::new();
        for _ in &panes {
            let index = state.panes.len();
            let new_pane = state.insert_pane(index);
            added.push((state.pane_uid(new_pane), state.pane_uid(PaneId(index - 1))));
        }
        (current_uid, added)
    };
    for (new_uid, source_uid) in &added {
        crate::insert_pane_ui_state(*new_uid, *source_uid);
    }

    let batch = crate::file_ops::new_batch();
    let focused_uid = open_group(
        ui,
        state,
        tile_cache,
        render_timer,
        current_uid,
        &tabs,
        batch,
        true,
        false,
    )
    .or(current_uid);
    for ((uid, _), files) in added.iter().zip(&panes) {
        open_group(
            ui,
            state,
            tile_cache,
            render_timer,
            *uid,
            files,
            batch,
            false,
            true,
        );
    }

    {
        let mut state = state.write();
        if let Some(pane) = focused_uid.and_then(|uid| state.pane_by_uid(uid)) {
            state.set_focused_pane(pane);
        }
    }
    {
        let state = state.read();
        ui.set_split_enabled(state.split_enabled);
        ui.set_focused_pane(state.focused_pane.as_index());
        crate::update_tabs(ui, &state);
    }
    report_problems(&problems);
    crate::request_render_loop(render_timer, &ui.as_weak(), state, tile_cache);
}

/// Opens a request from outside this process (another eov, or macOS) and
/// brings the window to the front (UI thread, no lock held).
pub(crate) fn open_handed_over(request: OpenRequest) -> Result<(), String> {
    crate::host_ui::with_host(|host| {
        let ui = host.ui().ok_or("the window is closed")?;
        open(
            &ui,
            &host.state,
            &host.tile_cache,
            &host.render_timer,
            request,
        );
        raise_window(&ui);
        Ok(())
    })
    .unwrap_or_else(|| Err("eov is still starting".to_string()))
}

/// Brings the window to the front after another eov handed it files.
/// Wayland lets no program raise itself; there the window asks for
/// attention instead.
pub(crate) fn raise_window(ui: &AppWindow) {
    use slint::winit_030::WinitWindowAccessor;
    use slint::winit_030::winit::window::UserAttentionType;
    ui.window().with_winit_window(|window| {
        if is_wayland(window) {
            window.request_user_attention(Some(UserAttentionType::Informational));
        } else {
            window.set_minimized(false);
            window.focus_window();
        }
    });
}

#[cfg(target_os = "linux")]
fn is_wayland(window: &slint::winit_030::winit::window::Window) -> bool {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    window
        .window_handle()
        .is_ok_and(|handle| matches!(handle.as_raw(), RawWindowHandle::Wayland(_)))
}

#[cfg(not(target_os = "linux"))]
fn is_wayland(_window: &slint::winit_030::winit::window::Window) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The folder the tests run "in".
    fn cwd() -> PathBuf {
        if cfg!(windows) {
            PathBuf::from(r"C:\cases")
        } else {
            PathBuf::from("/cases")
        }
    }

    /// An absolute path below [`cwd`].
    fn at(relative: &str) -> PathBuf {
        cwd().join(relative)
    }

    /// Percent-encodes everything but letters and digits.
    fn encode(text: &str) -> String {
        text.bytes()
            .map(|byte| {
                if byte.is_ascii_alphanumeric() {
                    (byte as char).to_string()
                } else {
                    format!("%{byte:02X}")
                }
            })
            .collect()
    }

    struct Disk(HashMap<PathBuf, Found>);

    impl Disk {
        fn new(entries: &[(&str, Found)]) -> Self {
            Self(
                entries
                    .iter()
                    .map(|(relative, found)| (at(relative), *found))
                    .collect(),
            )
        }

        fn lookup(&self) -> impl Fn(&Path) -> Option<Found> + '_ {
            |path| self.0.get(path).copied()
        }
    }

    fn os(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    fn request(disk: &Disk, files: &[&str], panes: &[&str], view: ViewRequest) -> OpenRequest {
        request_from_arguments(&os(files), &os(panes), view, &cwd(), &disk.lookup(), &[])
    }

    #[test]
    fn extra_files_are_tabs_of_one_pane_in_order() {
        let disk = Disk::new(&[
            ("a.svs", Found::File),
            ("b.svs", Found::File),
            ("c.svs", Found::File),
        ]);
        let request = request(
            &disk,
            &["a.svs", "b.svs", "c.svs"],
            &[],
            ViewRequest::default(),
        );
        assert_eq!(request.tabs, [at("a.svs"), at("b.svs"), at("c.svs")]);
        assert!(request.panes.is_empty());
        assert!(request.problems.is_empty());
    }

    #[test]
    fn existing_names_with_commas_are_taken_literally() {
        let disk = Disk::new(&[
            ("Case 12, H&E.svs", Found::File),
            ("a.svs", Found::File),
            ("b.svs", Found::File),
        ]);
        let request = request(
            &disk,
            &["Case 12, H&E.svs", "a.svs, b.svs"],
            &[],
            ViewRequest::default(),
        );
        assert_eq!(
            request.tabs,
            [at("Case 12, H&E.svs"), at("a.svs"), at("b.svs")]
        );
    }

    #[test]
    fn missing_files_are_problems_not_errors() {
        let disk = Disk::new(&[("a.svs", Found::File)]);
        let request = request(
            &disk,
            &["gone.svs", "a.svs", "a.svs,missing.svs"],
            &[],
            ViewRequest::default(),
        );
        assert_eq!(request.tabs, [at("a.svs"), at("a.svs")]);
        let subjects: Vec<_> = request
            .problems
            .iter()
            .map(|problem| problem.subject.as_str())
            .collect();
        assert_eq!(subjects, ["gone.svs", "missing.svs"]);
    }

    #[cfg(unix)]
    #[test]
    fn file_uris_from_file_managers_are_decoded() {
        let disk = Disk::new(&[]);
        let request = request(
            &disk,
            &[
                "file:///slides/Case%2012,%20H&E.svs",
                "FILE:///slides/caf%C3%A9.ndpi",
                "file://otherhost/share/x.svs",
            ],
            &[],
            ViewRequest::default(),
        );
        assert_eq!(
            request.tabs,
            ["/slides/Case 12, H&E.svs", "/slides/café.ndpi"].map(PathBuf::from)
        );
        assert_eq!(request.problems.len(), 1);
        assert!(request.problems[0].reason.contains("otherhost"));
    }

    #[cfg(windows)]
    #[test]
    fn file_uris_from_file_managers_are_decoded() {
        let disk = Disk::new(&[]);
        let request = request(
            &disk,
            &[
                "file:///C:/slides/Case%2012,%20H&E.svs",
                "file://fileserver/share/x.svs",
            ],
            &[],
            ViewRequest::default(),
        );
        assert_eq!(
            request.tabs,
            [r"C:\slides\Case 12, H&E.svs", r"\\fileserver\share\x.svs"].map(PathBuf::from)
        );
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_paths_survive() {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};
        let name = OsString::from_vec(b"/cases/\xff.svs".to_vec());
        let disk = Disk(HashMap::from([(PathBuf::from(&name), Found::File)]));
        let request = request_from_arguments(
            std::slice::from_ref(&name),
            &[],
            ViewRequest::default(),
            &cwd(),
            &disk.lookup(),
            &[],
        );
        assert_eq!(request.tabs[0].as_os_str().as_bytes(), name.as_bytes());

        let uri = request_from_arguments(
            &[OsString::from("file:///cases/%FF.svs")],
            &[],
            ViewRequest::default(),
            &cwd(),
            &disk.lookup(),
            &[],
        );
        assert_eq!(uri.tabs[0].as_os_str().as_bytes(), name.as_bytes());

        let (path, _) = parse_link("eov://open?path=%2Fcases%2F%FF.svs").unwrap();
        assert_eq!(path.as_os_str().as_bytes(), name.as_bytes());
    }

    #[test]
    fn panes_are_explicit() {
        let disk = Disk::new(&[
            ("a.svs", Found::File),
            ("b.svs", Found::File),
            ("c.svs", Found::File),
        ]);
        let request = request(
            &disk,
            &["a.svs"],
            &["b.svs,c.svs", "c.svs"],
            ViewRequest::default(),
        );
        assert_eq!(request.tabs, [at("a.svs")]);
        assert_eq!(
            request.panes,
            [vec![at("b.svs"), at("c.svs")], vec![at("c.svs")]]
        );
    }

    #[test]
    fn the_view_goes_to_the_first_slide() {
        let disk = Disk::new(&[
            ("series", Found::Folder),
            ("a.svs", Found::File),
            ("b.svs", Found::File),
        ]);
        let view = ViewRequest {
            center: Some(parse_point("1200,800").unwrap()),
            zoom: Some(Zoom::Magnification(20.0)),
        };
        let request = request(&disk, &["series", "a.svs", "b.svs"], &[], view);
        assert_eq!(request.views, [(at("a.svs"), view)]);

        let nothing =
            request_from_arguments(&os(&["series"]), &[], view, &cwd(), &disk.lookup(), &[]);
        assert!(nothing.views.is_empty());
        assert_eq!(nothing.problems.len(), 1);
    }

    #[test]
    fn coordinates_and_zooms_parse() {
        assert_eq!(
            parse_point("1200,800").unwrap(),
            Point {
                x: Coordinate::Px(1200.0),
                y: Coordinate::Px(800.0)
            }
        );
        assert_eq!(
            parse_point(" 300.5um , 200 \u{b5}m").unwrap(),
            Point {
                x: Coordinate::Um(300.5),
                y: Coordinate::Um(200.0)
            }
        );
        assert_eq!(parse_point("10px,20").unwrap().y, Coordinate::Px(20.0));
        for bad in ["1200um,800", "1200", "-5,3", "5mm,3mm", "nan,3", "um,1"] {
            assert!(parse_point(bad).is_err(), "{bad}");
        }

        assert_eq!(parse_zoom("20x").unwrap(), Zoom::Magnification(20.0));
        assert_eq!(parse_zoom("2.5X").unwrap(), Zoom::Magnification(2.5));
        assert_eq!(parse_zoom("0.5").unwrap(), Zoom::Factor(0.5));
        assert_eq!(parse_zoom("1:1").unwrap(), Zoom::Factor(1.0));
        assert_eq!(parse_zoom("Fit").unwrap(), Zoom::Fit);
        for bad in ["0", "-2x", "x", "inf", "20y", "1e9"] {
            assert!(parse_zoom(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn links_carry_a_slide_and_a_view() {
        let slide = at("Case 12, H&E.svs");
        let link = format!(
            "eov://open?path={}&x=300um&y=200um&zoom=20x",
            encode(&slide.to_string_lossy())
        );
        let (path, view) = parse_link(&link).unwrap();
        assert_eq!(path, slide);
        assert_eq!(
            view,
            ViewRequest {
                center: Some(Point {
                    x: Coordinate::Um(300.0),
                    y: Coordinate::Um(200.0)
                }),
                zoom: Some(Zoom::Magnification(20.0)),
            }
        );
        // `+` is a space; unknown parameters are ignored; no `//` needed.
        let plain = at("a b.svs");
        let link = format!(
            "EOV:open?path={}&future=1",
            encode(&plain.to_string_lossy()).replace("%20", "+")
        );
        let (path, view) = parse_link(&link).unwrap();
        assert_eq!(path, plain);
        assert!(view.is_empty());

        let absolute = encode(&at("a.svs").to_string_lossy());
        for bad in [
            "eov://open".to_string(),
            "eov://open?path=relative.svs".to_string(),
            format!("eov://delete?path={absolute}"),
            format!("eov://open?path={absolute}&x=1"),
            format!("eov://open?path={absolute}&x=1&y=2um"),
            "eov://open?path=%ZZ".to_string(),
            format!("https://open?path={absolute}"),
        ] {
            assert!(parse_link(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn link_views_merge_with_the_command_line() {
        let disk = Disk::new(&[]);
        let slide = at("a.svs");
        let link = format!(
            "eov://open?path={}&x=10&y=20&zoom=fit",
            encode(&slide.to_string_lossy())
        );
        let request = request(
            &disk,
            &[link.as_str()],
            &[],
            ViewRequest {
                center: None,
                zoom: Some(Zoom::Factor(2.0)),
            },
        );
        assert_eq!(request.tabs, std::slice::from_ref(&slide));
        assert_eq!(
            request.views,
            [(
                slide,
                ViewRequest {
                    center: Some(Point {
                        x: Coordinate::Px(10.0),
                        y: Coordinate::Px(20.0)
                    }),
                    zoom: Some(Zoom::Factor(2.0)),
                }
            )]
        );
    }

    #[cfg(windows)]
    #[test]
    fn links_to_network_paths_need_a_trusted_server() {
        let path = Path::new(r"\\fileserver\slides\a.svs");
        assert!(link_path_refusal(path, &[]).is_some());
        assert!(link_path_refusal(path, &["FILESERVER".to_string()]).is_none());
        assert!(link_path_refusal(Path::new(r"Z:\slides\a.svs"), &[]).is_none());
        assert!(link_path_refusal(Path::new(r"\\.\pipe\x"), &[]).is_some());
        let disk = Disk::new(&[]);
        let request = request(
            &disk,
            &["eov://open?path=%5C%5Cevil%5Cshare%5Ca.svs"],
            &[],
            ViewRequest::default(),
        );
        assert!(request.tabs.is_empty());
        assert_eq!(request.problems.len(), 1);
    }

    #[test]
    fn views_are_placed_with_the_slide_scale() {
        let slide = SlideScale {
            width: 100_000.0,
            height: 80_000.0,
            mpp_x: Some(0.25),
            mpp_y: Some(0.5),
            native_magnification: Some(40.0),
        };
        let fitted = (50_000.0, 40_000.0, 0.01);
        let view = resolve_view(
            &ViewRequest {
                center: Some(Point {
                    x: Coordinate::Um(1000.0),
                    y: Coordinate::Um(1000.0),
                }),
                zoom: Some(Zoom::Magnification(20.0)),
            },
            &slide,
            fitted,
            fitted,
        );
        assert_eq!(
            (view.center_x, view.center_y, view.zoom),
            (4000.0, 2000.0, 0.5)
        );
        assert!(view.warnings.is_empty());

        let unknown = SlideScale {
            mpp_x: None,
            mpp_y: None,
            native_magnification: None,
            ..slide
        };
        let view = resolve_view(
            &ViewRequest {
                center: Some(Point {
                    x: Coordinate::Um(1.0),
                    y: Coordinate::Um(1.0),
                }),
                zoom: Some(Zoom::Magnification(20.0)),
            },
            &unknown,
            fitted,
            fitted,
        );
        assert_eq!((view.center_x, view.center_y, view.zoom), fitted);
        assert_eq!(view.warnings.len(), 2);

        let outside = resolve_view(
            &ViewRequest {
                center: Some(Point {
                    x: Coordinate::Px(200_000.0),
                    y: Coordinate::Px(1.0),
                }),
                zoom: Some(Zoom::Factor(1.0)),
            },
            &slide,
            fitted,
            fitted,
        );
        assert_eq!(outside.zoom, 1.0);
        assert_eq!(outside.warnings.len(), 1);

        // A slide that is already open keeps what the request leaves open,
        // and `fit` frames it.
        let current = (1000.0, 2000.0, 0.5);
        let zoom_only = resolve_view(
            &ViewRequest {
                center: None,
                zoom: Some(Zoom::Factor(2.0)),
            },
            &slide,
            current,
            fitted,
        );
        assert_eq!(
            (zoom_only.center_x, zoom_only.center_y, zoom_only.zoom),
            (1000.0, 2000.0, 2.0)
        );
        let fit = resolve_view(
            &ViewRequest {
                center: None,
                zoom: Some(Zoom::Fit),
            },
            &slide,
            current,
            fitted,
        );
        assert_eq!((fit.center_x, fit.center_y, fit.zoom), fitted);
    }

    #[test]
    fn magnification_comes_from_the_objective_or_the_pixel_size() {
        assert_eq!(native_magnification(Some(40.0), Some(0.5)), Some(40.0));
        assert_eq!(native_magnification(None, Some(0.25)), Some(40.0));
        assert_eq!(native_magnification(Some(0.0), None), None);
    }
}

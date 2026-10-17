//! Plugins' slide sources: the slide formats a plugin
//! declares in its manifest (`[[slide_formats]]`) become readers of
//! [`eov_common::wsi::external`], so `WsiFile::open` opens their files when
//! eov's own readers cannot.
//!
//! Reads come from render workers, several threads at once, with none of
//! the host's locks held; the slide handles are `Send + Sync`. A call that
//! panics fails, marks the plugin's slides as stopped (later reads fail at
//! once) and is reported to the UI thread ([`take_panics`]), which
//! quarantines the plugin. Only files are opened; URI schemes in the
//! manifest are not offered yet.

use abi_stable::std_types::{RSliceMut, RStr};
use eov_common::wsi::SlideBounds;
use eov_common::wsi::external::{
    ExternalLevel, ExternalOpener, ExternalSlide, ExternalSlideInfo, register_opener,
    unregister_opener,
};
use eov_plugin_api::abi::{FfiError, FfiErrorKind, SlideBox, SlideRegion, SlideSourceBox};
use eov_plugin_api::manifest::SlideFormatSpec;
use eov_plugin_api::protocol::{SlideInfo, SlideOpenRequest};
use parking_lot::Mutex;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Panics in slide sources, for the UI thread to quarantine their plugins.
static PANICS: Mutex<Vec<(Arc<str>, String)>> = Mutex::new(Vec::new());

/// The panics reported since the last call: (plugin id, message).
pub fn take_panics() -> Vec<(Arc<str>, String)> {
    std::mem::take(&mut *PANICS.lock())
}

/// Makes the plugin's formats readable through its slide source.
pub fn register(plugin_id: &str, name: &str, formats: &[SlideFormatSpec], source: SlideSourceBox) {
    let opener = PluginOpener {
        plugin_id: Arc::from(plugin_id),
        name: name.to_string(),
        formats: formats.to_vec(),
        source: Arc::new(source),
        stopped: Arc::new(AtomicBool::new(false)),
    };
    let schemes: Vec<String> = formats
        .iter()
        .flat_map(|f| f.normalized_schemes())
        .collect();
    if !schemes.is_empty() {
        tracing::info!(
            "Plugin '{plugin_id}' declares the URI schemes {schemes:?}; eov opens files only for now"
        );
    }
    register_opener(plugin_id, Arc::new(opener));
}

/// The plugin stopped: no new slides open through it (open ones fail their
/// reads from now on, see [`PluginOpener::stopped`]).
pub fn unregister(plugin_id: &str) {
    unregister_opener(plugin_id);
    STOPPED.lock().push(Arc::from(plugin_id));
}

/// Plugins stopped since they opened slides (their slides fail fast).
static STOPPED: Mutex<Vec<Arc<str>>> = Mutex::new(Vec::new());

fn is_stopped(plugin_id: &str) -> bool {
    STOPPED.lock().iter().any(|id| &**id == plugin_id)
}

/// A plugin started again (enabled after being disabled).
pub fn restarted(plugin_id: &str) {
    STOPPED.lock().retain(|id| &**id != plugin_id);
}

struct PluginOpener {
    plugin_id: Arc<str>,
    name: String,
    formats: Vec<SlideFormatSpec>,
    source: Arc<SlideSourceBox>,
    /// Set when a call panicked: the source must not be called again.
    stopped: Arc<AtomicBool>,
}

/// Turns an FFI error into a message, reporting panics.
fn failed(plugin_id: &Arc<str>, stopped: &AtomicBool, error: FfiError) -> String {
    let message = error.message.to_string();
    if error.kind() == FfiErrorKind::Panicked {
        stopped.store(true, Ordering::Relaxed);
        PANICS.lock().push((Arc::clone(plugin_id), message.clone()));
    }
    message
}

impl ExternalOpener for PluginOpener {
    fn reader_name(&self) -> String {
        format!("the plugin '{}'", self.name)
    }

    fn extensions(&self) -> Vec<String> {
        self.formats
            .iter()
            .flat_map(SlideFormatSpec::normalized_extensions)
            .collect()
    }

    fn open(&self, path: &Path) -> Result<(ExternalSlideInfo, Box<dyn ExternalSlide>), String> {
        if self.stopped.load(Ordering::Relaxed) || is_stopped(&self.plugin_id) {
            return Err("the plugin was stopped".into());
        }
        let extension = path
            .extension()
            .map(|ext| ext.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        let format = self
            .formats
            .iter()
            .find(|format| format.normalized_extensions().contains(&extension))
            .ok_or_else(|| format!("no format of the plugin reads .{extension} files"))?;
        let location = dunce::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let request = SlideOpenRequest {
            format_id: format.id.clone(),
            location: location.to_string_lossy().into_owned(),
            is_uri: false,
        };
        let json = serde_json::to_string(&request).map_err(|e| e.to_string())?;
        let slide = self
            .source
            .open(RStr::from_str(&json))
            .into_result()
            .map_err(|error| failed(&self.plugin_id, &self.stopped, error))?;
        let info_json = slide
            .info()
            .into_result()
            .map_err(|error| failed(&self.plugin_id, &self.stopped, error))?;
        let info: SlideInfo = serde_json::from_str(info_json.as_str())
            .map_err(|error| format!("its description of the slide is not valid: {error}"))?;
        info.validate()?;
        Ok((
            external_info(&info, &format.id),
            Box::new(PluginSlide {
                plugin_id: Arc::clone(&self.plugin_id),
                slide,
                stopped: Arc::clone(&self.stopped),
            }),
        ))
    }
}

/// The checked [`SlideInfo`] as eov_common takes it.
fn external_info(info: &SlideInfo, format_id: &str) -> ExternalSlideInfo {
    let base = info.levels[0];
    let positive = |v: Option<f64>| v.filter(|v| v.is_finite() && *v > 0.0);
    let tile = |v: Option<u32>| v.filter(|v| (1..=8192).contains(v));
    let levels = info
        .levels
        .iter()
        .map(|level| {
            let ratio = (base.width as f64 / level.width as f64
                + base.height as f64 / level.height as f64)
                / 2.0;
            ExternalLevel {
                width: level.width,
                height: level.height,
                downsample: level.downsample.unwrap_or(ratio),
                tile_width: tile(level.tile_width),
                tile_height: tile(level.tile_height),
            }
        })
        .collect();
    let bounds = info.bounds.and_then(|rect| {
        let (x, y) = (rect.x.max(0.0), rect.y.max(0.0));
        let width = rect.width.min(base.width as f64 - x);
        let height = rect.height.min(base.height as f64 - y);
        (rect.x.is_finite() && rect.y.is_finite() && width >= 1.0 && height >= 1.0).then_some(
            SlideBounds {
                x: x as i64,
                y: y as i64,
                width: width as u64,
                height: height as u64,
            },
        )
    });
    ExternalSlideInfo {
        levels,
        vendor: info.vendor.clone().unwrap_or_else(|| format_id.to_string()),
        mpp_x: positive(info.mpp_x),
        mpp_y: positive(info.mpp_y),
        objective_power: positive(info.objective_power).filter(|p| *p <= 1000.0),
        scan_date: info.scan_date.clone(),
        comment: info.comment.clone(),
        stain: info.stain.clone(),
        background: info.background.unwrap_or([255, 255, 255]),
        bounds,
        properties: info
            .properties
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        associated_images: info
            .associated_images
            .iter()
            .map(|image| (image.name.clone(), image.width, image.height))
            .collect(),
    }
}

struct PluginSlide {
    plugin_id: Arc<str>,
    slide: SlideBox,
    stopped: Arc<AtomicBool>,
}

impl PluginSlide {
    fn check(&self) -> Result<(), String> {
        if self.stopped.load(Ordering::Relaxed) || is_stopped(&self.plugin_id) {
            Err("the plugin that reads this slide was stopped".into())
        } else {
            Ok(())
        }
    }
}

impl ExternalSlide for PluginSlide {
    fn read_region(
        &self,
        level: u32,
        x: u64,
        y: u64,
        width: u32,
        height: u32,
        out: &mut [u8],
    ) -> Result<(), String> {
        self.check()?;
        self.slide
            .read_region(
                SlideRegion {
                    level,
                    x,
                    y,
                    width,
                    height,
                },
                RSliceMut::from_mut_slice(out),
            )
            .into_result()
            .map_err(|error| failed(&self.plugin_id, &self.stopped, error))
    }

    fn read_associated_image(&self, name: &str) -> Result<(u32, u32, Vec<u8>), String> {
        self.check()?;
        let pixels = self
            .slide
            .read_associated_image(RStr::from_str(name))
            .into_result()
            .map_err(|error| failed(&self.plugin_id, &self.stopped, error))?;
        Ok((pixels.width, pixels.height, pixels.data.into_vec()))
    }
}

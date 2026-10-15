//! `eov probe`: everything eov knows about a slide, as text or JSON
//! (SH-17): size, pyramid levels, pixel size, objective, stain, scanned
//! bounds and regions, ICC profile, quickhash, associated images, the
//! channels of multichannel slides and every property.

use anyhow::Result;
use eov_common::{SlideBounds, WsiFile, format_decimal, format_optional_decimal};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Serialize)]
struct Report {
    path: String,
    filename: String,
    vendor: Option<String>,
    width: u64,
    height: u64,
    /// Micrometres per level-0 pixel.
    mpp_x: Option<f64>,
    mpp_y: Option<f64>,
    objective_power: Option<f64>,
    scan_date: Option<String>,
    stain: Option<String>,
    comment: Option<String>,
    /// `openslide.quickhash-1`.
    quickhash: Option<String>,
    /// The non-empty region (`openslide.bounds-*`).
    bounds: Option<Bounds>,
    /// Scanned regions (`openslide.region[i].*`).
    regions: Vec<Bounds>,
    /// `#rrggbb`.
    background_color: String,
    /// Size of the slide's ICC profile in bytes.
    icc_profile_size: Option<u64>,
    tile_size: u32,
    levels: Vec<Level>,
    associated_images: Vec<AssociatedImage>,
    /// The channels of a multichannel slide (fluorescence, multiplex), as
    /// eov composites them on open; `None` for colour slides.
    channels: Option<Channels>,
    properties: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct Channels {
    /// The stored sample type (`uint8`, `uint16`, `float32`, ...).
    sample_type: String,
    /// Bits actually used, when the file says (e.g. 12 of 16).
    significant_bits: Option<u32>,
    /// Focal planes and time points in the file; eov shows the first.
    size_z: u64,
    size_t: u64,
    channels: Vec<Channel>,
}

#[derive(Debug, Serialize)]
struct Channel {
    index: usize,
    name: String,
    fluor: Option<String>,
    emission_nm: Option<f64>,
    excitation_nm: Option<f64>,
    /// `#rrggbb`, the colour the channel is drawn in.
    color: String,
    visible: bool,
    /// The display window: `min` is drawn black, `max` in full colour.
    min: f64,
    max: f64,
    gamma: f64,
    /// The smallest and largest values of a low-resolution level.
    value_min: f64,
    value_max: f64,
}

fn channels(wsi: &WsiFile) -> Option<Channels> {
    let set = wsi.channels()?;
    let display = wsi.channel_display()?;
    Some(Channels {
        sample_type: set.sample_type.name().to_string(),
        significant_bits: set.significant_bits,
        size_z: set.size_z,
        size_t: set.size_t,
        channels: set
            .channels
            .iter()
            .zip(&display.channels)
            .map(|(info, settings)| {
                let [r, g, b] = settings.color;
                Channel {
                    index: info.index,
                    name: info.name.clone(),
                    fluor: info.fluor.clone(),
                    emission_nm: info.emission_nm,
                    excitation_nm: info.excitation_nm,
                    color: format!("#{r:02x}{g:02x}{b:02x}"),
                    visible: settings.visible,
                    min: settings.min,
                    max: settings.max,
                    gamma: settings.gamma,
                    value_min: info.stats.min,
                    value_max: info.stats.max,
                }
            })
            .collect(),
    })
}

#[derive(Debug, Serialize)]
struct Bounds {
    x: i64,
    y: i64,
    width: u64,
    height: u64,
}

impl From<&SlideBounds> for Bounds {
    fn from(bounds: &SlideBounds) -> Self {
        Self {
            x: bounds.x,
            y: bounds.y,
            width: bounds.width,
            height: bounds.height,
        }
    }
}

#[derive(Debug, Serialize)]
struct Level {
    level: u32,
    width: u64,
    height: u64,
    downsample: f64,
    tile_width: Option<u32>,
    tile_height: Option<u32>,
    /// The tile size eov reads this level in.
    preferred_tile_size: u32,
}

#[derive(Debug, Serialize)]
struct AssociatedImage {
    name: String,
    width: u64,
    height: u64,
    icc_profile_size: Option<u64>,
}

fn report(wsi: &WsiFile) -> Report {
    let properties = wsi.properties();
    let [r, g, b] = properties.background_color;
    Report {
        path: properties.path.display().to_string(),
        filename: properties.filename.clone(),
        vendor: properties.vendor.clone(),
        width: properties.width,
        height: properties.height,
        mpp_x: properties.mpp_x,
        mpp_y: properties.mpp_y,
        objective_power: properties.objective_power,
        scan_date: properties.scan_date.clone(),
        stain: properties.stain.clone(),
        comment: properties.comment.clone(),
        quickhash: properties.quickhash1.clone(),
        bounds: properties.bounds.as_ref().map(Bounds::from),
        regions: properties.regions.iter().map(Bounds::from).collect(),
        background_color: format!("#{r:02x}{g:02x}{b:02x}"),
        icc_profile_size: properties.icc_profile_size,
        tile_size: wsi.tile_size(),
        levels: properties
            .levels
            .iter()
            .map(|level| Level {
                level: level.level,
                width: level.width,
                height: level.height,
                downsample: level.downsample,
                tile_width: level.tile_width,
                tile_height: level.tile_height,
                preferred_tile_size: wsi.tile_size_for_level(level.level),
            })
            .collect(),
        associated_images: wsi
            .associated_images()
            .iter()
            .map(|image| AssociatedImage {
                name: image.name.clone(),
                width: image.width,
                height: image.height,
                icc_profile_size: image.icc_profile_size,
            })
            .collect(),
        channels: channels(wsi),
        properties: wsi.property_map().clone(),
    }
}

fn or_unknown(value: Option<&str>) -> &str {
    value.unwrap_or("unknown")
}

fn bounds_text(bounds: &Bounds) -> String {
    format!(
        "{}x{} at {},{}",
        bounds.width, bounds.height, bounds.x, bounds.y
    )
}

fn print_plain(report: &Report) {
    println!("File: {}", report.path);
    println!("Filename: {}", report.filename);
    println!("Dimensions: {}x{}", report.width, report.height);
    println!("Levels: {}", report.levels.len());
    println!("Vendor: {}", or_unknown(report.vendor.as_deref()));
    println!(
        "MPP: {} x {}",
        format_optional_decimal(report.mpp_x),
        format_optional_decimal(report.mpp_y)
    );
    println!(
        "Objective: {}",
        format_optional_decimal(report.objective_power)
    );
    println!("Scan date: {}", or_unknown(report.scan_date.as_deref()));
    println!("Stain: {}", or_unknown(report.stain.as_deref()));
    println!(
        "Bounds: {}",
        report
            .bounds
            .as_ref()
            .map(bounds_text)
            .unwrap_or_else(|| "the whole slide".to_string())
    );
    for (index, region) in report.regions.iter().enumerate() {
        println!("Region {index}: {}", bounds_text(region));
    }
    println!("Background: {}", report.background_color);
    println!(
        "ICC profile: {}",
        report
            .icc_profile_size
            .map(|size| format!("{size} bytes"))
            .unwrap_or_else(|| "none".to_string())
    );
    println!("Quickhash: {}", or_unknown(report.quickhash.as_deref()));
    println!("Base tile size: {}", report.tile_size);
    println!();
    println!("Levels:");
    for level in &report.levels {
        let tile = |value: Option<u32>| {
            value
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        };
        println!(
            "  L{}: {}x{} downsample {:.2} tile {}x{} preferred-tile {}",
            level.level,
            level.width,
            level.height,
            level.downsample,
            tile(level.tile_width),
            tile(level.tile_height),
            level.preferred_tile_size
        );
    }
    if let Some(channels) = &report.channels {
        println!();
        println!(
            "Channels: {} ({})",
            channels.channels.len(),
            channels.sample_type
        );
        for channel in &channels.channels {
            println!(
                "  C{}: {} colour {} window {}-{} values {}-{}{}",
                channel.index,
                channel.name,
                channel.color,
                format_decimal(channel.min),
                format_decimal(channel.max),
                format_decimal(channel.value_min),
                format_decimal(channel.value_max),
                if channel.visible { "" } else { " (hidden)" }
            );
        }
    }
    println!();
    println!("Associated images:");
    if report.associated_images.is_empty() {
        println!("  none");
    }
    for image in &report.associated_images {
        let icc = image
            .icc_profile_size
            .map(|size| format!(", ICC profile {size} bytes"))
            .unwrap_or_default();
        println!("  {}: {}x{}{icc}", image.name, image.width, image.height);
    }
    println!();
    println!("Properties:");
    for (name, value) in &report.properties {
        println!("  {name} = {value}");
    }
}

/// Runs `eov probe`.
pub(crate) fn run(path: &Path, json: bool) -> Result<()> {
    let wsi = WsiFile::open(path)?;
    let report = report(&wsi);
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_plain(&report);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_carry_every_property_and_associated_image() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.svs");
        let spec = eov_common::test_support::TiffSpec::aperio(512, 512, 256, 4);
        eov_common::test_support::write_tiff(&path, &spec, |_, _| [200, 120, 160]).unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        let json = serde_json::to_value(report(&wsi)).unwrap();
        assert_eq!(json["width"], 512);
        assert_eq!(json["vendor"], "aperio");
        assert_eq!(json["levels"].as_array().unwrap().len(), 2);
        assert_eq!(
            json["properties"].as_object().unwrap().len(),
            wsi.property_map().len()
        );
        let images: Vec<&str> = json["associated_images"]
            .as_array()
            .unwrap()
            .iter()
            .map(|image| image["name"].as_str().unwrap())
            .collect();
        assert!(
            images.contains(&"label") && images.contains(&"macro"),
            "{images:?}"
        );
        assert!(json["quickhash"].is_string());
        assert_eq!(json["background_color"], "#ffffff");
        assert!(json["channels"].is_null());
    }

    #[test]
    fn reports_list_the_channels_of_multichannel_slides() {
        use eov_common::test_support::{OmeTiffSpec, write_ome_tiff};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.ome.tif");
        let spec = OmeTiffSpec::channels(
            64,
            64,
            32,
            2,
            &[("DAPI", [0, 0, 255]), ("CD3", [0, 255, 0])],
        );
        write_ome_tiff(&path, &spec, |c, x, _| f64::from(x) * (c + 1) as f64).unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        let json = serde_json::to_value(report(&wsi)).unwrap();
        let channels = &json["channels"];
        assert_eq!(channels["sample_type"], "uint16");
        assert_eq!(channels["size_z"], 1);
        let list = channels["channels"].as_array().unwrap();
        let names: Vec<&str> = list.iter().map(|c| c["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["DAPI", "CD3"]);
        assert_eq!(list[0]["color"], "#0000ff");
        assert_eq!(list[1]["visible"], true);
        assert_eq!(list[1]["value_max"], 126.0);
        assert!(list[1]["max"].as_f64().unwrap() > list[1]["min"].as_f64().unwrap());
    }
}

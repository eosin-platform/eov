//! The sidebar surface: the properties the presenter pushes and the
//! callbacks the UI sends.
//!
//! Callbacks run on the UI thread; they change state, mark the presenter
//! dirty and start threads, never anything slow (no file writes on
//! keystrokes, VAE-17). Fields that can reject input return the value the
//! UI should show.

use crate::config::max_threads;
use crate::contract::{Scaling, ValueRange};
use crate::engine::{Backend, DevicePreference};
use crate::plugin::{Core, JobKind, JobRequest, ModelState, lock, run_context};
use crate::presenter::{Memo, PULSE};
use crate::render::Mode;
use crate::stats::{decimals_for, format_error};
use crate::tiling::{Resolution, TileCoord, choose_level};
use eov_plugin_api::prelude::*;
use std::sync::Arc;
use std::time::Instant;

/// The note under the overlay selector about the focused pane's display
/// adjustments (VAE-17): reconstructions get the pane's tone curve and
/// sharpening, but stain processing cannot be reproduced. Returns the text
/// and whether it is a warning.
pub fn adjustment_note(
    mode: Mode,
    display: Option<&DisplayAdjustments>,
    unsupported: bool,
) -> (String, bool) {
    match mode {
        Mode::Reconstruction if unsupported => (
            "This eov does not report its display adjustments: reconstructions are shown without them, so reset gamma, contrast and stain normalisation before comparing Original and Reconstruction.".to_string(),
            true,
        ),
        Mode::Reconstruction => match display.filter(|d| d.modified) {
            Some(display) if stain_processing(display) => (
                format!(
                    "Reconstructions get this pane's gamma, brightness, contrast and sharpening, but not its stain processing ({}): turn that off before comparing Original and Reconstruction.",
                    stain_label(display)
                ),
                true,
            ),
            Some(_) => (
                "Reconstructions are shown with this pane's display adjustments, like the slide.".to_string(),
                false,
            ),
            None => (String::new(), false),
        },
        Mode::ErrorMap | Mode::Difference if display.is_some_and(|d| d.modified) => (
            "Errors are measured on the slide's own pixels; display adjustments do not change them.".to_string(),
            false,
        ),
        _ => (String::new(), false),
    }
}

/// Whether the pane alters stain colours (normalisation, deconvolution
/// adjustments, an isolated stain), which eovae cannot reproduce.
fn stain_processing(display: &DisplayAdjustments) -> bool {
    if display.stain_normalization.is_some() {
        return true;
    }
    let badge = display.badge.to_lowercase();
    badge.contains("stain") || badge.contains("colour") || badge.contains("color")
}

fn stain_label(display: &DisplayAdjustments) -> String {
    match &display.stain_normalization {
        Some(method) => {
            let mut chars = method.chars();
            let method = chars
                .next()
                .map(|c| c.to_uppercase().chain(chars).collect::<String>())
                .unwrap_or_default();
            format!("{method} normalisation")
        }
        None => display.badge.clone(),
    }
}

const SCALING_OPTIONS: [&str; 4] = ["[-1, 1]", "[0, 1]", "[0, 255]", "ImageNet mean/std"];

fn scaling_index(scaling: &Scaling) -> i32 {
    match scaling {
        Scaling::Range {
            range: ValueRange::MinusOneOne,
        } => 0,
        Scaling::Range {
            range: ValueRange::ZeroOne,
        } => 1,
        Scaling::Range {
            range: ValueRange::Zero255,
        } => 2,
        Scaling::MeanStd { .. } => 3,
    }
}

fn scaling_at(index: i64) -> Option<Scaling> {
    match index {
        0 => Some(Scaling::range(ValueRange::MinusOneOne)),
        1 => Some(Scaling::range(ValueRange::ZeroOne)),
        2 => Some(Scaling::range(ValueRange::Zero255)),
        3 => Some(Scaling::IMAGENET),
        _ => None,
    }
}

fn mib(bytes: u64) -> String {
    let mb = bytes as f64 / (1024.0 * 1024.0);
    if mb >= 1024.0 {
        format!("{:.1} GB", mb / 1024.0)
    } else {
        format!("{mb:.0} MB")
    }
}

pub(crate) fn thousands(value: f64) -> String {
    let digits = format!("{:.0}", value.max(0.0));
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Every sidebar property.
pub fn properties<B: Backend>(core: &Arc<Core<B>>, memo: &mut Memo) -> Map<String, Value> {
    let mut p = Map::new();
    let mut set = |name: &str, value: Value| {
        p.insert(name.to_string(), value);
    };
    let state = lock(&core.state);
    let config = state.config.clone();

    // Sections.
    set("model-expanded", json!(config.sections.model));
    set("analysis-expanded", json!(config.sections.analysis));
    set("results-expanded", json!(config.sections.results));
    set("cache-expanded", json!(config.sections.cache));

    // Model.
    let model = state.model_info().cloned();
    let (model_state, model_name, model_message) = match &state.model {
        ModelState::None => (0, String::new(), "No model loaded.".to_string()),
        ModelState::Loading { path } => (
            1,
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            "Reading and hashing the model…".to_string(),
        ),
        ModelState::Ready { info } => (2, info.file_name.clone(), String::new()),
        ModelState::Failed { path, error } => (
            3,
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            error.clone(),
        ),
    };
    set("model-state", json!(model_state));
    set("model-name", json!(model_name));
    let request = state.engine_request();
    let engine_info = request.as_ref().and_then(|request| {
        state
            .engine
            .info
            .as_ref()
            .filter(|(sig, _)| *sig == crate::plugin::results_signature(request))
            .map(|(_, info)| info.clone())
    });
    let engine_error = request.as_ref().and_then(|request| {
        state
            .engine
            .error
            .as_ref()
            .filter(|(sig, _)| *sig == request.signature())
            .map(|(_, e)| e.clone())
    });
    let message = match (&state.engine.building, &engine_error) {
        (Some(building), _) if model_state == 2 => building.clone(),
        (_, Some(error)) if model_state == 2 => error.clone(),
        _ => model_message,
    };
    set("model-message", json!(message));
    set(
        "model-message-error",
        json!(
            model_state == 3
                || (model_state == 2 && engine_error.is_some() && state.engine.building.is_none())
        ),
    );
    set(
        "engine-label",
        json!(
            engine_info
                .as_ref()
                .map(|i| i.label.clone())
                .unwrap_or_default()
        ),
    );
    set(
        "engine-note",
        json!(
            engine_info
                .as_ref()
                .and_then(|i| i.fallback.clone())
                .map(|f| format!("CUDA unavailable: {f}"))
                .unwrap_or_default()
        ),
    );
    set("runtime-label", json!(core.backend.runtime_label()));
    match &model {
        Some(info) => {
            let mut detail = vec![
                format!("{} px", info.tile_size),
                info.layout.label().to_string(),
            ];
            detail.push(match info.fixed_batch {
                Some(batch) => format!("batch {batch}"),
                None => "dynamic batch".to_string(),
            });
            if let Some(mpp) = info.contract.mpp {
                detail.push(format!("{mpp:.3} µm/px"));
            }
            detail.push(format!("{} MB", (info.bytes as f64 / 1e6).round()));
            set(
                "model-detail",
                json!(match info.card {
                    Some(card) => format!("{card} · {}", detail.join(" · ")),
                    None => detail.join(" · "),
                }),
            );
            set(
                "input-text",
                json!(format!(
                    "{} · {} · {}",
                    info.input.name,
                    info.input_elem.label(),
                    info.input.shape_label()
                )),
            );
            let output = config.output_for(info);
            set(
                "output-options",
                json!(
                    info.outputs
                        .iter()
                        .map(|o| o.name.clone())
                        .collect::<Vec<_>>()
                ),
            );
            set(
                "output-index",
                json!(
                    info.outputs
                        .iter()
                        .position(|o| o.name == output)
                        .unwrap_or(0) as i32
                ),
            );
            set(
                "latent-text",
                json!(
                    info.latent
                        .as_ref()
                        .map(|l| format!(
                            "Embeddings from '{}' ({} values) are stored with results.",
                            l.name, l.dim
                        ))
                        .unwrap_or_default()
                ),
            );
            set("warnings", json!(info.warnings));
            let contract = config.contract_for(info);
            let overridden = config
                .overrides
                .get(&info.sha256)
                .is_some_and(|o| !o.is_empty());
            set(
                "contract-source",
                json!(if overridden {
                    format!(
                        "Changed by you (model declares: {})",
                        info.contract_source.label()
                    )
                } else {
                    format!("From {}", info.contract_source.label())
                }),
            );
            set("input-scaling-index", json!(scaling_index(&contract.input)));
            set(
                "output-scaling-index",
                json!(scaling_index(&contract.output)),
            );
            set("composite-white", json!(contract.composite_on_white));
            set("black-to-white", json!(contract.black_to_white_below > 0));
            set("contract-overridden", json!(overridden));
            set("contract-summary", json!(contract.summary()));
        }
        None => {
            set("model-detail", json!(""));
            set("input-text", json!(""));
            set("output-options", json!(Vec::<String>::new()));
            set("output-index", json!(0));
            set("latent-text", json!(""));
            set("warnings", json!(Vec::<String>::new()));
            set("contract-source", json!(""));
            set("input-scaling-index", json!(0));
            set("output-scaling-index", json!(0));
            set("composite-white", json!(true));
            set("black-to-white", json!(true));
            set("contract-overridden", json!(false));
            set("contract-summary", json!(""));
        }
    }
    set("scaling-options", json!(SCALING_OPTIONS));

    // Analysis settings.
    set(
        "device-options",
        json!(
            DevicePreference::ALL
                .iter()
                .map(|d| d.label())
                .collect::<Vec<_>>()
        ),
    );
    set(
        "device-index",
        json!(
            DevicePreference::ALL
                .iter()
                .position(|d| *d == config.device)
                .unwrap_or(0) as i32
        ),
    );
    set(
        "threads-text",
        json!(if config.threads == 0 {
            "auto".to_string()
        } else {
            config.threads.to_string()
        }),
    );
    set(
        "threads-hint",
        json!(format!("1–{}, or auto", max_threads())),
    );
    let fixed_batch = model.as_ref().and_then(|m| m.fixed_batch);
    set(
        "batch-text",
        json!(fixed_batch.unwrap_or(config.batch_size).to_string()),
    );
    set("batch-fixed", json!(fixed_batch.is_some()));
    set("tf32", json!(config.tf32));
    set("skip-background", json!(config.skip_background));
    set("auto-analyze", json!(config.auto_analyze));

    // The active slide.
    let active = state
        .active_file
        .and_then(|id| state.files.get(&id).map(|f| (id, f)));
    let model_mpp = model.as_ref().and_then(|m| config.contract_for(m).mpp);
    let (resolution_options, resolution_index, resolution_note, resolution_warning) = match active {
        Some((_, file)) => {
            let mut options = Vec::new();
            let auto_choice = choose_level(&file.geometry, Resolution::Auto, model_mpp);
            options.push(match (auto_choice, model_mpp) {
                (Some(choice), Some(mpp)) if choice.resample.is_some() => format!(
                    "Auto: the model's {mpp:.3} µm/px, resampled from level {}",
                    choice.level.index
                ),
                (Some(choice), Some(mpp)) => format!(
                    "Auto: the model's {mpp:.3} µm/px = level {}",
                    choice.level.index
                ),
                _ => "Auto (level 0)".to_string(),
            });
            for level in &file.geometry.levels {
                let mpp = file
                    .geometry
                    .level_mpp(level)
                    .map(|m| format!(" · {m:.2} µm/px"))
                    .unwrap_or_default();
                options.push(format!(
                    "Level {} · {}×{mpp}",
                    level.index,
                    if (level.downsample - level.downsample.round()).abs() < 0.01 {
                        format!("{:.0}", level.downsample)
                    } else {
                        format!("{:.3}", level.downsample)
                    }
                ));
            }
            let choice = choose_level(&file.geometry, config.resolution, model_mpp);
            let index = match (config.resolution, choice) {
                (Resolution::Auto, _) => 0,
                (_, Some(choice)) => {
                    1 + file
                        .geometry
                        .levels
                        .iter()
                        .position(|l| l.index == choice.level.index)
                        .unwrap_or(0)
                }
                _ => 0,
            };
            let level_mpp = choice.and_then(|c| c.level_mpp(&file.geometry));
            let (note, warning) = match choice {
                Some(choice) if choice.mismatched() && choice.resample.is_some() => (
                    format!(
                        "The slide's finest level is {:.3} µm/px, coarser than the model's {:.3} µm/px: tiles are interpolated and errors will be inflated.",
                        level_mpp.unwrap_or_default(),
                        model_mpp.unwrap_or_default()
                    ),
                    true,
                ),
                Some(choice) if choice.mismatched() => (
                    format!(
                        "Level {} is {:.2} µm/px but the model was trained at {:.2} µm/px; errors will be inflated.",
                        choice.level.index,
                        choice.mpp.unwrap_or_default(),
                        model_mpp.unwrap_or_default()
                    ),
                    true,
                ),
                Some(choice) if choice.resample.is_some() => (
                    format!(
                        "Tiles are read from level {} ({:.3} µm/px) and resampled to the model's {:.3} µm/px.",
                        choice.level.index,
                        level_mpp.unwrap_or_default(),
                        model_mpp.unwrap_or_default()
                    ),
                    false,
                ),
                Some(choice) => (
                    match (choice.mpp, model_mpp) {
                        (Some(level), Some(trained)) => format!(
                            "Level {} at {level:.2} µm/px (model {trained:.2} µm/px).",
                            choice.level.index
                        ),
                        (Some(level), None) => {
                            format!("Level {} at {level:.2} µm/px.", choice.level.index)
                        }
                        _ => format!(
                            "Level {} (the slide is not calibrated).",
                            choice.level.index
                        ),
                    },
                    false,
                ),
                None => (String::new(), false),
            };
            (options, index, note, warning)
        }
        None => (
            vec!["Auto".to_string()],
            0,
            "Open a slide.".to_string(),
            false,
        ),
    };
    set("resolution-options", json!(resolution_options));
    set("resolution-index", json!(resolution_index as i32));
    set("resolution-note", json!(resolution_note));
    set("resolution-warning", json!(resolution_warning));

    // Job.
    let ctx = active.and_then(|(id, _)| run_context(&state, id).ok());
    let can_analyze = active.is_some() && model_state == 2 && engine_error.is_none();
    set("can-analyze", json!(can_analyze));
    match &state.job {
        Some(job) => {
            set("job-running", json!(true));
            set(
                "job-title",
                json!(match job.request.kind {
                    JobKind::WholeSlide => "Whole slide",
                    JobKind::Viewport => "Viewport",
                    JobKind::Roi => "Region of interest",
                    JobKind::Auto => "Visible tiles",
                }),
            );
            let progress = job.progress;
            set(
                "job-status",
                json!(if progress.total > 0 {
                    format!(
                        "{} of {} tiles · {} background · {} unreadable",
                        progress.done,
                        progress.total,
                        progress.counts.background,
                        progress.counts.failed
                    )
                } else {
                    job.phase.clone()
                }),
            );
            set("job-progress", json!(progress.fraction() as f64));
            let mut detail = vec![job.phase.clone()];
            let eta = crate::plugin::format_eta(&progress);
            if !eta.is_empty() {
                detail.push(eta);
            }
            set("job-detail", json!(detail.join(" · ")));
        }
        None => {
            set("job-running", json!(false));
            set("job-title", json!(""));
            set("job-status", json!(""));
            set("job-progress", json!(0.0));
            set("job-detail", json!(""));
        }
    }
    set(
        "last-result",
        json!(
            state
                .last_job
                .as_ref()
                .map(|s| s.text.clone())
                .unwrap_or_default()
        ),
    );
    set(
        "last-result-error",
        json!(state.last_job.as_ref().is_some_and(|s| s.error)),
    );
    set(
        "auto-hint",
        json!(state.auto_hint.clone().unwrap_or_default()),
    );

    // Results of the active slide, as the focused pane shows it.
    let focused = state.focused_pane;
    let (slide_name, mode) = active
        .map(|(_, f)| {
            (
                f.info.filename.clone(),
                f.overlay(focused, state.pane_layers).mode,
            )
        })
        .unwrap_or_default();
    set("slide-name", json!(slide_name));
    set(
        "view-options",
        json!(Mode::ALL.iter().map(|m| m.label()).collect::<Vec<_>>()),
    );
    set("view-index", json!(mode.index() as i32));
    // Kept current by `RoiChanged` events.
    set(
        "has-roi",
        json!(active.is_some_and(|(_, f)| f.roi.is_some())),
    );
    let keep_pixels = config.keep_pixels;
    let memory_limit = config.memory_limit_bytes();
    let has_active = active.is_some();
    let active_id = active.map(|(id, _)| id);
    drop(state);
    let display = focused.and_then(|pane| memo.display(core, pane));
    let (note, warning) = adjustment_note(mode, display.as_ref(), memo.displays_unsupported);
    set("adjustment-note", json!(note));
    set("adjustment-warning", json!(warning));

    let stats = ctx.as_ref().and_then(|ctx| {
        core.ensure_loaded(ctx);
        memo.stats(core, &ctx.key)
    });
    match (&ctx, &stats) {
        (Some(ctx), Some(stats)) if stats.counts.total() > 0 => {
            let total = ctx.grid.tile_count().max(1);
            set(
                "coverage-text",
                json!(format!(
                    "{} tissue tiles scored, {} background, {} unreadable — {:.1} % of the {} tiles at level {}.",
                    stats.counts.analysed,
                    stats.counts.background,
                    stats.counts.failed,
                    stats.counts.total() as f64 * 100.0 / total as f64,
                    total,
                    ctx.grid.level
                )),
            );
            match &stats.summary {
                Some(summary) => {
                    let decimals =
                        decimals_for((summary.p95 - summary.p05).max(summary.max / 100.0) / 10.0);
                    let f = |v: f64| format_error(v, decimals);
                    set(
                        "stats-text",
                        json!(format!(
                            "Tile MAE: mean {} · median {} · p95 {} · max {}",
                            f(summary.mean),
                            f(summary.median),
                            f(summary.p95),
                            f(summary.max)
                        )),
                    );
                    let peak = stats.bins.iter().map(|b| b.count).max().unwrap_or(0).max(1);
                    let bin_decimals = stats
                        .bins
                        .first()
                        .map(|b| decimals_for(b.hi - b.lo))
                        .unwrap_or(4);
                    set(
                        "histogram",
                        json!(stats
                            .bins
                            .iter()
                            .map(|b| json!({
                                "label": format!("{}–{}: {}", format_error(b.lo, bin_decimals), format_error(b.hi, bin_decimals), b.count),
                                "count": b.count as i32,
                                "normalized": b.count as f64 / peak as f64,
                            }))
                            .collect::<Vec<_>>()),
                    );
                    set("histogram-peak", json!(peak as i32));
                    set(
                        "histogram-lo",
                        json!(
                            stats
                                .bins
                                .first()
                                .map(|b| format_error(b.lo, bin_decimals))
                                .unwrap_or_default()
                        ),
                    );
                    set(
                        "histogram-hi",
                        json!(
                            stats
                                .bins
                                .last()
                                .map(|b| format_error(b.hi, bin_decimals))
                                .unwrap_or_default()
                        ),
                    );
                    set(
                        "hot-regions",
                        json!(stats
                            .top
                            .iter()
                            .map(|(mae, coord)| {
                                let rect = ctx.grid.valid_footprint(*coord);
                                let max = lock(&core.results)
                                    .run(&ctx.key)
                                    .and_then(|run| run.state(*coord))
                                    .and_then(|s| match s {
                                        crate::results::TileState::Analysed { max, .. } => Some(max),
                                        _ => None,
                                    })
                                    .unwrap_or_default();
                                let (step_x, step_y) = (ctx.grid.step_x(), ctx.grid.step_y());
                                let size = if (step_x - step_y).abs() < 0.5 {
                                    format!("{} px", thousands(step_x))
                                } else {
                                    format!("{}×{} px", thousands(step_x), thousands(step_y))
                                };
                                json!({
                                    "id": coord.id(),
                                    "label": format!(
                                        "x {} · y {} · {size}",
                                        thousands(rect.x),
                                        thousands(rect.y),
                                    ),
                                    "score": format!("MAE {} · max {}", format_error(f64::from(*mae), decimals.max(4)), format_error(f64::from(max), 3)),
                                })
                            })
                            .collect::<Vec<_>>()),
                    );
                    let (legend_visible, title, lo, hi) = match mode {
                        Mode::ErrorMap => {
                            // The range the map is drawn with.
                            let (lo, hi) = active_id
                                .and_then(|id| memo.heat_range(id))
                                .unwrap_or_else(|| stats.heat_range());
                            (
                                true,
                                "Tile MAE (p05–p95)",
                                f(f64::from(lo)),
                                f(f64::from(hi)),
                            )
                        }
                        Mode::Difference => (
                            true,
                            "Pixel error",
                            "0".to_string(),
                            format_error(f64::from(stats.pixel_range()), 3),
                        ),
                        _ => (false, "", String::new(), String::new()),
                    };
                    set("legend-visible", json!(legend_visible));
                    set("legend-title", json!(title));
                    set("legend-lo", json!(lo));
                    set("legend-hi", json!(hi));
                }
                None => clear_stats(&mut set),
            }
            set("can-export", json!(true));
        }
        _ => {
            set(
                "coverage-text",
                json!(match &ctx {
                    Some(_) =>
                        "Nothing analysed on this slide with these settings yet.".to_string(),
                    None if !has_active => "Open a slide.".to_string(),
                    None => {
                        let state = lock(&core.state);
                        state
                            .active_file
                            .map(|id| {
                                run_context(&state, id)
                                    .err()
                                    .map(|r| r.message())
                                    .unwrap_or_default()
                            })
                            .unwrap_or_default()
                    }
                }),
            );
            clear_stats(&mut set);
            set("can-export", json!(false));
        }
    }

    // Cache.
    let cache = core.cache.status();
    set(
        "cache-text",
        json!(if cache.enabled {
            format!(
                "{} of {} on disk · {} runs",
                mib(cache.bytes),
                mib(cache.budget),
                cache.runs
            )
        } else {
            "Disk cache disabled".to_string()
        }),
    );
    set("cache-error", json!(cache.error.unwrap_or_default()));
    set("cache-enabled", json!(cache.enabled));
    set(
        "memory-text",
        json!(format!(
            "Tile pixels in memory: {} of {}",
            mib(lock(&core.results).payload_bytes() as u64),
            mib(memory_limit as u64)
        )),
    );
    set("keep-pixels", json!(keep_pixels));
    p
}

fn clear_stats(set: &mut impl FnMut(&str, Value)) {
    set("stats-text", json!(""));
    set("histogram", json!(Vec::<Value>::new()));
    set("histogram-peak", json!(0));
    set("histogram-lo", json!(""));
    set("histogram-hi", json!(""));
    set("hot-regions", json!(Vec::<Value>::new()));
    set("legend-visible", json!(false));
    set("legend-title", json!(""));
    set("legend-lo", json!(""));
    set("legend-hi", json!(""));
}

/// Everything in the settings that changes analysis results.
fn result_config<B: Backend>(
    core: &Arc<Core<B>>,
) -> (
    Option<String>,
    Option<crate::contract::Contract>,
    Resolution,
) {
    let state = lock(&core.state);
    (
        state
            .engine_request()
            .map(|request| crate::plugin::results_signature(&request)),
        state.model_info().map(|m| state.config.contract_for(m)),
        state.config.resolution,
    )
}

fn arg_int(args: &[Value]) -> Option<i64> {
    args.first().and_then(Value::as_i64)
}

fn arg_bool(args: &[Value]) -> Option<bool> {
    args.first().and_then(Value::as_bool)
}

fn arg_str(args: &[Value]) -> Option<&str> {
    args.first().and_then(Value::as_str)
}

/// Handles a sidebar callback.
pub fn callback<B: Backend>(
    core: &Arc<Core<B>>,
    name: &str,
    args: &[Value],
) -> Result<Value, String> {
    let before = result_config(core);
    let mut reply = Value::Null;
    let mut rebuild_engine = false;
    let mut refresh_overlays = false;
    match name {
        "load-model" => {
            let id = core
                .host
                .open_file_dialog(FileDialogArgs {
                    title: Some("Open an ONNX model".into()),
                    filters: vec![FileFilter {
                        name: "ONNX models".into(),
                        extensions: vec!["onnx".into()],
                    }],
                    ..Default::default()
                })
                .map_err(String::from)?;
            lock(&core.state).model_dialog = Some(id);
        }
        "choose-runtime" => {
            // Versioned Linux names (libonnxruntime.so.1.28.3) defeat an
            // extension filter, so Linux lists every file.
            let filters = if cfg!(target_os = "windows") {
                vec![FileFilter {
                    name: "ONNX Runtime".into(),
                    extensions: vec!["dll".into()],
                }]
            } else if cfg!(target_os = "macos") {
                vec![FileFilter {
                    name: "ONNX Runtime".into(),
                    extensions: vec!["dylib".into()],
                }]
            } else {
                Vec::new()
            };
            let id = core
                .host
                .open_file_dialog(FileDialogArgs {
                    title: Some("Choose the ONNX Runtime library".into()),
                    filters,
                    ..Default::default()
                })
                .map_err(String::from)?;
            lock(&core.state).runtime_dialog = Some(id);
        }
        "clear-model" => core.clear_model(),
        "output-selected" => {
            let mut state = lock(&core.state);
            if let (Some(index), Some(model)) = (arg_int(args), state.model_info().cloned())
                && let Some(output) = model.outputs.get(index.max(0) as usize)
            {
                state
                    .config
                    .outputs
                    .insert(model.sha256.clone(), output.name.clone());
                state.touch_config();
                rebuild_engine = true;
            }
            reply = json!(
                state
                    .model_info()
                    .map(|m| {
                        let output = state.config.output_for(m);
                        m.outputs.iter().position(|o| o.name == output).unwrap_or(0) as i32
                    })
                    .unwrap_or(0)
            );
        }
        "input-scaling-selected" | "output-scaling-selected" => {
            let input = name == "input-scaling-selected";
            let mut state = lock(&core.state);
            if let (Some(scaling), Some(model)) = (
                arg_int(args).and_then(scaling_at),
                state.model_info().cloned(),
            ) {
                let entry = state
                    .config
                    .overrides
                    .entry(model.sha256.clone())
                    .or_default();
                let declared = if input {
                    &model.contract.input
                } else {
                    &model.contract.output
                };
                let value = (scaling != *declared).then_some(scaling);
                if input {
                    entry.input = value;
                } else {
                    entry.output = value;
                }
                state.touch_config();
                refresh_overlays = true;
            }
            reply = json!(
                state
                    .model_info()
                    .map(|m| {
                        let contract = state.config.contract_for(m);
                        scaling_index(if input {
                            &contract.input
                        } else {
                            &contract.output
                        })
                    })
                    .unwrap_or(0)
            );
        }
        "composite-white-toggled" | "black-to-white-toggled" => {
            let mut state = lock(&core.state);
            if let (Some(on), Some(model)) = (arg_bool(args), state.model_info().cloned()) {
                let entry = state
                    .config
                    .overrides
                    .entry(model.sha256.clone())
                    .or_default();
                if name == "composite-white-toggled" {
                    entry.composite_on_white =
                        (on != model.contract.composite_on_white).then_some(on);
                } else {
                    let declared = model.contract.black_to_white_below;
                    let value = if on { declared.max(4) } else { 0 };
                    entry.black_to_white_below = (value != declared).then_some(value);
                }
                state.touch_config();
                refresh_overlays = true;
            }
        }
        "reset-preprocessing" => {
            let mut state = lock(&core.state);
            if let Some(model) = state.model_info().cloned() {
                state.config.overrides.remove(&model.sha256);
                state.touch_config();
                refresh_overlays = true;
            }
        }
        "device-selected" => {
            let mut state = lock(&core.state);
            if let Some(device) =
                arg_int(args).and_then(|i| DevicePreference::ALL.get(i.max(0) as usize).copied())
                && device != state.config.device
            {
                state.config.device = device;
                state.touch_config();
                rebuild_engine = true;
            }
            reply = json!(
                DevicePreference::ALL
                    .iter()
                    .position(|d| *d == state.config.device)
                    .unwrap_or(0) as i32
            );
        }
        "resolution-selected" => {
            let mut state = lock(&core.state);
            let index = arg_int(args).unwrap_or(0);
            let levels = state
                .active_file
                .and_then(|id| state.files.get(&id))
                .map(|f| f.geometry.levels.clone())
                .unwrap_or_default();
            let resolution = match index {
                0 => Some(Resolution::Auto),
                i => levels
                    .get((i - 1).max(0) as usize)
                    .map(|level| Resolution::Downsample {
                        value: level.downsample,
                    }),
            };
            if let Some(resolution) = resolution
                && resolution != state.config.resolution
            {
                state.config.resolution = resolution;
                state.touch_config();
                refresh_overlays = true;
            }
            reply = json!(index);
        }
        "threads-committed" => {
            let mut state = lock(&core.state);
            let threads = crate::config::Config::parse_threads(
                arg_str(args).unwrap_or(""),
                state.config.threads,
            );
            if threads != state.config.threads {
                state.config.threads = threads;
                state.touch_config();
                rebuild_engine = true;
            }
            reply = json!(if state.config.threads == 0 {
                "auto".to_string()
            } else {
                state.config.threads.to_string()
            });
        }
        "batch-committed" => {
            let mut state = lock(&core.state);
            let batch = crate::config::Config::parse_batch(
                arg_str(args).unwrap_or(""),
                state.config.batch_size,
            );
            if batch != state.config.batch_size {
                state.config.batch_size = batch;
                state.touch_config();
            }
            let fixed = state.model_info().and_then(|m| m.fixed_batch);
            reply = json!(fixed.unwrap_or(state.config.batch_size).to_string());
        }
        "tf32-toggled" => {
            let mut state = lock(&core.state);
            if let Some(on) = arg_bool(args)
                && on != state.config.tf32
            {
                state.config.tf32 = on;
                state.touch_config();
                rebuild_engine = true;
            }
        }
        "skip-background-toggled" => {
            let mut state = lock(&core.state);
            if let Some(on) = arg_bool(args) {
                state.config.skip_background = on;
                state.touch_config();
            }
        }
        "auto-analyze-toggled" => {
            let mut state = lock(&core.state);
            if let Some(on) = arg_bool(args) {
                state.config.auto_analyze = on;
                state.touch_config();
                if !on
                    && let Some(job) = state
                        .job
                        .as_ref()
                        .filter(|j| j.request.kind == JobKind::Auto)
                {
                    job.control.cancel();
                }
            }
        }
        "keep-pixels-toggled" => {
            let mut state = lock(&core.state);
            if let Some(on) = arg_bool(args) {
                state.config.keep_pixels = on;
                state.touch_config();
            }
        }
        "analyze-viewport" | "analyze-slide" | "analyze-roi" => {
            let (file_id, pane, view) = {
                let state = lock(&core.state);
                let pane = state.focused_pane;
                let view = pane.and_then(|p| state.panes.get(&p)).and_then(|p| p.view);
                (state.active_file, pane, view)
            };
            let Some(file_id) = file_id else {
                return Err("Open a slide first.".to_string());
            };
            core.start_job(JobRequest {
                kind: match name {
                    "analyze-slide" => JobKind::WholeSlide,
                    "analyze-roi" => JobKind::Roi,
                    _ => JobKind::Viewport,
                },
                file_id,
                pane,
                view,
            });
        }
        "cancel-job" => core.cancel_job(),
        "view-selected" => {
            // The focused pane's overlay.
            let (file_id, pane) = {
                let state = lock(&core.state);
                (state.active_file, state.focused_pane)
            };
            if let (Some(file_id), Some(mode)) = (
                file_id,
                arg_int(args).and_then(|i| Mode::ALL.get(i.max(0) as usize).copied()),
            ) {
                core.set_mode(file_id, pane, mode, None)?;
            }
            let state = lock(&core.state);
            reply = json!(
                state
                    .active_file
                    .map(|id| state.overlay(id, state.focused_pane).mode.index() as i32)
                    .unwrap_or(0)
            );
        }
        "jump-to-region" => {
            let Some(coord) = arg_str(args).and_then(TileCoord::parse_id) else {
                return Ok(Value::Null);
            };
            let target = {
                let mut state = lock(&core.state);
                let file_id = state.active_file;
                let rect = file_id
                    .and_then(|id| run_context(&state, id).ok())
                    .map(|ctx| ctx.grid.valid_footprint(coord));
                if let (Some(file_id), Some(_)) = (file_id, rect) {
                    state.pulse = Some((file_id, coord, Instant::now()));
                }
                rect.map(|rect| (state.focused_pane, rect, file_id))
            };
            if let Some((pane, rect, Some(file_id))) = target {
                // Frame the region (with context around it).
                let padded = Rect {
                    x: rect.x - rect.width,
                    y: rect.y - rect.height,
                    width: rect.width * 3.0,
                    height: rect.height * 3.0,
                };
                core.host.frame(pane, padded).map_err(String::from)?;
                core.dirty.mark(|d| {
                    d.overlays.insert(file_id);
                });
                let _ = PULSE;
            }
        }
        "hover-region" => {
            let coord = arg_str(args).and_then(TileCoord::parse_id);
            let mut state = lock(&core.state);
            let file_id = state.active_file;
            let hover = file_id.zip(coord);
            if state.hover != hover {
                state.hover = hover;
                drop(state);
                if let Some(file_id) = file_id {
                    core.dirty.mark(|d| {
                        d.overlays.insert(file_id);
                    });
                }
            }
        }
        "export-csv" => {
            let file = {
                let state = lock(&core.state);
                state
                    .active_file
                    .and_then(|id| state.files.get(&id).map(|f| (id, f.info.filename.clone())))
            };
            if let Some((file_id, filename)) = file {
                let stem = std::path::Path::new(&filename)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "slide".to_string());
                let id = core
                    .host
                    .save_file_dialog(FileDialogArgs {
                        title: Some("Export tile scores".into()),
                        filters: vec![FileFilter {
                            name: "CSV".into(),
                            extensions: vec!["csv".into()],
                        }],
                        default_name: Some(format!("{stem}-eovae.csv")),
                        ..Default::default()
                    })
                    .map_err(String::from)?;
                lock(&core.state).export_dialog = Some((id, file_id));
            }
        }
        "clear-cache" => {
            let status = core.cache.status();
            let id = core
                .host
                .confirm(ConfirmArgs {
                    title: "Clear the VAE result cache?".into(),
                    message: format!(
                        "This deletes {} of stored results for every slide. Results are recomputed when needed.",
                        mib(status.bytes)
                    ),
                    confirm_label: "Clear".into(),
                    cancel_label: "Cancel".into(),
                    destructive: true,
                })
                .map_err(String::from)?;
            lock(&core.state).clear_confirm = Some(id);
        }
        "section-toggled" => {
            let section = args.first().and_then(Value::as_str).unwrap_or("");
            let expanded = args.get(1).and_then(Value::as_bool).unwrap_or(true);
            let mut state = lock(&core.state);
            let sections = &mut state.config.sections;
            match section {
                "model" => sections.model = expanded,
                "analysis" => sections.analysis = expanded,
                "results" => sections.results = expanded,
                "cache" => sections.cache = expanded,
                _ => {}
            }
            state.touch_config();
        }
        _ => {}
    }
    // A change that alters results ends the running job: its tiles would
    // belong to the old configuration (VAE-12).
    if result_config(core) != before {
        let state = lock(&core.state);
        if let Some(job) = &state.job {
            job.control.cancel();
        }
    }
    if rebuild_engine {
        let has_model = lock(&core.state).model_info().is_some();
        if has_model {
            core.prebuild_engine();
        }
    }
    if refresh_overlays || rebuild_engine {
        core.dirty.mark(|d| {
            d.all_overlays = true;
            d.hud = true;
        });
    }
    core.dirty.mark(|d| d.sidebar = true);
    Ok(reply)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::ContractOverride;

    #[test]
    fn scaling_choices_round_trip() {
        for index in 0..4 {
            assert_eq!(scaling_index(&scaling_at(index).unwrap()), index as i32);
        }
        assert!(scaling_at(9).is_none());
    }

    #[test]
    fn the_adjustment_note_says_what_reconstructions_can_follow() {
        let neutral = DisplayAdjustments::default();
        assert_eq!(
            adjustment_note(Mode::Reconstruction, Some(&neutral), false),
            (String::new(), false)
        );
        let tone = DisplayAdjustments {
            gamma: 1.4,
            sharpness: 0.3,
            modified: true,
            badge: "Display-adjusted · Sharpened".into(),
            ..Default::default()
        };
        let (text, warning) = adjustment_note(Mode::Reconstruction, Some(&tone), false);
        assert!(text.contains("display adjustments") && !warning, "{text}");
        // Deconvolution without normalisation shows only in the badge.
        let deconvolved = DisplayAdjustments {
            modified: true,
            badge: "Isolated stain: Hematoxylin · Display-adjusted".into(),
            ..Default::default()
        };
        let (text, warning) = adjustment_note(Mode::Reconstruction, Some(&deconvolved), false);
        assert!(warning && text.contains("Isolated stain"), "{text}");
        let normalised = DisplayAdjustments {
            stain_normalization: Some("vahadane".into()),
            modified: true,
            ..Default::default()
        };
        let (text, _) = adjustment_note(Mode::Reconstruction, Some(&normalised), false);
        assert!(text.contains("Vahadane normalisation"), "{text}");
        // Errors do not depend on the display; old hosts cannot say.
        let (text, warning) = adjustment_note(Mode::ErrorMap, Some(&tone), false);
        assert!(text.contains("own pixels") && !warning, "{text}");
        assert!(adjustment_note(Mode::Reconstruction, None, true).1);
        assert_eq!(adjustment_note(Mode::Original, Some(&tone), false).0, "");
    }

    #[test]
    fn numbers_are_readable() {
        assert_eq!(thousands(1234567.4), "1,234,567");
        assert_eq!(thousands(12.0), "12");
        assert_eq!(mib(3 * 1024 * 1024 * 1024), "3.0 GB");
    }

    #[test]
    fn unchanged_overrides_stay_empty() {
        assert!(ContractOverride::default().is_empty());
    }
}

//! Plugin-level tests: the real plugin against the API's mock host, with a
//! synthetic slide served by the mock region reader and the test inference
//! backend (no ONNX Runtime needed).

use crate::Eovae;
use crate::engine::testing::TestBackend;
use crate::onnx_scan::encode::{ModelSpec, model};
use eov_plugin_api::abi::RegionRequest;
use eov_plugin_api::prelude::*;
use eov_plugin_api::testing::{MockHost, TestHarness, test_init};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

type Harness = TestHarness<Eovae<TestBackend>>;

const PANE: PaneId = 7;
const TILE: i64 = 32;

/// A 4096×2048 slide with Aperio-like levels (1, 4, 16.004) at 0.5 µm/px.
/// The left half is tissue, the right half glass.
fn slide_info(file_id: FileId, path: &Path) -> FileInfo {
    FileInfo {
        file_id,
        path: path.to_string_lossy().into_owned(),
        filename: path.file_name().unwrap().to_string_lossy().into_owned(),
        width: 4096,
        height: 2048,
        levels: vec![
            LevelInfo {
                level: 0,
                width: 4096,
                height: 2048,
                downsample: 1.0,
            },
            LevelInfo {
                level: 1,
                width: 1024,
                height: 512,
                downsample: 4.0,
            },
            LevelInfo {
                level: 2,
                width: 256,
                height: 128,
                downsample: 16.004,
            },
        ],
        mpp_x: Some(0.5),
        mpp_y: Some(0.5),
        ..Default::default()
    }
}

fn view(file_id: FileId, x: f64, y: f64, width: f64, height: f64) -> ViewInfo {
    let zoom = 0.5;
    ViewInfo {
        pane: PANE,
        file_id,
        center_x: x + width / 2.0,
        center_y: y + height / 2.0,
        zoom,
        width: width * zoom,
        height: height * zoom,
        scale_factor: 1.0,
        rotation_deg: 0.0,
        visible: Rect {
            x,
            y,
            width,
            height,
        },
    }
}

struct Setup {
    harness: Harness,
    requests: Arc<Mutex<Vec<RegionRequest>>>,
    dir: tempfile::TempDir,
    init: PluginInit,
}

/// What the region reader does.
#[derive(Clone, Copy, PartialEq)]
enum Reads {
    Slide,
    Fail,
}

fn write_fixtures(dir: &Path, model_name: &str) -> PathBuf {
    // A small model: 32-px tiles, fixed batch 4, trained at 2 µm/px.
    let model_path = dir.join(model_name);
    std::fs::write(
        &model_path,
        model(&ModelSpec {
            inputs: vec![("input", 1, vec![4, 3, TILE, TILE])],
            outputs: vec![("reconstruction", 1, vec![4, 3, TILE, TILE])],
            initializers: vec![],
            metadata: vec![("eovae.mpp", "2.0")],
            initializers_as_inputs: false,
        }),
    )
    .unwrap();
    std::fs::write(dir.join("slide-a.svs"), vec![1u8; 4096]).unwrap();
    std::fs::write(dir.join("slide-b.svs"), vec![2u8; 4096]).unwrap();
    model_path
}

/// A mock host showing slide A in one pane, serving synthetic pixels.
fn mock_host(dir: &Path, reads: Reads) -> (MockHost, Arc<Mutex<Vec<RegionRequest>>>) {
    let host = MockHost::new();
    let requests = Arc::new(Mutex::new(Vec::new()));
    {
        let mut state = host.state();
        state
            .snapshot
            .files
            .push(slide_info(1, &dir.join("slide-a.svs")));
        state.snapshot.panes.push(PaneInfo {
            pane: PANE,
            index: 0,
            file_id: Some(1),
            view: Some(view(1, 0.0, 0.0, 1024.0, 512.0)),
        });
        state.snapshot.focused_pane = Some(PANE);
        let log = Arc::clone(&requests);
        state.region_reader = Some(Box::new(move |request: &RegionRequest| {
            log.lock().unwrap().push(*request);
            if reads == Reads::Fail {
                return Err("corrupt JPEG data".to_string());
            }
            let n = (request.width * request.height) as usize;
            let mut data = Vec::with_capacity(n * 4);
            let glass = request.x >= 2048;
            for i in 0..n {
                if glass {
                    data.extend_from_slice(&[250, 250, 250, 255]);
                } else if i % 2 == 0 {
                    data.extend_from_slice(&[90, 60, 150, 255]);
                } else {
                    data.extend_from_slice(&[200, 120, 180, 255]);
                }
            }
            Ok(eov_plugin_api::sdk::Image {
                width: request.width,
                height: request.height,
                data,
            })
        }));
    }
    (host, requests)
}

fn setup_with(model_name: &str, config: Value, reads: Reads) -> Setup {
    let dir = tempfile::tempdir().unwrap();
    let model_path = write_fixtures(dir.path(), model_name);
    let mut init = test_init("eovae");
    let mut config = config;
    config
        .as_object_mut()
        .unwrap()
        .insert("model_path".into(), json!(model_path.to_string_lossy()));
    init.settings.insert("config".into(), config);
    let (host, requests) = mock_host(dir.path(), reads);
    let harness = Harness::with_host(host, init.clone()).unwrap();
    Setup {
        harness,
        requests,
        dir,
        init,
    }
}

fn setup() -> Setup {
    setup_with("vae.onnx", json!({"device": "cpu"}), Reads::Slide)
}

fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
    let started = Instant::now();
    while !condition() {
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "timed out waiting for {what}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

impl Setup {
    fn start(&self) -> String {
        self.harness
            .send(HostEvent::Ready {
                restore_sidebar: false,
            })
            .unwrap();
        self.harness
            .send(HostEvent::ToolbarButtonClicked {
                button_id: crate::plugin::TOOLBAR_BUTTON.into(),
            })
            .unwrap();
        let surface = self.sidebar_id().expect("the sidebar opened");
        self.wait_property("model ready", "model-state", |v| v == &json!(2));
        self.wait_property("engine", "engine-label", |v| {
            v.as_str().is_some_and(|s| !s.is_empty())
        });
        surface
    }

    fn sidebar_id(&self) -> Option<String> {
        self.harness
            .state()
            .surfaces
            .iter()
            .find(|(_, s)| s.open && s.spec.kind == SurfaceKind::Sidebar)
            .map(|(id, _)| id.clone())
    }

    fn property(&self, name: &str) -> Value {
        self.harness
            .state()
            .surfaces
            .values()
            .find(|s| s.open && s.spec.kind == SurfaceKind::Sidebar)
            .and_then(|s| s.properties.get(name).cloned())
            .unwrap_or(Value::Null)
    }

    fn wait_property(&self, what: &str, name: &str, check: impl Fn(&Value) -> bool) {
        wait_until(what, || check(&self.property(name)));
    }

    fn callback(&self, surface: &str, name: &str, args: Vec<Value>) -> Value {
        self.harness.ui_callback(surface, name, args).unwrap()
    }

    fn analyse_viewport(&self, surface: &str) {
        self.callback(surface, "analyze-viewport", vec![]);
        self.wait_finished();
    }

    fn wait_finished(&self) {
        self.wait_property("the job to finish", "last-result", |v| {
            v.as_str().is_some_and(|s| !s.is_empty())
        });
        self.wait_property("the job to stop", "job-running", |v| v == &json!(false));
    }

    fn reads(&self) -> Vec<RegionRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn layer_raster_ids(&self, layer: &str) -> Vec<String> {
        self.harness
            .state()
            .layers
            .get(layer)
            .map(|l| l.rasters.keys().cloned().collect())
            .unwrap_or_default()
    }
}

#[test]
fn registers_its_contributions_without_host_ui_special_cases() {
    let s = setup();
    let state = s.harness.state();
    assert!(
        state
            .toolbar_buttons
            .contains_key(crate::plugin::TOOLBAR_BUTTON)
    );
    assert!(state.hud_buttons.contains_key(crate::plugin::HUD_GRID));
    // The overlay menu comes from the plugin, not the host (PLG-8).
    let menu = &state.hud_buttons[crate::plugin::HUD_VIEW].menu;
    let ids: Vec<&str> = menu.iter().map(|m| m.item_id.as_str()).collect();
    assert_eq!(
        ids,
        ["original", "reconstruction", "error_map", "difference"]
    );
    // No CPU viewport filter: overlays are host-drawn rasters (VAE-16).
    assert!(state.filters.is_empty());
}

#[test]
fn analyses_the_real_pyramid_level_and_keys_requests_correctly() {
    let s = setup();
    let surface = s.start();
    // The model is trained at 2 µm/px; the slide is 0.5 µm/px at level 0,
    // so Auto picks level 1 (4×), not a fake "2^n" level (VAE-3, VAE-15).
    s.wait_property("resolution note", "resolution-note", |v| {
        v.as_str().is_some_and(|t| t.contains("Level 1"))
    });
    s.analyse_viewport(&surface);
    let reads = s.reads();
    assert!(!reads.is_empty());
    for request in &reads {
        assert_eq!(request.level, 1);
        assert_eq!((request.width, request.height), (32, 32));
        // Tile origins use the level's real downsample (128 level-0 px).
        assert_eq!(request.x % 128, 0, "{request:?}");
        assert_eq!(request.y % 128, 0, "{request:?}");
        // Read onto white, as the model was trained (VAE-4).
        assert_eq!(request.background, 0xFFFF_FFFF);
    }
    // The 1024×512 view covers 8×4 tiles of the 32-px grid at level 1.
    assert_eq!(reads.len(), 32);
    let result = s.property("last-result");
    assert!(result.as_str().unwrap().contains("32 analysed"), "{result}");
    // The view is centred: the first tiles read are in the middle.
    let first = reads[0];
    assert!(
        (256..=640).contains(&first.x) && (128..=384).contains(&first.y),
        "{first:?}"
    );
    // Analysing again finds everything done.
    let before = s.reads().len();
    s.analyse_viewport(&surface);
    assert_eq!(s.reads().len(), before, "results are reused");
}

#[test]
fn background_is_recorded_and_never_inferred_or_retried() {
    let s = setup();
    let surface = s.start();
    s.callback(&surface, "analyze-slide", vec![]);
    s.wait_finished();
    let result = s.property("last-result");
    let text = result.as_str().unwrap();
    // 32×16 tiles at level 1; the right half (x ≥ 2048) is glass.
    assert!(
        text.contains("256 analysed") && text.contains("256 background"),
        "{text}"
    );
    let coverage = s.property("coverage-text");
    assert!(
        coverage.as_str().unwrap().contains("256 background"),
        "{coverage}"
    );
    // Glass is not pending: a viewport run over it reads nothing more.
    s.harness
        .send(HostEvent::ViewChanged {
            view: view(1, 2048.0, 0.0, 1024.0, 512.0),
        })
        .unwrap();
    let before = s.reads().len();
    s.analyse_viewport(&surface);
    assert_eq!(
        s.reads().len(),
        before,
        "background tiles stay done (VAE-10)"
    );
}

#[test]
fn overlays_are_host_rasters_and_grid_lines_per_slide() {
    let s = setup();
    let surface = s.start();
    s.analyse_viewport(&surface);
    let layer = crate::plugin::raster_layer(1);
    s.harness
        .send(HostEvent::HudButtonClicked {
            button_id: crate::plugin::HUD_VIEW.into(),
            item_id: Some("error_map".into()),
            pane: PANE,
            view: Some(view(1, 0.0, 0.0, 1024.0, 512.0)),
        })
        .unwrap();
    wait_until("the heat map", || s.layer_raster_ids(&layer) == ["heat"]);
    // The HUD menu of that pane shows the mode, pushed by the plugin.
    wait_until("the HUD state", || {
        let state = s.harness.state();
        state
            .calls_to("hud.update")
            .iter()
            .any(|call| call["pane"] == json!(PANE) && call["active"] == json!(true))
    });
    s.harness
        .send(HostEvent::HudButtonClicked {
            button_id: crate::plugin::HUD_VIEW.into(),
            item_id: Some("reconstruction".into()),
            pane: PANE,
            view: Some(view(1, 0.0, 0.0, 1024.0, 512.0)),
        })
        .unwrap();
    wait_until("the mosaic", || {
        s.layer_raster_ids(&layer) == [format!("mosaic-{PANE}")]
    });
    {
        let state = s.harness.state();
        let raster = &state.layers[&layer].rasters[&format!("mosaic-{PANE}")];
        // Reconstructed pixels are opaque where tiles were analysed.
        let data = &raster.pixels.data;
        assert!(data.as_chunks::<4>().0.iter().any(|px| px[3] == 255));
        assert!(raster.x >= 0.0 && raster.width > 0.0);
    }
    // The grid is real polylines in its own layer (no "grid-" id hacks).
    s.harness
        .send(HostEvent::HudButtonClicked {
            button_id: crate::plugin::HUD_GRID.into(),
            item_id: None,
            pane: PANE,
            view: Some(view(1, 0.0, 0.0, 1024.0, 512.0)),
        })
        .unwrap();
    let grid = crate::plugin::grid_layer(1);
    wait_until("the grid", || !s.harness.state().objects(&grid).is_empty());
    let state = s.harness.state();
    let lines = state.objects(&grid);
    assert_eq!(lines.len(), (32 + 1) + (16 + 1));
    assert!(
        lines
            .iter()
            .all(|o| matches!(o.shape, Shape::Polyline { .. }))
    );
    assert!(state.layers[&grid].spec.min_zoom.is_some());
    drop(state);
    // Back to the original: the rasters go.
    s.harness
        .send(HostEvent::HudButtonClicked {
            button_id: crate::plugin::HUD_VIEW.into(),
            item_id: Some("original".into()),
            pane: PANE,
            view: Some(view(1, 0.0, 0.0, 1024.0, 512.0)),
        })
        .unwrap();
    wait_until("the rasters to go", || {
        s.layer_raster_ids(&layer).is_empty()
    });
}

#[test]
fn results_are_keyed_by_slide() {
    let s = setup();
    let surface = s.start();
    s.analyse_viewport(&surface);
    s.wait_property("results on slide A", "coverage-text", |v| {
        v.as_str().is_some_and(|t| t.contains("32 tissue tiles"))
    });
    // Slide B opens in the same pane: it must not show slide A's results.
    let other = slide_info(2, &s.dir.path().join("slide-b.svs"));
    s.harness.state().snapshot.files.push(other.clone());
    s.harness
        .send(HostEvent::FileOpened { file: other })
        .unwrap();
    s.harness
        .send(HostEvent::ActiveChanged {
            pane: Some(PANE),
            file_id: Some(2),
        })
        .unwrap();
    s.wait_property("slide B", "slide-name", |v| v == &json!("slide-b.svs"));
    s.wait_property("no results on B", "coverage-text", |v| {
        v.as_str().is_some_and(|t| t.contains("Nothing analysed"))
    });
    // Analysing B reads B (file 2), not the focused slide of another pane.
    s.harness
        .send(HostEvent::ViewChanged {
            view: view(2, 0.0, 0.0, 1024.0, 512.0),
        })
        .unwrap();
    let before = s.reads().len();
    s.callback(&surface, "analyze-viewport", vec![]);
    wait_until("slide B reads", || s.reads().len() >= before + 32);
    assert!(s.reads()[before..].iter().all(|r| r.file_id == 2));
}

#[test]
fn memory_for_tile_pixels_is_bounded() {
    let s = setup_with(
        "vae.onnx",
        json!({"device": "cpu", "memory_limit_mb": 1}),
        Reads::Slide,
    );
    let surface = s.start();
    s.callback(&surface, "analyze-slide", vec![]);
    s.wait_finished();
    // 256 analysed tiles of ~4 KiB each exceed 1 MiB: payloads were evicted,
    // scores were kept.
    let memory = s.property("memory-text");
    assert!(memory.as_str().unwrap().contains("of 1 MB"), "{memory}");
    let coverage = s.property("coverage-text");
    assert!(
        coverage.as_str().unwrap().contains("256 tissue tiles"),
        "{coverage}"
    );
}

#[test]
fn read_failures_are_reported_with_their_cause() {
    let s = setup_with("vae.onnx", json!({"device": "cpu"}), Reads::Fail);
    let surface = s.start();
    s.analyse_viewport(&surface);
    let result = s.property("last-result");
    let text = result.as_str().unwrap();
    assert!(
        text.contains("could not be read") && text.contains("corrupt JPEG"),
        "{text}"
    );
    assert!(
        !text.contains("cancel"),
        "never reported as a cancellation (VAE-7)"
    );
    assert_eq!(s.property("last-result-error"), json!(true));
    let finished = s.harness.calls_to("job.finish");
    assert!(
        finished.iter().any(|call| call["error"].is_string()),
        "{finished:?}"
    );
}

#[test]
fn inference_failures_are_reported_with_their_cause() {
    let s = setup_with("fail-vae.onnx", json!({"device": "cpu"}), Reads::Slide);
    let surface = s.start();
    s.analyse_viewport(&surface);
    let result = s.property("last-result");
    assert!(result.as_str().unwrap().contains("wrong shape"), "{result}");
}

#[test]
fn cancelling_stops_a_job_promptly() {
    let s = setup_with("hang-vae.onnx", json!({"device": "cpu"}), Reads::Slide);
    let surface = s.start();
    s.callback(&surface, "analyze-slide", vec![]);
    s.wait_property("the job", "job-running", |v| v == &json!(true));
    // The host's job panel can cancel it too.
    std::thread::sleep(Duration::from_millis(100));
    s.callback(&surface, "cancel-job", vec![]);
    s.wait_finished();
    let result = s.property("last-result");
    assert!(result.as_str().unwrap().contains("cancelled"), "{result}");
    assert_eq!(s.property("last-result-error"), json!(false));
}

#[test]
fn device_choice_is_independent_of_the_render_backend() {
    // CUDA requested but unavailable: a clear error, no crash, no CPU
    // fallback behind the user's back (VAE-8).
    let s = setup_with("vae.onnx", json!({"device": "cuda"}), Reads::Slide);
    s.harness
        .send(HostEvent::Ready {
            restore_sidebar: true,
        })
        .unwrap();
    s.wait_property("the CUDA error", "model-message", |v| {
        v.as_str()
            .is_some_and(|t| t.contains("CUDA is not available"))
    });
    assert_eq!(s.property("can-analyze"), json!(false));
    // Auto falls back to the CPU and says why.
    let surface = s.sidebar_id().unwrap();
    let index = s.callback(&surface, "device-selected", vec![json!(0)]);
    assert_eq!(index, json!(0));
    s.wait_property("the fallback note", "engine-note", |v| {
        v.as_str().is_some_and(|t| t.contains("no CUDA in tests"))
    });
    // The choice is saved in the host's settings (debounced, off the UI
    // thread), never in a home-directory file (VAE-16).
    wait_until("the saved setting", || {
        s.harness
            .state()
            .settings
            .get("config")
            .is_some_and(|c| c["device"] == json!("auto"))
    });
}

#[test]
fn typed_values_are_validated_by_the_plugin() {
    let s = setup();
    let surface = s.start();
    assert_eq!(
        s.callback(&surface, "threads-committed", vec![json!("")]),
        json!("auto")
    );
    assert_eq!(
        s.callback(&surface, "threads-committed", vec![json!("abc")]),
        json!("auto")
    );
    // A fixed-batch model ignores the batch setting.
    assert_eq!(
        s.callback(&surface, "batch-committed", vec![json!("64")]),
        json!("4")
    );
    let unchanged = s.callback(&surface, "output-selected", vec![json!(5)]);
    assert_eq!(unchanged, json!(0));
}

#[test]
fn the_sidebar_matches_its_slint_contract() {
    // Every property the plugin pushes exists in the .slint file, and every
    // callback the file declares is handled (checked textually here; the
    // types are checked with Slint itself by `scripts/check_ui.sh`).
    let s = setup();
    let surface = s.start();
    s.analyse_viewport(&surface);
    s.wait_property("stats", "stats-text", |v| {
        v.as_str().is_some_and(|t| !t.is_empty())
    });
    let ui = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("ui/eovae-sidebar.slint"),
    )
    .unwrap();
    let properties = s
        .harness
        .state()
        .surfaces
        .values()
        .find(|v| v.open)
        .unwrap()
        .properties
        .clone();
    let declared: Vec<&str> = ui
        .lines()
        .filter_map(|line| line.trim().split_once("property <"))
        .filter_map(|(_, rest)| rest.split_once('>'))
        .map(|(_, rest)| rest.trim().split([':', ';']).next().unwrap().trim())
        .collect();
    for name in properties.keys() {
        assert!(
            declared.contains(&name.as_str()),
            "property '{name}' is not declared in the sidebar"
        );
    }
    if let Some(dump) = std::env::var_os("EOVAE_DUMP_SIDEBAR") {
        std::fs::write(
            PathBuf::from(dump),
            serde_json::to_string_pretty(&properties).unwrap(),
        )
        .unwrap();
    }
    for line in ui.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("callback ") {
            let name = rest.split(['(', ';']).next().unwrap();
            let reply = s.harness.ui_callback(&surface, name, vec![]);
            assert!(
                reply.is_ok() || name == "load-model",
                "callback '{name}' failed: {reply:?}"
            );
        }
    }
}

#[test]
fn shutting_down_stops_every_thread() {
    let s = setup_with("hang-vae.onnx", json!({"device": "cpu"}), Reads::Slide);
    let surface = s.start();
    s.callback(&surface, "analyze-slide", vec![]);
    s.wait_property("the job", "job-running", |v| v == &json!(true));
    let started = Instant::now();
    drop(s.harness);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "shutdown joined promptly"
    );
}

#[test]
fn changing_what_results_mean_cancels_the_running_job() {
    let s = setup_with("hang-vae.onnx", json!({"device": "cpu"}), Reads::Slide);
    let surface = s.start();
    s.callback(&surface, "analyze-slide", vec![]);
    s.wait_property("the job", "job-running", |v| v == &json!(true));
    // Picking level 0 changes which tiles exist: the job ends (VAE-12).
    s.callback(&surface, "resolution-selected", vec![json!(1)]);
    s.wait_finished();
    let result = s.property("last-result");
    assert!(result.as_str().unwrap().contains("cancelled"), "{result}");
}

#[test]
fn results_persist_across_restarts_without_new_reads() {
    let s = setup();
    let surface = s.start();
    s.analyse_viewport(&surface);
    let Setup {
        harness, dir, init, ..
    } = s;
    // Shutting down flushes the cache.
    drop(harness);
    let (host, requests) = mock_host(dir.path(), Reads::Slide);
    let again = Setup {
        harness: Harness::with_host(host, init.clone()).unwrap(),
        requests,
        dir,
        init,
    };
    let surface = again.start();
    again.wait_property("stored results", "coverage-text", |v| {
        v.as_str().is_some_and(|t| t.contains("32 tissue tiles"))
    });
    again.analyse_viewport(&surface);
    assert!(
        again.reads().is_empty(),
        "everything came from the disk cache"
    );
    let result = again.property("last-result");
    assert!(
        result.as_str().unwrap().contains("already analysed"),
        "{result}"
    );
}

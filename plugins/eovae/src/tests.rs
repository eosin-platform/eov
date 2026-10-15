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
        flipped: false,
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

    /// Picks an overlay in a pane's HUD menu.
    fn hud_view(&self, pane: PaneId, item: &str) {
        let state = self.harness.state();
        let view = state
            .snapshot
            .panes
            .iter()
            .find(|p| p.pane == pane)
            .and_then(|p| p.view);
        drop(state);
        self.harness
            .send(HostEvent::HudButtonClicked {
                button_id: crate::plugin::HUD_VIEW.into(),
                item_id: Some(item.into()),
                pane,
                view,
            })
            .unwrap();
    }

    /// Shows slide `file_id` in a second pane, side by side.
    fn add_pane(&self, pane: PaneId, file_id: FileId) {
        let view = ViewInfo {
            pane,
            ..view(file_id, 0.0, 0.0, 1024.0, 512.0)
        };
        self.harness.state().snapshot.panes.push(PaneInfo {
            pane,
            index: 1,
            file_id: Some(file_id),
            view: Some(view),
        });
    }

    fn hidden_in(&self, layer: &str) -> Vec<PaneId> {
        self.harness
            .state()
            .layers
            .get(layer)
            .map(|l| l.hidden_panes.iter().copied().collect())
            .unwrap_or_default()
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
fn overlays_are_host_rasters_and_grid_lines() {
    let s = setup();
    let surface = s.start();
    s.analyse_viewport(&surface);
    let heat = crate::plugin::heat_layer(1);
    s.hud_view(PANE, "error_map");
    // The analysed 8×4 tiles fall in the first 64-px chunk of the heat map.
    wait_until("the heat map", || s.layer_raster_ids(&heat) == ["heat-0-0"]);
    // The HUD menu of that pane shows the mode, pushed by the plugin.
    wait_until("the HUD state", || {
        let state = s.harness.state();
        state
            .calls_to("hud.update")
            .iter()
            .any(|call| call["pane"] == json!(PANE) && call["active"] == json!(true))
    });
    s.hud_view(PANE, "reconstruction");
    let mosaic = crate::plugin::mosaic_layer(1, PANE);
    wait_until("the mosaic", || s.layer_raster_ids(&mosaic) == ["mosaic"]);
    wait_until("the heat map to go", || {
        s.layer_raster_ids(&heat).is_empty()
    });
    {
        let state = s.harness.state();
        let raster = &state.layers[&mosaic].rasters["mosaic"];
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
    // Back to the original: the pane's mosaic layer goes.
    s.hud_view(PANE, "original");
    wait_until("the mosaic to go", || {
        !s.harness.state().layers.contains_key(&mosaic)
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

#[test]
fn the_manifest_is_valid_for_plugin_api_1() {
    use eov_plugin_api::PluginManifest;
    let version = |text: &str| semver::Version::parse(text).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = PluginManifest::from_file(&root.join("plugin.toml")).unwrap();
    assert_eq!(manifest.id, "eovae");
    assert_eq!(
        manifest.version,
        env!("CARGO_PKG_VERSION"),
        "plugin.toml and Cargo.toml agree"
    );
    assert!(manifest.declares_environment);
    assert_eq!(manifest.api.as_deref(), Some(">=1.0, <2"));
    assert!(manifest.accepts_api(&version(eov_plugin_api::abi::API_VERSION)));
    assert!(manifest.accepts_host(&version("1.0.0")));
    assert!(!manifest.accepts_host(&version("2.0.0")));
    assert!(manifest.accepts_slint(&version("1.17.1")));
    manifest.validate_files(root).unwrap();
    // The host loads lib<id>.so (or the platform's name): the crate's
    // library must be named after the id.
    let cargo = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    assert!(cargo.contains("name = \"eovae\"\ncrate-type = [\"cdylib\", \"rlib\"]"));
    // No 0.x leftovers.
    let text = std::fs::read_to_string(root.join("plugin.toml")).unwrap();
    assert!(!text.contains("toolbar_buttons"));
}

#[test]
fn region_of_interest_runs_cover_only_the_roi() {
    let s = setup();
    // A host ROI over two tiles of the level-1 grid (128 level-0 px each).
    s.harness.state().roi.insert(
        1,
        Rect {
            x: 300.0,
            y: 140.0,
            width: 200.0,
            height: 100.0,
        },
    );
    let surface = s.start();
    s.wait_property("ROI button", "has-roi", |v| v == &json!(true));
    s.callback(&surface, "analyze-roi", vec![]);
    s.wait_finished();
    let reads = s.reads();
    // x 300..500 spans columns 2-3 (256..512); y 140..240 is row 1.
    assert_eq!(reads.len(), 2, "{reads:?}");
    assert!(
        reads
            .iter()
            .all(|r| (256..512).contains(&r.x) && r.y == 128)
    );
    let result = s.property("last-result");
    assert!(
        result.as_str().unwrap().contains("Region of interest"),
        "{result}"
    );
}

#[test]
fn a_host_identity_arriving_later_keeps_the_results() {
    let s = setup();
    let surface = s.start();
    s.analyse_viewport(&surface);
    // The host learns the slide's identity after eovae keyed the slide by
    // its own fingerprint.
    s.harness.state().snapshot.files[0].identity = Some("quickhash1:a".into());
    s.harness
        .send(HostEvent::ViewChanged {
            view: view(1, 1024.0, 512.0, 1024.0, 512.0),
        })
        .unwrap();
    s.callback(&surface, "analyze-viewport", vec![]);
    // The first results stay with the slide, next to the new ones.
    s.wait_property("both viewports", "coverage-text", |v| {
        v.as_str().is_some_and(|t| t.contains("64 tissue tiles"))
    });
    s.wait_property("the job to stop", "job-running", |v| v == &json!(false));
    let Setup {
        harness, dir, init, ..
    } = s;
    drop(harness);
    // New results were stored under the identity.
    let db =
        rusqlite::Connection::open(Path::new(&init.cache_dir).join(crate::db::DB_FILE)).unwrap();
    let identity: Option<String> = db
        .query_row("SELECT identity FROM slide", [], |row| row.get(0))
        .unwrap();
    assert_eq!(identity.as_deref(), Some("quickhash1:a"));
    drop(db);

    // After a restart everything is found again, with no new reads.
    let (host, requests) = mock_host(dir.path(), Reads::Slide);
    host.state().snapshot.files[0].identity = Some("quickhash1:a".into());
    let again = Setup {
        harness: Harness::with_host(host, init.clone()).unwrap(),
        requests,
        dir,
        init,
    };
    let surface = again.start();
    again.wait_property("stored results", "coverage-text", |v| {
        v.as_str().is_some_and(|t| t.contains("64 tissue tiles"))
    });
    again.analyse_viewport(&surface);
    assert!(again.reads().is_empty(), "{:?}", again.reads());
}

#[test]
fn each_pane_shows_its_own_overlay() {
    const OTHER: PaneId = 8;
    let s = setup();
    s.add_pane(OTHER, 1);
    let surface = s.start();
    s.analyse_viewport(&surface);
    let heat = crate::plugin::heat_layer(1);
    s.hud_view(PANE, "error_map");
    wait_until("the heat map", || !s.layer_raster_ids(&heat).is_empty());
    // Only the pane that chose it shows the error map.
    wait_until("hidden in the other pane", || s.hidden_in(&heat) == [OTHER]);
    // The other pane shows reconstructions, in a layer of its own.
    s.hud_view(OTHER, "reconstruction");
    let mosaic = crate::plugin::mosaic_layer(1, OTHER);
    wait_until("the other pane's mosaic", || {
        s.layer_raster_ids(&mosaic) == ["mosaic"]
    });
    assert_eq!(s.hidden_in(&mosaic), [PANE]);
    assert_eq!(s.hidden_in(&heat), [OTHER], "the error map stays");
    assert!(
        !s.harness
            .state()
            .layers
            .contains_key(&crate::plugin::mosaic_layer(1, PANE))
    );
    // The grid of one pane is not drawn in the other.
    s.harness
        .send(HostEvent::HudButtonClicked {
            button_id: crate::plugin::HUD_GRID.into(),
            item_id: None,
            pane: OTHER,
            view: None,
        })
        .unwrap();
    let grid = crate::plugin::grid_layer(1);
    wait_until("the grid", || !s.harness.state().objects(&grid).is_empty());
    assert_eq!(s.hidden_in(&grid), [PANE]);
    // Each pane's HUD shows its own overlay.
    wait_until("both HUDs", || {
        let calls = s.harness.calls_to("hud.update");
        let checked = |pane: PaneId, item: &str| {
            calls.iter().any(|call| {
                call["pane"] == json!(pane)
                    && call["menu"].as_array().is_some_and(|menu| {
                        menu.iter()
                            .any(|m| m["item_id"] == json!(item) && m["checked"] == json!(true))
                    })
            })
        };
        checked(PANE, "error_map") && checked(OTHER, "reconstruction")
    });
    // The sidebar follows the focused pane.
    s.wait_property("focused pane's overlay", "view-index", |v| v == &json!(2));
    s.harness
        .send(HostEvent::ActiveChanged {
            pane: Some(OTHER),
            file_id: Some(1),
        })
        .unwrap();
    s.wait_property("other pane's overlay", "view-index", |v| v == &json!(1));
}

#[test]
fn hosts_without_pane_visibility_share_overlays() {
    const OTHER: PaneId = 8;
    let s = setup();
    s.add_pane(OTHER, 1);
    s.harness
        .state()
        .unsupported
        .push("layer.set_pane_visibility".into());
    let surface = s.start();
    s.analyse_viewport(&surface);
    s.hud_view(PANE, "error_map");
    // The plugin notices and shows the overlay in every pane of the slide.
    wait_until("the shared overlay in the other pane's HUD", || {
        s.harness.calls_to("hud.update").iter().any(|call| {
            call["pane"] == json!(OTHER)
                && call["button_id"] == json!(crate::plugin::HUD_VIEW)
                && call["active"] == json!(true)
        })
    });
    let logs = s.harness.calls_to("log");
    assert!(
        logs.iter().any(|l| l["message"]
            .as_str()
            .is_some_and(|m| m.contains("same VAE overlay"))),
        "{logs:?}"
    );
}

#[test]
fn the_heat_map_resends_only_chunks_with_new_results() {
    let s = setup();
    let surface = s.start();
    s.analyse_viewport(&surface);
    let heat = crate::plugin::heat_layer(1);
    s.hud_view(PANE, "error_map");
    wait_until("the first chunk", || {
        s.layer_raster_ids(&heat) == ["heat-0-0"]
    });
    // Mark the chunk as the host stores it: re-sending it would undo this.
    s.harness
        .state()
        .layers
        .get_mut(&heat)
        .unwrap()
        .rasters
        .get_mut("heat-0-0")
        .unwrap()
        .pixels
        .data[3] = 1;
    // The next 8 tile columns (x 1024–2048) are analysed automatically in
    // the error-map view and fall in the second chunk.
    s.harness
        .send(HostEvent::ViewChanged {
            view: view(1, 1024.0, 0.0, 1024.0, 512.0),
        })
        .unwrap();
    wait_until("the second chunk", || {
        s.layer_raster_ids(&heat) == ["heat-0-0", "heat-1-0"]
    });
    s.wait_property("the job to stop", "job-running", |v| v == &json!(false));
    s.wait_property("all results", "coverage-text", |v| {
        v.as_str().is_some_and(|t| t.contains("64 tissue tiles"))
    });
    std::thread::sleep(Duration::from_millis(600));
    let state = s.harness.state();
    assert_eq!(
        state.layers[&heat].rasters["heat-0-0"].pixels.data[3], 1,
        "the unchanged chunk was not sent again"
    );
    // The chunks tile the heat map without overlapping.
    let a = &state.layers[&heat].rasters["heat-0-0"];
    let b = &state.layers[&heat].rasters["heat-1-0"];
    assert_eq!(a.x + a.width, b.x);
}

#[test]
fn reconstructions_follow_the_panes_display_adjustments() {
    let s = setup();
    let surface = s.start();
    s.analyse_viewport(&surface);
    s.hud_view(PANE, "reconstruction");
    let mosaic = crate::plugin::mosaic_layer(1, PANE);
    wait_until("the mosaic", || s.layer_raster_ids(&mosaic) == ["mosaic"]);
    let (index, plain) = {
        let state = s.harness.state();
        let data = &state.layers[&mosaic].rasters["mosaic"].pixels.data;
        let index = data
            .as_chunks::<4>()
            .0
            .iter()
            .position(|px| px[3] == 255)
            .expect("an analysed pixel");
        let px = &data[index * 4..index * 4 + 4];
        (index, [px[0], px[1], px[2]])
    };
    assert_eq!(s.property("adjustment-note"), json!(""));
    // The user raises the pane's gamma: the reconstruction follows (there
    // is no event; the plugin re-reads the pane's display).
    s.harness.state().display = DisplayAdjustments {
        gamma: 2.0,
        modified: true,
        badge: "Display-adjusted".into(),
        ..Default::default()
    };
    let tone = crate::render::DisplayTone {
        gamma: 2.0,
        brightness: 0.0,
        contrast: 1.0,
        sharpness: 0.0,
    };
    let expected = plain.map(|v| (tone.apply_unit(f32::from(v) / 255.0) * 255.0).round() as u8);
    wait_until("the adjusted mosaic", || {
        let state = s.harness.state();
        let data = &state.layers[&mosaic].rasters["mosaic"].pixels.data;
        data[index * 4..index * 4 + 3] == expected
    });
    s.wait_property("the note", "adjustment-note", |v| {
        v.as_str()
            .is_some_and(|t| t.contains("display adjustments"))
    });
    assert_eq!(s.property("adjustment-warning"), json!(false));
    // Stain normalisation cannot be reproduced: the note says so.
    s.harness.state().display = DisplayAdjustments {
        stain_normalization: Some("macenko".into()),
        modified: true,
        badge: "Colour-normalised (Macenko)".into(),
        ..Default::default()
    };
    s.wait_property("the warning", "adjustment-note", |v| {
        v.as_str()
            .is_some_and(|t| t.contains("Macenko normalisation"))
    });
    assert_eq!(s.property("adjustment-warning"), json!(true));
}

#[test]
fn roi_changes_arrive_as_events_without_polling() {
    let s = setup();
    let surface = s.start();
    assert_eq!(s.property("has-roi"), json!(false));
    let polls = s.harness.calls_to("roi.get").len();
    let roi = Rect {
        x: 300.0,
        y: 140.0,
        width: 200.0,
        height: 100.0,
    };
    s.harness.state().roi.insert(1, roi);
    s.harness
        .send(HostEvent::RoiChanged {
            file_id: 1,
            roi: Some(roi),
        })
        .unwrap();
    s.wait_property("ROI button", "has-roi", |v| v == &json!(true));
    // Sidebar refreshes do not ask the host for the ROI.
    for _ in 0..3 {
        s.callback(
            &surface,
            "section-toggled",
            vec![json!("cache"), json!(true)],
        );
        std::thread::sleep(Duration::from_millis(250));
    }
    assert_eq!(s.harness.calls_to("roi.get").len(), polls);
    s.harness.state().roi.clear();
    s.harness
        .send(HostEvent::RoiChanged {
            file_id: 1,
            roi: None,
        })
        .unwrap();
    s.wait_property("no ROI button", "has-roi", |v| v == &json!(false));
}

#[test]
fn a_new_grid_changes_the_layers_min_zoom_in_place() {
    let s = setup();
    let surface = s.start();
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
    let first = s.harness.state().layers[&grid].spec.min_zoom.unwrap();
    // Level 2 (16.004×) instead of level 1: other lines, another min zoom.
    s.callback(&surface, "resolution-selected", vec![json!(3)]);
    wait_until("the new lines", || {
        s.harness.state().objects(&grid).len() == (8 + 1) + (4 + 1)
    });
    let state = s.harness.state();
    let min_zoom = state.layers[&grid].spec.min_zoom.unwrap();
    assert!(min_zoom < first, "{min_zoom} vs {first}");
    assert_eq!(state.calls_to("layer.set_min_zoom").len(), 1);
    let created = state
        .calls_to("layer.create")
        .iter()
        .filter(|call| call["layer_id"] == json!(grid))
        .count();
    assert_eq!(created, 1, "the layer was not recreated");
}

// ---------------------------------------------------------------------------
// Headless batch runs (`eov batch run eovae:analyze`)
// ---------------------------------------------------------------------------

/// The plugin as `eov batch` starts it: no window, slide A published, no
/// panes. `model` false leaves the saved model unset.
fn batch_setup_with(model_name: &str, config: Value, model: bool) -> Setup {
    let dir = tempfile::tempdir().unwrap();
    let model_path = write_fixtures(dir.path(), model_name);
    let mut init = test_init("eovae");
    init.headless = true;
    let mut config = config;
    if model {
        config
            .as_object_mut()
            .unwrap()
            .insert("model_path".into(), json!(model_path.to_string_lossy()));
    }
    init.settings.insert("config".into(), config);
    let (host, requests) = mock_host(dir.path(), Reads::Slide);
    {
        let mut state = host.state();
        state.snapshot.panes.clear();
        state.snapshot.focused_pane = None;
    }
    let harness = Harness::with_host(host, init.clone()).unwrap();
    harness
        .send(HostEvent::Ready {
            restore_sidebar: false,
        })
        .unwrap();
    Setup {
        harness,
        requests,
        dir,
        init,
    }
}

fn batch_setup() -> Setup {
    batch_setup_with("vae.onnx", json!({"device": "cpu"}), true)
}

/// The parameters eov sends: the command's defaults with `given` over them.
fn batch_params<P: Plugin>(harness: &TestHarness<P>, given: Value) -> Map<String, Value> {
    let mut params: Map<String, Value> = harness.state().batch_commands["analyze"]
        .params
        .iter()
        .filter(|p| !p.default.is_null())
        .map(|p| (p.name.clone(), p.default.clone()))
        .collect();
    params.extend(given.as_object().cloned().unwrap_or_default());
    params
}

/// Analyses slide A as `eov batch run` does: open, run, wait for the
/// result, close.
fn run_batch<P: Plugin>(
    harness: &TestHarness<P>,
    run_id: &str,
    given: Value,
    out: &Path,
) -> BatchFinishArgs {
    let file = harness.state().snapshot.files[0].clone();
    let params = batch_params(harness, given);
    harness
        .send(HostEvent::FileOpened { file: file.clone() })
        .unwrap();
    harness
        .send(HostEvent::BatchRun {
            run_id: run_id.into(),
            command_id: "analyze".into(),
            file: file.clone(),
            output_dir: out.to_string_lossy().into_owned(),
            params,
        })
        .unwrap();
    let mut finished = None;
    wait_until("the slide's result", || {
        finished = harness
            .state()
            .batch_finished
            .iter()
            .find(|f| f.run_id == run_id)
            .cloned();
        finished.is_some()
    });
    harness
        .send(HostEvent::FileClosed {
            file_id: file.file_id,
        })
        .unwrap();
    finished.unwrap()
}

impl Setup {
    fn batch_params(&self, given: Value) -> Map<String, Value> {
        batch_params(&self.harness, given)
    }

    fn run_batch(&self, run_id: &str, given: Value, out: &Path) -> BatchFinishArgs {
        run_batch(&self.harness, run_id, given, out)
    }
}

#[test]
fn headless_starts_offer_the_batch_command_and_no_window_ui() {
    let s = batch_setup();
    let state = s.harness.state();
    let spec = &state.batch_commands["analyze"];
    assert!(spec.description.contains("tiles.csv"), "{spec:?}");
    // The saved settings are the defaults, so eov's manifest records them.
    let default = |name: &str| {
        spec.params
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.default.clone())
            .unwrap()
    };
    assert!(default("model").as_str().unwrap().ends_with("vae.onnx"));
    assert_eq!(default("device"), json!("cpu"));
    assert_eq!(default("cache"), json!("off"));
    // Nothing for a window.
    assert!(state.toolbar_buttons.is_empty() && state.hud_buttons.is_empty());
    assert!(state.surfaces.is_empty());
}

#[test]
fn a_batch_run_writes_tile_scores_a_heat_map_and_a_summary() {
    let s = batch_setup();
    let out = tempfile::tempdir().unwrap();
    let finish = s.run_batch("analyze-1", json!({}), out.path());
    assert!(finish.ok, "{finish:?}");
    assert_eq!(finish.outputs, ["tiles.csv", "heatmap.png", "summary.json"]);
    // tiles.csv: every tile of level 1 (32×16); the right half is glass.
    let csv = std::fs::read_to_string(out.path().join("tiles.csv")).unwrap();
    let mut lines = csv.lines();
    assert_eq!(
        lines.next().unwrap(),
        "col,row,x,y,width,height,width_um,height_um,level,state,background,mae,max_error"
    );
    let rows: Vec<Vec<&str>> = lines.map(|line| line.split(',').collect()).collect();
    assert_eq!(rows.len(), 32 * 16);
    let tissue: Vec<&Vec<&str>> = rows.iter().filter(|r| r[9] == "analysed").collect();
    assert_eq!(tissue.len(), 256);
    assert_eq!(rows.iter().filter(|r| r[10] == "1").count(), 256);
    // Level-0 position and size, and µm at 0.5 µm/px.
    assert_eq!(
        rows[0][..9],
        ["0", "0", "0", "0", "128", "128", "64.00", "64.00", "1"]
    );
    // The test engine is off by 0.02 in [-1, 1]: 2.55 grey levels.
    let mae: f64 = tissue[0][11].parse().unwrap();
    assert!((mae - 0.01).abs() < 0.001, "{mae}");
    // heatmap.png: one pixel per tile, glass transparent.
    let png = std::fs::read(out.path().join("heatmap.png")).unwrap();
    let map = crate::png::decode_rgba(&png).unwrap();
    assert_eq!((map.width, map.height), (32, 16));
    assert_eq!(map.rgba[3], 255, "tissue");
    assert_eq!(map.rgba[16 * 4 + 3], 0, "glass");
    assert!(
        map.text
            .iter()
            .any(|(k, v)| k == "eovae:level0_px_per_pixel" && v == "128,128"),
        "{:?}",
        map.text
    );
    // summary.json, the same as in eov's batch.json.
    let summary: Value =
        serde_json::from_str(&std::fs::read_to_string(out.path().join("summary.json")).unwrap())
            .unwrap();
    assert_eq!(summary, finish.summary);
    assert_eq!(summary["complete"], true);
    assert_eq!(summary["tiles"]["analysed"], 256);
    assert_eq!(summary["tiles"]["background"], 256);
    assert_eq!(summary["tiles"]["tile_um"], json!([64.0, 64.0]));
    assert_eq!(
        summary["resolution"]["resampled"], false,
        "level 1 is 2 µm/px"
    );
    assert_eq!(summary["resolution"]["level"], 1);
    assert_eq!(summary["resolution"]["model_mpp"], 2.0);
    assert_eq!(summary["device"]["used"], "cpu");
    assert_eq!(
        summary["device"]["batch_size"], 4,
        "the model's fixed batch"
    );
    assert_eq!(summary["onnx_runtime"], "test");
    assert_eq!(summary["model"]["file"], "vae.onnx");
    assert_eq!(summary["model"]["sha256"].as_str().unwrap().len(), 64);
    assert_eq!(summary["preprocessing"]["source"], "model metadata");
    assert_eq!(
        summary["heatmap"]["level0_px_per_pixel"],
        json!([128.0, 128.0])
    );
    let scores = &summary["scores"];
    assert_eq!(scores["count"], 256);
    for key in [
        "mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max",
    ] {
        assert!(scores[key].is_number(), "{key}: {scores}");
    }
    // Progress reached eov; every read was at level 1; nothing was saved.
    assert!(!s.harness.state().batch_progress.is_empty());
    let reads = s.reads();
    assert_eq!(reads.len(), 32 * 16);
    assert!(reads.iter().all(|r| r.level == 1));
    assert!(s.harness.calls_to("settings.set").is_empty());
}

#[test]
fn batch_parameters_apply_to_one_run_and_bad_ones_fail_the_slide() {
    let s = batch_setup();
    let out = tempfile::tempdir().unwrap();
    // CUDA is not available in tests: the slide fails and says why.
    let finish = s.run_batch("analyze-1", json!({"device": "cuda"}), out.path());
    assert!(!finish.ok);
    let message = finish.message.unwrap();
    assert!(message.contains("CUDA is not available"), "{message}");
    let finish = s.run_batch("analyze-2", json!({"cache": "sometimes"}), out.path());
    assert!(!finish.ok);
    assert!(finish.message.unwrap().contains("off, store or reuse"));
    // One colour scale for every slide.
    let finish = s.run_batch("analyze-3", json!({"heat_range": "0,0.02"}), out.path());
    assert!(finish.ok, "{finish:?}");
    assert_eq!(finish.summary["heatmap"]["range"], json!([0.0, 0.02]));
    let png = std::fs::read(out.path().join("heatmap.png")).unwrap();
    let rgba = crate::png::decode_rgba(&png).unwrap().rgba;
    let expected = crate::render::colormap(0.5);
    for c in 0..3 {
        assert!(
            (i32::from(rgba[c]) - i32::from(expected[c])).abs() <= 2,
            "{:?} vs {expected:?}",
            &rgba[..4]
        );
    }
    // The run's parameters were not saved as settings.
    drop(s.harness);
}

#[test]
fn batch_runs_without_a_usable_model_say_how_to_choose_one() {
    let s = batch_setup_with("vae.onnx", json!({"device": "cpu"}), false);
    let out = tempfile::tempdir().unwrap();
    let finish = s.run_batch("analyze-1", json!({}), out.path());
    assert!(!finish.ok);
    assert!(
        finish
            .message
            .as_deref()
            .unwrap()
            .contains("--param model="),
        "{finish:?}"
    );
    let missing = s.dir.path().join("missing.onnx");
    let finish = s.run_batch(
        "analyze-2",
        json!({"model": missing.to_string_lossy()}),
        out.path(),
    );
    assert!(!finish.ok);
    assert!(finish.message.unwrap().contains("cannot be used"));
    // Unknown commands are refused, not left waiting.
    let file = s.harness.state().snapshot.files[0].clone();
    s.harness
        .send(HostEvent::BatchRun {
            run_id: "nope-1".into(),
            command_id: "nope".into(),
            file,
            output_dir: out.path().to_string_lossy().into_owned(),
            params: Map::new(),
        })
        .unwrap();
    let finished = s.harness.state().batch_finished.clone();
    let refused = finished.iter().find(|f| f.run_id == "nope-1").unwrap();
    assert!(!refused.ok && refused.message.as_deref().unwrap().contains("analyze"));
}

#[test]
fn batch_results_can_be_stored_for_the_viewer_and_reused() {
    let s = batch_setup();
    let first = tempfile::tempdir().unwrap();
    let finish = s.run_batch("analyze-1", json!({"cache": "store"}), first.path());
    assert!(finish.ok, "{finish:?}");
    let reads = s.reads().len();
    // The same slide and settings again: nothing to compute.
    let second = tempfile::tempdir().unwrap();
    let finish = s.run_batch("analyze-2", json!({"cache": "reuse"}), second.path());
    assert!(finish.ok, "{finish:?}");
    assert_eq!(s.reads().len(), reads, "every tile came from the cache");
    assert_eq!(finish.summary["tiles"]["from_cache"], 512);
    assert_eq!(
        std::fs::read_to_string(first.path().join("tiles.csv")).unwrap(),
        std::fs::read_to_string(second.path().join("tiles.csv")).unwrap()
    );
    // The default computes every tile.
    let third = tempfile::tempdir().unwrap();
    let finish = s.run_batch("analyze-3", json!({}), third.path());
    assert!(finish.ok);
    assert_eq!(finish.summary["tiles"]["from_cache"], 0);
    assert_eq!(s.reads().len(), reads * 2);
}

#[test]
fn shutting_down_ends_a_running_batch_slide() {
    let s = batch_setup_with("hang-vae.onnx", json!({"device": "cpu"}), true);
    let host = s.harness.host().clone();
    let out = tempfile::tempdir().unwrap();
    let file = host.state().snapshot.files[0].clone();
    let params = s.batch_params(json!({}));
    s.harness
        .send(HostEvent::BatchRun {
            run_id: "analyze-1".into(),
            command_id: "analyze".into(),
            file,
            output_dir: out.path().to_string_lossy().into_owned(),
            params,
        })
        .unwrap();
    wait_until("inference", || s.reads().len() >= 4);
    let started = Instant::now();
    drop(s.harness);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "shutdown was prompt"
    );
    // eov heard that the slide was not finished.
    let state = host.state();
    let finish = state
        .batch_finished
        .iter()
        .find(|f| f.run_id == "analyze-1")
        .expect("a result");
    assert!(!finish.ok);
    assert!(
        finish.message.as_deref().unwrap().contains("cancelled"),
        "{finish:?}"
    );
}

/// `analyze` with ONNX Runtime and the HistoVAE reference model:
/// `EOVAE_ORT_LIB=/path/to/libonnxruntime.so.1 cargo test --release
/// batch_runs_with_onnx_runtime -- --ignored` (another model:
/// `EOVAE_BATCH_MODEL`).
#[cfg(feature = "onnx")]
#[test]
#[ignore = "needs ONNX Runtime (EOVAE_ORT_LIB) and the HistoVAE reference model"]
fn batch_runs_with_onnx_runtime() {
    type Real = TestHarness<Eovae<crate::ort_backend::OrtBackend>>;
    let model = std::env::var_os("EOVAE_BATCH_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("HistoVAE/pretrained/HistoVae_reconstruct.onnx")
        });
    assert!(model.exists(), "no model at {}", model.display());
    let dir = tempfile::tempdir().unwrap();
    write_fixtures(dir.path(), "unused.onnx");
    let mut init = test_init("eovae");
    init.headless = true;
    init.settings.insert(
        "config".into(),
        json!({"model_path": model.to_string_lossy(), "device": "cpu"}),
    );
    let (host, requests) = mock_host(dir.path(), Reads::Slide);
    {
        let mut state = host.state();
        state.snapshot.panes.clear();
        // 0.2 µm/px, finer than the model's 0.227: level 0, resampled.
        let file = &mut state.snapshot.files[0];
        file.mpp_x = Some(0.2);
        file.mpp_y = Some(0.2);
    }
    let harness = Real::with_host(host, init).unwrap();
    harness
        .send(HostEvent::Ready {
            restore_sidebar: false,
        })
        .unwrap();
    let out = tempfile::tempdir().unwrap();
    let started = Instant::now();
    let finish = run_batch(&harness, "analyze-1", json!({}), out.path());
    assert!(finish.ok, "{finish:?}");
    let summary = &finish.summary;
    println!(
        "{} tiles in {:.1} s on {}, ONNX Runtime {}: MAE mean {}",
        summary["tiles"]["analysed"],
        started.elapsed().as_secs_f64(),
        summary["device"]["label"],
        summary["onnx_runtime"],
        summary["scores"]["mean"]
    );
    // 256 px of 0.227 µm are 290.56 level-0 px: 15×8 tiles, 7 tissue
    // columns and one on the edge of the glass.
    let analysed = summary["tiles"]["analysed"].as_u64().unwrap();
    let background = summary["tiles"]["background"].as_u64().unwrap();
    assert_eq!(analysed + background, 120);
    assert!(analysed >= 56, "{analysed}");
    assert_eq!(summary["resolution"]["level"], 0);
    assert_eq!(summary["resolution"]["resampled"], true);
    assert_eq!(summary["resolution"]["mismatched"], false);
    assert_eq!(summary["model"]["card"], "HistoVAE reference (CAMELYON17)");
    assert!(
        summary["onnx_runtime"]
            .as_str()
            .is_some_and(|v| v.starts_with("1.")),
        "{summary}"
    );
    let mean = summary["scores"]["mean"].as_f64().unwrap();
    assert!(mean > 0.0 && mean < 0.5, "{mean}");
    assert_eq!(requests.lock().unwrap().len(), 120);
    for file in ["tiles.csv", "heatmap.png", "summary.json"] {
        assert!(out.path().join(file).is_file(), "{file}");
    }
}

/// VAE-15: a model trained at 1.6 µm/px gets windows of 32 × 1.6 = 51.2 µm
/// on a 0.5 and a 0.25 µm/px slide alike, read from the coarsest level that
/// is still finer than the model and resampled to its pixels.
#[test]
fn slides_at_other_resolutions_feed_windows_of_the_same_physical_size() {
    for (slide_mpp, level, downsample) in [(0.5, 0, 1.0), (0.25, 1, 4.0)] {
        let s = batch_setup();
        std::fs::write(s.dir.path().join("vae.onnx.eovae.json"), r#"{"mpp": 1.6}"#).unwrap();
        {
            let mut state = s.harness.state();
            let file = &mut state.snapshot.files[0];
            file.mpp_x = Some(slide_mpp);
            file.mpp_y = Some(slide_mpp);
        }
        let out = tempfile::tempdir().unwrap();
        let finish = s.run_batch("analyze-1", json!({}), out.path());
        assert!(finish.ok, "{finish:?}");
        let resolution = &finish.summary["resolution"];
        assert_eq!(resolution["resampled"], true, "{resolution}");
        assert_eq!(resolution["level"], level);
        assert_eq!(resolution["mpp"], 1.6);
        assert_eq!(resolution["mismatched"], false);
        let tile_um = &finish.summary["tiles"]["tile_um"];
        for axis in 0..2 {
            assert!(
                (tile_um[axis].as_f64().unwrap() - 51.2).abs() < 1e-9,
                "{tile_um}"
            );
        }
        // Every region read spans 51.2 µm, give or take the pixels on its
        // edges.
        let reads = s.reads();
        assert!(!reads.is_empty());
        let pixel_um = downsample * slide_mpp;
        for read in &reads {
            assert_eq!(read.level, level);
            for extent in [read.width, read.height] {
                let um = f64::from(extent) * pixel_um;
                assert!(
                    (um - 51.2).abs() <= 2.0 * pixel_um,
                    "{um} µm at {slide_mpp}"
                );
            }
        }
        // The CSV's tiles are 51.2 µm too (whole tiles inside the slide).
        let csv = std::fs::read_to_string(out.path().join("tiles.csv")).unwrap();
        let first: Vec<&str> = csv.lines().nth(1).unwrap().split(',').collect();
        assert_eq!(first[6..8], ["51.20", "51.20"], "{first:?}");
    }
}

/// VAE-15 end to end: on a slide whose colour encodes the level-0 position
/// (a sawtooth of 4 grey levels per pixel, period 64, in red along x and
/// green along y), resampled tiles (a model at 1.6 µm/px, square and
/// non-square slide pixels) put every reconstructed pixel back where it
/// was read: the mosaic's colour at a point is the colour at the centre of
/// the model pixel there (plus the test engine's 2.55 grey levels).
#[test]
fn resampled_pixels_land_where_they_belong() {
    fn saw(at: f64) -> f64 {
        at.rem_euclid(64.0) * 4.0
    }
    for (mpp_x, mpp_y) in [(0.5, 0.5), (0.5, 0.4)] {
        let dir = tempfile::tempdir().unwrap();
        let model_path = write_fixtures(dir.path(), "vae.onnx");
        std::fs::write(dir.path().join("vae.onnx.eovae.json"), r#"{"mpp": 1.6}"#).unwrap();
        let mut init = test_init("eovae");
        init.settings.insert(
            "config".into(),
            json!({"device": "cpu", "model_path": model_path.to_string_lossy()}),
        );
        let host = MockHost::new();
        let requests = Arc::new(Mutex::new(Vec::new()));
        {
            let mut state = host.state();
            let mut file = slide_info(1, &dir.path().join("slide-a.svs"));
            file.mpp_x = Some(mpp_x);
            file.mpp_y = Some(mpp_y);
            let downsamples: Vec<f64> = file.levels.iter().map(|l| l.downsample).collect();
            state.snapshot.files.push(file);
            state.snapshot.panes.push(PaneInfo {
                pane: PANE,
                index: 0,
                file_id: Some(1),
                view: Some(view(1, 0.0, 0.0, 1000.0, 1000.0)),
            });
            state.snapshot.focused_pane = Some(PANE);
            let log = Arc::clone(&requests);
            // OpenSlide's semantics: level pixel floor(x / downsample) first.
            state.region_reader = Some(Box::new(move |request: &RegionRequest| {
                log.lock().unwrap().push(*request);
                let ds = downsamples[request.level as usize];
                let (ox, oy) = (
                    (request.x as f64 / ds).floor(),
                    (request.y as f64 / ds).floor(),
                );
                let mut data = Vec::with_capacity((request.width * request.height * 4) as usize);
                for j in 0..request.height {
                    for i in 0..request.width {
                        let x = (ox + f64::from(i) + 0.5) * ds;
                        let y = (oy + f64::from(j) + 0.5) * ds;
                        data.extend_from_slice(&[saw(x) as u8, saw(y) as u8, 128, 255]);
                    }
                }
                Ok(eov_plugin_api::sdk::Image {
                    width: request.width,
                    height: request.height,
                    data,
                })
            }));
        }
        let s = Setup {
            harness: Harness::with_host(host, init.clone()).unwrap(),
            requests,
            dir,
            init,
        };
        let surface = s.start();
        s.wait_property("resampling", "resolution-note", |v| {
            v.as_str().is_some_and(|t| t.contains("resampled"))
        });
        s.analyse_viewport(&surface);
        s.hud_view(PANE, "reconstruction");
        let mosaic = crate::plugin::mosaic_layer(1, PANE);
        wait_until("the mosaic", || s.layer_raster_ids(&mosaic) == ["mosaic"]);
        let state = s.harness.state();
        let raster = &state.layers[&mosaic].rasters["mosaic"];
        let (w, h) = (
            f64::from(raster.pixels.width),
            f64::from(raster.pixels.height),
        );
        // Level-0 pixels per model pixel, and the centre of the model pixel
        // containing a point.
        let (scale_x, scale_y) = (1.6 / mpp_x, 1.6 / mpp_y);
        let centre = |at: f64, scale: f64| {
            let step = 32.0 * scale;
            let tile = (at / step).floor() * step;
            tile + (((at - tile) / scale).floor() + 0.5) * scale
        };
        let mut checked = 0;
        for my in (5..raster.pixels.height as usize).step_by(37) {
            for mx in (3..raster.pixels.width as usize).step_by(41) {
                let px = &raster.pixels.data[(my * raster.pixels.width as usize + mx) * 4..][..4];
                let x = raster.x + (mx as f64 + 0.5) * raster.width / w;
                let y = raster.y + (my as f64 + 0.5) * raster.height / h;
                let (cx, cy) = (centre(x, scale_x), centre(y, scale_y));
                // Away from the sawtooth's jumps (a model pixel spans up to
                // 4 level-0 pixels).
                let safe = |at: f64| (8.0..56.0).contains(&at.rem_euclid(64.0));
                if px[3] != 255 || !safe(cx) || !safe(cy) {
                    continue;
                }
                for (value, at) in [(px[0], cx), (px[1], cy)] {
                    let expected = saw(at) + 2.55;
                    assert!(
                        (f64::from(value) - expected).abs() <= 2.5,
                        "{mpp_x}/{mpp_y}: {value} at {at:.2} (expected {expected:.1})"
                    );
                }
                checked += 1;
            }
        }
        assert!(checked >= 20, "{checked} points checked");
    }
}

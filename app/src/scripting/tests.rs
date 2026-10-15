//! End to end: a headless window with a real UI host and slides, the
//! endpoint served on a temporary folder, and clients that speak the
//! protocol from its description alone, as `clients/python/eov_client.py`
//! does. That client runs too when `python3` is available.

use super::{Service, Timing, UiJob, UiRunner};
use crate::AppWindow;
use crate::host_ui::{self, InitOptions};
use crate::instance::{self, Claim, Endpoint, Handlers, Transport};
use crate::launch::OpenRequest;
use crate::state::{AppState, NewOpenFile, PaneId, TabActivation};
use crate::test_support::run_on_slint_ui_test_thread;
use eov_common::stain::StainPreset;
use eov_common::{TileCache, TileManager, ViewportState, WsiFile};
use eov_plugin_api::protocol::HostEvent;
use hmac::{Hmac, Mac};
use parking_lot::RwLock;
use serde_json::{Value, json};
use sha2::Sha256;
use slint::Timer;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

/// A protocol client written from the wire format alone.
struct WireClient {
    stream: Box<dyn Stream>,
    next_id: u64,
}

trait Stream: Read + Write + Send {}
impl<T: Read + Write + Send> Stream for T {}

fn hmac(secret: &[u8], role: &[u8], nonce: &[u8]) -> Vec<u8> {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret).unwrap();
    mac.update(role);
    mac.update(nonce);
    mac.finalize().into_bytes().to_vec()
}

impl WireClient {
    /// Reads `<name>.info` ("PORT PID SECRET"), connects to the socket next
    /// to it (port 0) or to the port, and proves the secret after eov did.
    fn connect(info: &Path) -> Self {
        let text = std::fs::read_to_string(info).unwrap();
        let fields: Vec<&str> = text.split_whitespace().collect();
        let port: u16 = fields[0].parse().unwrap();
        let secret: Vec<u8> = (0..fields[2].len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&fields[2][i..i + 2], 16).unwrap())
            .collect();
        assert_eq!(secret.len(), 32);
        let mut stream: Box<dyn Stream> = if port == 0 {
            #[cfg(unix)]
            {
                let socket =
                    std::os::unix::net::UnixStream::connect(info.with_extension("sock")).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(60)))
                    .unwrap();
                Box::new(socket)
            }
            #[cfg(not(unix))]
            unreachable!("Unix-domain sockets are Unix only")
        } else {
            let socket = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(60)))
                .unwrap();
            Box::new(socket)
        };
        let nonce = *uuid::Uuid::new_v4().as_bytes();
        let nonce = [nonce, *uuid::Uuid::new_v4().as_bytes()].concat();
        stream.write_all(b"eov-rpc1").unwrap();
        stream.write_all(&nonce).unwrap();
        let mut answer = [0u8; 64];
        stream.read_exact(&mut answer).unwrap();
        assert_eq!(
            answer[..32],
            hmac(&secret, b"rpc-server", &nonce)[..],
            "eov proves the secret"
        );
        stream
            .write_all(&hmac(&secret, b"rpc-client", &answer[32..]))
            .unwrap();
        Self { stream, next_id: 1 }
    }

    /// Sends `body` as one frame and returns the answer's frame.
    fn exchange(&mut self, body: &[u8]) -> Value {
        self.stream
            .write_all(&(body.len() as u32).to_be_bytes())
            .unwrap();
        self.stream.write_all(body).unwrap();
        let mut length = [0u8; 4];
        self.stream.read_exact(&mut length).unwrap();
        let mut answer = vec![0u8; u32::from_be_bytes(length) as usize];
        self.stream.read_exact(&mut answer).unwrap();
        serde_json::from_slice(&answer).unwrap()
    }

    fn call(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        let answer = self.exchange(
            &serde_json::to_vec(&json!({
                "jsonrpc": "2.0", "id": id, "method": method, "params": params
            }))
            .unwrap(),
        );
        assert_eq!(answer["id"], json!(id), "{answer}");
        assert_eq!(answer["jsonrpc"], json!("2.0"));
        answer
    }

    fn ok(&mut self, method: &str, params: Value) -> Value {
        let answer = self.call(method, params.clone());
        assert!(
            answer.get("error").is_none(),
            "{method} {params} failed: {answer}"
        );
        answer["result"].clone()
    }

    /// The error kind of a call that must fail.
    fn error(&mut self, method: &str, params: Value) -> String {
        let answer = self.call(method, params.clone());
        let error = answer
            .get("error")
            .unwrap_or_else(|| panic!("{method} {params} should fail: {answer}"));
        assert!(error["code"].is_i64() && error["message"].is_string());
        error["data"]["kind"].as_str().unwrap().to_string()
    }
}

fn decode(data: &Value) -> Vec<u8> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(data.as_str().unwrap())
        .unwrap()
}

/// DAB-positive and -negative colours of the H-DAB model.
fn ihc_colours() -> ([u8; 3], [u8; 3]) {
    let vectors = StainPreset::HematoxylinDab.basis().vectors();
    let pixel = |conc: [f32; 2]| {
        let mut rgb = [0u8; 3];
        for (c, out) in rgb.iter_mut().enumerate() {
            let od = vectors[0][c] * conc[0] + vectors[1][c] * conc[1];
            *out = (255.0 * (-od).exp()).round() as u8;
        }
        rgb
    };
    (pixel([0.3, 0.7]), pixel([0.8, 0.0]))
}

/// What the clients know about the test.
struct Fixtures {
    info: PathBuf,
    /// Open in the primary pane: 1024 × 512, 0.25 µm/px, three levels,
    /// DAB-positive left of x = 512.
    slide: PathBuf,
    slide_id: i64,
    pane: u32,
    /// Not open: 512 × 512, one level.
    other: PathBuf,
    /// Not a slide.
    text: PathBuf,
    out: PathBuf,
}

/// Runs jobs on the test's UI thread loop; drains Slint's event queue
/// (slides finish opening through it) without an event loop.
fn drain_slint_events() {
    if slint::quit_event_loop().is_ok() {
        let _ = slint::run_event_loop();
    }
}

#[test]
fn scripts_drive_a_running_eov() {
    run_on_slint_ui_test_thread(|| {
        // Events other tests left for their hosts run before this one has
        // any.
        drain_slint_events();
        struct ClearHost;
        impl Drop for ClearHost {
            fn drop(&mut self) {
                host_ui::clear_host();
            }
        }
        let _clear = ClearHost;
        let dir = tempfile::tempdir().unwrap();
        let (positive, negative) = ihc_colours();
        let slide = dir.path().join("ihc.tif");
        eov_common::test_support::write_tiff(
            &slide,
            &eov_common::test_support::TiffSpec {
                pixels_per_cm: Some(40_000),
                ..eov_common::test_support::TiffSpec::pyramid(1024, 512, 256, 3)
            },
            |x, _| if x < 512 { positive } else { negative },
        )
        .unwrap();
        let other = dir.path().join("other.tif");
        eov_common::test_support::write_tiled_tiff(&other, 512, 512, 256, |_, _| [90, 60, 30])
            .unwrap();
        let text = dir.path().join("notes.txt");
        std::fs::write(&text, "not a slide").unwrap();
        let out = dir.path().join("out");
        std::fs::create_dir(&out).unwrap();

        let ui = AppWindow::new().unwrap();
        let state = Arc::new(RwLock::new(AppState::new()));
        let tile_cache = Arc::new(TileCache::with_limits(256, 256 << 20));
        let timer = Rc::new(Timer::default());
        let plugins = tempfile::tempdir().unwrap();
        let host = host_ui::init(
            &ui,
            &state,
            &tile_cache,
            &timer,
            InitOptions {
                plugin_dir: plugins.path().to_path_buf(),
                safe_mode: true,
            },
        );
        let slide_id = {
            let mut state = state.write();
            // Far from other tests' file ids, whose late events may reach
            // this host.
            for _ in 0..1000 {
                state.allocate_file_id();
            }
            let id = state.allocate_file_id();
            let wsi = WsiFile::open(&slide).unwrap();
            let tile_manager = Arc::new(
                TileManager::new(wsi.with_synthetic_levels(), id)
                    .with_tile_cache(Arc::clone(&tile_cache)),
            );
            let tile_loader = Arc::new(crate::tile_loader::TileLoader::new(
                Arc::clone(&tile_manager),
                Arc::clone(&tile_cache),
            ));
            state.add_file(NewOpenFile {
                id,
                path: slide.clone(),
                wsi,
                tile_manager,
                tile_loader,
                viewport: ViewportState::new(400.0, 300.0, 1024.0, 512.0),
                thumbnail: None,
                target_pane: PaneId::PRIMARY,
                activation: TabActivation::Focus,
            });
            id
        };
        host.publish_now();
        let pane = state.read().pane_uid(PaneId::PRIMARY).unwrap();

        // The endpoint, served as main.rs serves it, but answering on this
        // loop instead of Slint's event loop.
        #[cfg(unix)]
        let transport = Transport::Unix;
        #[cfg(not(unix))]
        let transport = Transport::Tcp;
        let endpoint = Endpoint::new(dir.path().join("run"), "eov-e2e".into(), transport);
        let instance_timing = instance::Timing {
            idle: Duration::from_secs(1),
            ..instance::Timing::default()
        };
        let Claim::Primary(mut server) =
            instance::claim(&endpoint, &OpenRequest::default(), instance_timing)
        else {
            panic!("the endpoint is free");
        };
        let (jobs, queue) = mpsc::channel::<UiJob>();
        let jobs = parking_lot::Mutex::new(jobs);
        let runner: UiRunner = Arc::new(move |job| {
            let _ = jobs.lock().send(job);
        });
        let service = Arc::new(Service::new(
            true,
            runner,
            Arc::clone(&host.shared),
            Timing {
                answer: Duration::from_secs(20),
                open: Duration::from_secs(60),
            },
        ));
        server.serve(
            instance_timing,
            Handlers {
                open: Arc::new(|_, answer| {
                    let _ = answer.send(Err("not in this test".to_string()));
                }),
                script: Some(Arc::new(move |connection| service.serve(connection))),
            },
        );

        let fixtures = Fixtures {
            info: dir.path().join("run").join("eov-e2e.info"),
            slide: crate::paths::canonical(&slide),
            slide_id: i64::from(slide_id),
            pane,
            other: crate::paths::canonical(&other),
            text,
            out: out.clone(),
        };
        let python = python_commands(&fixtures);
        let client = std::thread::spawn(move || {
            wire_session(&fixtures);
            python.map(|commands| commands.map(run_for_at_most_two_minutes))
        });
        let deadline = Instant::now() + Duration::from_secs(300);
        while !client.is_finished() {
            assert!(Instant::now() < deadline, "the scripts did not finish");
            if let Ok(job) = queue.recv_timeout(Duration::from_millis(2)) {
                job();
            }
            drain_slint_events();
        }
        let python = match client.join() {
            Ok(python) => python,
            Err(panic) => std::panic::resume_unwind(panic),
        };
        if let Some([tests, example]) = python {
            let report = String::from_utf8_lossy(&tests.stderr);
            assert!(
                tests.status.success(),
                "the Python client's tests failed:\n{}\n{report}",
                String::from_utf8_lossy(&tests.stdout),
            );
            for test in ["test_a_script_drives_eov", "test_names_match_eov"] {
                assert!(
                    report
                        .lines()
                        .any(|line| line.starts_with(test) && line.ends_with("ok")),
                    "{test} did not run:\n{report}"
                );
            }
            let printed = String::from_utf8_lossy(&example.stdout);
            assert!(
                example.status.success() && printed.contains("exported"),
                "the example failed:\n{printed}\n{}",
                String::from_utf8_lossy(&example.stderr)
            );
            assert!(out.join("other-view.png").is_file(), "{printed}");
        } else {
            eprintln!("python3 is not available: the Python client was not run");
        }

        // What the scripts did went through the viewer's own paths.
        let file_id = i64::from(slide_id);
        assert!(
            host.undo.borrow().can_undo(Some(file_id)),
            "scripted edits can be undone"
        );
        let events = host.queued_events();
        assert!(
            events.iter().any(|event| matches!(
                event,
                HostEvent::RoiChanged { file_id: id, roi: Some(_) } if *id == file_id
            )),
            "plugins hear RoiChanged"
        );
        let key = WsiFile::open(&slide)
            .unwrap()
            .identity()
            .unwrap()
            .key()
            .to_string();
        let saved = crate::measurement_store::load(&key).expect("the ROI is kept for the slide");
        assert!(saved.roi.is_some());
        let names: Vec<String> = crate::bookmarks::list(&key)
            .into_iter()
            .map(|bookmark| bookmark.name)
            .collect();
        assert!(names.contains(&"Tumour edge".to_string()), "{names:?}");
        // Ctrl+Z undoes the last scripted edit.
        let before = state.read().get_file(slide_id).unwrap().measurements.len();
        state.write().set_focused_pane(PaneId::PRIMARY);
        host.undo();
        let after = state.read().get_file(slide_id).unwrap().measurements.len();
        assert_ne!(before, after, "the undo changed the measurements");
        drop(server);
        host.shutdown();
    });
}

/// Runs `command` and collects its output; a command that has not finished
/// after two minutes is killed (and fails).
fn run_for_at_most_two_minutes(mut command: std::process::Command) -> std::process::Output {
    use std::process::Stdio;
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Read both pipes while it runs, so a chatty child never blocks.
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let out = std::thread::spawn(move || {
        let mut text = Vec::new();
        let _ = stdout.read_to_end(&mut text);
        text
    });
    let err = std::thread::spawn(move || {
        let mut text = Vec::new();
        let _ = stderr.read_to_end(&mut text);
        text
    });
    let deadline = Instant::now() + Duration::from_secs(120);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            break child.wait().unwrap();
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    std::process::Output {
        status,
        stdout: out.join().unwrap(),
        stderr: err.join().unwrap(),
    }
}

/// `python3 clients/python/test_eov_client.py`, then
/// `clients/python/example.py`, against the test's eov, if `python3` runs
/// here. The example finds the endpoint as scripts do, by looking in the
/// runtime folder.
fn python_commands(fixtures: &Fixtures) -> Option<[std::process::Command; 2]> {
    let available = std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        return None;
    }
    let clients = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../clients/python");
    let home = crate::paths::home_dir();
    let config = home.join("python.toml");
    let plugins = home.join("python-plugins");
    let info = Endpoint::for_setup(&config, &plugins).info_file();
    let name = info.file_stem().unwrap().to_string_lossy().into_owned();
    let mut tests = std::process::Command::new("python3");
    tests
        .arg(clients.join("test_eov_client.py"))
        .env("EOV_TEST_INFO", &fixtures.info)
        .env("EOV_TEST_SLIDE", &fixtures.slide)
        .env("EOV_TEST_OUT", &fixtures.out)
        .env("EOV_TEST_IDLE", "1")
        .env("EOV_TEST_HOME", &home)
        .env("EOV_TEST_CONFIG", &config)
        .env("EOV_TEST_PLUGINS", &plugins)
        .env("EOV_TEST_NAME", name)
        .env("PYTHONDONTWRITEBYTECODE", "1");
    let mut example = std::process::Command::new("python3");
    example
        .arg(clients.join("example.py"))
        .arg(&fixtures.other)
        .arg(&fixtures.out)
        .env("EOV_RUNTIME_DIR", fixtures.info.parent().unwrap())
        .env("PYTHONDONTWRITEBYTECODE", "1");
    Some([tests, example])
}

/// Every method, its errors and its limits, over one connection (and a
/// second one at the same time).
fn wire_session(fx: &Fixtures) {
    let mut eov = WireClient::connect(&fx.info);
    let slide = fx.slide_id;
    let (positive, negative) = ihc_colours();

    // -- the viewer --
    let version = eov.ok("version", json!({}));
    assert_eq!(version["eov"], json!(crate::version::BUILD_VERSION));
    assert_eq!(version["protocol"], json!(1));
    assert_eq!(version["methods"], json!(super::METHODS));
    assert_eq!(
        version["limits"]["max_answer_bytes"],
        json!(64 << 20),
        "{version}"
    );

    let slides = eov.ok("slides", json!({}));
    let listed = slides.as_array().unwrap();
    assert_eq!(listed.len(), 1, "{slides}");
    let info = &listed[0];
    assert_eq!(info["file_id"], json!(slide));
    assert_eq!(info["filename"], json!("ihc.tif"));
    assert_eq!(info["path"], json!(fx.slide.to_string_lossy()));
    assert_eq!(
        (info["width"].clone(), info["height"].clone()),
        (json!(1024), json!(512))
    );
    assert_eq!(info["levels"].as_array().unwrap().len(), 3);
    assert!((info["mpp_x"].as_f64().unwrap() - 0.25).abs() < 1e-9);
    assert_eq!(info["native_magnification"], json!(40.0));
    assert_eq!(info["panes"], json!([fx.pane]));
    assert!(
        info["identity"]
            .as_str()
            .is_some_and(|key| key.starts_with("eov-slide")),
        "{info}"
    );

    let panes = eov.ok("panes", json!({}));
    assert_eq!(panes.as_array().unwrap().len(), 1);
    assert_eq!(panes[0]["pane"], json!(fx.pane));
    assert_eq!(panes[0]["focused"], json!(true));
    assert_eq!(panes[0]["tabs"], json!([slide]));
    assert_eq!(panes[0]["view"]["file_id"], json!(slide));

    // -- views --
    let view = eov.ok("view.set", json!({"x": 512, "y": 256, "zoom": 0.5}));
    assert_eq!(
        (view["center_x"].clone(), view["center_y"].clone()),
        (json!(512.0), json!(256.0))
    );
    assert_eq!(view["zoom"], json!(0.5));
    assert_eq!(view["magnification"], json!(20.0));
    let view = eov.ok("view.set", json!({"magnification": 10, "rotation": 90}));
    assert_eq!(view["zoom"], json!(0.25));
    assert_eq!(view["rotation_deg"], json!(90.0));
    let view = eov.ok(
        "view.set",
        json!({"pane": fx.pane, "x": "100um", "y": "50um", "rotation": 0}),
    );
    assert_eq!(
        (view["center_x"].clone(), view["center_y"].clone()),
        (json!(400.0), json!(200.0))
    );
    let got = eov.ok("view.get", json!({"pane": fx.pane}));
    assert_eq!(got["center_x"], view["center_x"]);
    assert_eq!(got["zoom"], view["zoom"]);
    assert_eq!(
        eov.error("view.set", json!({"rotation": 45})),
        "invalid_params"
    );
    assert_eq!(eov.error("view.set", json!({})), "invalid_params");
    assert_eq!(eov.error("view.set", json!({"x": 1})), "invalid_params");
    assert_eq!(eov.error("view.get", json!({"pane": 999_999})), "not_found");

    // -- pixels --
    let region = eov.ok(
        "region.read",
        json!({"slide": slide, "x": 500, "y": 0, "width": 24, "height": 2, "format": "rgba"}),
    );
    assert_eq!(region["level"], json!(0));
    let pixels = decode(&region["data"]);
    assert_eq!(pixels.len(), 24 * 2 * 4);
    assert_eq!(&pixels[..3], &positive, "x = 500 is positive");
    assert_eq!(
        &pixels[20 * 4..20 * 4 + 3],
        &negative,
        "x = 520 is negative"
    );
    let png = eov.ok(
        "region.read",
        json!({"x": 0, "y": 0, "width": 64, "height": 32, "mpp": 1.0}),
    );
    assert_eq!(png["format"], json!("png"));
    assert_eq!(
        (png["level"].clone(), png["downsample"].clone()),
        (json!(2), json!(4.0))
    );
    let image = image::load_from_memory(&decode(&png["data"])).unwrap();
    assert_eq!((image.width(), image.height()), (64, 32));
    let by_path = eov.ok(
        "region.read",
        json!({"path": fx.other.to_string_lossy(), "x": 10, "y": 10, "width": 4, "height": 4, "format": "rgb"}),
    );
    assert_eq!(&decode(&by_path["data"])[..3], &[90, 60, 30]);
    assert_eq!(by_path["mpp"], Value::Null);
    let resampled = eov.ok(
        "region.read",
        json!({"slide": slide, "x": 0, "y": 0, "width": 30, "height": 10, "downsample": 3}),
    );
    assert_eq!(resampled["level"], json!(1));
    for (params, kind) in [
        (
            json!({"x": 0, "y": 0, "width": 8192, "height": 8192}),
            "too_large",
        ),
        (
            json!({"x": 0, "y": 0, "width": 4, "height": 4, "level": 7}),
            "invalid_params",
        ),
        (
            json!({"path": "relative.tif", "x": 0, "y": 0, "width": 4, "height": 4}),
            "invalid_params",
        ),
        (
            json!({"path": fx.out.join("missing.tif").to_string_lossy(), "x": 0, "y": 0, "width": 4, "height": 4}),
            "not_found",
        ),
        (
            json!({"path": fx.text.to_string_lossy(), "x": 0, "y": 0, "width": 4, "height": 4}),
            "failed",
        ),
        (
            json!({"slide": slide, "path": fx.other.to_string_lossy(), "x": 0, "y": 0, "width": 4, "height": 4}),
            "invalid_params",
        ),
        (
            json!({"slide": 77, "x": 0, "y": 0, "width": 4, "height": 4}),
            "not_found",
        ),
        (
            json!({"x": 0, "y": 0, "width": 4, "height": 4, "format": "bmp"}),
            "invalid_params",
        ),
    ] {
        assert_eq!(eov.error("region.read", params.clone()), kind, "{params}");
    }
    #[cfg(unix)]
    assert_eq!(
        eov.error(
            "region.read",
            json!({"path": "/dev/zero", "x": 0, "y": 0, "width": 4, "height": 4})
        ),
        "refused"
    );

    // A second connection reads at the same time.
    let second_info = fx.info.clone();
    let second = std::thread::spawn(move || {
        let mut other = WireClient::connect(&second_info);
        (0..5)
            .map(|i| {
                other.ok(
                    "region.read",
                    json!({"slide": slide, "x": i * 100, "y": 0, "width": 50, "height": 50}),
                )["width"]
                    .clone()
            })
            .collect::<Vec<_>>()
    });

    // -- annotations --
    let roi = eov.ok("roi.get", json!({"slide": slide}));
    assert_eq!(roi, json!({"file_id": slide, "roi": null}));
    let roi = eov.ok(
        "roi.set",
        json!({"slide": slide, "x": 256, "y": 0, "width": 512, "height": 256}),
    );
    assert_eq!(
        roi["roi"],
        json!({"x": 256.0, "y": 0.0, "width": 512.0, "height": 256.0})
    );
    assert_eq!(eov.ok("roi.get", json!({}))["roi"], roi["roi"]);
    assert_eq!(
        eov.error("roi.set", json!({"x": 1, "y": 1, "width": 0, "height": 2})),
        "invalid_params"
    );
    assert_eq!(eov.error("roi.set", json!({"x": 1})), "invalid_params");

    let line = eov.ok(
        "measurements.add",
        json!({"mode": "line", "points": [[0, 0], [400, 0]]}),
    );
    assert!(
        (line["length_um"].as_f64().unwrap() - 100.0).abs() < 1e-9,
        "{line}"
    );
    assert_eq!(line["label"], json!("100 µm"));
    assert_eq!(line["mode"], json!("line"));
    let square = eov.ok(
        "measurements.add",
        json!({"slide": slide, "mode": "rectangle", "x": 0, "y": 0, "width": 40, "height": 40}),
    );
    assert!(
        (square["area_um2"].as_f64().unwrap() - 100.0).abs() < 1e-9,
        "{square}"
    );
    let angle = eov.ok(
        "measurements.add",
        json!({"mode": "angle", "points": [[10, 0], [0, 0], [0, 10]]}),
    );
    assert!(
        (angle["angle_deg"].as_f64().unwrap() - 90.0).abs() < 1e-9,
        "{angle}"
    );
    let list = eov.ok("measurements.list", json!({"slide": slide}));
    assert_eq!(list.as_array().unwrap().len(), 3);
    let removed = eov.ok(
        "measurements.remove",
        json!({"ids": [line["measurement_id"].clone()]}),
    );
    assert_eq!(removed["removed"], json!([line["measurement_id"].clone()]));
    assert_eq!(
        eov.ok("measurements.list", json!({}))
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        eov.error("measurements.remove", json!({"ids": ["nope"]})),
        "not_found"
    );
    assert_eq!(
        eov.error(
            "measurements.add",
            json!({"mode": "line", "points": [[0, 0]]})
        ),
        "invalid_params"
    );

    let mark = eov.ok("bookmarks.add", json!({"name": "Tumour edge"}));
    assert_eq!(mark["index"], json!(0));
    assert_eq!(mark["center_x"], json!(400.0), "the view the pane shows");
    let mark = eov.ok(
        "bookmarks.add",
        json!({"x": 100, "y": 100, "magnification": 40}),
    );
    assert_eq!(mark["name"], json!("Bookmark 2"));
    assert_eq!(mark["zoom"], json!(1.0));
    let marks = eov.ok("bookmarks.list", json!({"slide": slide}));
    assert_eq!(marks.as_array().unwrap().len(), 2);
    assert_eq!(marks[1]["magnification"], json!(40.0));
    assert_eq!(
        eov.error("bookmarks.add", json!({"name": "x".repeat(121)})),
        "invalid_params"
    );

    // -- analysis --
    let quantified = eov.ok("quantify", json!({}));
    assert_eq!(quantified["model"], json!("h-dab"));
    assert_eq!(quantified["stain_name"], json!("DAB"));
    assert_eq!(quantified["analysed_pixels"], json!(512 * 256));
    let fraction = quantified["positive_fraction"].as_f64().unwrap();
    assert!((fraction - 0.5).abs() < 0.02, "{quantified}");
    let hematoxylin = eov.ok(
        "quantify",
        json!({"model": "h-dab", "stain": 0, "region": {"x": 600, "y": 0, "width": 100, "height": 100}}),
    );
    assert_eq!(hematoxylin["stain_name"], json!("Hematoxylin"));
    assert_eq!(hematoxylin["region"]["x"], json!(600.0));
    assert_eq!(
        eov.error("quantify", json!({"model": "he-macenko"})),
        "invalid_params"
    );
    assert_eq!(
        eov.error("quantify", json!({"thresholds": [0.5, 0.2, 1.0]})),
        "invalid_params"
    );

    let figure = fx.out.join("figure.png");
    let exported = eov.ok(
        "export.view",
        json!({"path": figure.to_string_lossy(), "dpi": 96, "scale_bar": true}),
    );
    assert_eq!(exported["complete"], json!(true), "{exported}");
    assert_eq!(
        (exported["width"].clone(), exported["height"].clone()),
        (json!(400), json!(300))
    );
    let image = image::open(&figure).unwrap();
    assert_eq!((image.width(), image.height()), (400, 300));
    assert_eq!(
        eov.error("export.view", json!({"path": figure.to_string_lossy()})),
        "refused"
    );
    let replaced = eov.ok(
        "export.view",
        json!({"path": figure.to_string_lossy(), "dpi": 48, "overwrite": true}),
    );
    assert_eq!(replaced["width"], json!(200));
    assert_eq!(
        eov.error(
            "export.view",
            json!({"path": fx.out.join("figure.gif").to_string_lossy()})
        ),
        "invalid_params"
    );
    assert_eq!(
        eov.error(
            "export.view",
            json!({"path": fx.out.join("big.png").to_string_lossy(), "dpi": 2401})
        ),
        "invalid_params"
    );

    // -- opening and closing --
    let opened = eov.ok(
        "open",
        json!({"path": fx.other.to_string_lossy(), "view": {"x": 100, "y": 100, "zoom": 1}}),
    );
    let other_id = opened["file_id"].as_i64().unwrap();
    assert_ne!(other_id, slide);
    assert_eq!(opened["pane"], json!(fx.pane));
    assert_eq!(opened["view"]["zoom"], json!(1.0));
    assert_eq!(eov.ok("slides", json!({})).as_array().unwrap().len(), 2);
    let beside = eov.ok(
        "open",
        json!({"path": fx.slide.to_string_lossy(), "new_pane": true}),
    );
    assert_eq!(beside["file_id"], json!(slide));
    assert_ne!(beside["pane"], json!(fx.pane));
    let panes = eov.ok("panes", json!({}));
    assert_eq!(panes.as_array().unwrap().len(), 2, "{panes}");
    assert_eq!(
        eov.error("open", json!({"path": fx.text.to_string_lossy()})),
        "failed"
    );
    assert_eq!(
        eov.error("open", json!({"path": fx.out.to_string_lossy()})),
        "invalid_params"
    );
    assert_eq!(
        eov.ok("close", json!({"slide": other_id}))["closed"],
        json!(other_id)
    );
    assert_eq!(eov.ok("slides", json!({})).as_array().unwrap().len(), 1);
    assert_eq!(eov.error("close", json!({"slide": other_id})), "not_found");

    // -- the protocol --
    assert_eq!(eov.error("no.such.method", json!({})), "method_not_found");
    assert_eq!(
        eov.error("slides", json!({"verbose": true})),
        "invalid_params"
    );
    let answer = eov.exchange(b"{not json");
    assert_eq!(answer["error"]["data"]["kind"], json!("parse_error"));
    assert_eq!(answer["id"], Value::Null);
    let answer = eov.exchange(br#"{"jsonrpc": "1.0", "id": "a", "method": "version"}"#);
    assert_eq!(answer["error"]["data"]["kind"], json!("invalid_request"));
    assert_eq!(answer["id"], json!("a"));
    let answer = eov.exchange(br#"[{"method": "version"}]"#);
    assert_eq!(answer["error"]["data"]["kind"], json!("invalid_request"));
    let answer = eov.exchange(br#"{"id": "s", "method": "version", "params": [1]}"#);
    assert_eq!(answer["error"]["data"]["kind"], json!("invalid_params"));
    let answer = eov.exchange(br#"{"id": 7, "method": "version"}"#);
    assert_eq!(
        answer["result"]["protocol"],
        json!(1),
        "params and jsonrpc may be left out"
    );

    assert_eq!(
        second.join().unwrap(),
        vec![json!(50); 5],
        "the second connection's reads"
    );
}

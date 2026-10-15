use std::any::Any;
use std::sync::OnceLock;
use std::sync::mpsc;

type UiJob = Box<dyn FnOnce() + Send + 'static>;

fn slint_ui_test_sender() -> &'static mpsc::Sender<UiJob> {
    static SENDER: OnceLock<mpsc::Sender<UiJob>> = OnceLock::new();
    SENDER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<UiJob>();
        std::thread::Builder::new()
            .name("slint-ui-test-thread".to_string())
            .spawn(move || {
                i_slint_backend_testing::init_integration_test_with_mock_time();
                for job in rx {
                    job();
                }
            })
            .expect("failed to spawn Slint UI test thread");
        tx
    })
}

pub(crate) fn run_on_slint_ui_test_thread<R>(f: impl FnOnce() -> R + Send + 'static) -> R
where
    R: Send + 'static,
{
    // Plugins compiled as ordinary dependencies resolve their own files
    // through `eov_common::paths`; isolate before any of them run.
    crate::paths::isolate_for_tests();
    let (tx, rx) = mpsc::sync_channel::<std::thread::Result<R>>(1);
    let job = Box::new(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        let _ = tx.send(result);
    });
    slint_ui_test_sender()
        .send(job)
        .expect("failed to enqueue Slint UI test job");
    match rx.recv().expect("failed to receive Slint UI test result") {
        Ok(result) => result,
        Err(payload) => std::panic::resume_unwind(payload as Box<dyn Any + Send>),
    }
}

/// A UI host in a headless main window with one multichannel OME-TIFF
/// open in the focused pane, installed as `file_ops` installs slides (for
/// tests on the Slint test thread). Dropping it shuts the host down.
pub(crate) struct HostWithSlide {
    pub ui: crate::AppWindow,
    pub host: std::rc::Rc<crate::host_ui::UiHost>,
    pub state: std::sync::Arc<parking_lot::RwLock<crate::state::AppState>>,
    pub tile_cache: std::sync::Arc<eov_common::TileCache>,
    pub file_id: i32,
    pub wsi: eov_common::WsiFile,
    _dirs: (tempfile::TempDir, tempfile::TempDir),
}

impl HostWithSlide {
    /// Channels DAPI, CD45 and CD8 (uint16, 512 x 384, two levels), each
    /// a ramp in x.
    pub(crate) fn new() -> Self {
        use crate::state::{AppState, NewOpenFile};
        use crate::tile_loader::TileLoader;
        use eov_common::test_support::{OmeTiffSpec, write_ome_tiff};
        use eov_common::{TileCache, TileManager, ViewportState, WsiFile};
        use parking_lot::RwLock;
        use std::rc::Rc;
        use std::sync::Arc;

        let ui = crate::AppWindow::new().unwrap();
        let state = Arc::new(RwLock::new(AppState::new()));
        let tile_cache = Arc::new(TileCache::with_limits(4096, 256 << 20));
        let timer = Rc::new(slint::Timer::default());
        let plugin_dir = tempfile::tempdir().unwrap();
        let host = crate::host_ui::init(
            &ui,
            &state,
            &tile_cache,
            &timer,
            crate::host_ui::InitOptions {
                plugin_dir: plugin_dir.path().to_path_buf(),
                safe_mode: true,
            },
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multiplex.ome.tif");
        let spec = OmeTiffSpec::channels(
            512,
            384,
            128,
            2,
            &[
                ("DAPI", [0, 0, 255]),
                ("CD45", [0, 255, 0]),
                ("CD8", [255, 0, 0]),
            ],
        );
        write_ome_tiff(&path, &spec, |c, x, _| f64::from(x) * (c + 1) as f64).unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        let file_id = {
            let mut state = state.write();
            let id = state.allocate_file_id();
            let manager = Arc::new(
                TileManager::new(wsi.with_synthetic_levels(), id)
                    .with_tile_cache(Arc::clone(&tile_cache)),
            );
            let loader = Arc::new(TileLoader::new(
                Arc::clone(&manager),
                Arc::clone(&tile_cache),
            ));
            let target_pane = state.focused_pane;
            state.add_file(NewOpenFile {
                id,
                path: path.clone(),
                wsi: wsi.clone(),
                tile_manager: manager,
                tile_loader: loader,
                viewport: ViewportState::new(800.0, 600.0, 512.0, 384.0),
                thumbnail: None,
                target_pane,
                activation: crate::state::TabActivation::Focus,
            });
            id
        };
        Self {
            ui,
            host,
            state,
            tile_cache,
            file_id,
            wsi,
            _dirs: (plugin_dir, dir),
        }
    }
}

impl Drop for HostWithSlide {
    fn drop(&mut self) {
        self.host.shutdown();
        crate::host_ui::clear_host();
    }
}

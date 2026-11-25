//! Models plugins ask for (`model.fetch`, AUDIT §14.5): answered from
//! eov's model store at once when the model is there; otherwise eov says
//! who asks for what from where, downloads it once the user agrees (with
//! progress and Cancel in the status bar), verifies it and answers every
//! plugin that asked for it meanwhile. The administrator's policy applies
//! as on the command line (`eov model fetch`).

use super::{NotifyLevel, UiHost, with_host};
use crate::model_store::{
    self, DownloadPlan, FetchError, FetchProgress, ModelClient, ModelRequest, ModelStore,
    StoredModel,
};
use crate::plugin_runtime::EventTarget;
use crate::update::policy::redacted_url;
use eov_plugin_api::i18n::{format_decimal, isolate, tr};
use slint::{Timer, TimerMode};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How often the status bar shows a download's progress.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

/// A size as the window shows it: in KiB, MiB, GiB or TiB with one
/// decimal, written with the language's decimal separator, or in bytes
/// below 1 KiB (`crate::cli::format_bytes` writes it for the command line).
fn shown_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024 {
        // Translators: a size in bytes ("512 bytes"), below 1 KiB.
        return tr!("{n} byte" | "{n} bytes" % bytes);
    }
    let mut value = bytes as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{} {}", format_decimal(value, 1), UNITS[unit])
}

/// The models being asked about or downloaded, by the SHA-256 of their
/// main file.
#[derive(Default)]
pub(super) struct ModelFetches {
    pending: HashMap<String, Pending>,
}

struct Pending {
    label: String,
    /// Who asked: (plugin, request id).
    waiters: Vec<(Arc<str>, String)>,
    /// The status-bar job, once the download runs.
    job: Option<String>,
    cancel: Arc<AtomicBool>,
    /// What the download thread last reported.
    progress: Arc<Mutex<Option<FetchProgress>>>,
    timer: Option<Timer>,
}

impl ModelFetches {
    /// Whether a model is being asked about or downloaded.
    #[cfg(test)]
    pub(super) fn is_pending(&self, sha256: &str) -> bool {
        self.pending.contains_key(sha256)
    }
}

impl UiHost {
    /// A plugin's `model.fetch` (UI thread).
    pub(super) fn fetch_model(
        &self,
        plugin_id: &Arc<str>,
        request_id: String,
        request: ModelRequest,
    ) {
        let policy = crate::update::policy::current();
        let store = ModelStore::for_user(policy);
        let target = EventTarget::Plugin(Arc::clone(plugin_id));
        if let Some(model) = store.find_request(&request) {
            self.post(target, model_store::fetched_event(request_id, &Ok(model)));
            return;
        }
        let sha256 = request.main.sha256.clone();
        if let Some(pending) = self.model_fetches.borrow_mut().pending.get_mut(&sha256) {
            // Asked already: this plugin gets the same answer.
            pending.waiters.push((Arc::clone(plugin_id), request_id));
            return;
        }
        let name = self.plugin_name(plugin_id);
        let stored = store.user_model(&sha256);
        let plan = match model_store::plan_download(&request, policy, stored.as_ref()) {
            Ok(plan) => plan,
            Err(error) => {
                self.notify(
                    NotifyLevel::Warning,
                    &tr!("Cannot download {}", isolate(request.label())),
                    Some(&error.to_string()),
                    Some(&name),
                );
                self.post(target, model_store::fetched_event(request_id, &Err(error)));
                return;
            }
        };
        let label = request.label();
        self.model_fetches.borrow_mut().pending.insert(
            sha256.clone(),
            Pending {
                label: label.clone(),
                waiters: vec![(Arc::clone(plugin_id), request_id)],
                job: None,
                cancel: Arc::default(),
                progress: Arc::default(),
                timer: None,
            },
        );
        let urls: Vec<String> = plan
            .files
            .iter()
            .map(|file| redacted_url(&file.url))
            .collect();
        let message = match plan.known_size() {
            Some(bytes) => tr!(
                "{name} asks eov to download {model} ({size}) from {hosts}:\n{urls}\n\nThe files are checked against their SHA-256 and kept in eov's model store ({store}), where every plugin finds them.",
                name = isolate(&name),
                model = isolate(&label),
                size = isolate(shown_size(bytes)),
                hosts = plan.hosts().join(", "),
                urls = urls.join("\n"),
                store = isolate(store.root().display())
            ),
            None => tr!(
                "{name} asks eov to download {model} from {hosts}:\n{urls}\n\nThe files are checked against their SHA-256 and kept in eov's model store ({store}), where every plugin finds them.",
                name = isolate(&name),
                model = isolate(&label),
                hosts = plan.hosts().join(", "),
                urls = urls.join("\n"),
                store = isolate(store.root().display())
            ),
        };
        self.confirm(
            &tr!("Download {}?", isolate(&label)),
            &message,
            &tr!("Button" => "Download"),
            false,
            move |agreed| {
                with_host(|host| host.start_model_download(&sha256, plan, agreed));
            },
        );
    }

    fn start_model_download(&self, sha256: &str, plan: DownloadPlan, agreed: bool) {
        if !agreed {
            self.finish_model_fetch(sha256, Err(FetchError::Cancelled));
            return;
        }
        let (label, cancel, progress) = {
            let fetches = self.model_fetches.borrow();
            let Some(pending) = fetches.pending.get(sha256) else {
                return;
            };
            (
                pending.label.clone(),
                Arc::clone(&pending.cancel),
                Arc::clone(&pending.progress),
            )
        };
        let job = {
            let cancel = Arc::clone(&cancel);
            self.start_host_job(
                &tr!("Downloading {}", isolate(&label)),
                Some(Box::new(move || cancel.store(true, Ordering::SeqCst))),
            )
        };
        let timer = Timer::default();
        {
            let sha256 = sha256.to_string();
            timer.start(TimerMode::Repeated, PROGRESS_INTERVAL, move || {
                with_host(|host| host.show_model_progress(&sha256));
            });
        }
        if let Some(pending) = self.model_fetches.borrow_mut().pending.get_mut(sha256) {
            pending.job = Some(job);
            pending.timer = Some(timer);
        }
        let spawned = std::thread::Builder::new()
            .name("eov-model-download".to_string())
            .spawn({
                let sha256 = sha256.to_string();
                move || {
                    let policy = crate::update::policy::current();
                    let store = ModelStore::for_user(policy);
                    let result = store.fetch(
                        &ModelClient::for_build(),
                        &plan,
                        policy,
                        &|report| {
                            if let Ok(mut last) = progress.lock() {
                                *last = Some(report.clone());
                            }
                        },
                        &cancel,
                    );
                    let _ = slint::invoke_from_event_loop(move || {
                        with_host(|host| host.finish_model_fetch(&sha256, result));
                    });
                }
            });
        if let Err(error) = spawned {
            self.finish_model_fetch(
                sha256,
                Err(FetchError::Local {
                    path: std::path::PathBuf::from("eov-model-download"),
                    message: error.to_string(),
                }),
            );
        }
    }

    fn show_model_progress(&self, sha256: &str) {
        let (job, progress) = {
            let fetches = self.model_fetches.borrow();
            let Some(pending) = fetches.pending.get(sha256) else {
                return;
            };
            let progress = pending.progress.lock().ok().and_then(|p| p.clone());
            (pending.job.clone(), progress)
        };
        let (Some(job), Some(progress)) = (job, progress) else {
            return;
        };
        let done = shown_size(progress.done);
        let message = match progress.total {
            // Translators: how much of a download is done, e.g.
            // "12.3 MiB of 80.0 MiB".
            Some(total) => tr!(
                "{done} of {total}",
                done = isolate(&done),
                total = isolate(shown_size(total))
            ),
            // The status bar puts it after the job's title.
            None => isolate(done),
        };
        self.update_host_job(&job, progress.fraction(), Some(message));
    }

    /// The download (or the question) ended: tell the user and every
    /// plugin that asked.
    fn finish_model_fetch(&self, sha256: &str, result: Result<StoredModel, FetchError>) {
        let Some(pending) = self.model_fetches.borrow_mut().pending.remove(sha256) else {
            return;
        };
        drop(pending.timer);
        if let Some(job) = &pending.job {
            self.finish_host_job(job);
        }
        let names: Vec<String> = pending
            .waiters
            .iter()
            .map(|(plugin, _)| self.plugin_name(plugin))
            .collect();
        match &result {
            Ok(model) => self.notify(
                NotifyLevel::Success,
                &tr!("Downloaded {}", isolate(model.label())),
                Some(&tr!(
                    "Verified against its SHA-256 and kept in {}.",
                    isolate(model.dir.display())
                )),
                names.first().map(String::as_str),
            ),
            Err(error) if error.is_cancelled() => {}
            Err(error) => self.notify(
                NotifyLevel::Error,
                &tr!("Could not download {}", isolate(&pending.label)),
                Some(&error.to_string()),
                names.first().map(String::as_str),
            ),
        }
        for (plugin, request_id) in pending.waiters {
            self.post(
                EventTarget::Plugin(plugin),
                model_store::fetched_event(request_id, &result),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::test_server::{Route, TestServer};
    use crate::plugin_runtime::PluginInfo;
    use crate::plugin_runtime::loader::Module;
    use crate::test_support::run_on_slint_ui_test_thread;
    use eov_plugin_api::prelude::*;
    use sha2::{Digest, Sha256};

    /// What the test plugin asks for when its button is clicked, and the
    /// answers it got.
    static ASK: Mutex<Option<ModelFetchArgs>> = Mutex::new(None);
    static ANSWERS: Mutex<Vec<HostEvent>> = Mutex::new(Vec::new());

    struct Asker {
        host: Host,
    }

    impl Plugin for Asker {
        fn create(host: Host, _init: PluginInit) -> Result<Self, String> {
            Ok(Self { host })
        }

        fn on_event(&self, event: HostEvent) -> Result<(), String> {
            match event {
                HostEvent::ToolbarButtonClicked { .. } => {
                    let args = ASK.lock().unwrap().clone().expect("a request");
                    self.host.fetch_model(args).map_err(String::from)?;
                }
                event @ HostEvent::ModelFetched { .. } => ANSWERS.lock().unwrap().push(event),
                _ => {}
            }
            Ok(())
        }
    }

    /// Runs what worker threads queued for the UI thread.
    fn drain_slint_events() {
        if slint::quit_event_loop().is_ok() {
            let _ = slint::run_event_loop();
        }
    }

    fn click(host: &UiHost) {
        host.post(
            EventTarget::Plugin(Arc::from("asker")),
            HostEvent::ToolbarButtonClicked {
                button_id: "ask".into(),
            },
        );
        host.pump();
        host.pump();
    }

    fn answers() -> Vec<(Option<String>, Option<String>, bool)> {
        ANSWERS
            .lock()
            .unwrap()
            .iter()
            .filter_map(|event| match event {
                HostEvent::ModelFetched {
                    path,
                    error,
                    cancelled,
                    ..
                } => Some((path.clone(), error.clone(), *cancelled)),
                _ => None,
            })
            .collect()
    }

    /// A plugin's request asks the user, downloads with a status-bar job,
    /// answers every plugin that asked, and is then answered from the
    /// store; a declined one is answered as cancelled.
    #[test]
    fn plugins_get_models_after_the_user_agrees() {
        run_on_slint_ui_test_thread(|| {
            drain_slint_events();
            struct ClearHost;
            impl Drop for ClearHost {
                fn drop(&mut self) {
                    super::super::clear_host();
                }
            }
            let _clear = ClearHost;
            let server = TestServer::start();
            let body = b"a model the window downloads".repeat(1000);
            let sha256 = format!("{:x}", Sha256::digest(&body));
            server.route("/models/window.onnx", Route::file(body.clone()));
            let ui = crate::AppWindow::new().unwrap();
            let state = Arc::new(parking_lot::RwLock::new(crate::state::AppState::new()));
            let tile_cache = Arc::new(eov_common::TileCache::with_limits(64, 64 << 20));
            let timer = std::rc::Rc::new(Timer::default());
            let plugin_dir = tempfile::tempdir().unwrap();
            let host = super::super::init(
                &ui,
                &state,
                &tile_cache,
                &timer,
                super::super::InitOptions {
                    plugin_dir: plugin_dir.path().to_path_buf(),
                    safe_mode: false,
                },
            );
            host.start_module(
                PluginInfo {
                    id: "asker".into(),
                    name: "Asker".into(),
                    version: "1.0.0".into(),
                    ..Default::default()
                },
                Module::from_ref(eov_plugin_api::sdk::root_module::<Asker>()).unwrap(),
            )
            .unwrap();
            *ASK.lock().unwrap() = Some(ModelFetchArgs {
                url: Some(server.url("/models/window.onnx")),
                sha256: sha256.clone(),
                name: Some("Window test model".into()),
                ..Default::default()
            });
            ANSWERS.lock().unwrap().clear();

            click(&host);
            assert!(ui.get_plugin_confirm_visible(), "the user is asked first");
            assert!(ui.get_plugin_confirm_title().contains("Window test model"));
            let message = ui.get_plugin_confirm_message().to_string();
            assert!(
                message.contains("Asker") && message.contains("127.0.0.1"),
                "{message}"
            );
            assert!(server.seen().is_empty());
            // Asked again meanwhile: no second question.
            click(&host);
            assert!(host.model_fetches.borrow().is_pending(&sha256));

            host.answer_plugin_confirm(true);
            assert!(!ui.get_plugin_confirm_visible());
            let store = ModelStore::for_user(crate::update::policy::current());
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            while host.model_fetches.borrow().is_pending(&sha256) {
                assert!(std::time::Instant::now() < deadline, "the download ends");
                std::thread::sleep(Duration::from_millis(20));
                drain_slint_events();
            }
            host.pump();
            let stored = store.resolve(&sha256).expect("stored");
            let path = stored.path().to_string_lossy().into_owned();
            assert_eq!(
                answers(),
                [
                    (Some(path.clone()), None, false),
                    (Some(path.clone()), None, false)
                ],
                "both requests are answered"
            );
            // Stored now: answered without asking or downloading.
            let requests = server.seen().len();
            click(&host);
            assert!(!ui.get_plugin_confirm_visible());
            assert_eq!(answers().len(), 3);
            assert_eq!(server.seen().len(), requests);

            // Declined: the plugin hears it was cancelled.
            let other = b"another model".repeat(100);
            server.route("/models/other.onnx", Route::file(other.clone()));
            *ASK.lock().unwrap() = Some(ModelFetchArgs {
                url: Some(server.url("/models/other.onnx")),
                sha256: format!("{:x}", Sha256::digest(&other)),
                ..Default::default()
            });
            click(&host);
            host.answer_plugin_confirm(false);
            host.pump();
            let last = answers().pop().unwrap();
            assert_eq!((last.0, last.2), (None, true));
            host.shutdown();
        });
    }

    /// Sizes are written as the interface writes numbers, and the word
    /// for bytes is the language's.
    #[test]
    fn sizes_follow_the_interface_s_language() {
        use eov_plugin_api::i18n;
        struct English;
        impl Drop for English {
            fn drop(&mut self) {
                i18n::set_catalog(i18n::source(), None);
            }
        }
        assert_eq!(shown_size(0), "0 bytes");
        assert_eq!(shown_size(1), "1 byte");
        assert_eq!(shown_size(1023), "1023 bytes");
        assert_eq!(shown_size(1024), "1.0 KiB");
        assert_eq!(shown_size(12_900_000), "12.3 MiB");
        assert_eq!(shown_size(80 << 20), "80.0 MiB");
        assert_eq!(shown_size(3 << 39), "1.5 TiB");
        let _english = English;
        let german = i18n::by_tag("de").unwrap();
        let messages: [(Option<&str>, &str, &[&str]); 1] =
            [(None, "{n} byte", &["{n} Byte", "{n} Byte"])];
        i18n::set_catalog(
            german,
            Some(i18n::Catalog::from_messages(german, "", messages).unwrap()),
        );
        assert_eq!(shown_size(512), "512 Byte");
        assert_eq!(shown_size(12_900_000), "12,3 MiB");
    }
}

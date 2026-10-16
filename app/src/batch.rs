//! Headless batch runs of plugin analyses (AUDIT §14.5).
//!
//! `eov batch run <plugin>:<command> SLIDES --out DIR` loads one installed
//! plugin without a window (`PluginInit.headless`), sends it each slide as
//! a [`HostEvent::BatchRun`] and waits for its `batch.finish`, then writes
//! `DIR/batch.json`: a manifest of the run with eov's and the plugin's
//! versions, the package's SHA-256, the command and its parameters, and per
//! slide its identity, status, outputs and the plugin's summary (model,
//! hashes, resolution read, …), for reproducible results. `eov batch list`
//! lists the commands installed plugins offer.
//!
//! The plugin gets the host API it has in the window: slide reads,
//! properties, settings, jobs and notifications (printed to stderr). What
//! needs a window (tools, layers, surfaces, the view) is accepted and has no
//! effect.

use crate::host_ui::publish::{FileReader, wsi_file_info};
use crate::interaction::measure::Calibration;
use crate::plugin_runtime::filters::FilterChain;
use crate::plugin_runtime::loader;
use crate::plugin_runtime::settings::SettingsStore;
use crate::plugin_runtime::shared::{Command, HostShared, Published, PublishedFile};
use crate::plugin_runtime::{EventTarget, PluginInfo, PluginRuntime};
use anyhow::{Context, Result, bail};
use eov_common::WsiFile;
use eov_plugin_api::protocol::{
    BatchCommandSpec, BatchFinishArgs, FileId, HostEvent, HostSnapshot, NotifyLevel,
};
use serde::Serialize;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// How long a plugin has to register its batch commands after it starts.
const REGISTRATION_WAIT: Duration = Duration::from_secs(10);

/// `eov batch run` options.
pub(crate) struct RunOptions {
    /// `<plugin>:<command>`.
    pub target: String,
    pub inputs: Vec<PathBuf>,
    pub out: PathBuf,
    /// `name=value`; values are JSON when they parse, else strings.
    pub params: Vec<String>,
    /// Per slide; `None` waits for as long as the plugin works.
    pub timeout: Option<Duration>,
    /// Name slides by opaque ids and write no paths or file names.
    pub deidentify: bool,
}

/// A plugin running without a window.
struct Headless {
    shared: Arc<HostShared>,
    runtime: PluginRuntime,
    /// Set by plugins' calls (`set_wake`), so the loop sleeps until there is
    /// something to do.
    signal: Arc<(Mutex<bool>, Condvar)>,
}

impl Headless {
    fn new() -> Self {
        let shared = Arc::new(HostShared::new(
            Arc::new(crate::overlay::OverlayStore::new()),
            SettingsStore::new(),
            crate::version::BUILD_VERSION,
        ));
        let signal = Arc::new((Mutex::new(false), Condvar::new()));
        let waker = Arc::clone(&signal);
        shared.set_wake(move || {
            let (flag, condvar) = &*waker;
            if let Ok(mut set) = flag.lock() {
                *set = true;
                condvar.notify_all();
            }
        });
        let runtime =
            PluginRuntime::headless(Arc::clone(&shared), Arc::new(FilterChain::default()));
        Self {
            shared,
            runtime,
            signal,
        }
    }

    /// Loads and starts an installed plugin, with the checks the window
    /// makes (deployment policy for third-party plugins, the plugin API
    /// version). Returns its package's SHA-256 and origin.
    fn start(&mut self, plugin_dir: &Path, plugin_id: &str) -> Result<(String, String, String)> {
        let policy = crate::update::policy::current();
        let host = crate::update::compat::host_version();
        let inventory = crate::update::plugins::inventory(plugin_dir, &policy.plugin_dirs, &host)?;
        let Some(package) = inventory.selected(plugin_id) else {
            bail!("no compatible plugin '{plugin_id}' is installed (eov plugin ls)");
        };
        if package.disabled {
            bail!("plugin '{plugin_id}' is disabled (eov plugin enable {plugin_id})");
        }
        if !package.system && package.provenance.kind != "official" {
            policy.ensure_third_party_allowed("Running this plugin")?;
        }
        let descriptor = crate::plugins::discovery::load_descriptor(&package.package_path)
            .map_err(|error| anyhow::anyhow!("could not unpack the package: {error}"))?;
        let manifest = &descriptor.manifest;
        let api = semver::Version::parse(eov_plugin_api::abi::API_VERSION).expect("valid");
        if !manifest.accepts_api(&api) {
            bail!(
                "plugin '{plugin_id}' requires plugin API {}, this eov provides {api}",
                manifest.api.as_deref().unwrap_or("?")
            );
        }
        let info = PluginInfo {
            id: manifest.id.clone(),
            name: manifest.name.clone(),
            version: manifest.version.clone(),
            description: manifest.description.clone(),
            repository: manifest.repository.clone(),
            root: descriptor.root.clone(),
            package: Some(package.package_path.clone()),
        };
        let module = loader::library_path(&descriptor.root, &manifest.id)
            .and_then(|path| loader::load_library(&path));
        self.runtime
            .start(
                info,
                module,
                crate::paths::plugin_data_dir(&manifest.id),
                crate::paths::plugin_cache_dir(&manifest.id),
            )
            .map_err(|error| anyhow::anyhow!("plugin '{plugin_id}' did not start: {error}"))?;
        self.deliver(
            plugin_id,
            &HostEvent::Ready {
                restore_sidebar: false,
            },
        )?;
        Ok((
            manifest.version.clone(),
            package.sha256.clone(),
            package.provenance.describe(),
        ))
    }

    /// Sends an event; a plugin that panics is stopped and reported.
    fn deliver(&mut self, plugin_id: &str, event: &HostEvent) -> Result<()> {
        let target = EventTarget::Plugin(Arc::from(plugin_id));
        for delivery in self.runtime.deliver(&target, event) {
            if let Err(error) = delivery.result
                && error.kind() == eov_plugin_api::abi::FfiErrorKind::Panicked
            {
                bail!("plugin '{plugin_id}' crashed: {}", error.message);
            }
        }
        if !self
            .runtime
            .slot(plugin_id)
            .is_some_and(|slot| slot.is_active())
        {
            let status = self
                .runtime
                .slot(plugin_id)
                .map(|slot| slot.status.label())
                .unwrap_or_default();
            bail!("plugin '{plugin_id}' stopped: {status}");
        }
        Ok(())
    }

    /// Waits up to `timeout` for plugin calls, then handles them: prints
    /// notifications and progress, and returns the finished batch slides.
    fn pump(&mut self, timeout: Duration) -> Vec<BatchFinishArgs> {
        {
            let (flag, condvar) = &*self.signal;
            let mut set = flag.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if !*set && !self.shared.has_pending_commands() {
                set = condvar
                    .wait_timeout(set, timeout)
                    .map(|(guard, _)| guard)
                    .unwrap_or_else(|poisoned| poisoned.into_inner().0);
            }
            *set = false;
        }
        let mut finished = Vec::new();
        for (plugin_id, command) in self.shared.take_commands() {
            match command {
                Command::Notify(args) => {
                    let level = match args.level {
                        NotifyLevel::Error => "error",
                        NotifyLevel::Warning => "warning",
                        _ => "note",
                    };
                    let detail = args.detail.map(|d| format!(": {d}")).unwrap_or_default();
                    eprintln!("{plugin_id}: {level}: {}{detail}", args.message);
                }
                Command::JobStart { args, .. } => eprintln!("{plugin_id}: {}", args.title),
                Command::BatchProgress(args) => {
                    let percent = args
                        .fraction
                        .map(|f| format!(" {:.0} %", 100.0 * f.clamp(0.0, 1.0)))
                        .unwrap_or_default();
                    let message = args.message.map(|m| format!(" {m}")).unwrap_or_default();
                    eprint!("\r  {percent}{message}\u{1b}[K");
                }
                Command::BatchFinish(args) => finished.push(args),
                // Everything else needs a window.
                _ => {}
            }
        }
        finished
    }

    fn batch_commands(&self, plugin_id: Option<&str>) -> Vec<(Arc<str>, BatchCommandSpec)> {
        self.shared.contributions.lock().batch_commands(plugin_id)
    }

    /// Waits until the plugin offers `command_id` (or any command for
    /// `None`), for at most [`REGISTRATION_WAIT`].
    fn wait_for_command(&mut self, plugin_id: &str, command_id: Option<&str>) {
        let deadline = Instant::now() + REGISTRATION_WAIT;
        while Instant::now() < deadline {
            let commands = self.batch_commands(Some(plugin_id));
            let found = match command_id {
                Some(id) => commands.iter().any(|(_, spec)| spec.command_id == id),
                None => !commands.is_empty(),
            };
            if found {
                return;
            }
            self.pump(Duration::from_millis(100));
        }
    }

    fn shutdown(mut self) {
        self.runtime.shutdown_all();
    }
}

/// `eov batch list`: the batch commands of every installed plugin.
pub(crate) fn list(plugin_dir: &Path, json: bool) -> Result<()> {
    let policy = crate::update::policy::current();
    let host = crate::update::compat::host_version();
    let inventory = crate::update::plugins::inventory(plugin_dir, &policy.plugin_dirs, &host)?;
    let mut rows = Vec::new();
    for package in inventory
        .packages
        .iter()
        .filter(|p| p.selected && !p.disabled)
    {
        let mut headless = Headless::new();
        match headless.start(plugin_dir, &package.id) {
            Ok(_) => {
                headless.wait_for_command(&package.id, None);
                rows.extend(headless.batch_commands(Some(&package.id)));
            }
            Err(error) => eprintln!("{}: {error:#}", package.id),
        }
        headless.shutdown();
    }
    if json {
        let commands: Vec<Value> = rows
            .iter()
            .map(|(plugin, spec)| {
                serde_json::json!({
                    "command": format!("{plugin}:{}", spec.command_id),
                    "description": spec.description,
                    "params": spec.params,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&commands)?);
        return Ok(());
    }
    if rows.is_empty() {
        println!("No installed plugin offers batch commands.");
        return Ok(());
    }
    for (plugin, spec) in rows {
        println!("{plugin}:{}  {}", spec.command_id, spec.description);
        for param in spec.params {
            let default = if param.default.is_null() {
                String::new()
            } else {
                format!(" (default {})", param.default)
            };
            println!(
                "    --param {}=…  {}{default}",
                param.name, param.description
            );
        }
    }
    Ok(())
}

/// A `--param` value: JSON when it parses (numbers, booleans, lists), else
/// the text.
fn param_value(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
}

/// The command's defaults with the given `name=value` parameters over
/// them; unknown names are refused.
fn resolve_params(spec: &BatchCommandSpec, given: &[String]) -> Result<Map<String, Value>> {
    let mut params = Map::new();
    for param in &spec.params {
        if !param.default.is_null() {
            params.insert(param.name.clone(), param.default.clone());
        }
    }
    for item in given {
        let Some((name, value)) = item.split_once('=') else {
            bail!("--param {item}: expected name=value");
        };
        let name = name.trim();
        if !spec.params.iter().any(|param| param.name == name) {
            let known: Vec<&str> = spec.params.iter().map(|p| p.name.as_str()).collect();
            bail!(
                "{} takes no parameter '{name}' (it takes: {})",
                spec.command_id,
                if known.is_empty() {
                    "none".to_string()
                } else {
                    known.join(", ")
                }
            );
        }
        params.insert(name.to_string(), param_value(value));
    }
    Ok(params)
}

#[derive(Serialize)]
struct SlideResult {
    slide_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    identity: Option<String>,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    /// Relative to the run's output folder.
    outputs: Vec<String>,
    summary: Value,
    seconds: f64,
}

#[derive(Serialize)]
struct Manifest {
    format_version: u32,
    eov: String,
    plugin: Value,
    command: String,
    params: Map<String, Value>,
    deidentified: bool,
    started_unix: u64,
    finished_unix: Option<u64>,
    slides: Vec<SlideResult>,
}

/// The parameters as `batch.json` records them. In de-identified runs,
/// values that look like local paths (a model, a runtime) keep only their
/// file name: folders often carry user, project or patient names.
fn recorded_params(params: &Map<String, Value>, deidentify: bool) -> Map<String, Value> {
    if !deidentify {
        return params.clone();
    }
    params
        .iter()
        .map(|(name, value)| {
            let value = match value {
                Value::String(text)
                    if text.contains('/') || text.contains('\\') || text.starts_with('~') =>
                {
                    let name = text
                        .rsplit(['/', '\\'])
                        .next()
                        .filter(|name| !name.is_empty())
                        .unwrap_or("(a path)");
                    Value::String(name.to_string())
                }
                other => other.clone(),
            };
            (name.clone(), value)
        })
        .collect()
}

/// An error message without the slide's path or names, for de-identified
/// runs: they are replaced by the slide's id.
fn scrub(message: String, path: &Path, key: &str) -> String {
    let mut message = message.replace(&path.display().to_string(), key);
    let names = [path.file_name(), path.file_stem()]
        .into_iter()
        .chain(path.parent().and_then(Path::file_name).map(Some))
        .flatten();
    for name in names {
        let name = name.to_string_lossy();
        if name.len() > 2 {
            message = message.replace(&*name, key);
        }
    }
    message
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

fn write_manifest(out: &Path, manifest: &Manifest) -> Result<()> {
    let path = out.join("batch.json");
    let temp = out.join("batch.json.tmp");
    std::fs::write(&temp, serde_json::to_string_pretty(manifest)?)
        .with_context(|| format!("could not write {}", temp.display()))?;
    std::fs::rename(&temp, &path).with_context(|| format!("could not write {}", path.display()))
}

/// `eov batch run`. Returns whether every slide succeeded.
pub(crate) fn run(plugin_dir: &Path, options: &RunOptions) -> Result<bool> {
    let Some((plugin_id, _)) = options.target.split_once(':') else {
        bail!("expected <plugin>:<command>, e.g. example_plugin:tissue-stats (eov batch list)");
    };
    let mut headless = Headless::new();
    let package = headless.start(plugin_dir, plugin_id)?;
    run_started(headless, package, options)
}

/// Runs the batch with the plugin started in `headless`; `package` is its
/// version, SHA-256 and origin.
fn run_started(
    mut headless: Headless,
    (version, sha256, origin): (String, String, String),
    options: &RunOptions,
) -> Result<bool> {
    let Some((plugin_id, command_id)) = options.target.split_once(':') else {
        bail!("expected <plugin>:<command>");
    };
    let discovery = eov_common::dataset::discover_slides(&options.inputs);
    for (path, error) in &discovery.errors {
        eprintln!("{}: {error}", path.display());
    }
    if discovery.slides.is_empty() {
        headless.shutdown();
        bail!("no slides found in the inputs");
    }
    headless.wait_for_command(plugin_id, Some(command_id));
    let Some((_, spec)) = headless
        .batch_commands(Some(plugin_id))
        .into_iter()
        .find(|(_, spec)| spec.command_id == command_id)
    else {
        headless.shutdown();
        bail!("plugin '{plugin_id}' offers no batch command '{command_id}' (eov batch list)");
    };
    let params = match resolve_params(&spec, &options.params) {
        Ok(params) => params,
        Err(error) => {
            headless.shutdown();
            return Err(error);
        }
    };

    std::fs::create_dir_all(&options.out)
        .with_context(|| format!("could not create {}", options.out.display()))?;
    let mut manifest = Manifest {
        format_version: 1,
        eov: crate::version::BUILD_VERSION.to_string(),
        plugin: serde_json::json!({
            "id": plugin_id,
            "version": version,
            "sha256": sha256,
            "origin": origin,
        }),
        command: command_id.to_string(),
        params: recorded_params(&params, options.deidentify),
        deidentified: options.deidentify,
        started_unix: unix_now(),
        finished_unix: None,
        slides: Vec::new(),
    };
    crate::audit::record(
        "batch_started",
        serde_json::json!({
            "plugin": plugin_id,
            "version": manifest.plugin["version"],
            "sha256": sha256,
            "command": command_id,
            "slides": discovery.slides.len(),
        }),
    );

    let keys = eov_common::dataset::unique_slide_keys_for(&discovery.slides);
    let mut used = std::collections::HashSet::new();
    let mut all_ok = true;
    let count = discovery.slides.len();
    for (index, (input, key)) in discovery.slides.iter().zip(keys).enumerate() {
        let started = Instant::now();
        let opened = WsiFile::open(&input.path);
        let key = if options.deidentify {
            eov_common::dataset::deidentified_key(opened.as_ref().ok(), &input.path, &mut used)
        } else {
            key
        };
        eprintln!("[{}/{count}] {key}", index + 1);
        let result = match opened {
            Ok(wsi) => run_slide(
                &mut headless,
                plugin_id,
                command_id,
                &params,
                &wsi,
                &input.path,
                &key,
                index,
                options,
            ),
            Err(error) => Err(format!("could not open the slide: {error}")),
        };
        eprint!("\r\u{1b}[K");
        let (status, message, outputs, summary, identity) = match result {
            Ok(done) => {
                let outputs = done
                    .finish
                    .outputs
                    .iter()
                    .map(|file| format!("{key}/{file}"))
                    .collect();
                (
                    if done.finish.ok { "ok" } else { "failed" },
                    done.finish.message,
                    outputs,
                    done.finish.summary,
                    done.identity,
                )
            }
            Err(message) => ("failed", Some(message), Vec::new(), Value::Null, None),
        };
        let message = message.map(|message| {
            if options.deidentify {
                scrub(message, &input.path, &key)
            } else {
                message
            }
        });
        if status != "ok" {
            all_ok = false;
            eprintln!(
                "  failed: {}",
                message.as_deref().unwrap_or("no reason given")
            );
        }
        manifest.slides.push(SlideResult {
            slide_id: key,
            path: (!options.deidentify).then(|| input.path.display().to_string()),
            identity,
            status,
            message,
            outputs,
            summary,
            seconds: started.elapsed().as_secs_f64(),
        });
        write_manifest(&options.out, &manifest)?;
        if headless
            .runtime
            .slot(plugin_id)
            .is_none_or(|slot| !slot.is_active())
        {
            eprintln!("The plugin stopped; the remaining slides are not analysed.");
            all_ok = false;
            break;
        }
    }
    manifest.finished_unix = Some(unix_now());
    write_manifest(&options.out, &manifest)?;
    headless.shutdown();
    let ok = manifest.slides.iter().filter(|s| s.status == "ok").count();
    eprintln!(
        "{ok} of {count} slides analysed; results and batch.json in {}",
        options.out.display()
    );
    Ok(all_ok)
}

struct SlideDone {
    finish: BatchFinishArgs,
    identity: Option<String>,
}

/// Publishes one slide to the plugin, asks for the analysis and waits for
/// its result.
#[allow(clippy::too_many_arguments)]
fn run_slide(
    headless: &mut Headless,
    plugin_id: &str,
    command_id: &str,
    params: &Map<String, Value>,
    wsi: &WsiFile,
    path: &Path,
    key: &str,
    index: usize,
    options: &RunOptions,
) -> Result<SlideDone, String> {
    let file_id = FileId::from(index as i32 + 1);
    let identity = wsi
        .identity()
        .ok()
        .map(|identity| identity.key().to_string());
    let filename = if options.deidentify {
        key.to_string()
    } else {
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| key.to_string())
    };
    let shown_path = if options.deidentify {
        PathBuf::from(key)
    } else {
        path.to_path_buf()
    };
    let props = wsi.properties();
    let info = wsi_file_info(
        file_id,
        &shown_path,
        &filename,
        wsi,
        Calibration::new(props.mpp_x, props.mpp_y),
        identity.as_deref(),
    );
    let output_dir = options.out.join(key);
    std::fs::create_dir_all(&output_dir)
        .map_err(|error| format!("could not create {}: {error}", output_dir.display()))?;

    headless
        .shared
        .set_reader(file_id, Arc::new(FileReader::new(wsi.clone())));
    let mut published = Published {
        snapshot: HostSnapshot {
            host_version: crate::version::BUILD_VERSION.to_string(),
            files: vec![info.clone()],
            ..HostSnapshot::default()
        },
        ..Published::default()
    };
    published.files.insert(
        file_id,
        PublishedFile {
            info: info.clone(),
            properties: Arc::new(wsi.property_map().clone()),
            roi: None,
            measurements: Vec::new(),
            path: shown_path.clone(),
            ..PublishedFile::default()
        },
    );
    headless.shared.publish(published);

    let run_id = format!("{command_id}-{}", index + 1);
    let result = (|| {
        headless
            .deliver(plugin_id, &HostEvent::FileOpened { file: info.clone() })
            .map_err(|error| format!("{error:#}"))?;
        headless
            .deliver(
                plugin_id,
                &HostEvent::BatchRun {
                    run_id: run_id.clone(),
                    command_id: command_id.to_string(),
                    file: info.clone(),
                    output_dir: output_dir.to_string_lossy().into_owned(),
                    params: params.clone(),
                },
            )
            .map_err(|error| format!("{error:#}"))?;
        let deadline = options.timeout.map(|timeout| Instant::now() + timeout);
        loop {
            for finish in headless.pump(Duration::from_millis(200)) {
                if finish.run_id == run_id {
                    return Ok(finish);
                }
            }
            if !headless
                .runtime
                .slot(plugin_id)
                .is_some_and(|slot| slot.is_active())
            {
                return Err("the plugin stopped".to_string());
            }
            if deadline.is_some_and(|deadline| Instant::now() > deadline) {
                return Err("timed out".to_string());
            }
        }
    })();
    let _ = headless.deliver(plugin_id, &HostEvent::FileClosed { file_id });
    headless.shared.remove_reader(file_id);
    headless.shared.publish(Published::default());
    result.map(|finish| SlideDone { finish, identity })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_plugin_api::protocol::BatchParam;

    /// The example plugin's tissue-stats command over two slides, one of
    /// which cannot be read, with a manifest that names no paths.
    #[test]
    fn a_plugin_analyses_slides_without_a_window() {
        crate::paths::isolate_for_tests();
        let dir = tempfile::tempdir().unwrap();
        let slides = dir.path().join("case-42");
        std::fs::create_dir_all(&slides).unwrap();
        eov_common::test_support::write_tiled_tiff(
            &slides.join("tissue.tif"),
            512,
            512,
            256,
            |x, _| {
                if x < 256 {
                    [190, 110, 170]
                } else {
                    [240, 240, 240]
                }
            },
        )
        .unwrap();
        std::fs::write(slides.join("broken.svs"), b"not a slide").unwrap();
        let out = dir.path().join("out");

        let mut headless = Headless::new();
        headless
            .runtime
            .start(
                PluginInfo {
                    id: "example_plugin".into(),
                    name: "Example".into(),
                    version: "1.0.0".into(),
                    root: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("../plugin_api/example-plugin"),
                    ..Default::default()
                },
                loader::Module::from_ref(example_plugin::module()),
                dir.path().join("data"),
                dir.path().join("cache"),
            )
            .unwrap();
        let options = RunOptions {
            target: "example_plugin:tissue-stats".into(),
            inputs: vec![slides.clone()],
            out: out.clone(),
            params: vec!["max_pixels=1000000".into()],
            timeout: Some(Duration::from_secs(60)),
            deidentify: true,
        };
        let all_ok = run_started(
            headless,
            ("1.0.0".into(), "abc".into(), "local".into()),
            &options,
        )
        .unwrap();
        assert!(!all_ok, "the broken slide fails");

        let manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(out.join("batch.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["command"], "tissue-stats");
        assert_eq!(manifest["params"]["max_pixels"], 1_000_000);
        let results = manifest["slides"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        let ok: Vec<&Value> = results.iter().filter(|s| s["status"] == "ok").collect();
        assert_eq!(ok.len(), 1);
        let fraction = ok[0]["summary"]["tissue_fraction"].as_f64().unwrap();
        assert!((fraction - 0.5).abs() < 0.05, "{fraction}");
        let output = ok[0]["outputs"][0].as_str().unwrap();
        assert!(out.join(output).exists());
        // De-identified: neither the case folder nor the file names.
        let text = std::fs::read_to_string(out.join("batch.json")).unwrap();
        assert!(
            !text.contains("case-42") && !text.contains("tissue.tif"),
            "{text}"
        );
    }

    #[test]
    fn parameters_have_defaults_and_unknown_ones_are_refused() {
        let spec = BatchCommandSpec {
            command_id: "count".into(),
            description: String::new(),
            params: vec![
                BatchParam {
                    name: "level".into(),
                    description: String::new(),
                    default: serde_json::json!(2),
                },
                BatchParam {
                    name: "label".into(),
                    description: String::new(),
                    default: Value::Null,
                },
            ],
        };
        let params = resolve_params(&spec, &["label=tumour".into()]).unwrap();
        assert_eq!(params["level"], serde_json::json!(2));
        assert_eq!(params["label"], serde_json::json!("tumour"));
        let params = resolve_params(&spec, &["level=3".into()]).unwrap();
        assert_eq!(params["level"], serde_json::json!(3));
        assert!(resolve_params(&spec, &["colour=red".into()]).is_err());
        assert!(resolve_params(&spec, &["level".into()]).is_err());
    }

    #[test]
    fn deidentified_manifests_keep_only_the_file_names_of_paths() {
        let params: Map<String, Value> = serde_json::from_value(serde_json::json!({
            "model": "/home/alice/study-17/models/markers.onnx",
            "runtime": "C:\\Users\\Bob\\ort\\onnxruntime.dll",
            "cache": "~/scratch/",
            "region": "0,0,512,512",
            "level": 2,
        }))
        .unwrap();
        let recorded = recorded_params(&params, true);
        assert_eq!(recorded["model"], "markers.onnx");
        assert_eq!(recorded["runtime"], "onnxruntime.dll");
        assert_eq!(recorded["cache"], "(a path)");
        assert_eq!(recorded["region"], "0,0,512,512");
        assert_eq!(recorded["level"], 2);
        assert_eq!(
            recorded_params(&params, false),
            params,
            "kept as given otherwise"
        );
    }
}

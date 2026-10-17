//! `eov model`: fetch, list, verify, remove and import the models of AI
//! plugins.

use super::{
    CLI_REQUESTER, FetchError, FileRequest, ModelClient, ModelRequest, ModelStore, StoredModel,
    plan_download,
};
use crate::update::policy::{Policy, redacted_url};
use crate::update::prompt::Interaction;
use anyhow::{Result, anyhow, bail};
use clap::Subcommand;
use eov_plugin_api::protocol::is_sha256_hex;
use serde_json::{Value, json};
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Exit status of a command cancelled with Ctrl-C.
const EXIT_CANCELLED: i32 = 130;
/// Exit status of `verify` when a model is damaged.
const EXIT_DAMAGED: i32 = 1;

#[derive(Clone, Debug, Subcommand)]
pub(crate) enum ModelCommand {
    /// Download a model file, check its SHA-256 and keep it in the model
    /// store, where plugins find it.
    ///
    /// The file goes to a temporary file first and is kept only if its size
    /// and SHA-256 match; an interrupted download continues where it
    /// stopped when run again. The administrator's policy can name the
    /// site's copy of the file (model_urls), which is used instead of URL.
    #[command(after_help = "EXAMPLES:\n  \
    eov model fetch https://models.example.org/histovae/HistoVae_reconstruct.onnx \\\n      \
    --sha256 70a8deef1f01716efbc87feb181ea398425d08b5e4984ad9d4ab2b83fd354f92 \\\n      \
    --companion cd2735866bd69648eb17df97f196f266e6e2569f8c21fa1e9779fb7e3a233ef8=HistoVae_reconstruct.onnx.data \\\n      \
    --name \"HistoVAE reference (CAMELYON17)\"")]
    Fetch(FetchArgs),
    /// List the stored models.
    #[command(alias = "ls")]
    List {
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Hash every stored model again and report the damaged ones (exit
    /// status 1).
    Verify {
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Remove a stored model.
    #[command(alias = "rm")]
    Remove {
        /// Its SHA-256 (or the first digits of it), name or file name.
        model: String,
        /// Do not ask for confirmation.
        #[arg(short = 'y', long)]
        yes: bool,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Copy a model file into the store after checking it, for computers
    /// without network access (the file itself is left where it is).
    Import(ImportArgs),
}

/// Options of `eov model fetch`.
#[derive(Clone, Debug, clap::Args)]
pub(crate) struct FetchArgs {
    /// The model file's URL: https, or http on this computer or a server
    /// the administrator's model_url_prefixes names.
    url: String,
    /// The file's SHA-256 (64 hex digits).
    #[arg(long, value_name = "HEX")]
    sha256: String,
    /// What the model is called (shown by `eov model list`).
    #[arg(long, value_name = "NAME")]
    name: Option<String>,
    /// The model's version (`--version` is eov's own).
    #[arg(id = "model-version", long = "model-version", value_name = "VERSION")]
    version: Option<String>,
    /// The file's size in bytes (checked; a larger download stops early).
    #[arg(long, value_name = "BYTES")]
    size: Option<u64>,
    /// Store the file under this name (default: the URL's last part).
    #[arg(long, value_name = "NAME")]
    file_name: Option<String>,
    /// A file the model needs next to it, such as an ONNX model's external
    /// weights: its SHA-256, `=`, and its URL (relative to the model's, so
    /// a file name in the same folder works). Repeat for more.
    #[arg(long = "companion", value_name = "SHA256=URL")]
    companions: Vec<String>,
    /// Download without asking.
    #[arg(long)]
    allow_network: bool,
    /// Print machine-readable JSON.
    #[arg(long)]
    json: bool,
}

/// Options of `eov model import`.
#[derive(Clone, Debug, clap::Args)]
pub(crate) struct ImportArgs {
    /// The model file.
    file: PathBuf,
    /// Its expected SHA-256; the import is refused if the file differs.
    #[arg(long, value_name = "HEX")]
    sha256: Option<String>,
    /// What the model is called.
    #[arg(long, value_name = "NAME")]
    name: Option<String>,
    /// The model's version (`--version` is eov's own).
    #[arg(id = "model-version", long = "model-version", value_name = "VERSION")]
    version: Option<String>,
    /// A file the model needs next to it (FILE, or SHA256=FILE to check
    /// it). Repeat for more.
    #[arg(long = "companion", value_name = "[SHA256=]FILE")]
    companions: Vec<String>,
    /// Print machine-readable JSON.
    #[arg(long)]
    json: bool,
}

impl ModelCommand {
    fn json(&self) -> bool {
        match self {
            Self::Fetch(args) => args.json,
            Self::Import(args) => args.json,
            Self::List { json } | Self::Verify { json } | Self::Remove { json, .. } => *json,
        }
    }
}

/// What a command works with (tests give their own).
pub(crate) struct Context<'a> {
    pub store: ModelStore,
    pub client: ModelClient,
    pub policy: Policy,
    pub interaction: Interaction,
    pub out: &'a mut dyn Write,
    /// Notes and progress.
    pub err: &'a mut dyn Write,
    /// Draw a progress bar on standard error.
    pub progress_bar: bool,
    pub cancel: Arc<AtomicBool>,
}

/// Runs `eov model …` in this process (exits with the documented status).
pub(crate) fn run_cli(command: &ModelCommand) -> Result<()> {
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let cancel = Arc::clone(&cancel);
        let _ = ctrlc::set_handler(move || cancel.store(true, Ordering::SeqCst));
    }
    let policy = crate::update::policy::current().clone();
    let yes = matches!(command, ModelCommand::Remove { yes: true, .. });
    let json = command.json();
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    let progress_bar = !json && stderr.is_terminal();
    let mut ctx = Context {
        store: ModelStore::for_user(&policy),
        client: ModelClient::for_build(),
        policy,
        interaction: Interaction::terminal(yes),
        out: &mut stdout,
        err: &mut stderr,
        progress_bar,
        cancel,
    };
    match run(&mut ctx, command) {
        Ok(0) => Ok(()),
        Ok(status) => {
            let _ = std::io::stdout().flush();
            std::process::exit(status)
        }
        Err(error) => {
            if json {
                println!("{}", json!({ "error": format!("{error:#}") }));
            }
            Err(error)
        }
    }
}

/// Runs a command; returns the exit status.
pub(crate) fn run(ctx: &mut Context<'_>, command: &ModelCommand) -> Result<i32> {
    match command {
        ModelCommand::Fetch(args) => fetch(ctx, args),
        ModelCommand::List { json } => {
            list(ctx, *json)?;
            Ok(0)
        }
        ModelCommand::Verify { json } => verify(ctx, *json),
        ModelCommand::Remove { model, json, .. } => {
            remove(ctx, model, *json)?;
            Ok(0)
        }
        ModelCommand::Import(args) => import(ctx, args),
    }
}

fn sha256_arg(text: &str, what: &str) -> Result<String> {
    let text = text.trim();
    if !is_sha256_hex(text) {
        bail!("{what} must be a SHA-256 of 64 hex digits, not '{text}'");
    }
    Ok(text.to_ascii_lowercase())
}

/// `SHA256=REST` (the hash first, so `=` may appear in the rest), or just
/// `REST` when the hash is optional.
fn split_companion(text: &str) -> (Option<String>, String) {
    match text.split_once('=') {
        Some((hash, rest)) if is_sha256_hex(hash.trim()) => {
            (Some(hash.trim().to_ascii_lowercase()), rest.to_string())
        }
        _ => (None, text.to_string()),
    }
}

/// A stored model as JSON (what plugins get, with the store's folder).
fn model_json(model: &StoredModel) -> Value {
    let mut value = serde_json::to_value(model.to_protocol()).unwrap_or(Value::Null);
    if let Value::Object(map) = &mut value {
        map.insert("dir".into(), json!(model.dir.to_string_lossy()));
    }
    value
}

fn describe(out: &mut dyn Write, model: &StoredModel) -> std::io::Result<()> {
    let record = &model.record;
    let version = record
        .version
        .as_deref()
        .map(|version| format!(" {version}"))
        .unwrap_or_default();
    let read_only = if model.read_only {
        " (administrator, read-only)"
    } else {
        ""
    };
    writeln!(out, "{}{version}{read_only}", model.label())?;
    writeln!(out, "  sha256  {}", record.sha256)?;
    writeln!(
        out,
        "  file    {} ({})",
        model.path().display(),
        crate::cli::format_bytes(record.size)
    )?;
    for companion in &record.companions {
        writeln!(
            out,
            "  with    {} ({}, sha256 {})",
            companion.file_name,
            crate::cli::format_bytes(companion.size),
            companion.sha256
        )?;
    }
    if let Some(source) = &record.source {
        writeln!(out, "  from    {source}")?;
    } else {
        writeln!(out, "  from    imported")?;
    }
    let by = record
        .requested_by
        .as_deref()
        .map(|by| format!(" for {by}"))
        .unwrap_or_default();
    if let Some(when) = &record.fetched_at {
        writeln!(out, "  added   {when}{by}")?;
    }
    Ok(())
}

fn fetch(ctx: &mut Context<'_>, args: &FetchArgs) -> Result<i32> {
    let sha256 = sha256_arg(&args.sha256, "--sha256")?;
    let mut companions = Vec::new();
    for text in &args.companions {
        let (hash, url) = split_companion(text);
        let hash = hash.ok_or_else(|| {
            anyhow!("--companion takes SHA256=URL (the file's SHA-256, then its URL), not '{text}'")
        })?;
        companions.push(FileRequest {
            sha256: hash,
            size: None,
            file_name: None,
            url: Some(url),
        });
    }
    let request = ModelRequest {
        main: FileRequest {
            sha256,
            size: args.size,
            file_name: args.file_name.clone(),
            url: Some(args.url.clone()),
        },
        companions,
        name: args.name.clone().filter(|n| !n.trim().is_empty()),
        version: args.version.clone().filter(|v| !v.trim().is_empty()),
        requested_by: CLI_REQUESTER.to_string(),
    };
    if let Some(name) = &request.main.file_name {
        super::check_store_file_name(name).map_err(|e| anyhow!(e))?;
    }
    if let Some(stored) = ctx.store.find_request(&request) {
        if args.json {
            writeln!(
                ctx.out,
                "{}",
                serde_json::to_string_pretty(&model_json(&stored))?
            )?;
        } else {
            writeln!(ctx.out, "Already stored:")?;
            describe(ctx.out, &stored)?;
        }
        return Ok(0);
    }
    let stored = ctx.store.user_model(&request.main.sha256);
    let plan = plan_download(&request, &ctx.policy, stored.as_ref())?;
    for file in plan.files.iter().filter(|file| file.from_policy) {
        writeln!(
            ctx.err,
            "Using the administrator's copy of {}: {}",
            file.file_name,
            redacted_url(&file.url)
        )?;
    }
    if !args.allow_network {
        let size = plan
            .known_size()
            .map(|bytes| format!(" ({})", crate::cli::format_bytes(bytes)))
            .unwrap_or_default();
        let files: Vec<String> = plan
            .files
            .iter()
            .map(|file| format!("  {}", redacted_url(&file.url)))
            .collect();
        let question = format!(
            "Download {}{size} from {}?\n{}\n\nEOV never makes network requests without your permission and does not collect telemetry. The files are checked against their SHA-256 and kept in {}.",
            request.label(),
            plan.hosts().join(", "),
            files.join("\n"),
            ctx.store.root().display()
        );
        if !ctx.interaction.consent(&question) {
            if !ctx.interaction.interactive() {
                bail!(
                    "Downloading a model needs consent, but no terminal is attached. Re-run with --allow-network.{}",
                    crate::update::prompt::windows_hint()
                );
            }
            bail!("Download declined; nothing was changed.");
        }
    }
    let bar = ctx.progress_bar.then(|| {
        let bar = indicatif::ProgressBar::new(plan.known_size().unwrap_or(0));
        bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta} remaining) {msg}",
                )
                .unwrap()
                .progress_chars("█▓░"),
        );
        bar
    });
    let result = ctx.store.fetch(
        &ctx.client,
        &plan,
        &ctx.policy,
        &|progress| {
            if let Some(bar) = &bar {
                if let Some(total) = progress.total {
                    bar.set_length(total);
                }
                bar.set_position(progress.done);
                bar.set_message(progress.file_name.clone());
            }
        },
        &ctx.cancel,
    );
    if let Some(bar) = &bar {
        bar.finish_and_clear();
    }
    match result {
        Ok(model) => {
            if args.json {
                writeln!(
                    ctx.out,
                    "{}",
                    serde_json::to_string_pretty(&model_json(&model))?
                )?;
            } else {
                writeln!(ctx.out, "Stored:")?;
                describe(ctx.out, &model)?;
            }
            Ok(0)
        }
        Err(FetchError::Cancelled) => {
            writeln!(ctx.err, "Cancelled; nothing was stored.")?;
            Ok(EXIT_CANCELLED)
        }
        Err(error) => Err(anyhow!(error)),
    }
}

fn list(ctx: &mut Context<'_>, json: bool) -> Result<()> {
    let models = ctx.store.list();
    if json {
        let entries: Vec<Value> = models.iter().map(model_json).collect();
        writeln!(ctx.out, "{}", serde_json::to_string_pretty(&entries)?)?;
        return Ok(());
    }
    if models.is_empty() {
        writeln!(
            ctx.out,
            "No models are stored in {}. `eov model fetch` downloads one, `eov model import` adds a file.",
            ctx.store.root().display()
        )?;
        return Ok(());
    }
    for (index, model) in models.iter().enumerate() {
        if index > 0 {
            writeln!(ctx.out)?;
        }
        describe(ctx.out, model)?;
    }
    Ok(())
}

fn verify(ctx: &mut Context<'_>, json: bool) -> Result<i32> {
    let reports = {
        let err = &mut *ctx.err;
        let show = !json;
        let err = std::cell::RefCell::new(err);
        ctx.store.verify(&ctx.cancel, &|label| {
            if show {
                let _ = writeln!(err.borrow_mut(), "Checking {label}…");
            }
        })
    };
    if ctx.cancel.load(Ordering::Relaxed) {
        writeln!(ctx.err, "Cancelled.")?;
        return Ok(EXIT_CANCELLED);
    }
    let damaged = reports.iter().filter(|r| !r.problems.is_empty()).count();
    if json {
        let entries: Vec<Value> = reports
            .iter()
            .map(|report| {
                json!({
                    "dir": report.dir.to_string_lossy(),
                    "sha256": report.model.as_ref().map(|m| m.record.sha256.clone()),
                    "name": report.model.as_ref().map(StoredModel::label),
                    "ok": report.problems.is_empty(),
                    "problems": report.problems,
                })
            })
            .collect();
        writeln!(ctx.out, "{}", serde_json::to_string_pretty(&entries)?)?;
    } else {
        for report in &reports {
            let label = report
                .model
                .as_ref()
                .map(StoredModel::label)
                .unwrap_or_else(|| report.dir.display().to_string());
            if report.problems.is_empty() {
                writeln!(ctx.out, "ok       {label}")?;
            } else {
                writeln!(ctx.out, "DAMAGED  {label} ({})", report.dir.display())?;
                for problem in &report.problems {
                    writeln!(ctx.out, "         {problem}")?;
                }
            }
        }
        if reports.is_empty() {
            writeln!(ctx.out, "No models are stored.")?;
        } else if damaged > 0 {
            writeln!(
                ctx.out,
                "\n{damaged} of {} models are damaged: remove them (eov model remove) and fetch or import them again.",
                reports.len()
            )?;
        }
    }
    Ok(if damaged > 0 { EXIT_DAMAGED } else { 0 })
}

fn remove(ctx: &mut Context<'_>, query: &str, json: bool) -> Result<()> {
    let matches = ctx.store.find(query);
    let model = match matches.as_slice() {
        [] => bail!("No stored model matches '{query}' (eov model list)."),
        [model] => model.clone(),
        several => bail!(
            "'{query}' matches {} models; give the SHA-256:\n{}",
            several.len(),
            several
                .iter()
                .map(|m| format!("  {}  {}", m.record.sha256, m.label()))
                .collect::<Vec<_>>()
                .join("\n")
        ),
    };
    if model.read_only {
        bail!(
            "{} is provided by the administrator ({}) and cannot be removed.",
            model.label(),
            model.dir.display()
        );
    }
    ctx.interaction.confirm(&format!(
        "Remove {} ({}, {}) from the model store? Plugins that use it will ask for it again.",
        model.label(),
        &model.record.sha256[..12],
        crate::cli::format_bytes(model.total_size())
    ))?;
    ctx.store.remove(&model)?;
    if json {
        writeln!(
            ctx.out,
            "{}",
            json!({ "removed": model.record.sha256, "name": model.label() })
        )?;
    } else {
        writeln!(
            ctx.out,
            "Removed {} ({}).",
            model.label(),
            model.record.sha256
        )?;
    }
    Ok(())
}

fn import(ctx: &mut Context<'_>, args: &ImportArgs) -> Result<i32> {
    let expected = args
        .sha256
        .as_deref()
        .map(|text| sha256_arg(text, "--sha256"))
        .transpose()?;
    let companions: Vec<(PathBuf, Option<String>)> = args
        .companions
        .iter()
        .map(|text| {
            let (hash, path) = split_companion(text);
            (PathBuf::from(path), hash)
        })
        .collect();
    let result = ctx.store.import(
        &args.file,
        expected.as_deref(),
        &companions,
        args.name.as_deref(),
        args.version.as_deref(),
        &ctx.cancel,
    );
    match result {
        Ok(model) => {
            if args.json {
                writeln!(
                    ctx.out,
                    "{}",
                    serde_json::to_string_pretty(&model_json(&model))?
                )?;
            } else {
                writeln!(ctx.out, "Imported:")?;
                describe(ctx.out, &model)?;
                if expected.is_none() {
                    writeln!(
                        ctx.err,
                        "note: no --sha256 was given, so the file was not checked against a known hash; compare the SHA-256 above with the one its provider publishes."
                    )?;
                }
            }
            Ok(0)
        }
        Err(FetchError::Cancelled) => {
            writeln!(ctx.err, "Cancelled; nothing was stored.")?;
            Ok(EXIT_CANCELLED)
        }
        Err(error) => Err(anyhow!(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_server::{Route, TestServer};
    use super::*;
    use crate::update::prompt::ScriptedPrompter;
    use sha2::{Digest, Sha256};

    fn sha(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    /// Runs `eov model ARGS` against `store`; (status or error, stdout,
    /// stderr, the questions asked).
    fn eov_model(
        store: &ModelStore,
        policy: &Policy,
        answers: &[Option<&'static str>],
        args: &[&str],
    ) -> (Result<i32>, String, String, Vec<String>) {
        use clap::Parser;
        #[derive(Parser)]
        struct Wrapper {
            #[command(subcommand)]
            command: ModelCommand,
        }
        let command = Wrapper::try_parse_from(std::iter::once("model").chain(args.iter().copied()))
            .map(|wrapper| wrapper.command);
        let command = match command {
            Ok(command) => command,
            Err(error) => {
                return (
                    Err(anyhow!(error.to_string())),
                    String::new(),
                    String::new(),
                    vec![],
                );
            }
        };
        let prompter = ScriptedPrompter::new(!answers.is_empty(), answers);
        let asked = Arc::clone(&prompter.asked);
        let yes = matches!(command, ModelCommand::Remove { yes: true, .. });
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let status = {
            let mut ctx = Context {
                store: store.clone(),
                client: ModelClient::for_build(),
                policy: policy.clone(),
                interaction: Interaction::new(yes, Box::new(prompter)),
                out: &mut out,
                err: &mut err,
                progress_bar: false,
                cancel: Arc::new(AtomicBool::new(false)),
            };
            run(&mut ctx, &command)
        };
        let asked = asked.lock().unwrap().clone();
        (
            status,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
            asked,
        )
    }

    fn setup() -> (tempfile::TempDir, ModelStore) {
        crate::paths::isolate_for_tests();
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(dir.path().join("models"), Vec::new());
        (dir, store)
    }

    #[test]
    fn fetch_asks_downloads_and_lists() {
        let server = TestServer::start();
        let graph = b"onnx graph".repeat(100);
        let weights = b"weights".repeat(1000);
        server.route("/h/m.onnx", Route::file(graph.clone()));
        server.route("/h/m.onnx.data", Route::file(weights.clone()));
        let (_dir, store) = setup();
        let url = server.url("/h/m.onnx");
        let companion = format!("{}=m.onnx.data", sha(&weights));
        let args = [
            "fetch",
            url.as_str(),
            "--sha256",
            &sha(&graph),
            "--companion",
            companion.as_str(),
            "--name",
            "Reference",
        ];
        // Declined: nothing happens.
        let (status, _, _, asked) = eov_model(&store, &Policy::default(), &[Some("n")], &args);
        assert!(status.unwrap_err().to_string().contains("declined"));
        assert!(
            asked[0].contains("127.0.0.1") && asked[0].contains("m.onnx.data"),
            "{}",
            asked[0]
        );
        assert!(server.seen().is_empty());
        // No terminal and no --allow-network: refused with the remedy.
        let (status, ..) = eov_model(&store, &Policy::default(), &[], &args);
        assert!(status.unwrap_err().to_string().contains("--allow-network"));
        let (status, out, _, _) = eov_model(&store, &Policy::default(), &[Some("y")], &args);
        assert_eq!(status.unwrap(), 0);
        assert!(
            out.contains("Reference") && out.contains("m.onnx.data"),
            "{out}"
        );
        // Again: already stored, nothing downloaded or asked.
        let requests = server.seen().len();
        let mut again = args.to_vec();
        again.push("--json");
        let (status, out, _, asked) = eov_model(&store, &Policy::default(), &[], &again);
        assert_eq!(status.unwrap(), 0);
        assert!(asked.is_empty());
        assert_eq!(server.seen().len(), requests);
        let value: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(value["sha256"], sha(&graph));
        assert_eq!(value["companions"][0]["file_name"], "m.onnx.data");
        let (_, out, _, _) = eov_model(&store, &Policy::default(), &[], &["list", "--json"]);
        let listed: Vec<Value> = serde_json::from_str(&out).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0]["name"], "Reference");
        let (_, out, _, _) = eov_model(&store, &Policy::default(), &[], &["ls"]);
        assert!(out.contains(&sha(&graph)), "{out}");
    }

    #[test]
    fn fetch_reports_a_wrong_hash_and_obeys_the_policy() {
        let server = TestServer::start();
        let body = b"not the model".to_vec();
        server.route("/m.onnx", Route::file(body.clone()));
        let (_dir, store) = setup();
        let url = server.url("/m.onnx");
        let wrong = sha(b"the model");
        let (status, ..) = eov_model(
            &store,
            &Policy::default(),
            &[],
            &["fetch", &url, "--sha256", &wrong, "--allow-network"],
        );
        let message = status.unwrap_err().to_string();
        assert!(
            message.contains(&wrong) && message.contains(&sha(&body)),
            "{message}"
        );
        let offline = Policy {
            allow_network: false,
            ..Policy::default()
        };
        let (status, ..) = eov_model(
            &store,
            &offline,
            &[],
            &["fetch", &url, "--sha256", &sha(&body), "--allow-network"],
        );
        assert!(status.unwrap_err().to_string().contains("eov model import"));
        assert!(
            server.seen().len() == 1,
            "the policy refused before any request"
        );
        let (status, ..) = eov_model(
            &store,
            &Policy::default(),
            &[],
            &["fetch", &url, "--sha256", "abc"],
        );
        assert!(status.unwrap_err().to_string().contains("64 hex digits"));
    }

    #[test]
    fn import_verify_and_remove() {
        let (dir, store) = setup();
        let body = b"an imported model".to_vec();
        let file = dir.path().join("model.onnx");
        std::fs::write(&file, &body).unwrap();
        let path = file.to_string_lossy().into_owned();
        let (status, ..) = eov_model(
            &store,
            &Policy::default(),
            &[],
            &["import", &path, "--sha256", &sha(b"other")],
        );
        assert!(status.unwrap_err().to_string().contains("does not match"));
        let (status, out, _, _) = eov_model(
            &store,
            &Policy::default(),
            &[],
            &[
                "import",
                &path,
                "--sha256",
                &sha(&body),
                "--name",
                "Site model",
            ],
        );
        assert_eq!(status.unwrap(), 0, "{out}");
        assert!(
            out.contains("Site model") && out.contains("imported"),
            "{out}"
        );
        let (status, out, _, _) = eov_model(&store, &Policy::default(), &[], &["verify"]);
        assert_eq!(status.unwrap(), 0);
        assert!(out.contains("ok       Site model"), "{out}");
        // Damage it: verify says so and exits with 1.
        let stored = store.resolve(&sha(&body)).unwrap();
        std::fs::write(stored.path(), b"tampered model!!!").unwrap();
        let (status, out, _, _) = eov_model(&store, &Policy::default(), &[], &["verify", "--json"]);
        assert_eq!(status.unwrap(), EXIT_DAMAGED);
        let reports: Vec<Value> = serde_json::from_str(&out).unwrap();
        assert_eq!(reports[0]["ok"], false);
        // Removing asks unless -y; then it is gone.
        let (status, ..) = eov_model(&store, &Policy::default(), &[], &["remove", "Site model"]);
        assert!(status.unwrap_err().to_string().contains("-y"));
        let (status, out, _, _) = eov_model(
            &store,
            &Policy::default(),
            &[],
            &["rm", &sha(&body)[..10], "-y"],
        );
        assert_eq!(status.unwrap(), 0);
        assert!(out.contains("Removed Site model"), "{out}");
        assert!(store.list().is_empty());
        let (status, ..) = eov_model(
            &store,
            &Policy::default(),
            &[],
            &["remove", "Site model", "-y"],
        );
        assert!(status.unwrap_err().to_string().contains("No stored model"));
    }

    #[test]
    fn companions_split_at_a_leading_hash() {
        let hash = "a".repeat(64);
        assert_eq!(
            split_companion(&format!("{hash}=https://x/y?a=b")),
            (Some(hash.clone()), "https://x/y?a=b".to_string())
        );
        assert_eq!(
            split_companion("C:/models/w=1.data"),
            (None, "C:/models/w=1.data".to_string())
        );
    }
}

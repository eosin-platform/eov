//! eovae follows the language of eov's interface: a job's summary is a
//! whole sentence of that job (so that a translation can make the verb
//! agree with what finished), and problems reach the user in their
//! language while eov's log and batch output stay English.
//!
//! A test binary of its own: the language is global to a library, so it
//! must not change under the other tests.

use eov_plugin_api::i18n::{self, Catalog};
use eov_plugin_api::prelude::*;
use eov_plugin_api::testing::{MockHost, TestHarness, test_init};
use eovae::pipeline::JobOutcome;
use eovae::plugin::{JobKind, JobSummary, NoRun};
use eovae::results::Counts;
use eovae::tiling::ReadPlan;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

type H = TestHarness<eovae::Eovae<eovae::NoRuntime>>;

/// A few of eovae's texts in German, as its catalog would have them.
const GERMAN: &[(&str, &str)] = &[
    (
        "Viewport finished in {duration}: {counts}.",
        "Ansicht in {duration} analysiert: {counts}.",
    ),
    (
        "Region of interest finished in {duration}: {counts}.",
        "Interessenbereich in {duration} analysiert: {counts}.",
    ),
    ("{analysed} analysed", "{analysed} analysiert"),
    ("VAE analysis failed", "VAE-Analyse fehlgeschlagen"),
    (
        "Load an ONNX model first.",
        "Laden Sie zuerst ein ONNX-Modell.",
    ),
    (
        "eovae settings could not be saved: {}",
        "Die eovae-Einstellungen konnten nicht gespeichert werden: {}",
    ),
    (
        "only .onnx models are supported",
        "nur .onnx-Modelle werden unterstützt",
    ),
    (
        "The model could not be loaded: {}",
        "Das Modell konnte nicht geladen werden: {}",
    ),
];

/// eovae's texts with a number in German: the English singular, and the
/// German forms.
const GERMAN_PLURALS: &[(&str, &[&str])] = &[(
    "the host returned {n} byte for a {width}×{height} region",
    &[
        "der Host lieferte {n} Byte für einen Bereich von {width}×{height}",
        "der Host lieferte {n} Bytes für einen Bereich von {width}×{height}",
    ],
)];

fn speak_german() {
    let german = i18n::by_tag("de").unwrap();
    let messages = GERMAN
        .iter()
        .map(|(english, translated)| (None, *english, std::slice::from_ref(translated)))
        .chain(
            GERMAN_PLURALS
                .iter()
                .map(|(english, forms)| (None, *english, *forms)),
        );
    let catalog = Catalog::from_messages(german, "", messages).unwrap();
    i18n::set_catalog(german, Some(catalog));
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

/// What eovae wrote to eov's log at `level`.
fn logged(h: &H, level: &str) -> Vec<String> {
    h.calls_to("log")
        .iter()
        .filter(|l| l["level"] == json!(level))
        .filter_map(|l| l["message"].as_str().map(str::to_string))
        .collect()
}

fn notification(h: &H, message: impl Fn(&str) -> bool) -> Option<NotifyArgs> {
    h.state()
        .notifications
        .iter()
        .find(|n| message(&n.message))
        .cloned()
}

#[test]
fn summaries_and_problems_follow_the_language_of_the_interface() {
    let host = MockHost::new();
    {
        let mut state = host.state();
        state.snapshot.files.push(FileInfo {
            file_id: 1,
            path: "slide-a.svs".into(),
            filename: "slide-a.svs".into(),
            width: 4096,
            height: 2048,
            levels: vec![LevelInfo {
                level: 0,
                width: 4096,
                height: 2048,
                downsample: 1.0,
            }],
            ..Default::default()
        });
        state.snapshot.panes.push(PaneInfo {
            pane: 7,
            index: 0,
            file_id: Some(1),
            view: None,
        });
        state.snapshot.focused_pane = Some(7);
        // This eov cannot keep eovae's settings.
        state.unsupported.push(methods::SETTINGS_SET.to_string());
    }
    // Creating the plugin switches to the language eov names (none here).
    let h: H = TestHarness::with_host(host, test_init("eovae")).unwrap();
    speak_german();

    // Each kind of job has its own sentences.
    let summary = |kind| {
        JobSummary {
            kind,
            outcome: JobOutcome::Completed(Counts {
                analysed: 32,
                background: 0,
                failed: 0,
            }),
            elapsed: Duration::from_secs(12),
        }
        .text()
    };
    assert_eq!(
        summary(JobKind::Viewport),
        "Ansicht in 12 s analysiert: 32 analysiert."
    );
    assert_eq!(
        summary(JobKind::Roi),
        "Interessenbereich in 12 s analysiert: 32 analysiert."
    );
    // The whole slide's sentence is not translated here: it stays English.
    assert!(
        summary(JobKind::WholeSlide).starts_with("Whole slide finished in 12 s: "),
        "{}",
        summary(JobKind::WholeSlide)
    );

    // An error is kept in both languages, also when it is made of another
    // one: the user reads German, eov's log and batch output get English.
    let unusable = eovae::model::inspect(Path::new("model.pt"), &AtomicBool::new(false))
        .expect_err("only .onnx files are models");
    assert_eq!(unusable.shown, "nur .onnx-Modelle werden unterstützt");
    assert_eq!(unusable.english, "only .onnx models are supported");
    let failed = NoRun::ModelFailed(unusable).message();
    assert_eq!(
        failed.shown,
        "Das Modell konnte nicht geladen werden: nur .onnx-Modelle werden unterstützt"
    );
    assert_eq!(
        failed.english,
        "The model could not be loaded: only .onnx models are supported"
    );
    // A count of bytes takes its plural form in each language.
    let plan = ReadPlan {
        x: 0,
        y: 0,
        row: 0,
        width: 2,
        height: 2,
        origin_x: 0.0,
        origin_y: 0.0,
        step_x: 1.0,
        step_y: 1.0,
    };
    let short = |bytes: usize| {
        eovae::resample::resample(&vec![0; bytes], &plan, 2, [255; 3])
            .expect_err("too few bytes for 2×2 pixels")
    };
    assert_eq!(
        short(1).shown,
        "der Host lieferte 1 Byte für einen Bereich von 2×2"
    );
    assert_eq!(
        short(5).shown,
        "der Host lieferte 5 Bytes für einen Bereich von 2×2"
    );
    assert_eq!(
        short(1).english,
        "the host returned 1 byte for a 2×2 region"
    );
    assert_eq!(
        short(5).english,
        "the host returned 5 bytes for a 2×2 region"
    );

    // A failed analysis (no model is loaded): the user reads German, the
    // log has eovae's English line.
    h.send(HostEvent::Ready {
        restore_sidebar: false,
    })
    .unwrap();
    h.ui_callback("", "analyze-viewport", vec![]).unwrap();
    wait_until("the notification", || {
        notification(&h, |m| m == "VAE-Analyse fehlgeschlagen").is_some()
    });
    let failed = notification(&h, |m| m == "VAE-Analyse fehlgeschlagen").unwrap();
    assert_eq!(
        failed.detail.as_deref(),
        Some("Laden Sie zuerst ein ONNX-Modell.")
    );
    wait_until("the English line in the log", || {
        logged(&h, "info")
            .iter()
            .any(|m| m.starts_with("VAE analysis failed: "))
    });
    let failures: Vec<String> = logged(&h, "info")
        .into_iter()
        .filter(|m| m.starts_with("VAE analysis failed: "))
        .collect();
    assert_eq!(
        failures,
        ["VAE analysis failed: Load an ONNX model first."],
        "the log is English all through"
    );

    // Settings that cannot be saved: the same.
    h.ui_callback("", "section-toggled", vec![json!("cache"), json!(true)])
        .unwrap();
    wait_until("the warning", || {
        notification(&h, |m| {
            m.starts_with("Die eovae-Einstellungen konnten nicht gespeichert werden: ")
        })
        .is_some_and(|n| n.level == NotifyLevel::Warning)
    });
    assert!(
        logged(&h, "info")
            .iter()
            .any(|m| m.starts_with("eovae settings could not be saved: ")),
        "{:?}",
        logged(&h, "info")
    );
    assert!(
        notification(&h, |m| m.starts_with("eovae settings could not be saved: ")).is_none(),
        "the user reads German"
    );

    // eov would show a logged warning or error to the user as it is, in
    // English (PLG-15): eovae logs none.
    let problems = [logged(&h, "warn"), logged(&h, "error")].concat();
    assert!(problems.is_empty(), "{problems:?}");
}

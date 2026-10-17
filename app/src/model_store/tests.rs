//! The model store against a local HTTP server.

use super::test_server::{Route, TestServer};
use super::*;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Model-like bytes of `len` (not all equal, so offsets matter).
fn payload(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761).to_le_bytes()[1] ^ seed)
        .collect()
}

fn store() -> (tempfile::TempDir, ModelStore) {
    crate::paths::isolate_for_tests();
    let dir = tempfile::tempdir().unwrap();
    let store = ModelStore::new(dir.path().join("models"), Vec::new());
    (dir, store)
}

fn model_request(url: &str, sha256: &str, size: Option<u64>) -> ModelRequest {
    ModelRequest {
        main: FileRequest {
            sha256: sha256.to_string(),
            size,
            file_name: None,
            url: Some(url.to_string()),
        },
        companions: Vec::new(),
        name: Some("Test model".into()),
        version: Some("1".into()),
        requested_by: "eovae".into(),
    }
}

fn fetch(
    store: &ModelStore,
    request: &ModelRequest,
    policy: &Policy,
) -> Result<StoredModel, FetchError> {
    let plan = plan_download(
        request,
        policy,
        store.user_model(&request.main.sha256).as_ref(),
    )?;
    store.fetch(
        &ModelClient::for_build(),
        &plan,
        policy,
        &|_| {},
        &AtomicBool::new(false),
    )
}

fn partial_files(store: &ModelStore) -> Vec<String> {
    fs::read_dir(store.partial_dir())
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|name| name.ends_with(".part"))
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn a_good_download_is_verified_stored_and_recorded() {
    let server = TestServer::start();
    let body = payload(300_000, 1);
    server.route("/models/m.onnx", Route::file(body.clone()));
    let (_dir, store) = store();
    let url = format!("{}?token=secret", server.url("/models/m.onnx"));
    let request = model_request(&url, &sha(&body), Some(body.len() as u64));
    let progress = Mutex::new(Vec::new());
    let plan = plan_download(&request, &Policy::default(), None).unwrap();
    assert_eq!(plan.known_size(), Some(300_000));
    assert_eq!(plan.hosts(), ["127.0.0.1"]);
    let model = store
        .fetch(
            &ModelClient::for_build(),
            &plan,
            &Policy::default(),
            &|p| progress.lock().unwrap().push(p.done),
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(fs::read(model.path()).unwrap(), body);
    assert_eq!(model.path(), store.root().join(sha(&body)).join("m.onnx"));
    assert_eq!(*progress.lock().unwrap().last().unwrap(), 300_000);
    let record = &model.record;
    assert_eq!(record.size, 300_000);
    assert_eq!(record.name.as_deref(), Some("Test model"));
    assert_eq!(record.requested_by.as_deref(), Some("eovae"));
    assert!(record.fetched_at.as_deref().unwrap().ends_with('Z'));
    // The query (a signed URL's secret) is not recorded.
    assert_eq!(
        record.source.as_deref(),
        Some(server.url("/models/m.onnx").as_str())
    );
    assert!(partial_files(&store).is_empty());
    // Found by every way of asking, and not downloaded again.
    assert_eq!(store.resolve(&sha(&body)).unwrap(), model);
    assert_eq!(store.list(), std::slice::from_ref(&model));
    assert_eq!(store.find("Test model"), std::slice::from_ref(&model));
    assert_eq!(store.find(&sha(&body)[..8]), std::slice::from_ref(&model));
    assert_eq!(store.find_request(&request), Some(model.clone()));
    let protocol = model.to_protocol();
    assert_eq!(protocol.path, model.path().to_string_lossy());
    assert!(!protocol.read_only);
    // The audit log names it.
    let audit = fs::read_to_string(crate::paths::data_dir().join("audit.jsonl")).unwrap();
    assert!(
        audit
            .lines()
            .any(|line| line.contains("model_fetched") && line.contains(&sha(&body))),
        "{audit}"
    );
    assert!(!audit.contains("secret"));
}

#[test]
fn a_wrong_hash_deletes_the_file_and_says_what_was_expected() {
    let server = TestServer::start();
    let body = payload(10_000, 2);
    server.route("/m.onnx", Route::file(body.clone()));
    let (_dir, store) = store();
    let expected = sha(b"something else");
    let error = fetch(
        &store,
        &model_request(&server.url("/m.onnx"), &expected, None),
        &Policy::default(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        FetchError::Integrity {
            file: "m.onnx".into(),
            expected: expected.clone(),
            actual: sha(&body),
        }
    );
    let message = error.to_string();
    assert!(
        message.contains(&expected) && message.contains(&sha(&body)),
        "{message}"
    );
    assert!(partial_files(&store).is_empty());
    assert!(store.list().is_empty());
}

#[test]
fn a_wrong_size_is_refused_before_and_while_downloading() {
    let server = TestServer::start();
    let body = payload(50_000, 3);
    server.route("/m.onnx", Route::file(body.clone()));
    let (_dir, store) = store();
    // The server declares more than expected: refused before reading.
    let small = fetch(
        &store,
        &model_request(&server.url("/m.onnx"), &sha(&body), Some(1_000)),
        &Policy::default(),
    )
    .unwrap_err();
    assert!(
        matches!(
            small,
            FetchError::Size {
                expected: 1_000,
                actual: 50_000,
                ..
            }
        ),
        "{small:?}"
    );
    // Less arrives than expected.
    let large = fetch(
        &store,
        &model_request(&server.url("/m.onnx"), &sha(&body), Some(60_000)),
        &Policy::default(),
    )
    .unwrap_err();
    assert!(
        matches!(
            large,
            FetchError::Size {
                expected: 60_000,
                actual: 50_000,
                ..
            }
        ),
        "{large:?}"
    );
    assert!(large.to_string().contains("wrong size"), "{large}");
    assert!(partial_files(&store).is_empty());
    assert!(store.list().is_empty());
}

#[test]
fn an_interrupted_download_resumes_with_a_range_request() {
    let server = TestServer::start();
    let body = payload(200_000, 4);
    server.route(
        "/m.onnx",
        Route {
            cut_first: 1,
            cut_after: 70_000,
            ..Route::file(body.clone())
        },
    );
    let (_dir, store) = store();
    let request = model_request(&server.url("/m.onnx"), &sha(&body), Some(200_000));
    let error = fetch(&store, &request, &Policy::default()).unwrap_err();
    assert!(matches!(error, FetchError::Network { .. }), "{error:?}");
    let part = store.partial_dir().join(format!("{}.part", sha(&body)));
    assert_eq!(fs::metadata(&part).unwrap().len(), 70_000);
    let model = fetch(&store, &request, &Policy::default()).unwrap();
    assert_eq!(fs::read(model.path()).unwrap(), body);
    let seen = server.seen();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].1, None);
    assert_eq!(seen[1].1.as_deref(), Some("bytes=70000-"));
    assert!(!part.exists());
}

#[test]
fn a_server_without_ranges_sends_the_file_again() {
    let server = TestServer::start();
    let body = payload(120_000, 5);
    server.route(
        "/m.onnx",
        Route {
            ranges: false,
            cut_first: 1,
            cut_after: 50_000,
            ..Route::file(body.clone())
        },
    );
    let (_dir, store) = store();
    let request = model_request(&server.url("/m.onnx"), &sha(&body), None);
    assert!(fetch(&store, &request, &Policy::default()).is_err());
    let model = fetch(&store, &request, &Policy::default()).unwrap();
    assert_eq!(fs::read(model.path()).unwrap(), body);
    assert_eq!(server.seen()[1].1.as_deref(), Some("bytes=50000-"));
}

#[test]
fn a_complete_partial_download_is_used_without_the_network() {
    let server = TestServer::start();
    let body = payload(20_000, 6);
    let (_dir, store) = store();
    fs::create_dir_all(store.partial_dir()).unwrap();
    fs::write(
        store.partial_dir().join(format!("{}.part", sha(&body))),
        &body,
    )
    .unwrap();
    let request = model_request(&server.url("/gone.onnx"), &sha(&body), Some(20_000));
    let model = fetch(&store, &request, &Policy::default()).unwrap();
    assert_eq!(fs::read(model.path()).unwrap(), body);
    assert!(server.seen().is_empty());
}

#[test]
fn cancelling_stops_the_download_and_deletes_it() {
    let server = TestServer::start();
    let body = payload(400_000, 7);
    server.route(
        "/slow.onnx",
        Route {
            trickle: Some(std::time::Duration::from_millis(20)),
            ..Route::file(body.clone())
        },
    );
    let (_dir, store) = store();
    let request = model_request(&server.url("/slow.onnx"), &sha(&body), None);
    let plan = plan_download(&request, &Policy::default(), None).unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let started = std::time::Instant::now();
    let error = store
        .fetch(
            &ModelClient::for_build(),
            &plan,
            &Policy::default(),
            &|progress| {
                if progress.done > 0 {
                    cancel.store(true, Ordering::SeqCst);
                }
            },
            &cancel,
        )
        .unwrap_err();
    assert!(error.is_cancelled(), "{error:?}");
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
    assert!(partial_files(&store).is_empty());
    assert!(store.list().is_empty());
}

#[test]
fn the_policy_refuses_downloads_with_a_clear_message() {
    let server = TestServer::start();
    let body = payload(1_000, 8);
    server.route("/m.onnx", Route::file(body.clone()));
    let (_dir, store) = store();
    let request = model_request(&server.url("/m.onnx"), &sha(&body), None);
    let offline = Policy {
        allow_network: false,
        source: Some(PathBuf::from("/etc/eov/policy.toml")),
        ..Policy::default()
    };
    let error = plan_download(&request, &offline, None).unwrap_err();
    let message = error.to_string();
    assert!(matches!(error, FetchError::Refused(_)));
    assert!(
        message.contains("/etc/eov/policy.toml") && message.contains("eov model import"),
        "{message}"
    );
    let disabled = Policy {
        allow_model_downloads: Some(false),
        ..Policy::default()
    };
    assert!(plan_download(&request, &disabled, None).is_err());
    let elsewhere = Policy {
        model_url_prefixes: vec!["https://models.example.org/".into()],
        ..Policy::default()
    };
    assert!(
        plan_download(&request, &elsewhere, None)
            .unwrap_err()
            .to_string()
            .contains("models.example.org")
    );
    // Plain HTTP is for this computer only, and every redirect is checked.
    let remote = model_request("http://models.example.org/m.onnx", &sha(&body), None);
    assert!(plan_download(&remote, &Policy::default(), None).is_err());
    server.route(
        "/moved.onnx",
        Route {
            redirect: Some("http://models.example.org/m.onnx".into()),
            ..Route::default()
        },
    );
    let moved = model_request(&server.url("/moved.onnx"), &sha(&body), None);
    let error = fetch(&store, &moved, &Policy::default()).unwrap_err();
    assert!(matches!(error, FetchError::Refused(_)), "{error:?}");
    assert_eq!(server.seen().len(), 1, "nothing beyond the redirect");
    // Nothing says where to download it from.
    let nowhere = ModelRequest {
        main: FileRequest {
            url: None,
            ..request.main.clone()
        },
        ..request.clone()
    };
    let error = plan_download(&nowhere, &Policy::default(), None).unwrap_err();
    assert!(error.to_string().contains("model_urls"), "{error}");
}

#[test]
fn the_sites_copy_wins_over_the_given_url() {
    let server = TestServer::start();
    let body = payload(5_000, 9);
    server.route("/site/m.onnx", Route::file(body.clone()));
    server.route(
        "/redirect",
        Route {
            redirect: Some("/site/m.onnx".into()),
            ..Route::default()
        },
    );
    let (_dir, store) = store();
    let policy = Policy {
        model_urls: [(sha(&body), server.url("/redirect"))]
            .into_iter()
            .collect(),
        ..Policy::default()
    };
    let request = model_request("https://unreachable.invalid/m.onnx", &sha(&body), None);
    let plan = plan_download(&request, &policy, None).unwrap();
    assert!(plan.files[0].from_policy);
    // The file keeps the name the request gives it.
    assert_eq!(plan.main_file_name, "m.onnx");
    let model = fetch(&store, &request, &policy).unwrap();
    assert_eq!(fs::read(model.path()).unwrap(), body);
    assert_eq!(
        server
            .seen()
            .iter()
            .map(|s| s.0.as_str())
            .collect::<Vec<_>>(),
        ["/redirect", "/site/m.onnx"]
    );
}

#[test]
fn companions_are_stored_next_to_the_main_file() {
    let server = TestServer::start();
    let graph = payload(2_000, 10);
    let weights = payload(30_000, 11);
    let other = payload(1_000, 12);
    server.route("/h/model.onnx", Route::file(graph.clone()));
    server.route("/h/model.onnx.data", Route::file(weights.clone()));
    server.route("/h/extra.bin", Route::file(other.clone()));
    let (_dir, store) = store();
    let mut with_weights = model_request(&server.url("/h/model.onnx"), &sha(&graph), None);
    with_weights.companions.push(FileRequest {
        sha256: sha(&weights),
        size: Some(30_000),
        file_name: Some("model.onnx.data".into()),
        url: None,
    });
    let plan = plan_download(&with_weights, &Policy::default(), None).unwrap();
    assert_eq!(plan.files[1].url.as_str(), server.url("/h/model.onnx.data"));
    let model = fetch(&store, &with_weights, &Policy::default()).unwrap();
    assert_eq!(
        fs::read(model.dir.join("model.onnx.data")).unwrap(),
        weights
    );
    assert!(model.satisfies(&with_weights));
    assert_eq!(model.total_size(), 32_000);
    // Another companion joins the stored model; only it is downloaded.
    let mut with_extra = with_weights.clone();
    with_extra.companions.push(FileRequest {
        sha256: sha(&other),
        size: None,
        file_name: None,
        url: Some("extra.bin".into()),
    });
    let before = server.seen().len();
    let model = fetch(&store, &with_extra, &Policy::default()).unwrap();
    assert_eq!(model.record.companions.len(), 2);
    assert_eq!(server.seen().len(), before + 1);
    assert_eq!(store.resolve(&sha(&graph)).unwrap(), model);
    // A different file under a name the model has is refused.
    server.route("/h/fake.data", Route::file(other.clone()));
    let mut conflicting = with_weights.clone();
    conflicting.companions[0] = FileRequest {
        sha256: sha(&other),
        size: None,
        file_name: Some("model.onnx.data".into()),
        url: Some("fake.data".into()),
    };
    let error = fetch(&store, &conflicting, &Policy::default()).unwrap_err();
    assert!(matches!(error, FetchError::Conflict(_)), "{error:?}");
    assert_eq!(
        fs::read(model.dir.join("model.onnx.data")).unwrap(),
        weights,
        "the stored weights are untouched"
    );
}

#[test]
fn models_are_imported_after_verification() {
    let (dir, store) = store();
    let graph = payload(3_000, 13);
    let weights = payload(9_000, 14);
    let source = dir.path().join("usb");
    fs::create_dir_all(&source).unwrap();
    fs::write(source.join("model.onnx"), &graph).unwrap();
    fs::write(source.join("model.onnx.data"), &weights).unwrap();
    let never = AtomicBool::new(false);
    let error = store
        .import(
            &source.join("model.onnx"),
            Some(&sha(b"nope")),
            &[],
            None,
            None,
            &never,
        )
        .unwrap_err();
    assert!(matches!(error, FetchError::Integrity { .. }), "{error:?}");
    assert!(store.list().is_empty());
    let model = store
        .import(
            &source.join("model.onnx"),
            Some(&sha(&graph).to_ascii_uppercase()),
            &[(source.join("model.onnx.data"), Some(sha(&weights)))],
            Some("Imported"),
            None,
            &never,
        )
        .unwrap();
    assert_eq!(model.record.source, None);
    assert_eq!(model.record.requested_by.as_deref(), Some(CLI_REQUESTER));
    assert_eq!(fs::read(model.path()).unwrap(), graph);
    assert_eq!(
        fs::read(model.dir.join("model.onnx.data")).unwrap(),
        weights
    );
    // The originals stay where they were.
    assert!(source.join("model.onnx").exists());
    assert!(partial_files(&store).is_empty());
    let wrong_companion = store.import(
        &source.join("model.onnx"),
        None,
        &[(source.join("model.onnx.data"), Some(sha(b"x")))],
        None,
        None,
        &never,
    );
    assert!(wrong_companion.is_err());
}

#[test]
fn verify_finds_damaged_files_and_remove_deletes_models() {
    let (dir, store) = store();
    let body = payload(4_000, 15);
    let file = dir.path().join("m.onnx");
    fs::write(&file, &body).unwrap();
    let never = AtomicBool::new(false);
    let model = store.import(&file, None, &[], None, None, &never).unwrap();
    let reports = store.verify(&never, &|_| {});
    assert_eq!(reports.len(), 1);
    assert!(reports[0].problems.is_empty());
    let mut damaged = body.clone();
    damaged[0] ^= 0xff;
    fs::write(model.path(), &damaged).unwrap();
    let reports = store.verify(&never, &|_| {});
    assert_eq!(reports[0].problems.len(), 1);
    assert!(
        reports[0].problems[0].contains(&sha(&body)),
        "{:?}",
        reports[0]
    );
    // A model missing a file is not offered to plugins.
    fs::remove_file(model.path()).unwrap();
    assert_eq!(store.resolve(&sha(&body)), None);
    store.remove(&model).unwrap();
    assert!(store.list().is_empty() && !model.dir.exists());
}

#[test]
fn read_only_stores_are_offered_but_never_changed() {
    let (dir, user) = store();
    let body = payload(2_500, 16);
    let file = dir.path().join("m.onnx");
    fs::write(&file, &body).unwrap();
    let never = AtomicBool::new(false);
    // The administrator's store, prepared the same way.
    let admin_root = dir.path().join("admin");
    ModelStore::new(admin_root.clone(), Vec::new())
        .import(&file, None, &[], Some("Site model"), None, &never)
        .unwrap();
    let store = ModelStore::new(user.root().to_path_buf(), vec![admin_root]);
    let model = store.resolve(&sha(&body)).unwrap();
    assert!(model.read_only && model.to_protocol().read_only);
    assert_eq!(store.list().len(), 1);
    assert!(matches!(store.remove(&model), Err(FetchError::Refused(_))));
    assert!(model.path().exists());
}

#[test]
fn requests_from_plugins_are_checked() {
    let args = ModelFetchArgs {
        url: Some("https://models.example.org/m.onnx".into()),
        sha256: "70A8DEEF1F01716EFBC87FEB181EA398425D08B5E4984AD9D4AB2B83FD354F92".into(),
        name: Some("  Reference  ".into()),
        ..Default::default()
    };
    let request = ModelRequest::from_args(&args, "eovae").unwrap();
    assert_eq!(
        request.main.sha256,
        "70a8deef1f01716efbc87feb181ea398425d08b5e4984ad9d4ab2b83fd354f92"
    );
    assert_eq!(request.name.as_deref(), Some("Reference"));
    let reserved = ModelFetchArgs {
        file_name: Some(RECORD_FILE.into()),
        ..args.clone()
    };
    assert!(ModelRequest::from_args(&reserved, "eovae").is_err());
    // A URL without a file name needs one given.
    let bare = ModelRequest::from_args(
        &ModelFetchArgs {
            url: Some("https://models.example.org/".into()),
            ..args
        },
        "eovae",
    )
    .unwrap();
    assert!(plan_download(&bare, &Policy::default(), None).is_err());
}

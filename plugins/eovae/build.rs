use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux" => println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN"),
        "macos" => println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path"),
        _ => {}
    }

    if let Err(error) = stage_vendored_ort_sidecars() {
        println!("cargo:warning=failed to stage vendored ONNX Runtime sidecars: {error}");
    }
}

fn stage_vendored_ort_sidecars() -> Result<(), String> {
    let target = env::var("TARGET").map_err(|error| error.to_string())?;
    let out_dir = PathBuf::from(env::var("OUT_DIR").map_err(|error| error.to_string())?);
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .ok_or_else(|| format!("unexpected OUT_DIR layout: {}", out_dir.display()))?
        .to_path_buf();

    let Some(lib_dir) = locate_ort_lib_dir(&target)? else {
        return Ok(());
    };
    let sidecars = collect_sidecars(&lib_dir)?;
    if sidecars.is_empty() {
        return Ok(());
    }

    for source in sidecars {
        println!("cargo:rerun-if-changed={}", source.display());
        for destination_dir in [&profile_dir, &profile_dir.join("deps")] {
            fs::create_dir_all(destination_dir).map_err(|error| error.to_string())?;
            let destination =
                destination_dir.join(source.file_name().ok_or_else(|| {
                    format!("sidecar path has no file name: {}", source.display())
                })?);
            copy_if_needed(&source, &destination)?;
        }
    }

    Ok(())
}

fn locate_ort_lib_dir(target: &str) -> Result<Option<PathBuf>, String> {
    let cache_root = ort_cache_root()?
        .join("ort.pyke.io")
        .join("dfbin")
        .join(target);
    if !cache_root.is_dir() {
        return Ok(None);
    }

    let mut candidates = fs::read_dir(&cache_root)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path().join("onnxruntime").join("lib"))
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.reverse();

    Ok(candidates.into_iter().find(|path| has_sidecars(path)))
}

fn ort_cache_root() -> Result<PathBuf, String> {
    if let Ok(root) = env::var("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(root));
    }
    let home = env::var("HOME").map_err(|error| error.to_string())?;
    Ok(PathBuf::from(home).join(".cache"))
}

fn has_sidecars(lib_dir: &Path) -> bool {
    collect_sidecars(lib_dir)
        .map(|entries| !entries.is_empty())
        .unwrap_or(false)
}

fn collect_sidecars(lib_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut entries = fs::read_dir(lib_dir)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| {
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                return false;
            };
            (name.starts_with("libonnxruntime")
                && (name.contains(".so") || name.ends_with(".dylib")))
                || (name.starts_with("onnxruntime") && name.ends_with(".dll"))
        })
        .collect::<Vec<_>>();
    entries.sort();
    Ok(entries)
}

fn copy_if_needed(source: &Path, destination: &Path) -> Result<(), String> {
    let should_copy = match fs::metadata(destination) {
        Ok(metadata) => {
            metadata.len()
                != fs::metadata(source)
                    .map_err(|error| error.to_string())?
                    .len()
        }
        Err(_) => true,
    };
    if should_copy {
        fs::copy(source, destination).map_err(|error| error.to_string())?;
    }
    Ok(())
}

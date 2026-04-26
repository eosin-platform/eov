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

    if let Err(error) = stage_histovae_runtime_libraries() {
        println!("cargo:warning=failed to stage bundled HistoVAE ONNX Runtime libraries: {error}");
    }
}

fn stage_histovae_runtime_libraries() -> Result<(), String> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").map_err(|error| error.to_string())?);
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .ok_or_else(|| format!("unexpected OUT_DIR layout: {}", out_dir.display()))?
        .to_path_buf();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").map_err(|error| error.to_string())?);

    let Some(runtime_dirs) = locate_histovae_runtime_dirs(&manifest_dir)? else {
        return Ok(());
    };
    let runtime_libraries = collect_runtime_libraries(&runtime_dirs)?;
    if runtime_libraries.is_empty() {
        return Ok(());
    }

    for source in runtime_libraries {
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

#[derive(Clone, Debug)]
struct RuntimeDirs {
    ort_capi_dir: PathBuf,
    cuda_dir: Option<PathBuf>,
    cudnn_dir: Option<PathBuf>,
    cuda_toolkit_dir: Option<PathBuf>,
}

fn locate_histovae_runtime_dirs(manifest_dir: &Path) -> Result<Option<RuntimeDirs>, String> {
    let venv_lib_dir = manifest_dir.join("HistoVAE").join(".venv").join("lib");
    if !venv_lib_dir.is_dir() {
        return Ok(None);
    }

    let mut site_packages = fs::read_dir(&venv_lib_dir)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path().join("site-packages"))
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    site_packages.sort();
    site_packages.reverse();

    for site_packages_dir in site_packages {
        let ort_capi_dir = site_packages_dir.join("onnxruntime").join("capi");
        if !ort_capi_dir.is_dir() {
            continue;
        }

        let cuda_dir = [
            site_packages_dir.join("nvidia").join("cu12").join("lib"),
            site_packages_dir.join("nvidia").join("cu13").join("lib"),
            site_packages_dir.join("nvidia").join("cuda_runtime").join("lib"),
        ]
        .into_iter()
        .find(|path| path.is_dir());
        let cudnn_dir = [
            site_packages_dir.join("nvidia").join("cudnn").join("lib"),
        ]
        .into_iter()
        .find(|path| path.is_dir());

        return Ok(Some(RuntimeDirs {
            ort_capi_dir,
            cuda_dir,
            cudnn_dir,
            cuda_toolkit_dir: locate_cuda_toolkit_dir(),
        }));
    }

    Ok(None)
}

fn locate_cuda_toolkit_dir() -> Option<PathBuf> {
    [
        "/usr/local/cuda-12.9/targets/x86_64-linux/lib",
        "/usr/local/cuda-12/targets/x86_64-linux/lib",
        "/usr/local/cuda/targets/x86_64-linux/lib",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| path.is_dir())
}

fn collect_runtime_libraries(runtime_dirs: &RuntimeDirs) -> Result<Vec<PathBuf>, String> {
    let mut entries = Vec::new();
    entries.extend(collect_matching_files(&runtime_dirs.ort_capi_dir, |name| {
        (name.starts_with("libonnxruntime") && (name.contains(".so") || name.ends_with(".dylib")))
            || (name.starts_with("onnxruntime") && name.ends_with(".dll"))
    })?);

    if let Some(cuda_toolkit_dir) = &runtime_dirs.cuda_toolkit_dir {
        entries.extend(collect_matching_files(cuda_toolkit_dir, |name| {
            name.starts_with("libcublasLt.so.12")
                || name.starts_with("libcublas.so.12")
                || name.starts_with("libcufft.so.11")
                || name.starts_with("libcudart.so.12")
        })?);
    }

    if let Some(cuda_dir) = &runtime_dirs.cuda_dir {
        entries.extend(collect_matching_files(cuda_dir, |name| {
            name.starts_with("libcurand.so.10")
        })?);
    }

    if let Some(cudnn_dir) = &runtime_dirs.cudnn_dir {
        entries.extend(collect_matching_files(cudnn_dir, |name| {
            name.starts_with("libcudnn") && name.contains(".so.9")
        })?);
    }

    entries.sort();
    entries.dedup();
    Ok(entries)
}

fn collect_matching_files(
    directory: &Path,
    predicate: impl Fn(&str) -> bool,
) -> Result<Vec<PathBuf>, String> {
    let mut entries = fs::read_dir(directory)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(&predicate)
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

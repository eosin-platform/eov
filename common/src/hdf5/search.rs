//! Where eov looks for the HDF5 library, and what it says when there is
//! none.
//!
//! In order: the library named with `--hdf5-library` or
//! [`LIBRARY_ENV`] (only that one, when given), then `$HDF5_DIR/lib`,
//! `$CONDA_PREFIX/lib`, and the platform's names through the system's
//! loader (plus Homebrew's and MacPorts' folders on macOS).

use std::fmt;
use std::path::{Path, PathBuf};

/// The environment variable that names the library, like `--hdf5-library`.
pub const LIBRARY_ENV: &str = "EOV_HDF5_LIBRARY";

/// The library's file names, the unversioned development links first, then
/// the sonames of 1.14, 1.12 and 1.10 and their Debian `_serial` variants.
#[cfg(all(unix, not(target_os = "macos")))]
const NAMES: &[&str] = &[
    "libhdf5.so",
    "libhdf5_serial.so",
    "libhdf5.so.310",
    "libhdf5_serial.so.310",
    "libhdf5.so.200",
    "libhdf5_serial.so.200",
    "libhdf5.so.103",
    "libhdf5_serial.so.103",
];
#[cfg(target_os = "macos")]
const NAMES: &[&str] = &[
    "libhdf5.dylib",
    "libhdf5.310.dylib",
    "libhdf5.200.dylib",
    "libhdf5.103.dylib",
];
#[cfg(windows)]
const NAMES: &[&str] = &["hdf5.dll", "libhdf5.dll"];
#[cfg(not(any(unix, windows)))]
const NAMES: &[&str] = &[];

/// Folders tried after the loader's own search, which on macOS does not
/// include Homebrew's or MacPorts' prefix.
#[cfg(target_os = "macos")]
const SYSTEM_FOLDERS: &[&str] = &["/opt/homebrew/lib", "/usr/local/lib", "/opt/local/lib"];
#[cfg(not(target_os = "macos"))]
const SYSTEM_FOLDERS: &[&str] = &[];

/// Where the library is below `$HDF5_DIR` and `$CONDA_PREFIX`. Windows
/// keeps DLLs in `bin` (and conda in `Library\bin`).
#[cfg(windows)]
const HDF5_DIR_FOLDERS: &[&str] = &["bin", "lib"];
#[cfg(not(windows))]
const HDF5_DIR_FOLDERS: &[&str] = &["lib", "lib64"];
#[cfg(windows)]
const CONDA_FOLDERS: &[&str] = &["Library\\bin"];
#[cfg(not(windows))]
const CONDA_FOLDERS: &[&str] = &["lib"];

/// A library file to try, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Candidate {
    /// A path, or a bare file name for the system's loader to find.
    pub(crate) path: PathBuf,
    /// `--hdf5-library`, `EOV_HDF5_LIBRARY`, `HDF5_DIR`, `CONDA_PREFIX` or
    /// `system`.
    pub(crate) origin: &'static str,
}

/// Where to look for the library.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LibrarySearch {
    /// The library (or a folder with it) named by `--hdf5-library` or
    /// `EOV_HDF5_LIBRARY`, and which of the two. When set, nothing else is
    /// tried: a library that was asked for by name is never replaced by
    /// another one.
    pub(crate) explicit: Option<(PathBuf, &'static str)>,
    /// `$HDF5_DIR`, an HDF5 installation prefix.
    pub(crate) hdf5_dir: Option<PathBuf>,
    /// `$CONDA_PREFIX`, the active conda environment.
    pub(crate) conda_prefix: Option<PathBuf>,
    /// Whether to try the platform's names through the system's loader.
    pub(crate) system: bool,
}

impl LibrarySearch {
    /// The search for `--hdf5-library` (when given) and the environment.
    pub(crate) fn from_env(option: Option<&Path>) -> Self {
        let var = |name: &str| {
            std::env::var_os(name)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        };
        let explicit = match option {
            Some(path) => Some((path.to_path_buf(), "--hdf5-library")),
            None => var(LIBRARY_ENV).map(|path| (path, LIBRARY_ENV)),
        };
        Self {
            explicit,
            hdf5_dir: var("HDF5_DIR"),
            conda_prefix: var("CONDA_PREFIX"),
            system: true,
        }
    }

    /// The files to try, in order.
    pub(crate) fn candidates(&self) -> Vec<Candidate> {
        if let Some((path, origin)) = &self.explicit {
            if path.is_dir() {
                return names_in(path, origin).collect();
            }
            // A file that exists is loaded by its absolute path; a bare name
            // is left to the system's loader.
            let path = if path.is_file() {
                std::path::absolute(path).unwrap_or_else(|_| path.clone())
            } else {
                path.clone()
            };
            return vec![Candidate { path, origin }];
        }
        let mut candidates = Vec::new();
        if let Some(prefix) = &self.hdf5_dir {
            for folder in HDF5_DIR_FOLDERS {
                candidates.extend(names_in(&prefix.join(folder), "HDF5_DIR"));
            }
        }
        if let Some(prefix) = &self.conda_prefix {
            for folder in CONDA_FOLDERS {
                candidates.extend(names_in(&prefix.join(folder), "CONDA_PREFIX"));
            }
        }
        if self.system {
            candidates.extend(names_in(Path::new(""), "system"));
            for folder in SYSTEM_FOLDERS {
                candidates.extend(names_in(Path::new(folder), "system"));
            }
        }
        candidates
    }
}

/// The platform's names in `folder` (bare names for an empty one).
fn names_in(folder: &Path, origin: &'static str) -> impl Iterator<Item = Candidate> {
    let folder = folder.to_path_buf();
    NAMES.iter().map(move |name| Candidate {
        path: folder.join(name),
        origin,
    })
}

/// No usable HDF5 library: every file tried and why it was not used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoadError {
    pub(crate) tried: Vec<(Candidate, String)>,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.tried.as_slice() {
            [] => writeln!(
                f,
                "eov could not load the HDF5 library: it knows no library names on this platform."
            )?,
            [(candidate, reason)] if matches!(candidate.origin, "--hdf5-library" | LIBRARY_ENV) => {
                writeln!(
                    f,
                    "eov could not load the HDF5 library {} (from {}): {}",
                    candidate.path.display(),
                    candidate.origin,
                    without_path(reason, &candidate.path)
                )?;
            }
            tried => {
                writeln!(f, "eov could not load the HDF5 library. It tried:")?;
                for (candidate, reason) in tried {
                    writeln!(
                        f,
                        "  {} ({}): {}",
                        candidate.path.display(),
                        candidate.origin,
                        without_path(reason, &candidate.path)
                    )?;
                }
            }
        }
        writeln!(f, "HDF5 1.10 or newer is needed. To install it:")?;
        for line in INSTALL {
            writeln!(f, "  {line}")?;
        }
        writeln!(
            f,
            "Or name the library file with --hdf5-library <PATH> or {LIBRARY_ENV}=<PATH>."
        )?;
        write!(
            f,
            "--format zarr writes the same arrays and needs no library."
        )
    }
}

impl std::error::Error for LoadError {}

/// The loader's message usually starts with the file name, which the
/// listing already shows.
fn without_path<'a>(reason: &'a str, path: &Path) -> &'a str {
    let path = path.to_string_lossy();
    reason
        .strip_prefix(path.as_ref())
        .and_then(|rest| rest.strip_prefix(": "))
        .unwrap_or(reason)
}

#[cfg(all(unix, not(target_os = "macos")))]
const INSTALL: &[&str] = &[
    "Debian, Ubuntu: sudo apt install libhdf5-310 (Debian 13, Ubuntu 25.04 and later), \
     libhdf5-103-1t64 (Ubuntu 24.04) or libhdf5-103-1 (Debian 12, Ubuntu 22.04)",
    "Fedora: sudo dnf install hdf5",
    "Arch: sudo pacman -S hdf5",
    "conda: conda install -c conda-forge hdf5 (eov looks in $CONDA_PREFIX/lib)",
];
#[cfg(target_os = "macos")]
const INSTALL: &[&str] = &[
    "Homebrew: brew install hdf5",
    "MacPorts: sudo port install hdf5",
    "conda: conda install -c conda-forge hdf5 (eov looks in $CONDA_PREFIX/lib)",
];
#[cfg(windows)]
const INSTALL: &[&str] = &[
    "conda: conda install -c conda-forge hdf5 (eov looks in %CONDA_PREFIX%\\Library\\bin)",
    "the HDF Group's installer (https://github.com/HDFGroup/hdf5/releases), \
     with its bin folder on PATH or in HDF5_DIR",
];
#[cfg(not(any(unix, windows)))]
const INSTALL: &[&str] = &["build HDF5 from https://github.com/HDFGroup/hdf5"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_library_named_explicitly_is_the_only_one_tried() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("libhdf5-custom.so");
        std::fs::write(&file, b"").unwrap();
        let search = LibrarySearch {
            explicit: Some((file.clone(), LIBRARY_ENV)),
            hdf5_dir: Some(temp.path().into()),
            conda_prefix: Some(temp.path().into()),
            system: true,
        };
        assert_eq!(
            search.candidates(),
            [Candidate {
                path: file,
                origin: LIBRARY_ENV
            }]
        );

        // A folder: the platform's names in it.
        let search = LibrarySearch {
            explicit: Some((temp.path().into(), "--hdf5-library")),
            ..LibrarySearch::default()
        };
        let candidates = search.candidates();
        assert_eq!(candidates.len(), NAMES.len());
        assert!(
            candidates
                .iter()
                .all(|c| c.path.parent() == Some(temp.path()))
        );
    }

    #[test]
    fn prefixes_come_before_the_system_names() {
        let search = LibrarySearch {
            explicit: None,
            hdf5_dir: Some("/opt/hdf5".into()),
            conda_prefix: Some("/opt/conda".into()),
            system: true,
        };
        let origins: Vec<&str> = search.candidates().iter().map(|c| c.origin).collect();
        let first = |origin| origins.iter().position(|o| *o == origin);
        if NAMES.is_empty() {
            return;
        }
        assert_eq!(first("HDF5_DIR"), Some(0));
        assert!(first("HDF5_DIR") < first("CONDA_PREFIX"));
        assert!(first("CONDA_PREFIX") < first("system"));
        let candidates = search.candidates();
        assert!(
            candidates[0]
                .path
                .starts_with(Path::new("/opt/hdf5").join(HDF5_DIR_FOLDERS[0]))
        );
        // The system's names are bare, for the loader to find.
        let system = candidates.iter().find(|c| c.origin == "system").unwrap();
        assert_eq!(system.path, Path::new(NAMES[0]));
    }

    #[test]
    fn the_error_lists_what_was_tried_and_what_to_install() {
        let error = LoadError {
            tried: vec![
                (
                    Candidate {
                        path: "/x/libhdf5.so".into(),
                        origin: "HDF5_DIR",
                    },
                    "/x/libhdf5.so: cannot open shared object file".into(),
                ),
                (
                    Candidate {
                        path: "libhdf5.so.103".into(),
                        origin: "system",
                    },
                    "HDF5 1.8.22 is too old".into(),
                ),
            ],
        };
        let text = error.to_string();
        assert!(text.contains("/x/libhdf5.so (HDF5_DIR): cannot open shared object file"));
        assert!(text.contains("libhdf5.so.103 (system): HDF5 1.8.22 is too old"));
        assert!(text.contains("--hdf5-library"));
        assert!(text.contains("--format zarr"));
        assert!(text.contains(INSTALL[0]));
    }
}

//! Where release metadata and artifacts come from: GitHub, or an offline
//! bundle for air-gapped machines.
//!
//! An offline bundle is a directory holding an eov `release.toml`, ideally
//! its `release.toml.minisig`, and any of the artifacts the manifest lists,
//! under their published file names (plugin packages may also sit in a
//! `plugins/` subdirectory). Every file is verified against the SHA-256 in
//! the manifest, and the manifest itself must carry a signature that
//! verifies against eov's embedded release keys unless `--allow-unsigned`
//! is given.

use super::manifest::{MANIFEST_FILE, PlatformArtifact, release_tag};
use super::network::{MAX_METADATA_BYTES, NetworkPermission, ReleaseClient};
use super::plugins::{self, Catalog, PackageFetch, ResolvedPackage};
use super::signature::{SIGNATURE_SUFFIX, SignatureStatus, Verifier};
use anyhow::{Context, Result, bail};
use semver::Version;
use std::fs;
use std::path::{Path, PathBuf};

/// The origin of an update.
pub enum ReleaseSource {
    Network { permission: NetworkPermission },
    Offline { dir: PathBuf, allow_unsigned: bool },
}

impl ReleaseSource {
    /// An offline bundle: a directory, or the `release.toml` inside it.
    pub fn offline(from: &Path, allow_unsigned: bool) -> Result<Self> {
        let dir = if from.is_dir() {
            from.to_path_buf()
        } else if from.file_name().is_some_and(|name| name == MANIFEST_FILE) {
            from.parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."))
        } else {
            bail!(
                "{} is neither an offline update directory nor its {MANIFEST_FILE}",
                from.display()
            );
        };
        if !dir.join(MANIFEST_FILE).is_file() {
            bail!("{} contains no {MANIFEST_FILE}", dir.display());
        }
        Ok(Self::Offline {
            dir,
            allow_unsigned,
        })
    }

    pub fn permission(&self) -> Option<&NetworkPermission> {
        match self {
            Self::Network { permission } => Some(permission),
            Self::Offline { .. } => None,
        }
    }

    pub fn is_offline(&self) -> bool {
        matches!(self, Self::Offline { .. })
    }

    pub fn describe(&self, version: &Version) -> String {
        match self {
            Self::Network { .. } => format!("GitHub release {}", release_tag(version)),
            Self::Offline { dir, .. } => format!("offline bundle {}", dir.display()),
        }
    }

    /// The verified eov manifest (`release` selects a version; the latest
    /// release otherwise).
    pub fn catalog(
        &self,
        client: &ReleaseClient,
        verifier: &Verifier,
        release: Option<&Version>,
    ) -> Result<Catalog> {
        match self {
            Self::Network { permission } => plugins::fetch_catalog(
                client,
                permission,
                verifier,
                release.map(release_tag).as_deref(),
                release,
            ),
            Self::Offline {
                dir,
                allow_unsigned,
            } => {
                let text = read_limited(&dir.join(MANIFEST_FILE))?;
                let signature_path = dir.join(format!("{MANIFEST_FILE}{SIGNATURE_SUFFIX}"));
                let signature = match signature_path.is_file() {
                    true => Some(read_limited(&signature_path)?),
                    false => None,
                };
                let catalog =
                    plugins::catalog_from_text(&text, signature.as_deref(), verifier, release)?;
                if !matches!(catalog.signature, SignatureStatus::Verified { .. }) && !allow_unsigned
                {
                    bail!(
                        "The offline bundle's {MANIFEST_FILE} is not signed with a trusted eov release key ({}). Offline bundles are only trusted when signed; if you verified this bundle yourself, re-run with --allow-unsigned.",
                        catalog.signature.describe()
                    );
                }
                Ok(catalog)
            }
        }
    }

    /// Fetches a verified artifact into a temporary file in `directory`.
    pub fn fetch(
        &self,
        client: &ReleaseClient,
        artifact: &PlatformArtifact,
        directory: &Path,
    ) -> Result<tempfile::TempPath> {
        fs::create_dir_all(directory)
            .with_context(|| format!("could not create {}", directory.display()))?;
        match self {
            Self::Network { permission } => Ok(client
                .download_verified(permission, &artifact.url, &artifact.sha256, directory)?
                .into_temp_path()),
            Self::Offline { dir, .. } => {
                let path = self.locate(dir, artifact)?;
                plugins::copy_verified(&path, &artifact.sha256, directory)
            }
        }
    }

    /// Points a resolved plugin package at the offline bundle.
    pub fn localize(&self, package: &mut ResolvedPackage) -> Result<()> {
        let Self::Offline { dir, .. } = self else {
            return Ok(());
        };
        let PackageFetch::Download { url, sha256 } = &package.fetch else {
            return Ok(());
        };
        let artifact = PlatformArtifact {
            version: package.version.to_string(),
            sha256: sha256.clone(),
            url: url.clone(),
            environment: None,
        };
        package.fetch = PackageFetch::LocalFile {
            path: self.locate(dir, &artifact)?,
            sha256: sha256.clone(),
        };
        Ok(())
    }

    fn locate(&self, dir: &Path, artifact: &PlatformArtifact) -> Result<PathBuf> {
        let name = artifact
            .file_name()
            .with_context(|| format!("artifact URL {} has no file name", artifact.url))?;
        [dir.join(name), dir.join("plugins").join(name)]
            .into_iter()
            .find(|path| path.is_file())
            .with_context(|| format!("the offline bundle {} has no {name}", dir.display()))
    }
}

fn read_limited(path: &Path) -> Result<String> {
    let length = fs::metadata(path)
        .with_context(|| format!("could not read {}", path.display()))?
        .len();
    if length > MAX_METADATA_BYTES {
        bail!(
            "{} is larger than the {MAX_METADATA_BYTES}-byte limit",
            path.display()
        );
    }
    fs::read_to_string(path).with_context(|| format!("could not read {}", path.display()))
}

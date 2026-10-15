//! Verification of signed release manifests.
//!
//! A release may publish `release.toml.minisig`, a minisign signature over
//! the exact bytes of `release.toml`, produced by
//! `scripts/release_signing.py` (or `minisign`). Its signed *trusted comment*
//! names the manifest kind, version and repository, so a valid signature
//! cannot be replayed for a different release (fields are separated by tabs):
//!
//! ```text
//! timestamp:1760000000 file:release.toml kind:eov version:1.0.1 repository:https://github.com/eosin-platform/eov
//! ```
//!
//! Rules, in order:
//!
//! 1. A signature that is present must verify against a trusted key and
//!    match the manifest identity; otherwise nothing is changed.
//! 2. A missing signature is refused when the manifest declares one
//!    (`[manifest] signature = "minisign"`), when the deployment policy sets
//!    `require_signatures`, or when this build embeds release keys and the
//!    release is at or after [`SIGNED_SINCE`](super::keys::SIGNED_SINCE).
//! 3. Otherwise the release is accepted and reported as unsigned. A build
//!    without embedded keys reports present signatures as unverifiable.

use super::keys::{RELEASE_KEYS, SIGNED_SINCE, TrustedKey};
use super::manifest::{canonical_repository, normalize_version};
use super::policy::Policy;
use anyhow::{Context, Result, bail};
use semver::Version;
use serde::Serialize;
use std::collections::BTreeMap;

/// File-name suffix of a manifest signature.
pub const SIGNATURE_SUFFIX: &str = ".minisig";

/// The result of a successful verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SignatureStatus {
    /// Signed by a trusted key.
    Verified { key: String },
    /// Not signed, and no signature is required.
    Unsigned,
    /// Signed, but this build has no trusted keys to check it with.
    Unverifiable,
}

impl SignatureStatus {
    pub fn describe(&self) -> String {
        match self {
            Self::Verified { key } => format!("verified (key {key})"),
            Self::Unsigned => "none (this release is not signed)".to_string(),
            Self::Unverifiable => {
                "present, but this eov build has no release keys to verify it".to_string()
            }
        }
    }
}

/// What a manifest claims to be; must match its signature.
#[derive(Debug, Clone, Copy)]
pub struct ManifestIdentity<'a> {
    pub kind: &'a str,
    pub version: &'a Version,
    pub repository: &'a str,
}

/// Verifies manifest signatures with a fixed key set.
pub struct Verifier {
    keys: Vec<(String, minisign_verify::PublicKey)>,
    signed_since: Option<Version>,
    require_by_policy: bool,
}

impl Verifier {
    /// The keys compiled into this build, with the deployment policy.
    pub fn for_build(policy: &Policy) -> Result<Self> {
        Self::new(
            RELEASE_KEYS,
            Some(Version::parse(SIGNED_SINCE).context("invalid SIGNED_SINCE")?),
            policy.require_signatures,
        )
    }

    pub fn new(
        keys: &[TrustedKey],
        signed_since: Option<Version>,
        require_by_policy: bool,
    ) -> Result<Self> {
        let keys = keys
            .iter()
            .map(|key| {
                minisign_verify::PublicKey::from_base64(key.public_key)
                    .map(|public_key| (key.name.to_string(), public_key))
                    .map_err(|error| anyhow::anyhow!("invalid release key {}: {error}", key.name))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            keys,
            signed_since,
            require_by_policy,
        })
    }

    /// Whether this verifier would refuse an unsigned manifest of `version`.
    pub fn requires_signature(&self, version: &Version, declared: bool) -> bool {
        declared
            || self.require_by_policy
            || (!self.keys.is_empty()
                && self
                    .signed_since
                    .as_ref()
                    .is_some_and(|since| version >= since))
    }

    /// Checks `signature` (the `.minisig` text, if any) over `manifest`.
    pub fn verify(
        &self,
        manifest: &[u8],
        signature: Option<&str>,
        declared: bool,
        identity: ManifestIdentity<'_>,
    ) -> Result<SignatureStatus> {
        let what = format!(
            "the {} {} release manifest",
            identity.kind, identity.version
        );
        let Some(signature) = signature else {
            if declared {
                bail!(
                    "{what} declares a signature, but release.toml{SIGNATURE_SUFFIX} is missing. Nothing was changed."
                );
            }
            if self.requires_signature(identity.version, false) {
                bail!(
                    "{what} is not signed, but {} requires signed release metadata. Nothing was changed.",
                    if self.require_by_policy {
                        "the administrator policy"
                    } else {
                        "this eov build"
                    }
                );
            }
            return Ok(SignatureStatus::Unsigned);
        };

        let signature = minisign_verify::Signature::decode(signature)
            .map_err(|error| anyhow::anyhow!("{what} has a malformed signature: {error}"))?;
        if self.keys.is_empty() {
            if self.require_by_policy {
                bail!(
                    "{what} is signed, but this eov build has no release keys and the administrator policy requires verified signatures."
                );
            }
            return Ok(SignatureStatus::Unverifiable);
        }
        let key = self
            .keys
            .iter()
            .find(|(_, key)| key.verify(manifest, &signature, false).is_ok())
            .map(|(name, _)| name.clone())
            .with_context(|| {
                format!(
                    "the signature of {what} does not verify against the trusted eov release keys. The metadata may have been tampered with; nothing was changed."
                )
            })?;
        check_trusted_comment(signature.trusted_comment(), identity)
            .with_context(|| format!("the signature of {what} belongs to a different release"))?;
        Ok(SignatureStatus::Verified { key })
    }
}

fn check_trusted_comment(comment: &str, identity: ManifestIdentity<'_>) -> Result<()> {
    let fields: BTreeMap<&str, &str> = comment
        .split('\t')
        .filter_map(|field| field.split_once(':'))
        .collect();
    let field = |name: &str| {
        fields
            .get(name)
            .copied()
            .with_context(|| format!("the signed comment has no '{name}' field"))
    };
    if field("kind")? != identity.kind {
        bail!(
            "it was made for a '{}' manifest, not '{}'",
            field("kind")?,
            identity.kind
        );
    }
    if normalize_version(field("version")?)? != *identity.version {
        bail!("it was made for version {}", field("version")?);
    }
    if canonical_repository(field("repository")?)? != canonical_repository(identity.repository)? {
        bail!("it was made for repository {}", field("repository")?);
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod test_keys {
    //! A key pair used only by tests; its secret half is not stored anywhere.
    use super::TrustedKey;

    pub(crate) const TEST_KEY: TrustedKey = TrustedKey {
        name: "test",
        public_key: include_str!("../../tests/data/signing/test-key.pub.b64"),
    };
}

#[cfg(test)]
mod tests {
    use super::test_keys::TEST_KEY;
    use super::*;

    const MANIFEST: &str = include_str!("../../tests/data/signing/release.toml");
    const SIGNATURE: &str = include_str!("../../tests/data/signing/release.toml.minisig");
    const WRONG_IDENTITY: &str =
        include_str!("../../tests/data/signing/release.toml.other-version.minisig");
    const REPOSITORY: &str = "https://github.com/eosin-platform/eov";

    fn identity(version: &Version) -> ManifestIdentity<'_> {
        ManifestIdentity {
            kind: "eov",
            version,
            repository: REPOSITORY,
        }
    }

    fn verifier(require: bool) -> Verifier {
        Verifier::new(&[TEST_KEY], Some(Version::new(1, 0, 1)), require).unwrap()
    }

    #[test]
    fn valid_signature_is_verified() {
        let version = Version::new(1, 0, 1);
        let status = verifier(false)
            .verify(
                MANIFEST.as_bytes(),
                Some(SIGNATURE),
                true,
                identity(&version),
            )
            .unwrap();
        assert_eq!(status, SignatureStatus::Verified { key: "test".into() });
    }

    #[test]
    fn tampered_manifest_is_rejected() {
        let version = Version::new(1, 0, 1);
        let tampered = MANIFEST.replace("sha256 = \"a", "sha256 = \"b");
        assert_ne!(tampered, MANIFEST);
        let error = verifier(false)
            .verify(
                tampered.as_bytes(),
                Some(SIGNATURE),
                true,
                identity(&version),
            )
            .unwrap_err();
        assert!(format!("{error:#}").contains("tampered"), "{error:#}");
    }

    #[test]
    fn signature_for_another_release_is_rejected() {
        let version = Version::new(1, 0, 1);
        let error = verifier(false)
            .verify(
                MANIFEST.as_bytes(),
                Some(WRONG_IDENTITY),
                true,
                identity(&version),
            )
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("different release"),
            "{error:#}"
        );
    }

    #[test]
    fn missing_signature_rules() {
        let old = Version::new(1, 0, 0);
        let new = Version::new(1, 0, 1);
        // Declared but missing.
        assert!(
            verifier(false)
                .verify(MANIFEST.as_bytes(), None, true, identity(&new))
                .is_err()
        );
        // Embedded keys require signatures from SIGNED_SINCE on.
        assert!(
            verifier(false)
                .verify(MANIFEST.as_bytes(), None, false, identity(&new))
                .is_err()
        );
        // Older releases are accepted unsigned.
        assert_eq!(
            verifier(false)
                .verify(MANIFEST.as_bytes(), None, false, identity(&old))
                .unwrap(),
            SignatureStatus::Unsigned
        );
        // Policy requires signatures for every release.
        assert!(
            verifier(true)
                .verify(MANIFEST.as_bytes(), None, false, identity(&old))
                .is_err()
        );
    }

    #[test]
    fn builds_without_keys_accept_but_flag_signatures() {
        let version = Version::new(1, 0, 1);
        let keyless = Verifier::new(&[], Some(Version::new(1, 0, 1)), false).unwrap();
        assert_eq!(
            keyless
                .verify(
                    MANIFEST.as_bytes(),
                    Some(SIGNATURE),
                    true,
                    identity(&version)
                )
                .unwrap(),
            SignatureStatus::Unverifiable
        );
        assert_eq!(
            keyless
                .verify(MANIFEST.as_bytes(), None, false, identity(&version))
                .unwrap(),
            SignatureStatus::Unsigned
        );
        let strict = Verifier::new(&[], None, true).unwrap();
        assert!(
            strict
                .verify(
                    MANIFEST.as_bytes(),
                    Some(SIGNATURE),
                    true,
                    identity(&version)
                )
                .is_err()
        );
    }

    #[test]
    fn build_keys_parse() {
        Verifier::for_build(&Policy::default()).unwrap();
    }
}

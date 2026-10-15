//! Public keys trusted for eov release metadata.
//!
//! Release manifests (`release.toml`) are signed with
//! [minisign](https://jedisct1.github.io/minisign/) keys. eov verifies
//! `release.toml.minisig` against the keys listed here, which are compiled
//! into the binary so a compromised GitHub account or release token cannot
//! substitute its own key.
//!
//! No release key exists yet. To enable signing (repository owner):
//!
//! 1. Create a key pair offline:
//!    `python3 scripts/release_signing.py keygen --secret-key eov-release.key --public-key eov-release.pub`
//!    (or `minisign -G`).
//! 2. Store the secret key file's contents and its password as the GitHub
//!    Actions secrets `EOV_RELEASE_SIGNING_KEY` and
//!    `EOV_RELEASE_SIGNING_KEY_PASSWORD`; the release workflow then signs
//!    every `release.toml` automatically.
//! 3. Add the public key (second line of `eov-release.pub`) to
//!    [`RELEASE_KEYS`] and set [`SIGNED_SINCE`] to the first release that
//!    will be signed. Builds that contain a key refuse unsigned metadata for
//!    that release and every later one.
//!
//! To rotate a key, add the new key next to the old one for at least one
//! release before removing the old key.

/// A trusted minisign public key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrustedKey {
    /// A human-readable name shown in diagnostics.
    pub name: &'static str,
    /// The base64 public key (the second line of a minisign `.pub` file).
    pub public_key: &'static str,
}

/// Keys whose signatures are accepted on eov release manifests.
pub const RELEASE_KEYS: &[TrustedKey] = &[];

/// The first eov release whose metadata must be signed once [`RELEASE_KEYS`]
/// is non-empty. Older releases (manual downgrades) are accepted unsigned and
/// reported as such.
pub const SIGNED_SINCE: &str = "1.0.1";

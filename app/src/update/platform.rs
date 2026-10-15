//! The operating system and CPU architecture that release artifacts target.

use super::manifest::Architecture;
use anyhow::{Result, bail};
use serde::Serialize;

/// A release platform such as `linux.x86_64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Platform {
    /// `linux`, `macos` or `windows`.
    pub os: &'static str,
    #[serde(serialize_with = "serialize_architecture")]
    pub arch: Architecture,
}

fn serialize_architecture<S: serde::Serializer>(
    arch: &Architecture,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(arch.as_str())
}

impl Platform {
    /// The platform this eov runs on.
    pub fn current() -> Result<Self> {
        let os = if cfg!(target_os = "windows") {
            "windows"
        } else if cfg!(target_os = "macos") {
            "macos"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else {
            bail!("unsupported operating system for eov releases")
        };
        Ok(Self {
            os,
            arch: Architecture::current()?,
        })
    }

    #[cfg(test)]
    pub const fn linux_x86_64() -> Self {
        Self {
            os: "linux",
            arch: Architecture::X86_64,
        }
    }

    /// `linux.x86_64`, as recorded in lockfiles.
    pub fn short_name(&self) -> String {
        format!("{}.{}", self.os, self.arch.as_str())
    }

    /// The release-manifest key of a native plugin artifact.
    pub fn plugin_artifact_key(&self) -> String {
        format!("platform.{}", self.short_name())
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.short_name())
    }
}

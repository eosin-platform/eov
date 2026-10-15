//! macOS bundle checks for tree updates.
//!
//! The release build raises a bundle's `LSMinimumSystemVersion` to the
//! newest deployment target of anything it ships (packaging/macos/build.sh),
//! so a downloaded bundle states which macOS it needs. Installing a bundle
//! the Mac cannot run would replace a working eov with one that macOS refuses
//! to open, so the update is refused before anything is replaced.

use anyhow::{Result, bail};
use std::path::Path;

/// `LSMinimumSystemVersion` from an XML property list.
fn minimum_from_xml(plist: &str) -> Option<String> {
    let after_key = plist.split_once("<key>LSMinimumSystemVersion</key>")?.1;
    let value = after_key.trim_start().strip_prefix("<string>")?;
    let (version, _) = value.split_once("</string>")?;
    let version = version.trim();
    is_dotted_version(version).then(|| version.to_string())
}

fn is_dotted_version(value: &str) -> bool {
    !value.is_empty()
        && value.split('.').all(|part| {
            !part.is_empty() && part.len() <= 5 && part.bytes().all(|b| b.is_ascii_digit())
        })
}

/// Compares dotted numeric versions (`"10.15" < "11.0"`, `"15" == "15.0"`).
fn version_cmp(left: &str, right: &str) -> std::cmp::Ordering {
    let parse = |value: &str| -> Vec<u32> {
        value
            .split('.')
            .map(|part| part.parse().unwrap_or(0))
            .collect()
    };
    let (left, right) = (parse(left), parse(right));
    let length = left.len().max(right.len());
    for index in 0..length {
        let ordering = left
            .get(index)
            .copied()
            .unwrap_or(0)
            .cmp(&right.get(index).copied().unwrap_or(0));
        if ordering.is_ne() {
            return ordering;
        }
    }
    std::cmp::Ordering::Equal
}

/// The minimum macOS version a bundle declares, if it declares one.
pub fn bundle_minimum_system_version(bundle: &Path) -> Option<String> {
    let path = bundle.join("Contents/Info.plist");
    let bytes = std::fs::read(&path).ok()?;
    if let Some(version) = std::str::from_utf8(&bytes).ok().and_then(minimum_from_xml) {
        return Some(version);
    }
    // A binary property list: let plutil convert it.
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("/usr/bin/plutil")
            .args(["-convert", "xml1", "-o", "-"])
            .arg(&path)
            .output()
            .ok()?;
        if output.status.success() {
            return std::str::from_utf8(&output.stdout)
                .ok()
                .and_then(minimum_from_xml);
        }
    }
    None
}

/// The running macOS version (`kern.osproductversion`).
#[cfg(target_os = "macos")]
pub fn current_version() -> Option<String> {
    let name = c"kern.osproductversion";
    let mut buffer = [0u8; 32];
    let mut length = buffer.len();
    // SAFETY: the name is NUL-terminated, the buffer and its length are
    // valid, and no new value is set.
    let result = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buffer.as_mut_ptr().cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if result != 0 {
        return None;
    }
    let text = std::str::from_utf8(&buffer[..length]).ok()?;
    let version = text.trim_end_matches('\0').trim();
    is_dotted_version(version).then(|| version.to_string())
}

#[cfg(not(target_os = "macos"))]
pub fn current_version() -> Option<String> {
    None
}

/// Refuses a bundle that needs a newer macOS than `running` (when both are
/// known).
pub fn ensure_runnable(bundle: &Path, target: &str, running: Option<&str>) -> Result<()> {
    let (Some(required), Some(running)) = (bundle_minimum_system_version(bundle), running) else {
        return Ok(());
    };
    if version_cmp(running, &required).is_lt() {
        bail!(
            "eov {target} needs macOS {required} or newer, but this Mac runs macOS {running}. \
             Nothing was changed; update macOS first, or stay on this version of eov."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle_with(info_plist: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let contents = dir.path().join("eov.app/Contents");
        std::fs::create_dir_all(&contents).unwrap();
        std::fs::write(contents.join("Info.plist"), info_plist).unwrap();
        dir
    }

    const PLIST: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key>
  <string>io.eosin.eov</string>
  <key>LSMinimumSystemVersion</key>
  <string>15.0</string>
</dict>
</plist>
"#;

    #[test]
    fn reads_the_declared_minimum() {
        let dir = bundle_with(PLIST);
        assert_eq!(
            bundle_minimum_system_version(&dir.path().join("eov.app")).as_deref(),
            Some("15.0")
        );
        assert_eq!(
            minimum_from_xml("<key>LSMinimumSystemVersion</key>\n\t<string> 11.0 </string>")
                .as_deref(),
            Some("11.0")
        );
        assert_eq!(
            minimum_from_xml("<key>LSMinimumSystemVersion</key><string>eleven</string>"),
            None
        );
        assert_eq!(
            minimum_from_xml("<key>Other</key><string>11.0</string>"),
            None
        );
    }

    #[test]
    fn compares_versions_numerically() {
        assert!(version_cmp("10.15", "11.0").is_lt());
        assert!(version_cmp("10.9", "10.15").is_lt());
        assert!(version_cmp("15", "15.0").is_eq());
        assert!(version_cmp("26.0.1", "26.0").is_gt());
    }

    #[test]
    fn refuses_bundles_this_mac_cannot_run() {
        let dir = bundle_with(PLIST);
        let bundle = dir.path().join("eov.app");
        let error = ensure_runnable(&bundle, "1.2.0", Some("14.6.1")).unwrap_err();
        assert!(error.to_string().contains("needs macOS 15.0"), "{error}");
        ensure_runnable(&bundle, "1.2.0", Some("15.0")).unwrap();
        ensure_runnable(&bundle, "1.2.0", Some("26.1")).unwrap();
        // Unknown on either side: nothing to compare.
        ensure_runnable(&bundle, "1.2.0", None).unwrap();
        ensure_runnable(&dir.path().join("missing.app"), "1.2.0", Some("11.0")).unwrap();
    }
}

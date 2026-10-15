//! What `eov register-file-types` changes in the Windows registry, as a
//! plan (tested on every platform; `windows.rs` carries it out).
//!
//! Everything is below `HKEY_CURRENT_USER\Software\Classes`, so no
//! administrator rights are needed and other users are not affected:
//!
//! - `eov.Slide`: the ProgID ("Whole-slide image"), its icon and
//!   `shell\open\command`;
//! - `.svs`, `.ndpi`, …: `OpenWithProgids\eov.Slide`, so "Open with" offers
//!   eov; the default program only for vendor formats that have none;
//! - `Applications\eov.exe`: eov's name, icon and supported types for "Open
//!   with";
//! - `eov`: the URL protocol of eov links.
//!
//! The commands pass `--` before the path or link, so that nothing in it
//! can be taken for an option.

use super::{SLIDE_TYPES, URL_SCHEME};
use std::fmt;
use std::path::Path;

pub(crate) const CLASSES: &str = r"Software\Classes";
pub(crate) const PROG_ID: &str = "eov.Slide";
const PROG_ID_NAME: &str = "Whole-slide image";
const APPLICATION: &str = "eov.exe";
const PROTOCOL_NAME: &str = "URL:eov link";

/// One change below `HKEY_CURRENT_USER`. A `name` of `None` is the key's
/// default value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Change {
    /// Sets a string value, creating the key.
    Set {
        key: String,
        name: Option<String>,
        value: String,
    },
    /// Sets a string value that is missing or empty.
    SetIfUnset {
        key: String,
        name: Option<String>,
        value: String,
    },
    /// Deletes a value.
    DeleteValue { key: String, name: String },
    /// Deletes a value that equals `value`.
    DeleteValueIf {
        key: String,
        name: Option<String>,
        value: String,
    },
    /// Deletes a key with everything below it, if its default value is
    /// `value` (or always, for `None`).
    DeleteTree {
        key: String,
        if_default: Option<String>,
    },
}

fn set(key: &str, name: Option<&str>, value: &str) -> Change {
    Change::Set {
        key: key.to_string(),
        name: name.map(str::to_string),
        value: value.to_string(),
    }
}

fn keys() -> (String, String, String) {
    (
        format!(r"{CLASSES}\{PROG_ID}"),
        format!(r"{CLASSES}\Applications\{APPLICATION}"),
        format!(r"{CLASSES}\{URL_SCHEME}"),
    )
}

/// The changes that make `exe` (eov.exe, without a `\\?\` prefix) the
/// program for slides and eov links.
pub(crate) fn registration(exe: &Path) -> Vec<Change> {
    let exe = exe.display();
    let command = format!("\"{exe}\" -- \"%1\"");
    let icon = format!("\"{exe}\",0");
    let (prog_id, application, protocol) = keys();
    let mut changes = vec![
        set(&prog_id, None, PROG_ID_NAME),
        set(&format!(r"{prog_id}\DefaultIcon"), None, &icon),
        set(&format!(r"{prog_id}\shell\open\command"), None, &command),
        set(&application, Some("FriendlyAppName"), "eov"),
        set(&format!(r"{application}\DefaultIcon"), None, &icon),
        set(
            &format!(r"{application}\shell\open\command"),
            None,
            &command,
        ),
    ];
    for slide in SLIDE_TYPES {
        let extension = format!(".{}", slide.extension);
        let key = format!(r"{CLASSES}\{extension}");
        changes.push(set(&format!(r"{key}\OpenWithProgids"), Some(PROG_ID), ""));
        changes.push(set(
            &format!(r"{application}\SupportedTypes"),
            Some(&extension),
            "",
        ));
        if slide.eov_owns {
            changes.push(Change::SetIfUnset {
                key,
                name: None,
                value: PROG_ID.to_string(),
            });
        }
    }
    changes.extend([
        set(&protocol, None, PROTOCOL_NAME),
        set(&protocol, Some("URL Protocol"), ""),
        set(&format!(r"{protocol}\DefaultIcon"), None, &icon),
        set(&format!(r"{protocol}\shell\open\command"), None, &command),
    ]);
    changes
}

/// The changes that undo [`registration`], leaving what other programs
/// set alone.
pub(crate) fn unregistration() -> Vec<Change> {
    let (prog_id, application, protocol) = keys();
    let mut changes = vec![
        Change::DeleteTree {
            key: prog_id,
            if_default: None,
        },
        Change::DeleteTree {
            key: application,
            if_default: None,
        },
    ];
    for slide in SLIDE_TYPES {
        let key = format!(r"{CLASSES}\.{}", slide.extension);
        changes.push(Change::DeleteValue {
            key: format!(r"{key}\OpenWithProgids"),
            name: PROG_ID.to_string(),
        });
        if slide.eov_owns {
            changes.push(Change::DeleteValueIf {
                key,
                name: None,
                value: PROG_ID.to_string(),
            });
        }
    }
    changes.push(Change::DeleteTree {
        key: protocol,
        if_default: Some(PROTOCOL_NAME.to_string()),
    });
    changes
}

fn value_name(name: &Option<String>) -> &str {
    name.as_deref().unwrap_or("(default)")
}

impl fmt::Display for Change {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Change::Set { key, name, value } => {
                write!(f, r"set HKCU\{key} {} = {value:?}", value_name(name))
            }
            Change::SetIfUnset { key, name, value } => write!(
                f,
                r"set HKCU\{key} {} = {value:?} unless another program set it",
                value_name(name)
            ),
            Change::DeleteValue { key, name } => write!(f, r"delete HKCU\{key} {name}"),
            Change::DeleteValueIf { key, name, value } => write!(
                f,
                r"delete HKCU\{key} {} if it is {value:?}",
                value_name(name)
            ),
            Change::DeleteTree {
                key,
                if_default: None,
            } => write!(f, r"delete HKCU\{key}"),
            Change::DeleteTree {
                key,
                if_default: Some(value),
            } => write!(f, r"delete HKCU\{key} if it is eov's ({value:?})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_offers_eov_for_every_format_and_for_links() {
        let changes = registration(Path::new(r"C:\Tools\eov\eov.exe"));
        let command = r#""C:\Tools\eov\eov.exe" -- "%1""#.to_string();
        assert!(changes.contains(&Change::Set {
            key: r"Software\Classes\eov.Slide\shell\open\command".into(),
            name: None,
            value: command.clone(),
        }));
        assert!(changes.contains(&Change::Set {
            key: r"Software\Classes\eov\shell\open\command".into(),
            name: None,
            value: command,
        }));
        assert!(changes.contains(&Change::Set {
            key: r"Software\Classes\eov".into(),
            name: Some("URL Protocol".into()),
            value: String::new(),
        }));
        for slide in SLIDE_TYPES {
            let key = format!(r"Software\Classes\.{}\OpenWithProgids", slide.extension);
            assert!(changes.contains(&set(&key, Some(PROG_ID), "")), "{key}");
            let claims = changes.iter().any(|change| {
                matches!(change, Change::SetIfUnset { key, .. }
                    if *key == format!(r"Software\Classes\.{}", slide.extension))
            });
            assert_eq!(claims, slide.eov_owns, ".{}", slide.extension);
        }
        // TIFF and DICOM keep their default programs.
        assert!(!changes.iter().any(|change| matches!(change,
            Change::Set { key, name: None, .. } if key.ends_with(r"\.tif"))));
        for change in &changes {
            let key = match change {
                Change::Set { key, .. } | Change::SetIfUnset { key, .. } => key,
                other => panic!("registration only sets values, not {other}"),
            };
            assert!(key.starts_with(r"Software\Classes\"), "{key}");
        }
    }

    #[test]
    fn unregistration_removes_only_what_registration_added() {
        let changes = unregistration();
        assert!(changes.contains(&Change::DeleteTree {
            key: r"Software\Classes\eov.Slide".into(),
            if_default: None,
        }));
        assert!(changes.contains(&Change::DeleteTree {
            key: r"Software\Classes\eov".into(),
            if_default: Some(PROTOCOL_NAME.into()),
        }));
        for change in &changes {
            match change {
                Change::DeleteValue { key, name } => {
                    assert!(key.ends_with("OpenWithProgids") && name == PROG_ID)
                }
                Change::DeleteValueIf { value, .. } => assert_eq!(value, PROG_ID),
                Change::DeleteTree { key, .. } => assert!(
                    key.ends_with(PROG_ID)
                        || key.ends_with(APPLICATION)
                        || key.ends_with(URL_SCHEME)
                ),
                other => panic!("unregistration only deletes, not {other}"),
            }
        }
        // Never a whole extension key: other programs keep theirs.
        assert!(!changes.iter().any(|change| matches!(change,
            Change::DeleteTree { key, .. } if key.contains(r"Classes\."))));
    }
}

//! `eov register-file-types` on Linux, for an AppImage or a `cargo
//! install`, which no package manager installed: the MIME package, a
//! desktop entry that starts this eov, and its icon, in the user's data
//! folder (`$XDG_DATA_HOME`, `~/.local/share`), then the desktop's caches.
//! The Flatpak exports the same files itself.

use super::{APP_ID, desktop_entry, mime_package};
use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};

/// The icon of the desktop entry (eov's window icon).
const ICON: &[u8] = include_bytes!("../../ui/icons/logo_full.png");

/// Marks the desktop entry as written by `register-file-types`.
const MARKER: &str = "X-Eov-Registered=true";

/// The files `register-file-types` manages below a data folder.
struct Targets {
    data_home: PathBuf,
}

impl Targets {
    fn mime_dir(&self) -> PathBuf {
        self.data_home.join("mime")
    }

    fn mime_package(&self) -> PathBuf {
        self.mime_dir()
            .join("packages")
            .join(format!("{APP_ID}.xml"))
    }

    fn applications_dir(&self) -> PathBuf {
        self.data_home.join("applications")
    }

    fn desktop_entry(&self) -> PathBuf {
        self.applications_dir().join(format!("{APP_ID}.desktop"))
    }

    fn icon(&self) -> PathBuf {
        self.data_home
            .join("icons/hicolor/256x256/apps")
            .join(format!("{APP_ID}.png"))
    }
}

/// What registering or unregistering did, or would do.
#[derive(Debug, PartialEq, Eq)]
enum Step {
    Write(PathBuf),
    Remove(PathBuf),
    /// A desktop entry eov did not write.
    Keep(PathBuf),
    Note(String),
}

impl Step {
    fn describe(&self, dry_run: bool) -> String {
        let would = if dry_run { "would " } else { "" };
        match self {
            Step::Write(path) if dry_run => format!("would write {}", path.display()),
            Step::Write(path) => format!("wrote {}", path.display()),
            Step::Remove(path) if dry_run => format!("would remove {}", path.display()),
            Step::Remove(path) => format!("removed {}", path.display()),
            Step::Keep(path) => format!("{would}leave {}: eov did not write it", path.display()),
            Step::Note(note) => note.clone(),
        }
    }
}

/// `program` as the program of an `Exec=` key, quoted. Characters that
/// need escaping twice over (Desktop Entry Specification, "The Exec key")
/// are refused rather than risked.
fn exec_program(program: &Path) -> Result<String> {
    let text = program
        .to_str()
        .context("the path of eov is not valid UTF-8")?;
    if text.contains(['"', '`', '$', '\\', '\n']) {
        bail!(
            "the path of eov ({text}) contains characters a desktop entry cannot start \
             reliably; move eov elsewhere"
        );
    }
    Ok(format!("\"{}\"", text.replace('%', "%%")))
}

/// The program the desktop entry starts: the AppImage itself (not the
/// binary in its mount, which disappears when eov exits), or this one.
fn this_program() -> Result<PathBuf> {
    if let Some(appimage) = std::env::var_os("APPIMAGE")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
    {
        return Ok(appimage);
    }
    std::env::current_exe().context("cannot tell where eov is")
}

fn write(path: &Path, contents: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    fs::write(path, contents).with_context(|| format!("cannot write {}", path.display()))
}

fn remove(path: &Path, dry_run: bool) -> Result<()> {
    if dry_run {
        return Ok(());
    }
    fs::remove_file(path).with_context(|| format!("cannot remove {}", path.display()))
}

/// Writes the files (unless `dry_run`).
fn register_in(targets: &Targets, program: &Path, dry_run: bool) -> Result<Vec<Step>> {
    let entry = desktop_entry(&exec_program(program)?, true);
    let files = [
        (targets.mime_package(), mime_package().into_bytes()),
        (targets.desktop_entry(), entry.into_bytes()),
        (targets.icon(), ICON.to_vec()),
    ];
    let mut steps = Vec::new();
    for (path, contents) in files {
        if !dry_run {
            write(&path, &contents)?;
        }
        steps.push(Step::Write(path));
    }
    Ok(steps)
}

/// Removes what `register_in` wrote (unless `dry_run`); a desktop entry
/// without eov's marker is someone else's and stays, with its files.
fn unregister_in(targets: &Targets, dry_run: bool) -> Result<Vec<Step>> {
    let entry = targets.desktop_entry();
    let text = match fs::read_to_string(&entry) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Vec::new());
        }
        Err(error) => {
            return Err(error).with_context(|| format!("cannot read {}", entry.display()));
        }
    };
    if !text.lines().any(|line| line.trim() == MARKER) {
        return Ok(vec![Step::Keep(entry)]);
    }
    remove(&entry, dry_run)?;
    let mut steps = vec![Step::Remove(entry)];
    for path in [targets.mime_package(), targets.icon()] {
        if path.exists() {
            remove(&path, dry_run)?;
            steps.push(Step::Remove(path));
        }
    }
    Ok(steps)
}

/// Rebuilds the desktop's caches of MIME types and applications.
fn refresh(targets: &Targets) -> Vec<Step> {
    let mut steps = Vec::new();
    for (program, dir) in [
        ("update-mime-database", targets.mime_dir()),
        ("update-desktop-database", targets.applications_dir()),
    ] {
        if !dir.exists() {
            continue;
        }
        let status = crate::desktop::command(program).arg(&dir).status();
        steps.push(Step::Note(match status {
            Ok(status) if status.success() => format!("ran {program} {}", dir.display()),
            Ok(status) => format!("{program} failed ({status})"),
            Err(_) => format!(
                "{program} is not installed: file managers see the change after the next login"
            ),
        }));
    }
    steps
}

pub(super) fn run(register: bool, dry_run: bool) -> Result<()> {
    if std::env::var_os("FLATPAK_ID").is_some() {
        bail!("the Flatpak declares eov's file types and eov:// links itself");
    }
    let data_home =
        dirs::data_dir().context("no data folder ($XDG_DATA_HOME or ~/.local/share)")?;
    let targets = Targets { data_home };
    let mut steps = if register {
        register_in(&targets, &this_program()?, dry_run)?
    } else {
        unregister_in(&targets, dry_run)?
    };
    if steps.is_empty() {
        println!("eov was not registered for this user.");
        return Ok(());
    }
    if !dry_run && steps.iter().any(|step| !matches!(step, Step::Keep(_))) {
        steps.extend(refresh(&targets));
    }
    for step in steps {
        println!("{}", step.describe(dry_run));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registering_writes_a_desktop_entry_for_this_program_and_removes_it_again() {
        let dir = tempfile::tempdir().unwrap();
        let targets = Targets {
            data_home: dir.path().to_path_buf(),
        };
        let program = Path::new("/home/u/Apps/eov 1.0 (100%).AppImage");
        assert_eq!(register_in(&targets, program, true).unwrap().len(), 3);
        assert!(
            !targets.desktop_entry().exists(),
            "a dry run writes nothing"
        );

        register_in(&targets, program, false).unwrap();
        let entry = fs::read_to_string(targets.desktop_entry()).unwrap();
        assert!(entry.contains("Exec=\"/home/u/Apps/eov 1.0 (100%%).AppImage\" %U\n"));
        assert!(entry.contains(MARKER));
        assert_eq!(
            fs::read_to_string(targets.mime_package()).unwrap(),
            mime_package()
        );
        assert_eq!(fs::read(targets.icon()).unwrap(), ICON);

        assert_eq!(unregister_in(&targets, true).unwrap().len(), 3);
        assert!(
            targets.desktop_entry().exists(),
            "a dry run removes nothing"
        );
        unregister_in(&targets, false).unwrap();
        for path in [
            targets.desktop_entry(),
            targets.mime_package(),
            targets.icon(),
        ] {
            assert!(!path.exists(), "{} was left", path.display());
        }
        assert!(unregister_in(&targets, false).unwrap().is_empty());
    }

    #[test]
    fn a_desktop_entry_eov_did_not_write_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let targets = Targets {
            data_home: dir.path().to_path_buf(),
        };
        write(&targets.desktop_entry(), b"[Desktop Entry]\nName=eov\n").unwrap();
        write(&targets.mime_package(), b"<mime-info/>").unwrap();
        let steps = unregister_in(&targets, false).unwrap();
        assert_eq!(steps, [Step::Keep(targets.desktop_entry())]);
        assert!(targets.desktop_entry().exists() && targets.mime_package().exists());
    }

    #[test]
    fn programs_with_unquotable_paths_are_refused() {
        assert!(exec_program(Path::new("/opt/eov/eov")).is_ok());
        for bad in ["/opt/$HOME/eov", "/opt/e\"ov/eov", r"/opt/a\b/eov"] {
            assert!(exec_program(Path::new(bad)).is_err(), "{bad}");
        }
    }
}

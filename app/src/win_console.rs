//! Console output and fatal-error reporting for the Windows GUI build.
//!
//! Release builds of `eov.exe` use the Windows GUI subsystem (no console
//! window flashes up when it is started from Explorer), so Windows gives the
//! process no standard handles: CLI output, prompts, `--help` and startup
//! errors would all vanish. [`init`], called first thing in `main`, fixes
//! that:
//!
//! - **CLI use** (a subcommand, a flag such as `--help`, or an argument that
//!   is not an existing file): attach to the console of the parent process
//!   (`AttachConsole(ATTACH_PARENT_PROCESS)`) and reopen the standard handles
//!   that are not redirected. Standard input is only attached when eov was
//!   started through the console launcher `eov.com` (see
//!   `win_console_launcher.rs`): cmd.exe and PowerShell do not wait for GUI
//!   programs, so both would read the same keystrokes. Without the launcher,
//!   prompts are therefore refused with a hint to use `eov` (which resolves
//!   to `eov.com`) or `-y`.
//! - **GUI launches** (no console): panics on the main thread and errors
//!   returned from `main` are shown in a native message box, because nobody
//!   would ever see them otherwise.
//!
//! On other platforms, and in debug builds (console subsystem), [`init`]
//! does nothing.

#![cfg_attr(not(all(windows, not(debug_assertions))), allow(dead_code))]

use std::ffi::OsString;
use std::path::Path;

/// Environment variable the console launcher sets for `eov.exe`.
pub const LAUNCHER_ENV: &str = "EOV_CONSOLE_LAUNCHER";

/// Subcommands (and aliases) that produce console output.
pub(crate) const SUBCOMMANDS: &[&str] = &[
    "update",
    "upgrade",
    "plugin",
    "probe",
    "recent",
    "config-path",
    "paths",
    "register-file-types",
    "unregister-file-types",
    "dataset",
    "call",
    "script-endpoint",
    "help",
    "plugin-window",
];

/// Whether an argument is a URI eov decodes (`file:` from file managers,
/// `eov:` links), rather than a path.
fn is_uri(text: &str) -> bool {
    let starts_with = |prefix: &str| {
        text.get(..prefix.len())
            .is_some_and(|start| start.eq_ignore_ascii_case(prefix))
    };
    starts_with("eov:") || starts_with("file:")
}

/// Whether the command line asks for console behaviour rather than just
/// opening files in the viewer: any flag (`--help`, `--version`,
/// `--log-level ...`), any subcommand, or anything that is not an existing
/// file or directory (eov reports it, see `launch.rs`). As in `launch.rs`,
/// an argument that exists is a path even with commas in its name.
pub fn wants_console(args: &[OsString], exists: impl Fn(&Path) -> bool) -> bool {
    args.iter().any(|arg| {
        let text = arg.to_string_lossy();
        if text.starts_with('-') || SUBCOMMANDS.contains(&text.as_ref()) {
            return true;
        }
        if is_uri(&text) || exists(Path::new(arg)) {
            return false;
        }
        text.split(',').any(|part| {
            let part = part.trim();
            !part.is_empty() && !is_uri(part) && !exists(Path::new(part))
        })
    })
}

/// Attaches the console or installs fatal-error dialogs (Windows release
/// builds only). Call before anything prints.
pub fn init() {
    #[cfg(all(windows, not(debug_assertions)))]
    imp::init();
}

/// Whether standard input is a console eov may prompt on (on Windows release
/// builds only when started through `eov.com`).
pub fn prompts_supported() -> bool {
    !(cfg!(all(windows, not(debug_assertions))) && std::env::var_os(LAUNCHER_ENV).is_none())
}

#[cfg(all(windows, not(debug_assertions)))]
mod imp {
    use super::{LAUNCHER_ENV, wants_console};
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};
    use windows_sys::Win32::Foundation::{
        CloseHandle, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, ReadFile,
    };
    use windows_sys::Win32::System::Console::{
        ATTACH_PARENT_PROCESS, AttachConsole, GetStdHandle, STD_ERROR_HANDLE, STD_HANDLE,
        STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, SetStdHandle,
    };
    use windows_sys::Win32::System::Pipes::{CreatePipe, PeekNamedPipe};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        MB_ICONERROR, MB_OK, MB_SETFOREGROUND, MessageBoxW,
    };

    /// The last bytes written to standard error in GUI mode.
    static CAPTURED: Mutex<Vec<u8>> = Mutex::new(Vec::new());
    static CAPTURE_PIPE: OnceLock<usize> = OnceLock::new();
    const CAPTURE_LIMIT: usize = 64 * 1024;

    unsafe extern "C" {
        fn atexit(callback: extern "C" fn()) -> i32;
    }

    pub fn init() {
        let args: Vec<OsString> = std::env::args_os().skip(1).collect();
        let launcher = std::env::var_os(LAUNCHER_ENV).is_some();
        if (launcher || wants_console(&args, |path| path.exists()))
            // SAFETY: plain Win32 call.
            && unsafe { AttachConsole(ATTACH_PARENT_PROCESS) } != 0
        {
            reopen(STD_OUTPUT_HANDLE, "CONOUT$");
            reopen(STD_ERROR_HANDLE, "CONOUT$");
            if launcher {
                reopen(STD_INPUT_HANDLE, "CONIN$");
            } else {
                // The shell already printed its next prompt; start on a new
                // line.
                println!();
            }
            return;
        }
        install_fatal_error_reporting();
    }

    fn valid(handle: HANDLE) -> bool {
        handle != 0 && handle != INVALID_HANDLE_VALUE
    }

    /// Opens the console device for a standard handle that is not
    /// redirected (redirected handles, e.g. `eov plugin ls > file`, stay).
    fn reopen(id: STD_HANDLE, device: &str) {
        // SAFETY: plain Win32 calls with a NUL-terminated UTF-16 name.
        unsafe {
            if valid(GetStdHandle(id)) {
                return;
            }
            let name: Vec<u16> = device.encode_utf16().chain(std::iter::once(0)).collect();
            let handle = CreateFileW(
                name.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                0,
            );
            if valid(handle) {
                SetStdHandle(id, handle);
            }
        }
    }

    pub(super) fn message_box(title: &str, text: &str) {
        let wide = |value: &str| {
            value
                .encode_utf16()
                .chain(std::iter::once(0))
                .collect::<Vec<u16>>()
        };
        let (title, text) = (wide(title), wide(text));
        // SAFETY: NUL-terminated buffers that outlive the call.
        unsafe {
            MessageBoxW(
                0,
                text.as_ptr(),
                title.as_ptr(),
                MB_OK | MB_ICONERROR | MB_SETFOREGROUND,
            );
        }
    }

    fn install_fatal_error_reporting() {
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            default_hook(info);
            // Panics on other threads do not end the process.
            if std::thread::current().name() == Some("main") {
                message_box(
                    "eov stopped unexpectedly",
                    &format!("eov stopped because of an internal error:\n\n{info}"),
                );
            }
        }));
        capture_stderr();
    }

    /// Redirects standard error into a pipe whose tail is inspected when the
    /// process exits: `main` returning an error prints `Error: ...` there.
    fn capture_stderr() {
        // SAFETY: plain Win32 calls; the handles are owned by this module.
        unsafe {
            if valid(GetStdHandle(STD_ERROR_HANDLE)) {
                return;
            }
            let mut read: HANDLE = 0;
            let mut write: HANDLE = 0;
            if CreatePipe(&mut read, &mut write, std::ptr::null(), 64 * 1024) == 0 {
                return;
            }
            SetStdHandle(STD_ERROR_HANDLE, write);
            let _ = CAPTURE_PIPE.set(read as usize);
            let reader = read as usize;
            let spawned = std::thread::Builder::new()
                .name("eov-stderr-capture".into())
                .spawn(move || {
                    let mut buffer = [0u8; 4096];
                    loop {
                        let mut read_bytes = 0u32;
                        let ok = ReadFile(
                            reader as HANDLE,
                            buffer.as_mut_ptr().cast(),
                            buffer.len() as u32,
                            &mut read_bytes,
                            std::ptr::null_mut(),
                        );
                        if ok == 0 || read_bytes == 0 {
                            break;
                        }
                        if let Ok(mut captured) = CAPTURED.lock() {
                            captured.extend_from_slice(&buffer[..read_bytes as usize]);
                            if captured.len() > CAPTURE_LIMIT {
                                let excess = captured.len() - CAPTURE_LIMIT;
                                captured.drain(..excess);
                            }
                        }
                    }
                });
            if spawned.is_err() {
                SetStdHandle(STD_ERROR_HANDLE, 0);
                CloseHandle(write);
                CloseHandle(read);
                return;
            }
            atexit(report_at_exit);
        }
    }

    extern "C" fn report_at_exit() {
        // Give the reader a moment to drain what `main` printed last; never
        // block on end-of-file, which a child process may delay forever.
        if let Some(&read) = CAPTURE_PIPE.get() {
            for _ in 0..50 {
                let mut available = 0u32;
                // SAFETY: peeking at our own pipe.
                let ok = unsafe {
                    PeekNamedPipe(
                        read as HANDLE,
                        std::ptr::null_mut(),
                        0,
                        std::ptr::null_mut(),
                        &mut available,
                        std::ptr::null_mut(),
                    )
                };
                if ok == 0 || available == 0 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let text = CAPTURED
            .lock()
            .map(|captured| String::from_utf8_lossy(&captured).into_owned())
            .unwrap_or_default();
        if let Some(message) = super::fatal_error_text(&text) {
            message_box("eov could not start", &message);
        }
    }
}

/// Extracts the report of an error returned from `main` (`Error: ...`) from
/// captured standard-error text.
fn fatal_error_text(captured: &str) -> Option<String> {
    let start = if captured.starts_with("Error: ") {
        0
    } else {
        captured.rfind("\nError: ")? + 1
    };
    let text = captured[start..].trim();
    let text = text.strip_prefix("Error: ").unwrap_or(text);
    let limit = 2000;
    Some(if text.chars().count() > limit {
        format!("{}...", text.chars().take(limit).collect::<String>())
    } else {
        text.to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<OsString> {
        list.iter().map(OsString::from).collect()
    }

    #[test]
    fn cli_invocations_want_a_console() {
        let none = |_: &Path| false;
        let all = |_: &Path| true;
        assert!(wants_console(&args(&["--help"]), all));
        assert!(wants_console(&args(&["-V"]), all));
        assert!(wants_console(&args(&["plugin", "ls"]), all));
        assert!(wants_console(&args(&["update", "-y"]), all));
        assert!(wants_console(&args(&["missing.svs"]), none));
        assert!(wants_console(&args(&["a.svs,missing.svs"]), |path| path
            == Path::new("a.svs")));
    }

    #[test]
    fn opening_existing_files_does_not() {
        let all = |_: &Path| true;
        assert!(!wants_console(&args(&[]), all));
        assert!(!wants_console(&args(&["C:\\slides\\a.svs"]), all));
        assert!(!wants_console(&args(&["a.svs,b.svs", "c.tif"]), all));
        assert!(!wants_console(&args(&["file:///C:/slides/a.svs"]), |_| {
            false
        }));
        assert!(!wants_console(
            &args(&["eov://open?path=C%3A%5Ca.svs&zoom=20x"]),
            |_| false
        ));
        assert!(!wants_console(&args(&["Case 12, H&E.svs"]), |path| path
            == Path::new("Case 12, H&E.svs")));
    }

    #[test]
    fn fatal_error_text_takes_the_last_error_report() {
        assert_eq!(
            fatal_error_text("Error: no Vulkan adapter\n\nCaused by:\n    x\n").unwrap(),
            "no Vulkan adapter\n\nCaused by:\n    x"
        );
        assert_eq!(
            fatal_error_text("WARN something\nError: bad config\n").unwrap(),
            "bad config"
        );
        assert!(fatal_error_text("thread 'main' panicked at x").is_none());
    }

    #[test]
    fn prompts_are_supported_outside_windows_release_builds() {
        if !cfg!(all(windows, not(debug_assertions))) {
            assert!(prompts_supported());
        }
    }
}

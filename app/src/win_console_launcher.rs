//! `eov.com`: the console launcher shipped next to `eov.exe` on Windows.
//!
//! `eov.exe` is a GUI-subsystem program, so cmd.exe and PowerShell neither
//! give it a console nor wait for it. The Windows release also ships this
//! console-subsystem program as `eov.com`; because `.COM` precedes `.EXE` in
//! the default `PATHEXT`, typing `eov ...` in a terminal runs the launcher,
//! which starts `eov.exe` with the same arguments and console handles, waits
//! for it and returns its exit code, so output, prompts (`eov update`) and
//! scripts behave like any console program. Plain viewer launches
//! (`eov slide.svs`) are started without waiting.

#[cfg(windows)]
#[path = "win_console.rs"]
#[allow(dead_code)]
mod win_console;

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    use std::process::{Command, ExitCode};
    use windows_sys::Win32::System::Console::SetConsoleCtrlHandler;

    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let exe = match std::env::current_exe() {
        Ok(path) => path.with_file_name("eov.exe"),
        Err(error) => {
            eprintln!("eov: cannot locate eov.exe: {error}");
            return ExitCode::FAILURE;
        }
    };
    let mut command = Command::new(&exe);
    command.args(&args).env(win_console::LAUNCHER_ENV, "1");
    if !win_console::wants_console(&args, |path| path.exists()) {
        // Opening slides: do not keep the terminal busy until the viewer
        // closes.
        return match command.spawn() {
            Ok(_) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("eov: could not start {}: {error}", exe.display());
                ExitCode::FAILURE
            }
        };
    }
    // Ctrl+C reaches eov.exe directly (it shares this console); the
    // launcher only waits for it.
    // SAFETY: a null handler with TRUE makes this process ignore Ctrl+C.
    unsafe { SetConsoleCtrlHandler(None, 1) };
    match command.status() {
        Ok(status) => ExitCode::from(status.code().unwrap_or(1).clamp(0, 255) as u8),
        Err(error) => {
            eprintln!("eov: could not start {}: {error}", exe.display());
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(windows))]
fn main() -> std::process::ExitCode {
    eprintln!("eov-console is the Windows console launcher (eov.com); run eov directly.");
    std::process::ExitCode::FAILURE
}

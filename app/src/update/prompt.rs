//! Interactive mutation confirmation helpers.

use anyhow::{Result, bail};
use std::io::{IsTerminal, Write};

pub fn confirm(prompt: &str, yes: bool) -> Result<()> {
    if yes {
        return Ok(());
    }
    if !std::io::stdin().is_terminal() {
        bail!("{prompt}\nRe-run with -y/--yes in a noninteractive environment.");
    }
    let mut stdout = std::io::stdout();
    writeln!(stdout, "{prompt}\n\nProceed? (Y/n)")?;
    stdout.flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    let answer = answer.trim();
    if answer.is_empty() || answer.eq_ignore_ascii_case("y") || answer.eq_ignore_ascii_case("yes") {
        Ok(())
    } else {
        bail!("Operation cancelled.");
    }
}

//! Interactive confirmation and consent.
//!
//! All questions go through a [`Prompter`] so tests can script answers and a
//! future GUI can answer them with dialogs. The terminal prompter writes
//! questions to stderr (stdout stays machine-readable, e.g. with `--json`)
//! and treats end-of-input (Ctrl-D, a closed pipe) and read errors as "no".

use anyhow::{Result, bail};
use std::io::{BufRead, IsTerminal, Write};

/// An answer to a yes/no question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answer {
    Yes,
    No,
}

/// Asks yes/no questions.
pub trait Prompter: Send {
    /// Whether questions can be answered at all (a terminal is attached).
    fn interactive(&self) -> bool;

    /// Asks `question`. An empty answer selects `default`; end of input and
    /// read errors always answer [`Answer::No`].
    fn ask(&mut self, question: &str, default: Answer) -> Answer;
}

/// Questions on the terminal.
#[derive(Debug, Default)]
pub struct TerminalPrompter;

impl Prompter for TerminalPrompter {
    fn interactive(&self) -> bool {
        std::io::stdin().is_terminal()
    }

    fn ask(&mut self, question: &str, default: Answer) -> Answer {
        let hint = match default {
            Answer::Yes => "(Y/n)",
            Answer::No => "(y/N)",
        };
        let mut stderr = std::io::stderr();
        if writeln!(stderr, "{question} {hint}").is_err() || stderr.flush().is_err() {
            return Answer::No;
        }
        let mut line = String::new();
        match std::io::stdin().lock().read_line(&mut line) {
            Ok(0) | Err(_) => {
                // Ctrl-D or a closed stdin is never consent.
                let _ = writeln!(stderr);
                Answer::No
            }
            Ok(_) => parse_answer(&line, default),
        }
    }
}

/// On Windows, prompts need the `eov.com` console launcher.
pub fn windows_hint() -> &'static str {
    if crate::win_console::prompts_supported() {
        ""
    } else {
        "\n(On Windows, run `eov` rather than `eov.exe` from a terminal: the eov.com launcher lets eov ask questions.)"
    }
}

/// Interprets a typed answer.
pub fn parse_answer(line: &str, default: Answer) -> Answer {
    let answer = line.trim();
    if answer.is_empty() {
        default
    } else if answer.eq_ignore_ascii_case("y") || answer.eq_ignore_ascii_case("yes") {
        Answer::Yes
    } else {
        Answer::No
    }
}

/// How a command interacts with the user: `-y` and the prompter.
pub struct Interaction {
    /// `-y/--yes`: skip mutation confirmations (never network consent).
    pub assume_yes: bool,
    prompter: Box<dyn Prompter>,
}

impl Interaction {
    pub fn new(assume_yes: bool, prompter: Box<dyn Prompter>) -> Self {
        Self {
            assume_yes,
            prompter,
        }
    }

    /// Terminal interaction.
    pub fn terminal(assume_yes: bool) -> Self {
        Self::new(assume_yes, Box::new(TerminalPrompter))
    }

    /// Whether a person can answer questions.
    pub fn interactive(&self) -> bool {
        self.prompter.interactive()
    }

    /// Asks before mutating anything. `-y` answers yes; a non-interactive
    /// session without `-y` is refused rather than assumed to agree.
    pub fn confirm(&mut self, message: &str) -> Result<()> {
        if self.assume_yes {
            return Ok(());
        }
        if !self.prompter.interactive() {
            bail!(
                "{message}\n\nConfirmation is required, but no terminal is attached.\nRe-run with -y/--yes to proceed without a prompt.{}",
                windows_hint()
            );
        }
        match self
            .prompter
            .ask(&format!("{message}\n\nProceed?"), Answer::Yes)
        {
            Answer::Yes => Ok(()),
            Answer::No => bail!("Operation cancelled."),
        }
    }

    /// Asks for consent to something `-y` must not imply (network access).
    /// Returns `false` when declined or when nobody can answer.
    pub fn consent(&mut self, message: &str) -> bool {
        self.prompter.interactive() && self.prompter.ask(message, Answer::Yes) == Answer::Yes
    }
}

/// A prompter with scripted answers, for tests. `None` simulates end of
/// input.
#[cfg(test)]
pub struct ScriptedPrompter {
    pub interactive: bool,
    pub answers: std::collections::VecDeque<Option<&'static str>>,
    pub asked: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

#[cfg(test)]
impl ScriptedPrompter {
    pub fn new(interactive: bool, answers: &[Option<&'static str>]) -> Self {
        Self {
            interactive,
            answers: answers.iter().copied().collect(),
            asked: Default::default(),
        }
    }
}

#[cfg(test)]
impl Prompter for ScriptedPrompter {
    fn interactive(&self) -> bool {
        self.interactive
    }

    fn ask(&mut self, question: &str, default: Answer) -> Answer {
        self.asked.lock().unwrap().push(question.to_string());
        match self.answers.pop_front().flatten() {
            Some(line) => parse_answer(line, default),
            None => Answer::No,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_answer_selects_default_and_other_text_declines() {
        assert_eq!(parse_answer("\n", Answer::Yes), Answer::Yes);
        assert_eq!(parse_answer("  yes ", Answer::No), Answer::Yes);
        assert_eq!(parse_answer("Y", Answer::No), Answer::Yes);
        assert_eq!(parse_answer("n", Answer::Yes), Answer::No);
        assert_eq!(parse_answer("sure", Answer::Yes), Answer::No);
    }

    #[test]
    fn end_of_input_never_confirms() {
        let mut interaction =
            Interaction::new(false, Box::new(ScriptedPrompter::new(true, &[None])));
        let error = interaction.confirm("Replace eov?").unwrap_err();
        assert_eq!(error.to_string(), "Operation cancelled.");
        let mut interaction =
            Interaction::new(false, Box::new(ScriptedPrompter::new(true, &[None])));
        assert!(!interaction.consent("Use the network?"));
    }

    #[test]
    fn yes_flag_skips_confirmation_but_not_consent() {
        let mut interaction = Interaction::new(true, Box::new(ScriptedPrompter::new(false, &[])));
        interaction.confirm("Replace eov?").unwrap();
        assert!(!interaction.consent("Use the network?"));
    }

    #[test]
    fn non_interactive_confirmation_is_refused_with_guidance() {
        let mut interaction = Interaction::new(false, Box::new(ScriptedPrompter::new(false, &[])));
        let error = interaction.confirm("Replace eov?").unwrap_err().to_string();
        assert!(error.contains("-y/--yes"), "{error}");
    }
}

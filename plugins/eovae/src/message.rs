//! Texts eovae both shows and records: why a job, a model or the inference
//! engine failed. The user reads them in the language of the interface;
//! eov's log and batch output get them in English (see the README).

use eov_plugin_api::i18n::{self, tr};
use std::fmt;

/// The same text in the language of the interface ([`Self::shown`]) and in
/// English ([`Self::english`]), written by the same code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    /// In the language of the interface when it was made: for the user.
    pub shown: String,
    /// In English: for eov's log and for batch output.
    pub english: String,
}

impl Message {
    /// The text `write` makes, in both languages (it is called twice, see
    /// [`i18n::shown_and_english`]). Make names, numbers and other
    /// messages part of it inside `write`: `i18n::isolate(name)`, and a
    /// [`Message`] by its `Display`.
    pub fn new(write: impl Fn() -> String) -> Self {
        let (shown, english) = i18n::shown_and_english(write);
        Self { shown, english }
    }

    /// A text that is the same in every language: eov's, ONNX Runtime's
    /// or the system's own words (which are English), a name.
    pub fn plain(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            shown: text.clone(),
            english: text,
        }
    }
}

/// The form for the language being written: English inside
/// [`i18n::in_english`] (where [`Message::new`] writes the English form),
/// the shown one otherwise. So a message made part of another one appears
/// in each of its forms in that form's language.
impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if i18n::current().is_source() {
            &self.english
        } else {
            &self.shown
        })
    }
}

/// The user (or eov, or a newer job) cancelled the operation.
pub fn cancelled() -> Message {
    Message::new(|| {
        // Translators: an error: the operation was cancelled
        tr!("cancelled")
    })
}

// Texts in other languages are checked in tests/language.rs: the language is
// global to the library.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_english_both_forms_are_the_same() {
        let inner = Message::plain("libonnxruntime.so: not found");
        let outer = Message::new(|| format!("ONNX Runtime could not be loaded: {inner}"));
        assert_eq!(outer.shown, outer.english);
        assert_eq!(
            outer.english,
            "ONNX Runtime could not be loaded: libonnxruntime.so: not found"
        );
        assert_eq!(outer.to_string(), outer.english);
        assert_eq!(cancelled(), Message::plain("cancelled"));
    }
}

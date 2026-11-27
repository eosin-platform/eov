//! Texts eovae both shows and records: why a job, a model or the inference
//! engine failed. The user reads them in the language of the interface,
//! also after it changes; eov's log and batch output get them in English
//! (see the README).

use eov_plugin_api::i18n::{self, tr};
use std::fmt;
use std::sync::Arc;

/// A text the user reads ([`Self::shown`]) and eov's log and batch output
/// record in English ([`Self::english`]), written by the same code.
///
/// It keeps the code that writes it, so it is written again each time it
/// is shown: in the language of the interface at that moment, also when
/// the user picks another one while the text stays on screen (a model that
/// could not be loaded).
#[derive(Clone)]
pub struct Message {
    /// Writes the text in the language being written.
    write: Arc<dyn Fn() -> String + Send + Sync>,
    /// In English: for eov's log and for batch output.
    pub english: String,
}

impl Message {
    /// The text `write` makes: in English now (see [`i18n::in_english`]),
    /// in the language of the interface whenever it is shown. `write` is
    /// kept, so it owns what it inserts (a `move` closure). Make names,
    /// numbers and other messages part of it inside `write`:
    /// `i18n::isolate(&name)`, and a [`Message`] by its `Display`.
    pub fn new(write: impl Fn() -> String + Send + Sync + 'static) -> Self {
        let english = i18n::in_english(&write);
        Self {
            write: Arc::new(write),
            english,
        }
    }

    /// A text that is the same in every language: eov's, ONNX Runtime's
    /// or the system's own words (which are English), a name.
    pub fn plain(text: impl Into<String>) -> Self {
        let english = text.into();
        let text: Arc<str> = Arc::from(english.as_str());
        Self {
            write: Arc::new(move || text.to_string()),
            english,
        }
    }

    /// In the language of the interface now: for the user.
    pub fn shown(&self) -> String {
        self.to_string()
    }
}

/// The form for the language being written: English inside
/// [`i18n::in_english`] (where [`Message::new`] writes the English form)
/// and in an English interface, the interface's language otherwise. So a
/// message made part of another one appears in each of its forms in that
/// form's language.
impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if i18n::current().is_source() {
            f.write_str(&self.english)
        } else {
            f.write_str(&(self.write)())
        }
    }
}

impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Message")
            .field("shown", &self.shown())
            .field("english", &self.english)
            .finish()
    }
}

/// Messages are equal when they say the same in English.
impl PartialEq for Message {
    fn eq(&self, other: &Self) -> bool {
        self.english == other.english
    }
}

impl Eq for Message {}

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
        let outer = Message::new(move || format!("ONNX Runtime could not be loaded: {inner}"));
        assert_eq!(outer.shown(), outer.english);
        assert_eq!(
            outer.english,
            "ONNX Runtime could not be loaded: libonnxruntime.so: not found"
        );
        assert_eq!(outer.to_string(), outer.english);
        assert_eq!(cancelled(), Message::plain("cancelled"));
    }
}

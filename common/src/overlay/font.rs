//! The embedded overlay font.
//!
//! `fonts/DejaVuSans-eov.ttf` is a 62 KB subset of DejaVu Sans 2.37
//! (Basic Latin, Latin-1 Supplement, Latin Extended-A, Greek, common
//! punctuation, arrows and mathematical symbols; hinting removed), embedded
//! so that text burned into exports never depends on system font paths
//! (EXP-5). Its licence — the Bitstream Vera and Arev font licences; DejaVu
//! changes are public domain — is in `fonts/LICENSE-DejaVu.txt`, reproduced
//! from the font's own `name` table (ID 13). The subset keeps the family name,
//! which the licence permits as it contains neither "Bitstream", "Vera",
//! "Arev" nor "Tavmjong Bah".
//!
//! Regenerate with fontTools:
//!
//! ```text
//! pyftsubset DejaVuSans.ttf \
//!   --unicodes="U+0020-007E,U+00A0-017F,U+0391-03A9,U+03B1-03C9,U+2010-2027,U+2030,\
//! U+2032-2033,U+2039-203A,U+2122,U+2190-2195,U+2212,U+221E,U+2248,U+2260,U+2264-2265,\
//! U+25A0-25A1,U+25CF,U+FFFD" \
//!   --no-hinting --layout-features='kern' --legacy-kern \
//!   --name-IDs='*' --name-languages='*' --output-file=DejaVuSans-eov.ttf
//! ```

use std::sync::OnceLock;

use ab_glyph::FontArc;

static FONT_BYTES: &[u8] = include_bytes!("fonts/DejaVuSans-eov.ttf");

/// The embedded sans-serif font used for all burned-in text.
pub fn embedded_font() -> &'static FontArc {
    static FONT: OnceLock<FontArc> = OnceLock::new();
    FONT.get_or_init(|| FontArc::try_from_slice(FONT_BYTES).expect("embedded font is valid"))
}

/// Pre-1.0 entry point. Always returns the embedded font now, so overlay
/// text can no longer be silently dropped on systems without the expected
/// font files.
#[deprecated(since = "1.0.0", note = "use `overlay::embedded_font()`")]
pub fn load_system_font() -> Option<FontArc> {
    Some(embedded_font().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ab_glyph::Font;

    #[test]
    fn embedded_font_covers_the_characters_labels_use() {
        let font = embedded_font();
        for ch in "0123456789 .,:;-–—+±×·%° µmμ≈≤≥ inmm px γ Δ AaZz ÄÉØßé Łł…→".chars()
        {
            assert_ne!(font.glyph_id(ch).0, 0, "missing glyph for {ch:?}");
        }
    }
}

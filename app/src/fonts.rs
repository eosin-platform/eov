//! Chinese, Japanese and Korean text in the font of its language.
//!
//! Slint 1.19 lays text out with parley and fontique and gives them no
//! language: a text without a `font-family` takes each character from the
//! first family of the generic families `system-ui`, then `sans-serif`, that
//! has it, in the order fontique got them from the system (on Linux,
//! fontconfig's order for no language in particular). So 、 and 。 come from
//! Noto Sans Mongolian or PhagsPa, which have them and come before any CJK
//! font, and are drawn as small marks at mid-height; and Chinese, Japanese
//! and Korean all take the first CJK font of the list, with its glyph forms.
//!
//! While the interface is in Japanese, Chinese or Korean, the font of that
//! language (the first of [`candidates`] that is installed) comes right after
//! the first family of both lists, before every other family that has CJK
//! glyphs. The first family keeps drawing Latin text, unless it has CJK
//! glyphs itself: then the language's font comes first. Other languages get
//! the lists as they were. Slint's font collection is shared, so every
//! window (plugin windows too) uses the new order from its next frame;
//! texts that are not laid out again keep their glyphs until they are.

use eov_plugin_api::i18n::Language;
use slint::fontique_012::fontique::{
    Collection, FamilyId, GenericFamily, QueryFamily, QueryStatus, SourceCache,
};
use std::cell::RefCell;

/// The generic families Slint takes text without a `font-family` from, in
/// its order.
const GENERIC_FAMILIES: [GenericFamily; 2] = [GenericFamily::SystemUi, GenericFamily::SansSerif];

/// Characters only fonts with Chinese, Japanese or Korean text have, or CJK
/// punctuation (as Noto Sans Mongolian has).
const CJK_SAMPLES: [char; 5] = ['\u{3001}', '\u{3002}', 'あ', '一', '가'];

/// Where eov runs: which fonts the system comes with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Platform {
    /// Linux and the other systems that have fontconfig.
    Linux,
    Windows,
    MacOs,
}

impl Platform {
    const ALL: [Platform; 3] = [Platform::Linux, Platform::Windows, Platform::MacOs];

    const CURRENT: Platform = if cfg!(target_os = "windows") {
        Platform::Windows
    } else if cfg!(target_os = "macos") {
        Platform::MacOs
    } else {
        Platform::Linux
    };
}

/// The fonts for the language `tag` that `platform` comes with or its
/// distributions package, preferred first.
fn platform_candidates(tag: &str, platform: Platform) -> &'static [&'static str] {
    use Platform::*;
    match (tag, platform) {
        // "Source Han Sans" alone is the Japanese face of the Source Han
        // Sans collection (the others are K, SC, TC and HC).
        ("ja", Linux) => &[
            "Noto Sans CJK JP",
            "Source Han Sans JP",
            "Source Han Sans",
            "Noto Sans JP",
            "IPAexGothic",
            "IPAPGothic",
            "TakaoPGothic",
            "VL PGothic",
        ],
        ("ja", Windows) => &["Yu Gothic UI", "Meiryo UI", "Yu Gothic", "Meiryo"],
        ("ja", MacOs) => &["Hiragino Sans", "Hiragino Kaku Gothic ProN"],
        ("zh-Hans", Linux) => &[
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "Source Han Sans CN",
            "Noto Sans SC",
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "Droid Sans Fallback",
        ],
        ("zh-Hans", Windows) => &["Microsoft YaHei UI", "Microsoft YaHei"],
        // fontique cannot draw PingFang where it has hvgl outlines (recent
        // macOS) and leaves it out; Heiti is what fontique falls back to.
        ("zh-Hans", MacOs) => &["PingFang SC", "Hiragino Sans GB", "Heiti SC"],
        ("zh-Hant", Linux) => &[
            "Noto Sans CJK TC",
            "Source Han Sans TC",
            "Source Han Sans TW",
            "Noto Sans TC",
            // Hong Kong forms are closer to Taiwan's than any other.
            "Noto Sans CJK HK",
            "Source Han Sans HC",
            "Source Han Sans HK",
        ],
        ("zh-Hant", Windows) => &["Microsoft JhengHei UI", "Microsoft JhengHei"],
        ("zh-Hant", MacOs) => &["PingFang TC", "Heiti TC"],
        ("ko", Linux) => &[
            "Noto Sans CJK KR",
            "Source Han Sans KR",
            "Source Han Sans K",
            "Noto Sans KR",
            "NanumGothic",
            "UnDotum",
        ],
        ("ko", Windows) => &["Malgun Gothic"],
        ("ko", MacOs) => &["Apple SD Gothic Neo"],
        _ => &[],
    }
}

/// The fonts to look for when the interface is in `tag`: `platform`'s
/// first, then those of the other systems (a font copied over from another
/// system is still a font for the language). None for languages that are
/// not written with CJK characters.
fn candidates(tag: &str, platform: Platform) -> impl Iterator<Item = &'static str> + '_ {
    std::iter::once(platform)
        .chain(
            Platform::ALL
                .into_iter()
                .filter(move |other| *other != platform),
        )
        .flat_map(move |platform| platform_candidates(tag, platform).iter().copied())
}

/// The first font for `tag` that `installed` says is there.
fn choose(
    tag: &str,
    platform: Platform,
    mut installed: impl FnMut(&str) -> bool,
) -> Option<&'static str> {
    candidates(tag, platform).find(|name| installed(name))
}

/// `families` with `font` right after the first family, or first when the
/// first family has CJK glyphs itself (or is `font`), and nowhere else. An
/// empty list stays empty: the text goes on to the next generic family.
fn reorder<T: Copy + PartialEq>(families: &[T], font: T, first_has_cjk: bool) -> Vec<T> {
    let Some(&first) = families.first() else {
        return Vec::new();
    };
    let mut reordered = Vec::with_capacity(families.len() + 1);
    let rest = if first == font || first_has_cjk {
        families
    } else {
        reordered.push(first);
        &families[1..]
    };
    reordered.push(font);
    reordered.extend(rest.iter().copied().filter(|family| *family != font));
    reordered
}

/// The generic family lists as they were before eov changed them.
#[derive(Clone)]
struct Original {
    /// The families the collection itself puts first (Slint's: the fonts
    /// `SLINT_DEFAULT_FONT` and `SLINT_FONT_PATH` name, usually none); the
    /// system's follow them.
    own: [Vec<FamilyId>; 2],
    /// The whole lists, the collection's then the system's.
    whole: [Vec<FamilyId>; 2],
}

impl Original {
    /// Reads the lists of `collection`, which eov has not changed yet. The
    /// collection only gives the whole lists; without its own part, what
    /// is left is the system's.
    fn read(collection: &mut Collection) -> Self {
        let whole: [Vec<FamilyId>; 2] =
            GENERIC_FAMILIES.map(|generic| collection.generic_families(generic).collect());
        let own = std::array::from_fn(|index| {
            let generic = GENERIC_FAMILIES[index];
            collection.set_generic_families(generic, std::iter::empty());
            let system: Vec<FamilyId> = collection.generic_families(generic).collect();
            let whole = whole[index].as_slice();
            whole
                .strip_suffix(system.as_slice())
                .unwrap_or(whole)
                .to_vec()
        });
        Self { own, whole }
    }
}

thread_local! {
    /// The lists before eov first changed them on this thread (Slint's
    /// collection is per thread); `None` until then.
    static ORIGINAL: RefCell<Option<Original>> = const { RefCell::new(None) };
}

/// Puts the font of `language` first for CJK text, or the system's order
/// back for languages without one. Costs nothing until the interface is
/// first in Japanese, Chinese or Korean. Needs Slint's platform (a window).
pub(crate) fn apply(language: &Language) {
    let tag = language.tag;
    let cjk = candidates(tag, Platform::CURRENT).next().is_some();
    if !cjk && ORIGINAL.with_borrow(Option::is_none) {
        return;
    }
    let mut collection = slint::fontique_012::shared_collection();
    let original = ORIGINAL.with_borrow_mut(|original| {
        original
            .get_or_insert_with(|| Original::read(&mut collection))
            .clone()
    });
    let font = if cjk {
        choose(tag, Platform::CURRENT, |name| {
            collection.family_id(name).is_some()
        })
        .and_then(|name| Some((name, collection.family_id(name)?)))
    } else {
        None
    };
    let Some((name, font)) = font else {
        if cjk {
            tracing::info!(
                "No font for {tag} that eov knows is installed; CJK text uses the system's font order"
            );
        }
        for (generic, own) in GENERIC_FAMILIES.into_iter().zip(original.own) {
            collection.set_generic_families(generic, own.into_iter());
        }
        return;
    };
    tracing::info!("CJK text in {tag} uses {name}");
    for (generic, whole) in GENERIC_FAMILIES.into_iter().zip(&original.whole) {
        let first_has_cjk = whole
            .first()
            .is_some_and(|first| *first != font && has_cjk_glyphs(&mut collection, *first));
        let reordered = reorder(whole, font, first_has_cjk);
        collection.set_generic_families(generic, reordered.into_iter());
    }
}

/// Whether the regular font of `family` has any of [`CJK_SAMPLES`].
fn has_cjk_glyphs(collection: &mut Collection, family: FamilyId) -> bool {
    let mut source_cache = SourceCache::default();
    let mut query = collection.query(&mut source_cache);
    query.set_families([QueryFamily::Id(family)]);
    let mut found = false;
    query.matches_with(|font| {
        found = font.charmap().is_some_and(|charmap| {
            CJK_SAMPLES
                .iter()
                .any(|&sample| charmap.map(sample).is_some_and(|glyph| glyph != 0))
        });
        QueryStatus::Stop
    });
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINUX_NOTO: [&str; 5] = [
        "Noto Sans",
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK KR",
    ];

    fn installed(fonts: &[&str]) -> impl FnMut(&str) -> bool {
        move |name| fonts.contains(&name)
    }

    #[test]
    fn each_cjk_language_gets_its_own_font_on_each_system() {
        let windows = [
            "Segoe UI",
            "Yu Gothic UI",
            "Meiryo UI",
            "Microsoft YaHei UI",
            "Microsoft JhengHei UI",
            "Malgun Gothic",
        ];
        let macos = [
            "Helvetica",
            "Hiragino Sans",
            "PingFang SC",
            "PingFang TC",
            "Apple SD Gothic Neo",
        ];
        for (tag, linux_font, windows_font, macos_font) in [
            ("ja", "Noto Sans CJK JP", "Yu Gothic UI", "Hiragino Sans"),
            (
                "zh-Hans",
                "Noto Sans CJK SC",
                "Microsoft YaHei UI",
                "PingFang SC",
            ),
            (
                "zh-Hant",
                "Noto Sans CJK TC",
                "Microsoft JhengHei UI",
                "PingFang TC",
            ),
            (
                "ko",
                "Noto Sans CJK KR",
                "Malgun Gothic",
                "Apple SD Gothic Neo",
            ),
        ] {
            assert_eq!(
                choose(tag, Platform::Linux, installed(&LINUX_NOTO)),
                Some(linux_font)
            );
            assert_eq!(
                choose(tag, Platform::Windows, installed(&windows)),
                Some(windows_font)
            );
            assert_eq!(
                choose(tag, Platform::MacOs, installed(&macos)),
                Some(macos_font)
            );
        }
    }

    #[test]
    fn the_system_s_own_font_wins_and_others_stand_in() {
        let both = ["Noto Sans CJK JP", "Yu Gothic UI", "Hiragino Sans"];
        assert_eq!(
            choose("ja", Platform::Linux, installed(&both)),
            Some("Noto Sans CJK JP")
        );
        assert_eq!(
            choose("ja", Platform::Windows, installed(&both)),
            Some("Yu Gothic UI")
        );
        assert_eq!(
            choose("ja", Platform::MacOs, installed(&both)),
            Some("Hiragino Sans")
        );
        // Older Windows, distributions' spellings, copied fonts.
        assert_eq!(
            choose("ja", Platform::Windows, installed(&["Meiryo UI"])),
            Some("Meiryo UI")
        );
        assert_eq!(
            choose(
                "ja",
                Platform::Linux,
                installed(&["Source Han Sans", "Source Han Sans K"])
            ),
            Some("Source Han Sans")
        );
        assert_eq!(
            choose("ko", Platform::Linux, installed(&["Source Han Sans K"])),
            Some("Source Han Sans K")
        );
        assert_eq!(
            choose("zh-Hans", Platform::Linux, installed(&["Microsoft YaHei"])),
            Some("Microsoft YaHei")
        );
        // macOS without a PingFang fontique can draw.
        assert_eq!(
            choose("zh-Hant", Platform::MacOs, installed(&["Heiti TC"])),
            Some("Heiti TC")
        );
        // A font of another language is never taken.
        assert_eq!(
            choose("ja", Platform::Linux, installed(&["Noto Sans CJK SC"])),
            None
        );
        assert_eq!(choose("ko", Platform::Linux, installed(&[])), None);
    }

    #[test]
    fn other_languages_have_no_cjk_font() {
        for language in eov_plugin_api::i18n::LANGUAGES {
            let cjk = matches!(language.tag, "ja" | "zh-Hans" | "zh-Hant" | "ko");
            for platform in Platform::ALL {
                assert_eq!(
                    candidates(language.tag, platform).next().is_some(),
                    cjk,
                    "{} on {platform:?}",
                    language.tag
                );
                if !cjk {
                    assert_eq!(choose(language.tag, platform, installed(&LINUX_NOTO)), None);
                }
            }
        }
    }

    #[test]
    fn the_font_comes_right_after_the_first_family() {
        // Noto Sans, then fonts with CJK punctuation before the CJK font.
        let list = [
            "Latin",
            "Arabic",
            "Mongolian",
            "PhagsPa",
            "CJK JP",
            "DejaVu",
        ];
        assert_eq!(
            reorder(&list, "CJK SC", false),
            [
                "Latin",
                "CJK SC",
                "Arabic",
                "Mongolian",
                "PhagsPa",
                "CJK JP",
                "DejaVu"
            ]
        );
        // A font already in the list moves instead of appearing twice.
        assert_eq!(
            reorder(&list, "CJK JP", false),
            [
                "Latin",
                "CJK JP",
                "Arabic",
                "Mongolian",
                "PhagsPa",
                "DejaVu"
            ]
        );
        assert_eq!(
            reorder(&["Segoe UI"], "Yu Gothic UI", false),
            ["Segoe UI", "Yu Gothic UI"]
        );
    }

    #[test]
    fn a_first_family_with_cjk_glyphs_goes_after_the_font() {
        let list = ["CJK SC", "Latin", "CJK JP"];
        assert_eq!(
            reorder(&list, "CJK JP", true),
            ["CJK JP", "CJK SC", "Latin"]
        );
        // The font itself first: the order stays.
        assert_eq!(reorder(&list, "CJK SC", false), list);
        assert_eq!(reorder(&list, "CJK SC", true), list);
        // No family at all: the next generic family is asked.
        assert!(reorder(&[], "CJK SC", false).is_empty());
    }
}

/// The font order in the window as the interface changes language.
#[cfg(test)]
mod window_tests {
    use super::*;
    use eov_plugin_api::i18n;
    use slint::ComponentHandle;

    fn lists() -> Vec<Vec<FamilyId>> {
        let mut collection = slint::fontique_012::shared_collection();
        GENERIC_FAMILIES
            .iter()
            .map(|generic| collection.generic_families(*generic).collect())
            .collect()
    }

    /// The order fonts are tried in: `list` without repeats (eov's part of
    /// a list comes before the system's, which has the same families).
    fn tried(list: &[FamilyId]) -> Vec<FamilyId> {
        let mut seen = Vec::new();
        for family in list {
            if !seen.contains(family) {
                seen.push(*family);
            }
        }
        seen
    }

    #[test]
    fn cjk_languages_put_their_font_first_and_others_restore_the_order() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let ui = crate::AppWindow::new().unwrap();
            ui.show().unwrap();
            struct English<'a>(&'a crate::AppWindow);
            impl Drop for English<'_> {
                fn drop(&mut self) {
                    crate::i18n::switch_to(self.0, i18n::source());
                }
            }
            let _english = English(&ui);
            crate::i18n::switch_to(&ui, i18n::source());
            let before = lists();
            for tag in ["ja", "zh-Hans", "zh-Hant", "ko", "ru", "ja", "en"] {
                let language = i18n::by_tag(tag).unwrap();
                crate::i18n::switch_to(&ui, language);
                let mut collection = slint::fontique_012::shared_collection();
                let font = choose(tag, Platform::CURRENT, |name| {
                    collection.family_id(name).is_some()
                })
                .and_then(|name| collection.family_id(name));
                let after = lists();
                let Some(font) = font else {
                    // Not a CJK language, or no font for it here: the lists
                    // as they were.
                    assert_eq!(after, before, "{tag}");
                    continue;
                };
                for (before, after) in before.iter().zip(&after) {
                    let order = tried(after);
                    let position = order.iter().position(|family| *family == font);
                    assert!(
                        matches!(position, Some(0 | 1)),
                        "{tag}: the font is tried at {position:?}"
                    );
                    if position == Some(1) {
                        assert_eq!(order[0], before[0], "{tag}: the first family stays first");
                    }
                    let others: Vec<FamilyId> =
                        order.iter().copied().filter(|f| *f != font).collect();
                    let original: Vec<FamilyId> =
                        tried(before).into_iter().filter(|f| *f != font).collect();
                    assert_eq!(
                        others, original,
                        "{tag}: the other families keep their order"
                    );
                }
            }
        });
    }
}

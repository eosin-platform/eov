"""Tests of the rules i18n.py checks translations against.

    python3 -m unittest discover -s tools/i18n -p 'test_*.py'
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location("i18n", Path(__file__).with_name("i18n.py"))
i18n = importlib.util.module_from_spec(_spec)
sys.modules["i18n"] = i18n  # dataclasses look their module up by name
_spec.loader.exec_module(i18n)

RLM = "\u200f"
FSI = "\u2068"
PDI = "\u2069"


def language(tag: str):
    return next(entry for entry in i18n.languages() if entry.tag == tag)


def entry(msgid: str, msgid_plural: str = ""):
    return SimpleNamespace(msgid=msgid, msgid_plural=msgid_plural)


class RightToLeft(unittest.TestCase):
    def test_a_text_that_starts_with_its_own_letters_needs_no_mark(self):
        self.assertFalse(i18n.starts_left_to_right("فتح {}"))
        self.assertFalse(i18n.starts_left_to_right("פתיחת הקובץ {name}"))

    def test_a_latin_word_or_a_value_first_needs_the_mark(self):
        self.assertTrue(i18n.starts_left_to_right("TIFF هرمي"))
        self.assertTrue(i18n.starts_left_to_right("{} شرائح"))
        self.assertEqual(i18n.right_to_left("TIFF هرمي"), RLM + "TIFF هرمي")
        self.assertFalse(i18n.starts_left_to_right(RLM + "TIFF هرمي"))

    def test_isolated_runs_are_skipped(self):
        self.assertFalse(i18n.starts_left_to_right(f"{FSI}--language{PDI} الخيار"))
        self.assertTrue(i18n.starts_left_to_right(f"{FSI}فتح{PDI} TIFF"))

    def test_joined_values_need_the_mark_without_letters_of_the_language(self):
        for text in (
            "{plugin}: {status}",
            "Cellaris: {}",
            "{mean}; {dark}",
            "{reason}.\n{hint}",
            "{a} · {b}",
            "{path} ({source}): {what}",
        ):
            with self.subTest(text=text):
                self.assertTrue(i18n.starts_left_to_right(text))

    def test_single_values_and_names_keep_their_own_direction(self):
        for text in ("{} × {} px", "{}%", "{hours}:{minutes}", "OME-TIFF", "{tooltip} ({key})", "{x}, {y}", "Lanczos-3"):
            with self.subTest(text=text):
                self.assertFalse(i18n.starts_left_to_right(text))

    def test_each_line_is_judged_on_its_own(self):
        # Slint lays out every line of a text as a paragraph of its own.
        text = "فشل التحميل.\nONNX Runtime 1.20 مطلوب"
        self.assertEqual(i18n.left_to_right_lines(text), [1])
        self.assertEqual(i18n.right_to_left(text), text.replace("\nONNX", "\n" + RLM + "ONNX"))
        self.assertEqual(i18n.left_to_right_lines("{reason}.\n{hint}"), [0])
        self.assertEqual(i18n.left_to_right_lines("سطر\neov update --force"), [])

    def test_search_word_lists_get_no_mark(self):
        keywords = "whole-slide;wsi;pathology;microscopy;شريحة;علم الأمراض;"
        self.assertFalse(i18n.starts_left_to_right(keywords))
        self.assertEqual(i18n.right_to_left(keywords), keywords)


class Isolates(unittest.TestCase):
    def test_balanced(self):
        for text in ("", "a", f"{FSI}a{PDI}", f"{FSI}{FSI}a{PDI}{PDI} {FSI}b{PDI}"):
            with self.subTest(text=text):
                self.assertFalse(i18n.unbalanced_isolates(text))

    def test_unbalanced(self):
        for text in (f"{FSI}a", f"a{PDI}", f"{PDI}{FSI}", f"{FSI}{FSI}a{PDI}", f"{FSI}eov update\n--force{PDI}"):
            with self.subTest(text=text):
                self.assertTrue(i18n.unbalanced_isolates(text))


class Validation(unittest.TestCase):
    def test_right_to_left_rules_apply_to_right_to_left_languages_only(self):
        self.assertIn("U+200F", i18n.validate_translation(entry("Pyramidal TIFF"), "TIFF هرمي", language("ar")))
        self.assertIsNone(i18n.validate_translation(entry("Pyramidal TIFF"), RLM + "TIFF هرمي", language("ar")))
        self.assertIsNone(i18n.validate_translation(entry("Pyramidal TIFF"), "TIFF pyramidal", language("fr")))

    def test_an_open_isolate_is_refused_in_every_language(self):
        for tag, text in (("he", f"{RLM}סיומת {FSI}.eop"), ("de", f"Endung {FSI}.eop")):
            with self.subTest(tag=tag):
                self.assertIn("U+2069", i18n.validate_translation(entry("extension .eop"), text, language(tag)))

    def test_a_later_line_must_start_right_to_left_too(self):
        text = "فشل التحميل.\nONNX Runtime {} مطلوب"
        why = i18n.validate_translation(entry("Loading failed.\nONNX Runtime {} is needed"), text, language("ar"))
        self.assertIn("line 2", why)

    def test_embeddings_are_refused(self):
        why = i18n.validate_translation(entry("extension .eop"), "סיומת \u202a.eop\u202c", language("he"))
        self.assertIn("U+202A", why)

    def test_placeholders_must_match(self):
        self.assertIn("missing", i18n.validate_translation(entry("Open {name}"), "Öffnen", language("de")))
        self.assertIn("unknown", i18n.validate_translation(entry("Open"), "Öffnen {name}", language("de")))


if __name__ == "__main__":
    unittest.main()

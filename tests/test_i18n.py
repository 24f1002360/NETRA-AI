"""Prevent internal localisation keys from appearing in operator screens."""

import json
from pathlib import Path

import pytest

from app.i18n import load_translations, t


ROOT = Path(__file__).resolve().parents[1]
I18N_DIR = ROOT / "configs" / "i18n"
LANGUAGES = ("en", "hi", "ta", "te")

# Exact messages emitted by core.iqa.quality and core.inference fallbacks.
IQA_OPERATOR_KEYS = {
    "iqa.pass",
    "iqa.corrected",
    "iqa.retake.blur_high",
    "iqa.retake.too_dark",
    "iqa.retake.too_bright",
    "iqa.retake.fov_partial",
    "iqa.retake.low_contrast",
    "iqa.retake.off_centre",
    "iqa.retake.iqa_error",
}


def test_iqa_operator_messages_exist_for_every_supported_language():
    """Every backend IQA outcome must have operator-facing wording."""
    translations = {
        lang: json.loads((I18N_DIR / f"{lang}.json").read_text(encoding="utf-8"))
        for lang in LANGUAGES
    }

    for lang, catalogue in translations.items():
        missing = IQA_OPERATOR_KEYS - catalogue.keys()
        assert not missing, f"{lang}: missing operator messages: {sorted(missing)}"


@pytest.mark.parametrize("lang", LANGUAGES)
def test_missing_translation_never_exposes_an_internal_key(lang):
    load_translations(str(I18N_DIR))
    rendered = t("iqa.retake.future_internal_code", lang)
    assert rendered != "iqa.retake.future_internal_code"
    assert "iqa." not in rendered

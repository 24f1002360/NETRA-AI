import json
import os

from markupsafe import Markup, escape

_translations = {}
_translations_dir = None

def load_translations(i18n_dir):
    global _translations, _translations_dir
    _translations_dir = i18n_dir
    if not os.path.exists(i18n_dir):
        return
    for fname in os.listdir(i18n_dir):
        if fname.endswith('.json'):
            lang = fname.replace('.json', '')
            with open(os.path.join(i18n_dir, fname), 'r', encoding='utf-8') as f:
                _translations[lang] = json.load(f)

def _lookup(key, lang):
    return _translations.get(lang, {}).get(
        key,
        _translations.get('en', {}).get(key),
    )


def t(key, lang='en'):
    value = _lookup(key, lang)
    if value is not None:
        return value

    # Templates reload immediately in Flask development mode, while JSON
    # dictionaries are normally loaded at app startup. Refresh once on a
    # missing key so a new translated label never appears as its raw key.
    if _translations_dir:
        load_translations(_translations_dir)
        value = _lookup(key, lang)
        if value is not None:
            return value

    # Never reveal an implementation key (for example ``iqa.retake.blur_high``)
    # to an operator. Known IQA failures get an actionable, localised fallback;
    # every other missing label gets a neutral localised message instead.
    fallback_key = (
        "iqa.retake.unknown"
        if str(key).startswith("iqa.")
        else "ui.text_unavailable"
    )
    return _lookup(fallback_key, lang) or "Text unavailable"


def bilingual(key, lang='en'):
    """Return an operator-first label with a compact English helper line.

    English is intentionally not repeated when English is the selected
    language. Translation values are escaped before the small markup wrapper
    is added, so the helper is safe to render in Jinja templates.
    """
    primary = t(key, lang)
    english = t(key, 'en')
    if lang == 'en' or primary == english:
        return escape(primary)
    return Markup(
        '<span class="bilingual-label">'
        f'<span class="bilingual-label__primary">{escape(primary)}</span>'
        f'<span class="bilingual-label__english">{escape(english)}</span>'
        '</span>'
    )

def get_audio_path(key, lang='hi'):
    path = os.path.join('app', 'audio', lang, f"{key.replace('.', '_')}.mp3")
    return path if os.path.exists(path) else None

def available_languages():
    return list(_translations.keys())

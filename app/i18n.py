import json
import os

from markupsafe import Markup, escape

_translations = {}

def load_translations(i18n_dir):
    global _translations
    if not os.path.exists(i18n_dir):
        return
    for fname in os.listdir(i18n_dir):
        if fname.endswith('.json'):
            lang = fname.replace('.json', '')
            with open(os.path.join(i18n_dir, fname), 'r', encoding='utf-8') as f:
                _translations[lang] = json.load(f)

def t(key, lang='en'):
    return _translations.get(lang, {}).get(key, _translations.get('en', {}).get(key, key))


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

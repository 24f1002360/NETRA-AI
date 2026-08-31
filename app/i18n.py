import json
import os

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

def get_audio_path(key, lang='hi'):
    path = os.path.join('app', 'audio', lang, f"{key.replace('.', '_')}.mp3")
    return path if os.path.exists(path) else None

def available_languages():
    return list(_translations.keys())

import os
from flask import Flask, session
from app.i18n import load_translations, t
from app.routes import bp

def create_app(config):
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(config.get("app", {}))
    app.secret_key = app.config.get("secret_key", "dev")

    i18n_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "i18n")
    if os.path.exists(i18n_path):
        load_translations(i18n_path)

    @app.context_processor
    def inject_t():
        def t_jinja(key):
            lang = session.get("lang", app.config.get("default_language", "en"))
            return t(key, lang)
        return dict(t=t_jinja)

    app.register_blueprint(bp)
    return app

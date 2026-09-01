import os

from flask import Flask, render_template, request, session
from werkzeug.exceptions import RequestEntityTooLarge

from app.i18n import available_languages, bilingual, load_translations, t
from app.routes import bp, screening_bp

def create_app(config):
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(config.get("app", {}))
    # The capture screen validates 10 MB per image. Keep a small server-side
    # allowance for multipart metadata while preventing accidental huge uploads.
    if app.config.get("MAX_CONTENT_LENGTH") is None:
        app.config["MAX_CONTENT_LENGTH"] = 22 * 1024 * 1024
    app.secret_key = app.config.get("secret_key", "dev")

    i18n_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "i18n")
    if os.path.exists(i18n_path):
        load_translations(i18n_path)

    @app.context_processor
    def inject_t():
        def t_jinja(key):
            lang = session.get("lang", app.config.get("default_language", "en"))
            return t(key, lang)
        lang = session.get("lang", app.config.get("default_language", "en"))
        endpoint = request.endpoint or ""
        return {
            "t": t_jinja,
            "bt": lambda key: bilingual(key, lang),
            "lang": lang,
            "languages": available_languages(),
            "active_page": "history" if endpoint.endswith("history") else "capture",
        }

    app.register_blueprint(bp)
    app.register_blueprint(screening_bp, url_prefix="/api")

    @app.errorhandler(RequestEntityTooLarge)
    def file_too_large(_error):
        return (
            render_template(
                "_error.html",
                title="Images are too large",
                detail="Use JPG or PNG retinal images smaller than 10 MB each, then try again.",
            ),
            413,
        )
    
    return app

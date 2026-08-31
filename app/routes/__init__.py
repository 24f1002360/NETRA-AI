from __future__ import annotations

from flask import Blueprint

from app.routes.screening import screening_bp


# Existing application blueprint.
# app/__init__.py imports this as `bp`.
bp = Blueprint(
    "main",
    __name__,
)


# Backend API routes.
def register_routes(app):
    app.register_blueprint(
        screening_bp,
        url_prefix="/api",
    )
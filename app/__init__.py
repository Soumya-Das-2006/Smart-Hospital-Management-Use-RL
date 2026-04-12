"""
app/__init__.py — Flask application factory.

WHY THIS EXISTS: The factory pattern (create_app()) lets us create multiple
app instances with different configurations — one for production, one for
testing. This is the Django equivalent of having separate settings files.

Industry standards applied:
  - AppConfig.validate() called at startup — fail fast on missing secrets
  - SECRET_KEY loaded from environment variable, never hardcoded
  - Structured logging configured once here, not scattered across modules
  - Background cleanup thread started as daemon (auto-killed when main process exits)
  - All blueprints registered in one place for easy discovery
"""

import logging
import threading
import time

from flask import Flask

from core.config import AppConfig


def create_app() -> Flask:
    """
    WHY THIS EXISTS: Creates and configures the Flask application. Called once
    at startup by run.py and gunicorn. Test suites can call it with a test config.
    """
    # Validate config at startup — better to crash here than on first request
    AppConfig.validate()

    app = Flask(__name__, template_folder="templates", static_folder="static")

    # ── Flask configuration ────────────────────────────────────────────────────
    # SECRET_KEY from environment — NEVER hardcode this in source code
    app.config["SECRET_KEY"]      = AppConfig.SECRET_KEY
    app.config["JSON_SORT_KEYS"]  = False
    app.config["DEBUG"]           = AppConfig.DEBUG

    # ── Logging setup ─────────────────────────────────────────────────────────
    # Structured format with timestamps so logs are parseable by log aggregators
    logging.basicConfig(
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.DEBUG if AppConfig.DEBUG else logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Smart Hospital Allocator (debug=%s)", AppConfig.DEBUG)

    # ── Blueprint registration ─────────────────────────────────────────────────
    from app.routes.api   import api_bp, openenv_bp
    from app.routes.views import views_bp

    app.register_blueprint(openenv_bp)   # GET /health  POST /reset  POST /step  GET /state
    app.register_blueprint(api_bp)       # /api/*  — rich dashboard endpoints
    app.register_blueprint(views_bp)     # /  /dashboard  /admin  /docs  + error pages

    # ── Background stale-session cleanup ──────────────────────────────────────
    # WHY THIS EXISTS: Prevents memory leaks from abandoned sessions.
    # Runs as a daemon thread so it is automatically killed when the process exits.
    from app.services.hospital_service import get_service

    def _cleanup_loop() -> None:
        """Periodically purges sessions idle longer than SESSION_TTL_SECONDS."""
        svc = get_service()
        while True:
            time.sleep(300)   # check every 5 minutes
            try:
                purged = svc.purge_stale()
                if purged:
                    logging.getLogger(__name__).info(
                        "Background cleanup: purged %d stale session(s)", purged
                    )
            except Exception as exc:
                logging.getLogger(__name__).exception("Cleanup loop error: %s", exc)

    cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True, name="session-cleanup")
    cleanup_thread.start()

    return app

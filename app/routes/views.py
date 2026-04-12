"""
app/routes/views.py — HTML page routes.

WHY THIS EXISTS: Thin view layer that renders templates. Views here do no
business logic — they only pass static data to templates that the template
engine needs to render correctly (e.g. action metadata for the dashboard).

Industry standards applied:
  - Try/except on every route so a template rendering error returns a clean 500 page
  - Fat Service / Thin View: no business logic in views
  - Docstrings explain WHY each route exists
"""

import logging

from flask import Blueprint, render_template, abort

from core.env import HospitalEnv
from core.config import ENV_CFG, REWARD_CFG

logger     = logging.getLogger(__name__)
views_bp   = Blueprint("views", __name__)


@views_bp.get("/")
def index():
    """WHY THIS EXISTS: Landing page — architecture overview and quick-start links."""
    try:
        return render_template("index.html")
    except Exception as exc:
        logger.exception("Failed to render index: %s", exc)
        abort(500)


@views_bp.get("/dashboard")
def dashboard():
    """
    WHY THIS EXISTS: Live interactive RL dashboard where users can manually
    step through an episode or watch the heuristic agent play in real time.
    Pre-computes action metadata here so the template stays logic-free.
    """
    try:
        actions = [
            {"id": i, "name": name, "description": _action_description(i)}
            for i, name in HospitalEnv.ACTION_NAMES.items()
        ]
        return render_template("dashboard.html", actions=actions)
    except Exception as exc:
        logger.exception("Failed to render dashboard: %s", exc)
        abort(500)


@views_bp.get("/admin")
def admin():
    """WHY THIS EXISTS: Admin panel for downloads, episode exports, and server stats."""
    try:
        return render_template("admin.html")
    except Exception as exc:
        logger.exception("Failed to render admin: %s", exc)
        abort(500)


@views_bp.get("/docs")
def docs():
    """WHY THIS EXISTS: Full API reference so developers can integrate without guessing."""
    try:
        return render_template("docs.html")
    except Exception as exc:
        logger.exception("Failed to render docs: %s", exc)
        abort(500)


# ── Custom error pages ────────────────────────────────────────────────────────

@views_bp.app_errorhandler(404)
def not_found(error):
    """WHY THIS EXISTS: Returns a branded 404 instead of the default Flask/Werkzeug page."""
    return render_template("errors/404.html"), 404


@views_bp.app_errorhandler(500)
def internal_error(error):
    """
    WHY THIS EXISTS: Never expose raw Python tracebacks to end users.
    Returns a friendly 500 page and logs the real error for developers.
    """
    logger.exception("Unhandled 500 error: %s", error)
    return render_template("errors/500.html"), 500


# ── Private helpers ───────────────────────────────────────────────────────────

def _action_description(action_id: int) -> str:
    """
    WHY THIS EXISTS: Keeps human-readable action descriptions out of the template
    and in Python where they can be tested and updated without touching HTML.
    """
    descriptions = {
        0: "Admit the top-queue patient to an ICU bed (requires doctor + nurse + free ICU bed)",
        1: "Admit the top-queue patient to a general ward bed",
        2: "Dispatch an ambulance to bring in a critical patient (60% success rate)",
        3: "Escalate a deteriorating general-ward patient to ICU",
        4: "Discharge a low-severity recovered patient and free their bed",
        5: "Call extra staff: +2 doctors and +3 nurses for the next 60 timesteps",
        6: "Hold — monitor all patients and wait this timestep",
    }
    return descriptions.get(action_id, "Unknown action")

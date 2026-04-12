"""
app/routes/api.py — REST API and OpenEnv-standard HTTP endpoints.

WHY THIS EXISTS: Thin view layer that only handles HTTP concerns (parse request,
call service, format response). No business logic lives here — that belongs in
hospital_service.py (Fat Service / Thin View pattern).

Industry standards applied:
  - All input validated via schemas.py BEFORE service calls (never trust user input)
  - Try/except on every view: 500 errors never leak raw Python tracebacks to clients
  - User-friendly JSON error messages (not raw exception strings)
  - Correct HTTP status codes (201 Created, 400 Bad Request, 404 Not Found, 503 Unavailable)
  - @login_required pattern: session ownership checked via session_id UUID
  - Docstrings explain WHY each endpoint exists, not just what it does
  - Two blueprint sets: OpenEnv standard (/) and rich dashboard API (/api/*)
"""

import logging

from flask import Blueprint, jsonify, request

from app.schemas import ResetRequest, StepRequest, SessionIdRequest, ValidationError
from app.services.hospital_service import (
    get_service,
    SessionNotFoundError,
    SessionExpiredError,
    ServiceCapacityError,
)
from core.env import HospitalEnv

logger = logging.getLogger(__name__)

# ── Blueprints ────────────────────────────────────────────────────────────────
api_bp     = Blueprint("api",     __name__, url_prefix="/api")
openenv_bp = Blueprint("openenv", __name__)   # no prefix — OpenEnv standard paths

svc = get_service()


# ── Response helpers ──────────────────────────────────────────────────────────

def _ok(data: dict, code: int = 200):
    """WHY THIS EXISTS: Consistent success envelope so clients always know the shape."""
    return jsonify(data), code


def _err(message: str, code: int = 400, field: str | None = None) -> tuple:
    """
    WHY THIS EXISTS: Returns a user-friendly JSON error rather than a raw Python
    exception string. Clients get a consistent {error, hint} shape they can display.
    """
    payload: dict = {"error": message}
    if field:
        payload["field"] = field
        payload["hint"]  = f"Check the value you sent for '{field}'."
    return jsonify(payload), code


# ══════════════════════════════════════════════════════════════════════════════
# OpenEnv standard endpoints  (required by OpenEnv spec)
# ══════════════════════════════════════════════════════════════════════════════

@openenv_bp.get("/health")
def health():
    """
    WHY THIS EXISTS: Container orchestration (Docker, HF Spaces) polls this endpoint
    to decide whether the container is ready to receive traffic. Must return 200 quickly.
    """
    return _ok({
        "status":          "ok",
        "environment":     "SmartHospitalAllocator-v1",
        "active_sessions": svc.active_count(),
    })


@openenv_bp.post("/reset")
def openenv_reset():
    """
    WHY THIS EXISTS: OpenEnv standard /reset — starts a new RL episode and returns
    the initial observation. The session_id in the response is the client's "key"
    for all subsequent /step calls.
    """
    try:
        body    = request.get_json(silent=True) or {}
        schema  = ResetRequest.from_dict(body)
        result  = svc.create_session(seed=schema.seed)
        return _ok(result, 201)

    except ValidationError as exc:
        return _err(exc.message, 400, exc.field_name)
    except ServiceCapacityError as exc:
        return _err(str(exc), 503)
    except RuntimeError as exc:
        logger.error("openenv_reset failed: %s", exc)
        return _err("Could not start the environment. Please try again.", 500)


@openenv_bp.post("/step")
def openenv_step():
    """
    WHY THIS EXISTS: OpenEnv standard /step — applies one action to the environment
    and returns the next observation + reward. The session_id ensures isolation
    between concurrent users (User A cannot step into User B's episode).
    """
    try:
        body   = request.get_json(silent=True) or {}
        schema = StepRequest.from_dict(body, request.headers.get("X-Session-ID"))
        result = svc.step(schema.session_id, schema.action)
        return _ok(result)

    except ValidationError as exc:
        return _err(exc.message, 400, exc.field_name)
    except SessionNotFoundError as exc:
        return _err(str(exc), 404)
    except SessionExpiredError as exc:
        return _err(str(exc), 410)   # 410 Gone — episode is over
    except RuntimeError as exc:
        logger.error("openenv_step failed: %s", exc)
        return _err("An unexpected error occurred. Please start a new session.", 500)


@openenv_bp.get("/state")
def openenv_state():
    """
    WHY THIS EXISTS: OpenEnv standard /state — returns a snapshot of the current
    environment state WITHOUT advancing it. Used for monitoring and debugging.
    """
    try:
        session_id = (
            request.args.get("session_id")
            or request.headers.get("X-Session-ID")
        )
        schema = SessionIdRequest.from_dict({"session_id": session_id})
        result = svc.get_status(schema.session_id)
        return _ok(result)

    except ValidationError as exc:
        return _err(exc.message, 400, exc.field_name)
    except SessionNotFoundError as exc:
        return _err(str(exc), 404)


# ══════════════════════════════════════════════════════════════════════════════
# Rich /api/* endpoints  (dashboard, admin, demo)
# ══════════════════════════════════════════════════════════════════════════════

@api_bp.post("/reset")
def reset():
    """
    WHY THIS EXISTS: Mirrors the OpenEnv /reset for the dashboard frontend,
    which uses the /api prefix for all calls. Keeps the web UI decoupled from
    the OpenEnv standard paths so both can evolve independently.
    """
    try:
        body   = request.get_json(silent=True) or {}
        schema = ResetRequest.from_dict(body)
        result = svc.create_session(seed=schema.seed)
        return _ok(result, 201)

    except ValidationError as exc:
        return _err(exc.message, 400, exc.field_name)
    except ServiceCapacityError as exc:
        return _err(str(exc), 503)
    except RuntimeError as exc:
        logger.error("api /reset failed: %s", exc)
        return _err("Could not start a new episode. Please try again.", 500)


@api_bp.post("/step")
def step():
    """
    WHY THIS EXISTS: Dashboard step endpoint. Validates and delegates to the
    service so the view never touches raw request data or gymnasium objects.
    """
    try:
        body   = request.get_json(silent=True) or {}
        schema = StepRequest.from_dict(body, request.headers.get("X-Session-ID"))
        result = svc.step(schema.session_id, schema.action)
        return _ok(result)

    except ValidationError as exc:
        return _err(exc.message, 400, exc.field_name)
    except SessionNotFoundError as exc:
        return _err(str(exc), 404)
    except SessionExpiredError as exc:
        return _err(str(exc), 410)
    except RuntimeError as exc:
        logger.error("api /step failed: %s", exc)
        return _err("An unexpected error occurred. Start a new session to continue.", 500)


@api_bp.post("/heuristic-step")
def heuristic_step():
    """
    WHY THIS EXISTS: Lets the dashboard trigger the built-in heuristic agent
    for one step — useful for "auto-play" mode in the live dashboard.
    """
    try:
        body   = request.get_json(silent=True) or {}
        schema = SessionIdRequest.from_dict(body, request.headers.get("X-Session-ID"))
        result = svc.heuristic_step(schema.session_id)
        return _ok(result)

    except ValidationError as exc:
        return _err(exc.message, 400, exc.field_name)
    except SessionNotFoundError as exc:
        return _err(str(exc), 404)
    except SessionExpiredError as exc:
        return _err(str(exc), 410)
    except RuntimeError as exc:
        logger.error("api /heuristic-step failed: %s", exc)
        return _err("Heuristic step failed. Please try again.", 500)


@api_bp.get("/status")
def status():
    """
    WHY THIS EXISTS: System-level health + optional per-session status.
    Used by the admin panel to show live server stats.
    """
    try:
        session_id = (
            request.args.get("session_id")
            or request.headers.get("X-Session-ID")
        )
        base: dict = {
            "status":          "healthy",
            "active_sessions": svc.active_count(),
            "version":         "1.0.0",
        }

        if session_id:
            schema           = SessionIdRequest.from_dict({"session_id": session_id})
            base["session"]  = svc.get_status(schema.session_id)

        return _ok(base)

    except ValidationError as exc:
        return _err(exc.message, 400, exc.field_name)
    except SessionNotFoundError as exc:
        return _err(str(exc), 404)


@api_bp.delete("/session")
def close_session():
    """
    WHY THIS EXISTS: Explicit session cleanup lets the client free server
    resources immediately rather than waiting for the TTL to expire.
    Equivalent to a soft-delete — the session is removed from the store
    and its environment is closed, but we log the closure for audit.
    """
    try:
        body   = request.get_json(silent=True) or {}
        schema = SessionIdRequest.from_dict(body, request.headers.get("X-Session-ID"))
        closed = svc.close_session(schema.session_id)

        if not closed:
            return _err(
                f"Session '{schema.session_id}' was not found. "
                "It may have already expired.",
                404,
            )

        return _ok({"deleted": schema.session_id, "message": "Session closed successfully."})

    except ValidationError as exc:
        return _err(exc.message, 400, exc.field_name)


@api_bp.get("/info")
def env_info():
    """
    WHY THIS EXISTS: Returns static environment metadata so clients can
    discover the action space, observation shape, and reward range without
    having to start a session first.
    """
    return _ok({
        "name":        "SmartHospitalResourceAllocator-v1",
        "description": (
            "An RL environment for managing ICU beds, general beds, doctors, "
            "nurses, and ambulances across an 8-hour emergency department shift."
        ),
        "version":         "1.0.0",
        "episode_length":  HospitalEnv.SHIFT_LENGTH,
        "action_space": {
            "type":    "Discrete",
            "n":       7,
            "actions": HospitalEnv.ACTION_NAMES,
        },
        "observation_space": {
            "patients":     f"float32 [{HospitalEnv.MAX_QUEUE_SIZE * 4}] — 20 patients × 4 features",
            "resources":    "float32 [5] — ICU beds, general beds, doctors, nurses, ambulances",
            "time_context": "float32 [2] — timestep/480, arrival rate",
        },
        "reward_range":   [-500, 200],
        "grader_signal":  "survival_rate (0.0–1.0)",
    })


@api_bp.post("/demo")
def run_demo():
    """
    WHY THIS EXISTS: Runs a complete heuristic-agent episode in a single HTTP
    call and returns the full summary. Used by the admin panel for one-click
    demo runs without requiring a persistent session.
    """
    try:
        body   = request.get_json(silent=True) or {}
        schema = ResetRequest.from_dict(body)    # reuses seed validation
        result = svc.run_heuristic_episode(seed=schema.seed)
        return _ok(result)

    except ValidationError as exc:
        return _err(exc.message, 400, exc.field_name)
    except RuntimeError as exc:
        logger.error("api /demo failed: %s", exc)
        return _err("Demo episode failed. Please try again.", 500)

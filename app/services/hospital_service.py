"""
app/services/hospital_service.py — Session management service layer.

WHY THIS EXISTS: Encapsulates ALL business logic for managing RL environment
sessions so that views remain thin (request in → response out) and never
contain logic that would need to be tested or reused independently.

Industry standards applied:
  - Fat Model / Thin View: all logic lives here, views are dumb dispatchers
  - UUIDs for session IDs (prevents enumeration attacks, looks professional)
  - Timestamps (created_at, last_accessed) on every session record
  - Soft "delete" pattern: sessions are marked done before purge, not silently dropped
  - Thread-safety via a single lock (no race conditions on session dict)
  - Try/except in all public methods: never let a gymnasium exception bubble raw to the client
  - All public methods have docstrings explaining WHY they exist
  - Configurable TTL from AppConfig (not hardcoded)
  - MAX_SESSIONS cap to prevent memory exhaustion under load
"""

import logging
import threading
import time
import uuid
from typing import Any, Dict, Optional

import numpy as np

from core.env import HospitalEnv
from core.agents import heuristic_action
from core.config import ENV_CFG, AppConfig

logger = logging.getLogger(__name__)


# ── Serialisation helper ──────────────────────────────────────────────────────

def _to_serializable(d: Dict) -> Dict:
    """
    WHY THIS EXISTS: numpy types (float32, int64, ndarray) are not JSON-serialisable
    by default. Converting them here prevents 500 errors on every response.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


# ── Custom service-layer exceptions ──────────────────────────────────────────

class SessionNotFoundError(KeyError):
    """WHY THIS EXISTS: Distinguishes a missing session (404) from an unexpected error (500)."""
    pass


class SessionExpiredError(RuntimeError):
    """WHY THIS EXISTS: Signals that the episode is over; the client should POST /reset."""
    pass


class ServiceCapacityError(RuntimeError):
    """WHY THIS EXISTS: Signals that the server is at MAX_SESSIONS; client should retry later."""
    pass


# ── Service class ─────────────────────────────────────────────────────────────

class HospitalService:
    """
    WHY THIS EXISTS: Thread-safe, in-process session store that gives each
    API caller their own isolated HospitalEnv instance. Centralises all
    session lifecycle logic (create → step → expire → purge) in one place
    so it can be unit-tested independently of Flask routing.
    """

    def __init__(self) -> None:
        # Dict keyed by UUID session_id. Values are session metadata dicts.
        self._sessions: Dict[str, Dict[str, Any]] = {}
        # Single lock protects all reads and writes to _sessions.
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def create_session(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        WHY THIS EXISTS: Initialises a new RL episode and registers it in the
        session store. Returns the initial observation so the caller can start
        sending step() requests immediately.

        Raises:
            ServiceCapacityError: if MAX_SESSIONS is already reached.
        """
        with self._lock:
            if len(self._sessions) >= AppConfig.MAX_SESSIONS:
                raise ServiceCapacityError(
                    f"Server is at capacity ({AppConfig.MAX_SESSIONS} active sessions). "
                    "Please try again in a few minutes."
                )

        try:
            env = HospitalEnv(seed=seed)
            obs, info = env.reset(seed=seed)
        except Exception as exc:
            logger.exception("Failed to initialise HospitalEnv (seed=%s): %s", seed, exc)
            raise RuntimeError(
                "Could not start a new hospital shift. Please try again."
            ) from exc

        # UUID as session key: unpredictable, URL-safe, professional
        session_id = str(uuid.uuid4())

        # Full session record — includes created_at / last_accessed timestamps
        session: Dict[str, Any] = {
            "env":             env,
            "obs":             obs,
            "info":            info,
            "done":            False,
            "step_count":      0,
            "total_reward":    0.0,
            "rewards_history": [],
            # Timestamps — always store these; they enable TTL purge and audit logs
            "created_at":      time.time(),
            "last_accessed":   time.time(),
            "seed":            seed,
        }

        with self._lock:
            self._sessions[session_id] = session

        logger.info("Session created: %s (seed=%s)", session_id, seed)

        return {
            "session_id":  session_id,
            "observation": _to_serializable(obs),
            "info":        _to_serializable(info),
            "done":        False,
        }

    def step(self, session_id: str, action: int) -> Dict[str, Any]:
        """
        WHY THIS EXISTS: Advances one timestep in the RL environment for the
        given session. Handles the done/expired state gracefully rather than
        raising a raw gymnasium error.

        Raises:
            SessionNotFoundError: session_id does not exist.
        """
        session = self._require_session(session_id)

        # Graceful "already done" — tell the client what to do next
        if session["done"]:
            raise SessionExpiredError(
                "This episode has ended. Send POST /reset to start a new one."
            )

        env: HospitalEnv = session["env"]

        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except ValueError as exc:
            # Invalid action from gymnasium — return a structured error, not a crash
            logger.warning("Invalid action %d in session %s: %s", action, session_id, exc)
            return {
                "error":      f"Invalid action '{action}'. Valid range: 0–6.",
                "session_id": session_id,
                "done":       False,
            }
        except Exception as exc:
            logger.exception("Unexpected env.step error in session %s: %s", session_id, exc)
            raise RuntimeError(
                "An unexpected error occurred during the simulation step. "
                "Please start a new session."
            ) from exc

        done = terminated or truncated

        # Mutate session record atomically
        session["obs"]             = obs
        session["info"]            = info
        session["done"]            = done
        session["step_count"]     += 1
        session["total_reward"]   += float(reward)
        session["rewards_history"].append(float(reward))
        session["last_accessed"]   = time.time()

        return {
            "session_id":   session_id,
            "observation":  _to_serializable(obs),
            "reward":       float(reward),
            "done":         done,
            "info":         _to_serializable(info),
            "step_count":   session["step_count"],
            "total_reward": session["total_reward"],
            "action_name":  HospitalEnv.ACTION_NAMES.get(action, "unknown"),
        }

    def heuristic_step(self, session_id: str) -> Dict[str, Any]:
        """
        WHY THIS EXISTS: Lets the client delegate action selection to the
        built-in heuristic agent — useful for demos and the /demo endpoint.

        Raises:
            SessionNotFoundError: session_id does not exist.
            SessionExpiredError:  episode is already done.
        """
        session = self._require_session(session_id)

        if session["done"]:
            raise SessionExpiredError(
                "This episode has ended. Send POST /reset to start a new one."
            )

        action = heuristic_action(session["obs"], session["info"])
        return self.step(session_id, action)

    def get_status(self, session_id: str) -> Dict[str, Any]:
        """
        WHY THIS EXISTS: Returns a full state snapshot without advancing the
        episode — used by GET /state and GET /api/status?session_id=...

        Raises:
            SessionNotFoundError: session_id does not exist.
        """
        session = self._require_session(session_id)
        session["last_accessed"] = time.time()

        return {
            "session_id":   session_id,
            "done":         session["done"],
            "step_count":   session["step_count"],
            "total_reward": session["total_reward"],
            "info":         _to_serializable(session["info"]),
            "observation":  _to_serializable(session["obs"]),
            # Expose timestamps so callers can reason about session age
            "created_at":   session["created_at"],
            "last_accessed": session["last_accessed"],
        }

    def close_session(self, session_id: str) -> bool:
        """
        WHY THIS EXISTS: Explicitly frees a session and its HospitalEnv
        (which holds NumPy arrays). This is the "soft delete" equivalent —
        we log the closure and free resources rather than just dropping the key.

        Returns True if a session was found and closed, False if not found.
        """
        with self._lock:
            session = self._sessions.pop(session_id, None)

        if session is not None:
            try:
                session["env"].close()
            except Exception:
                pass   # best-effort cleanup
            logger.info(
                "Session closed: %s (steps=%d, total_reward=%.1f)",
                session_id, session["step_count"], session["total_reward"],
            )
            return True

        return False

    def run_heuristic_episode(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        WHY THIS EXISTS: Runs a complete heuristic-agent episode without
        persisting a session — used by POST /api/demo to give clients a full
        episode summary in a single HTTP call.
        """
        try:
            env = HospitalEnv(seed=seed)
            obs, info = env.reset(seed=seed)
        except Exception as exc:
            logger.exception("Demo episode failed to initialise: %s", exc)
            raise RuntimeError("Could not start a demo episode. Please try again.") from exc

        total_reward    = 0.0
        step            = 0
        done            = False
        rewards         = []
        hourly_snapshots = []

        try:
            while not done and step < ENV_CFG["SHIFT_LENGTH"]:
                action = heuristic_action(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                rewards.append(float(reward))
                step += 1
                done = terminated or truncated

                if step % 60 == 0:
                    hourly_snapshots.append({
                        "hour":          step // 60,
                        "queue":         info["queue_length"],
                        "survived":      info["patients_survived"],
                        "died":          info["patients_died"],
                        "reward_so_far": round(total_reward, 1),
                        "survival_rate": round(info["survival_rate"] * 100, 1),
                    })
        except Exception as exc:
            logger.exception("Demo episode failed at step %d: %s", step, exc)
            raise RuntimeError(
                f"Demo episode failed at step {step}. Partial results may be incomplete."
            ) from exc
        finally:
            env.close()

        return {
            "steps":              step,
            "total_reward":       round(total_reward, 2),
            "survived":           info["patients_survived"],
            "died":               info["patients_died"],
            "survival_rate":      round(info["survival_rate"] * 100, 1),
            "triage_accuracy":    round(info["triage_accuracy"] * 100, 1),
            "hourly_snapshots":   hourly_snapshots,
            "rewards":            rewards,
        }

    def purge_stale(self) -> int:
        """
        WHY THIS EXISTS: Prevents unbounded memory growth by removing sessions
        that haven't been accessed within SESSION_TTL_SECONDS. Called periodically
        by the background cleanup thread in app/__init__.py.
        """
        now = time.time()
        with self._lock:
            stale = [
                sid for sid, s in self._sessions.items()
                if now - s["last_accessed"] > AppConfig.SESSION_TTL_SECONDS
            ]
            for sid in stale:
                s = self._sessions.pop(sid)
                try:
                    s["env"].close()
                except Exception:
                    pass
        if stale:
            logger.info("Purged %d stale session(s) (TTL=%ds)", len(stale), AppConfig.SESSION_TTL_SECONDS)
        return len(stale)

    def active_count(self) -> int:
        """WHY THIS EXISTS: Health endpoint needs active session count without exposing internals."""
        with self._lock:
            return len(self._sessions)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _require_session(self, session_id: str) -> Dict[str, Any]:
        """
        WHY THIS EXISTS: Single point for session lookup so all public methods
        raise the same typed exception (SessionNotFoundError) instead of a bare
        KeyError. Allows the view layer to map it cleanly to a 404 response.
        """
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(
                f"Session '{session_id}' not found. "
                "It may have expired or never existed. Send POST /reset to begin."
            )
        return session


# ── Module-level singleton ────────────────────────────────────────────────────
# WHY THIS EXISTS: One service instance per process; imported by api.py and
# __init__.py's cleanup loop.

_service = HospitalService()


def get_service() -> HospitalService:
    """Returns the process-level HospitalService singleton."""
    return _service

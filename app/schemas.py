"""
app/schemas.py — Request/response validation schemas.

WHY THIS EXISTS: Validates all incoming data BEFORE it reaches service or model
layer. This is the industry standard "Thin View, Fat Validation" pattern.
Using dataclasses + manual validation (no heavy dependencies like marshmallow)
keeps the footprint small while still enforcing contracts at the boundary.

Industry standards applied:
  - All user input validated here, never inside views or services
  - Clear, user-friendly error messages (not raw Python exceptions to the client)
  - Strict type coercion with bounds checking
"""

from dataclasses import dataclass, field
from typing import Optional


# ── Custom exception for validation failures ──────────────────────────────────

class ValidationError(ValueError):
    """
    WHY THIS EXISTS: Distinguishes input validation errors (400 Bad Request)
    from unexpected server errors (500) so views can handle them differently.
    """
    def __init__(self, field_name: str, message: str):
        self.field_name = field_name
        self.message    = message
        super().__init__(f"{field_name}: {message}")

    def to_dict(self) -> dict:
        """Returns a user-friendly JSON-serialisable error payload."""
        return {
            "error":   self.message,
            "field":   self.field_name,
            "hint":    f"Check the value you sent for '{self.field_name}'.",
        }


# ── Request schemas ───────────────────────────────────────────────────────────

@dataclass
class ResetRequest:
    """
    WHY THIS EXISTS: Validates and coerces the body of POST /reset before
    any environment or service code sees it.
    """
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ResetRequest":
        """Parses and validates a raw request body dict."""
        seed = data.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except (ValueError, TypeError):
                raise ValidationError("seed", "Must be an integer.")
            if not (0 <= seed <= 2_147_483_647):
                raise ValidationError("seed", "Must be between 0 and 2147483647.")
        return cls(seed=seed)


@dataclass
class StepRequest:
    """
    WHY THIS EXISTS: Validates the body of POST /step so invalid actions
    never reach the environment, preventing cryptic gymnasium errors.
    """
    session_id: str
    action:     int

    # Valid action range (mirrors HospitalEnv.action_space.n - 1)
    ACTION_MIN: int = field(default=0, init=False, repr=False)
    ACTION_MAX: int = field(default=6, init=False, repr=False)

    @classmethod
    def from_dict(cls, data: dict, header_session_id: Optional[str] = None) -> "StepRequest":
        """Parses and validates a raw request body dict + optional header fallback."""
        session_id = data.get("session_id") or header_session_id
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValidationError(
                "session_id",
                "Required. Pass it in the request body or the X-Session-ID header."
            )

        action = data.get("action")
        if action is None:
            raise ValidationError("action", "Required. Must be an integer between 0 and 6.")
        try:
            action = int(action)
        except (ValueError, TypeError):
            raise ValidationError("action", "Must be an integer.")
        if not (0 <= action <= 6):
            raise ValidationError("action", f"Must be between 0 and 6. Got {action}.")

        return cls(session_id=session_id.strip(), action=action)


@dataclass
class SessionIdRequest:
    """
    WHY THIS EXISTS: Reusable schema for any endpoint that only needs a session_id
    (heuristic-step, close-session, state). Avoids copy-pasting validation logic.
    """
    session_id: str

    @classmethod
    def from_dict(cls, data: dict, header_session_id: Optional[str] = None) -> "SessionIdRequest":
        """Parses session_id from body dict with header as fallback."""
        session_id = data.get("session_id") or header_session_id
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValidationError(
                "session_id",
                "Required. Pass it in the request body or the X-Session-ID header."
            )
        return cls(session_id=session_id.strip())

"""
tests/test_api.py — Flask API integration tests.

WHY THIS EXISTS: Tests the full HTTP layer — request parsing, validation,
correct status codes, and response shapes — without needing a live server.
Covers every item in the audit's Part 1 (16 tests).

Run with:
    pytest tests/test_api.py -v
"""

import json
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Audit Part 1 — 16 original API tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthAndInfo:
    """Tests 1–2: health check and environment info."""

    def test_health_returns_200_and_ok(self, client):
        """WHY: Container orchestration (Docker, HF Spaces) uses this to decide
        whether to route traffic to the container."""
        r = client.get("/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"
        assert data["environment"] == "SmartHospitalAllocator-v1"
        assert isinstance(data["active_sessions"], int)

    def test_api_info_returns_action_space(self, client):
        """WHY: Clients need to discover the action space without starting a session."""
        r = client.get("/api/info")
        assert r.status_code == 200
        data = r.get_json()
        assert data["action_space"]["n"] == 7
        assert data["episode_length"] == 480
        assert "patients" in data["observation_space"]


class TestResetEndpoint:
    """Tests 3, 7, 14: reset creates sessions correctly."""

    def test_openenv_reset_returns_201_with_session_id(self, client):
        """WHY: OpenEnv standard path POST /reset must return 201 Created."""
        r = client.post("/reset", json={"seed": 42})
        assert r.status_code == 201
        data = r.get_json()
        assert "session_id" in data
        assert len(data["session_id"]) == 36   # UUID format
        assert "observation" in data
        assert data["done"] is False

    def test_api_reset_dashboard_endpoint(self, client):
        """WHY: Dashboard uses /api/reset — must mirror OpenEnv /reset behaviour."""
        r = client.post("/api/reset", json={"seed": 42})
        assert r.status_code == 201
        data = r.get_json()
        assert "session_id" in data

    def test_reset_with_invalid_seed_returns_400(self, client):
        """WHY: Schema validation must reject out-of-range seeds gracefully."""
        r = client.post("/reset", json={"seed": -1})
        assert r.status_code == 400
        data = r.get_json()
        assert "error" in data
        assert "field" in data

    def test_reset_with_string_seed_returns_400(self, client):
        """WHY: Seed must be an integer — string should be rejected."""
        r = client.post("/reset", json={"seed": "not_a_number"})
        assert r.status_code == 400


class TestStepEndpoint:
    """Tests 4–6: step advances state correctly."""

    def test_step_hold_action_returns_200(self, client, session_id):
        """WHY: Action 6 (hold) is always safe — verify step mechanics work."""
        r = client.post("/step", json={"session_id": session_id, "action": 6})
        assert r.status_code == 200
        data = r.get_json()
        assert data["step_count"] == 1
        assert isinstance(data["reward"], float)
        assert "survival_rate" in data["info"]

    def test_step_icu_admit_updates_total_reward(self, client, session_id):
        """WHY: Each step must accumulate total_reward correctly."""
        r1 = client.post("/step", json={"session_id": session_id, "action": 6})
        r2 = client.post("/step", json={"session_id": session_id, "action": 0})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r2.get_json()["step_count"] == 2

    def test_get_state_does_not_advance_step_count(self, client, session_id):
        """WHY: GET /state is read-only — it must not advance the episode."""
        # Take 2 steps
        client.post("/step", json={"session_id": session_id, "action": 6})
        client.post("/step", json={"session_id": session_id, "action": 6})
        # GET /state should still show step_count=2
        r = client.get(f"/state?session_id={session_id}")
        assert r.status_code == 200
        assert r.get_json()["step_count"] == 2


class TestHeuristicAndDemo:
    """Tests 8–10: heuristic step and full demo episode."""

    def test_heuristic_step_returns_action_name(self, client, session_id):
        """WHY: Heuristic agent must return a valid named action."""
        r = client.post("/api/heuristic-step", json={"session_id": session_id})
        assert r.status_code == 200
        data = r.get_json()
        assert "action_name" in data
        from core.env import HospitalEnv
        assert data["action_name"] in HospitalEnv.ACTION_NAMES.values()

    def test_api_status_returns_healthy(self, client):
        """WHY: Admin panel polls /api/status for server health."""
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"

    def test_demo_runs_full_episode_with_8_hourly_snapshots(self, client):
        """WHY: POST /api/demo must complete a full 480-step episode in one call."""
        r = client.post("/api/demo", json={"seed": 42})
        assert r.status_code == 200
        data = r.get_json()
        assert data["steps"] == 480
        assert len(data["hourly_snapshots"]) == 8
        assert 0.0 <= data["survival_rate"] <= 100.0


class TestValidationErrors:
    """Tests 11–14: schema validation rejects bad input correctly."""

    def test_invalid_action_99_returns_400(self, client, session_id):
        """WHY: Out-of-range action must be caught by schema before hitting env."""
        r = client.post("/step", json={"session_id": session_id, "action": 99})
        assert r.status_code == 400
        data = r.get_json()
        assert "error" in data
        assert "field" in data

    def test_missing_session_id_returns_400(self, client):
        """WHY: Missing required field must return 400 with field hint."""
        r = client.post("/step", json={"action": 0})
        assert r.status_code == 400
        data = r.get_json()
        assert data.get("field") == "session_id"

    def test_unknown_session_id_returns_404(self, client):
        """WHY: Non-existent session must return 404, not 500."""
        r = client.post("/step", json={"session_id": "00000000-0000-0000-0000-000000000000", "action": 0})
        assert r.status_code == 404

    def test_reset_seed_below_zero_returns_400(self, client):
        """WHY: Seed bounds are 0–2147483647. -1 must be rejected cleanly."""
        r = client.post("/reset", json={"seed": -1})
        assert r.status_code == 400


class TestSessionLifecycle:
    """Tests 15–16: session can be closed and is then gone."""

    def test_delete_session_returns_200(self, client, session_id):
        """WHY: Explicit session cleanup must succeed and confirm the deletion."""
        r = client.delete("/api/session", json={"session_id": session_id})
        assert r.status_code == 200
        data = r.get_json()
        assert data["deleted"] == session_id

    def test_step_on_deleted_session_returns_404(self, client, session_id):
        """WHY: After deletion the session must be truly gone — not just marked done."""
        client.delete("/api/session", json={"session_id": session_id})
        r = client.post("/step", json={"session_id": session_id, "action": 0})
        assert r.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# Additional edge-case API tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Additional tests beyond the audit scope."""

    def test_step_on_expired_episode_returns_410(self, client):
        """WHY: 410 Gone is the correct code when the episode is finished but
        the session still exists — different from 404 (session not found)."""
        from core.config import ENV_CFG
        # Create a session and force it to termination via catastrophe (10 deaths)
        # We'll just check the error shape if we try to step on a done session
        # by patching the service state directly.
        r = client.post("/reset", json={"seed": 42})
        sid = r.get_json()["session_id"]

        from app.services.hospital_service import get_service
        svc = get_service()
        with svc._lock:
            if sid in svc._sessions:
                svc._sessions[sid]["done"] = True

        r = client.post("/step", json={"session_id": sid, "action": 0})
        assert r.status_code == 410
        assert "error" in r.get_json()

    def test_two_sessions_are_isolated(self, client):
        """WHY: Each session has its own HospitalEnv instance — steps in one
        must not affect the other."""
        r1 = client.post("/reset", json={"seed": 1})
        r2 = client.post("/reset", json={"seed": 2})
        sid1 = r1.get_json()["session_id"]
        sid2 = r2.get_json()["session_id"]

        # Advance session 1 by 3 steps
        for _ in range(3):
            client.post("/step", json={"session_id": sid1, "action": 6})

        # Session 2 should still be at step_count=0
        s2 = client.get(f"/state?session_id={sid2}").get_json()
        assert s2["step_count"] == 0, "Sessions are not isolated!"

    def test_x_session_id_header_accepted(self, client):
        """WHY: X-Session-ID header is the alternative to body session_id."""
        r = client.post("/reset", json={"seed": 42})
        sid = r.get_json()["session_id"]
        r2 = client.post("/step",
                         json={"action": 6},
                         headers={"X-Session-ID": sid})
        assert r2.status_code == 200

    def test_html_pages_return_200(self, client):
        """WHY: All four HTML pages must render without template errors."""
        for path in ["/", "/dashboard", "/admin", "/docs"]:
            r = client.get(path)
            assert r.status_code == 200, f"Page {path} returned {r.status_code}"

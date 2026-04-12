"""
tests/conftest.py — Shared pytest fixtures.

WHY THIS EXISTS: Centralises test setup so individual test files don't each
create their own Flask app or HospitalEnv instances. Fixtures are reused
across all test modules, keeping tests fast and consistent.
"""

import os
import sys

import pytest

# Ensure project root is on path so `core.*` and `app.*` resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set minimal env vars so AppConfig.validate() doesn't emit warnings in tests
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("HF_TOKEN",   "hf_test_token_placeholder")


@pytest.fixture(scope="session")
def flask_app():
    """
    WHY THIS EXISTS: Creates one Flask application for the entire test session.
    scope="session" means it's created once and reused — much faster than
    creating a new app per test.
    """
    from app import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture()
def client(flask_app):
    """
    WHY THIS EXISTS: Provides a Flask test client that can make HTTP requests
    without starting a real server. Each test gets a fresh client context.
    """
    with flask_app.test_client() as c:
        yield c


@pytest.fixture()
def env():
    """
    WHY THIS EXISTS: Provides a fresh, seeded HospitalEnv for unit tests.
    Uses seed=42 for reproducibility — same seed always gives same episode.
    Calls env.close() after each test to free resources.
    """
    from core.env import HospitalEnv
    e = HospitalEnv(seed=42)
    e.reset(seed=42)
    yield e
    e.close()


@pytest.fixture()
def session_id(client):
    """
    WHY THIS EXISTS: Creates a real API session so integration tests don't
    need to repeat the POST /reset boilerplate. Returns the session_id string.
    """
    response = client.post("/reset", json={"seed": 42})
    assert response.status_code == 201
    return response.get_json()["session_id"]

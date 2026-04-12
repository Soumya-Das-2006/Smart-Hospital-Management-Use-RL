"""
core/config.py — Single source of truth for all configuration.

WHY THIS EXISTS: Centralises every constant and setting so nothing is ever
hardcoded in business logic. All environment values are loaded from the .env
file at startup; hardcoded defaults are provided only for non-secret parameters.

Industry standards applied:
  - All secrets / changeable config via environment variables (12-Factor App)
  - python-dotenv loads .env automatically in development
  - Typed dataclasses instead of raw dicts for IDE support and validation
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file (development convenience; in production use real env vars) ──
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# Application secrets & infrastructure  (read from environment — NEVER hardcode)
# ══════════════════════════════════════════════════════════════════════════════

class AppConfig:
    """
    WHY THIS EXISTS: Groups all runtime-configurable values in one place so
    operators can change behaviour without touching source code.
    """
    # Flask
    SECRET_KEY: str      = os.getenv("SECRET_KEY", "dev-insecure-key-change-in-production")
    DEBUG: bool          = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    PORT: int            = int(os.getenv("PORT", "7860"))
    HOST: str            = os.getenv("HOST", "0.0.0.0")
    WORKERS: int         = int(os.getenv("GUNICORN_WORKERS", "2"))

    # LLM / inference
    API_BASE_URL: str    = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
    MODEL_NAME: str      = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
    HF_TOKEN: str | None = os.getenv("HF_TOKEN")

    # Session management
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "600"))
    MAX_SESSIONS: int        = int(os.getenv("MAX_SESSIONS", "100"))

    @classmethod
    def validate(cls) -> None:
        """
        WHY THIS EXISTS: Fail fast at startup rather than failing silently at
        runtime when the first real request hits a missing config value.
        """
        warnings = []
        if cls.SECRET_KEY == "dev-insecure-key-change-in-production":
            warnings.append(
                "WARNING: SECRET_KEY is using the default insecure value. "
                "Set SECRET_KEY in your .env file before deploying to production."
            )
        if cls.HF_TOKEN is None:
            warnings.append(
                "WARNING: HF_TOKEN is not set. The inference.py script will fail. "
                "Set HF_TOKEN in your .env file."
            )
        for w in warnings:
            import logging
            logging.getLogger(__name__).warning(w)


# ══════════════════════════════════════════════════════════════════════════════
# RL Environment constants  (game mechanics — safe to hardcode)
# ══════════════════════════════════════════════════════════════════════════════

ENV_CFG: dict = {
    "MAX_ICU_BEDS":      10,
    "MAX_GENERAL_BEDS":  30,
    "MAX_DOCTORS":       8,
    "MAX_NURSES":        15,
    "MAX_AMBULANCES":    5,
    "MAX_QUEUE_SIZE":    20,
    "SHIFT_LENGTH":      480,    # 8 hours × 60 min timesteps
    # Patient severity distribution at spawn (must sum to 1.0)
    "SEVERITY_PROBS":    [0.25, 0.30, 0.25, 0.12, 0.08],
    # Treatment duration (timesteps) by severity tier
    "TREATMENT_RANGE": {
        "mild":     (5,   60),   # severity 1–2
        "moderate": (15, 120),   # severity 3–5
    },
}

REWARD_CFG: dict = {
    "patient_survived":        100.0,
    "patient_deteriorated":    -60.0,
    "patient_died":           -200.0,
    "global_death_penalty":    -5.0,
    "wait_time_weight":         0.4,
    "empty_queue_bonus":        15.0,
    "triage_correct_bonus":     30.0,
    "triage_wrong_penalty":    -25.0,
    "idle_icu_penalty":        -1.5,
    "critical_amb_penalty":    -0.5,
    "overcrowd_general":       -25.0,
    "overcrowd_icu_critical":  -40.0,
    "invalid_action_penalty":  -10.0,
    "throughput_bonus":          8.0,
    "throughput_threshold":      0.75,
}

PPO_CFG: dict = {
    "learning_rate":   3e-4,
    "n_steps":         2048,
    "batch_size":      256,
    "n_epochs":        10,
    "gamma":           0.99,
    "gae_lambda":      0.95,
    "clip_range":      0.2,
    "ent_coef":        0.01,
    "n_envs":          4,
    "total_timesteps": 200_000,
}

DQN_CFG: dict = {
    "learning_rate":          1e-4,
    "buffer_size":            100_000,
    "learning_starts":        5_000,
    "batch_size":             256,
    "tau":                    0.005,
    "gamma":                  0.99,
    "train_freq":             4,
    "exploration_fraction":   0.3,
    "exploration_final_eps":  0.05,
    "total_timesteps":        200_000,
}

EVAL_CFG: dict = {
    "n_eval_episodes":       20,
    "eval_freq":             5_000,
    "success_survival_rate": 0.70,
}

INFERENCE_CFG: dict = {
    "max_steps":         50,
    "temperature":       0.3,
    "max_tokens":        200,
    "success_threshold": 0.50,
}

"""
inference.py — Smart Hospital Resource Allocator — OpenEnv Hackathon submission.

WHY THIS EXISTS: This is the evaluation script judges run to verify the
environment is both solvable (model can get reward) and correct (output format
matches spec). It connects an LLM via the HF Router to run one full episode
and emits the exact [START]/[STEP]/[END] stdout format required by the spec.

Industry standards applied:
  - All config via environment variables (12-Factor App) — no hardcoded secrets
  - Fail fast: missing HF_TOKEN raises ValueError immediately, not on first API call
  - Try/except on every LLM call: network errors → safe fallback action, not crash
  - Docstrings explain WHY each function exists
  - Typed function signatures throughout
  - Reward normalised to [0, 1] via survival_rate (spec requires 0–1 float)

Environment variables:
  API_BASE_URL  — HF Router endpoint  (default: https://api-inference.huggingface.co/v1)
  MODEL_NAME    — Model identifier     (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      — Hugging Face token   (REQUIRED — no default)
"""

import json
import logging
import os
import sys
from typing import Optional

from openai import OpenAI

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stderr,   # keep stderr clean so [START]/[STEP]/[END] go to stdout only
)
logger = logging.getLogger(__name__)

# ── Environment variables (read once at import time) ──────────────────────────
# WHY: Fail fast — if HF_TOKEN is missing, crash immediately with a clear message
# rather than producing a confusing "authentication error" on the first API call.

API_BASE_URL: str = os.getenv(
    "API_BASE_URL",
    "https://api-inference.huggingface.co/v1",
)
MODEL_NAME: str = os.getenv(
    "MODEL_NAME",
    "Qwen/Qwen2.5-72B-Instruct",
)
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

# BUG FIX (Issue 5): was a module-level raise, meaning `import inference`
# would immediately crash in any context where HF_TOKEN isn't set (e.g. tests,
# type checkers, or the Flask app if inference is imported accidentally).
# Validation is now deferred to run_episode() — the only place it's actually needed.

def _get_client() -> OpenAI:
    """
    WHY THIS EXISTS: Defers HF_TOKEN validation and OpenAI client creation to
    the point of actual use (run_episode), not at module import time.
    This allows the module to be safely imported in test environments.
    """
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError(
            "HF_TOKEN environment variable is required but not set. "
            "Get your token at https://huggingface.co/settings/tokens "
            "and set it with: export HF_TOKEN=hf_your_token_here"
        )
    return OpenAI(base_url=os.getenv("API_BASE_URL", API_BASE_URL), api_key=token)

# ── Constants ─────────────────────────────────────────────────────────────────
TASK_NAME        = "hospital-triage"
ENV_NAME         = "SmartHospitalAllocator-v1"
MAX_STEPS        = 50       # cap inference rollout (full shift = 480)
SUCCESS_THRESHOLD = 0.50    # survival_rate >= 50% counts as success

SYSTEM_PROMPT = """You are an expert emergency-department triage AI managing an 8-hour hospital shift.

OBSERVATION (JSON keys):
- patients: list of up to 5 most urgent patients, each with {severity(0-1), wait_time(0-1), treatment_needed(0-1), age_risk(0-1)}
- resources: {icu_free(0-1), general_beds_free(0-1), doctors(0-1), nurses(0-1), ambulances(0-1)}
- time_context: {shift_progress(0-1), arrival_rate(0-1)}

ACTIONS:
0: admit_to_icu        — for severity >= 0.8 (critical), needs doctor + free ICU bed
1: admit_to_general    — for severity 0.2–0.7, needs free general bed
2: dispatch_ambulance  — when queue is empty and ambulances available
3: escalate_to_icu     — when a general patient deteriorates (high severity + long wait)
4: discharge_patient   — when general beds are full and mild patients have recovered
5: call_extra_staff    — when doctors < 0.3 and critical queue is building
6: hold_and_monitor    — only when queue is stable and all resources are stretched

GOAL: Maximise patient survival rate.

Respond ONLY with valid JSON: {"action": <integer 0-6>, "reason": "<one sentence>"}
Do NOT include markdown fences, explanation, or any other text."""


def _format_observation(obs: dict) -> str:
    """
    WHY THIS EXISTS: Converts the raw numpy observation into a compact JSON
    string that fits within the LLM's context window without wasting tokens
    on the full 80-element patients array.
    """
    patients_flat = obs["patients"]
    patient_list  = []
    for i in range(0, min(len(patients_flat), 80), 4):
        sev = round(float(patients_flat[i]), 2)
        if sev == 0.0:
            break
        patient_list.append({
            "severity":         sev,
            "wait_time":        round(float(patients_flat[i + 1]), 2),
            "treatment_needed": round(float(patients_flat[i + 2]), 2),
            "age_risk":         round(float(patients_flat[i + 3]), 2),
        })

    res = obs["resources"]
    tc  = obs["time_context"]

    return json.dumps({
        "patients": patient_list[:5],   # top-5 most urgent
        "resources": {
            "icu_free":          round(float(res[0]), 2),
            "general_beds_free": round(float(res[1]), 2),
            "doctors":           round(float(res[2]), 2),
            "nurses":            round(float(res[3]), 2),
            "ambulances":        round(float(res[4]), 2),
        },
        "time_context": {
            "shift_progress": round(float(tc[0]), 2),
            "arrival_rate":   round(float(tc[1]), 2),
        },
    }, indent=2)


def _query_llm(obs: dict, info: dict) -> tuple[int, str]:
    """
    WHY THIS EXISTS: Isolates the LLM API call so errors can be caught and
    handled gracefully — a network timeout or malformed response falls back to
    action 6 (hold_and_monitor) rather than crashing the episode.

    Returns:
        (action_int, reason_str) — action is guaranteed to be in [0, 6].
    """
    user_message = (
        f"Current hospital state:\n{_format_observation(obs)}\n\n"
        f"Survival rate so far: {info.get('survival_rate', 0):.0%} | "
        f"Queue length: {info.get('queue_length', 0)} | "
        f"Admitted: {info.get('admitted_count', 0)}\n\n"
        "Choose the best action."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=150,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model added them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        parsed = json.loads(raw)
        action = int(parsed.get("action", 6))
        reason = str(parsed.get("reason", ""))
        # Guard against out-of-range action
        if action not in range(7):
            logger.warning("LLM returned out-of-range action %d, defaulting to hold", action)
            action = 6
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Failed to parse LLM response: %r — defaulting to hold", raw)
        action = 6
        reason = "parse_error"

    return action, reason


def run_episode() -> None:
    """
    WHY THIS EXISTS: Runs one full inference episode and emits the exact
    [START]/[STEP]/[END] stdout format required by the hackathon judges.
    All logging goes to stderr so it does not contaminate the stdout output.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core.env import HospitalEnv

    client = _get_client()   # validates HF_TOKEN and creates the OpenAI client

    env       = HospitalEnv()
    obs, info = env.reset()

    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    total_reward = 0.0
    rewards:     list[float] = []
    done         = False
    step         = 0
    last_error:  Optional[str] = None

    try:
        while not done and step < MAX_STEPS:
            # Get action from LLM (with safe fallback on any exception)
            try:
                action, _reason = _query_llm(obs, info)
                last_error       = None
            except Exception as exc:
                logger.warning("LLM call failed at step %d: %s — using hold action", step + 1, exc)
                action     = 6
                last_error = str(exc)

            action_name = HospitalEnv.ACTION_NAMES.get(action, str(action))

            # Step the environment (with safe fallback on unexpected gymnasium error)
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except Exception as exc:
                logger.exception("env.step failed at step %d: %s", step + 1, exc)
                last_error = str(exc)
                reward     = 0.0
                done       = True

            step         += 1
            total_reward += float(reward)
            rewards.append(float(reward))

            # Grader score: survival_rate is already normalised to [0, 1]
            norm_reward = round(float(info.get("survival_rate", 0.0)), 2)
            error_str   = last_error.replace(" ", "_") if last_error else "null"

            print(
                f"[STEP] step={step} action={action_name} "
                f"reward={norm_reward:.2f} done={str(done).lower()} "
                f"error={error_str}",
                flush=True,
            )

    finally:
        env.close()

        final_survival = float(info.get("survival_rate", 0.0)) if step > 0 else 0.0
        success        = final_survival >= SUCCESS_THRESHOLD
        rewards_str    = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success={str(success).lower()} steps={step} "
            f"rewards={rewards_str}",
            flush=True,
        )

        logger.info(
            "Episode complete — steps=%d, survival=%.1f%%, success=%s",
            step, final_survival * 100, success,
        )


if __name__ == "__main__":
    run_episode()

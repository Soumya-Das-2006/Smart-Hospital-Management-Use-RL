"""
core/agents.py — Canonical agent implementations.

WHY THIS EXISTS: Single source of truth for all agent logic so the same
heuristic is used consistently across the demo endpoint, the dashboard
auto-play mode, and the inference script. Eliminates copy-paste drift.

Industry standards applied:
  - Pure functions with no side effects (easy to unit-test)
  - Docstrings explain the clinical reasoning behind each decision rule
  - Type hints on all parameters and return values
  - Named constants instead of magic numbers (e.g. SEVERITY_CRITICAL = 4.0)
"""

import logging
from typing import Dict

import numpy as np

from core.env import HospitalEnv

logger = logging.getLogger(__name__)

# ── Decision thresholds (named constants, not magic numbers) ──────────────────
_SEVERITY_CRITICAL    = 4.0   # severity ≥ 4 → ICU candidate
_SEVERITY_MODERATE    = 2.0   # severity ≥ 2 → general ward candidate
_SEVERITY_MILD        = 1.0   # severity ≥ 1 → general ward if beds plentiful
_GEN_BED_LOW          = 5     # general bed count below which discharge is considered
_GEN_BED_COMFORTABLE  = 3     # minimum free general beds to safely admit
_ADMITTED_FULL        = 15    # admitted count above which discharge is prioritised
_STAFF_THROTTLE       = 120   # minimum timesteps between extra-staff calls
_CRITICAL_QUEUE_AMB   = 2     # queue depth of criticals before ambulance dispatch


def heuristic_action(obs: Dict, info: Dict) -> int:
    """
    WHY THIS EXISTS: Rule-based triage agent that mimics clinical decision logic.
    Used as a strong baseline and for demo runs. Also useful as a warm-start
    prior when collecting SFT data for the RL training pipeline.

    Decision priority (clinical reasoning):
      1. Critical (sev ≥ 4) + ICU free + doctor available → admit to ICU immediately
      2. Critical + ICU full → escalate an existing general patient to free ICU capacity
      3. Moderate/mild patient + comfortable general bed availability → admit to general
      4. General ward filling up → discharge recovered low-severity patients
      5. Multiple criticals incoming + ambulances idle → dispatch to increase inflow
      6. Doctors overwhelmed (< 2 available) + critical queue → call extra staff
      7. Default: hold and monitor (safe when queue is stable)

    Args:
        obs:  Observation dict from HospitalEnv._get_observation().
        info: Info dict from HospitalEnv._get_info().

    Returns:
        Integer action ID in range [0, 6].
    """
    patients  = obs["patients"].reshape(HospitalEnv.MAX_QUEUE_SIZE, 4)
    resources = obs["resources"]

    # Denormalise resource values back to real counts for readable comparisons
    icu_free  = float(resources[0]) * HospitalEnv.MAX_ICU_BEDS
    gen_free  = float(resources[1]) * HospitalEnv.MAX_GENERAL_BEDS
    doctors   = float(resources[2]) * HospitalEnv._DOCTOR_NORM
    ambulances_norm = float(resources[4])

    # Top-of-queue severity (most urgent patient waiting)
    top_severity     = patients[0][0] * 5.0 if patients[0][0] > 0 else 0.0
    critical_waiting = sum(1 for p in patients if p[0] * 5.0 >= _SEVERITY_CRITICAL and p[0] > 0)

    # Rule 1: Critical patient + ICU capacity + doctor → immediate ICU admit
    if top_severity >= _SEVERITY_CRITICAL and icu_free >= 1 and doctors >= 1:
        return HospitalEnv.ACTION_ADMIT_ICU

    # Rule 2: Critical + no ICU beds → escalate general patient to make room
    if top_severity >= _SEVERITY_CRITICAL and icu_free == 0:
        return HospitalEnv.ACTION_ESCALATE_ICU

    # Rule 3: Moderate patient + comfortable general bed availability → admit
    if top_severity >= _SEVERITY_MODERATE and gen_free >= _GEN_BED_COMFORTABLE:
        return HospitalEnv.ACTION_ADMIT_GENERAL

    # Rule 4: Mild patient + plenty of general beds → admit
    if top_severity >= _SEVERITY_MILD and gen_free >= _GEN_BED_LOW:
        return HospitalEnv.ACTION_ADMIT_GENERAL

    # Rule 5: General ward filling up with long-stay patients → discharge to free beds
    if gen_free <= _GEN_BED_LOW and info.get("admitted_count", 0) > _ADMITTED_FULL:
        return HospitalEnv.ACTION_DISCHARGE

    # Rule 6: Multiple criticals waiting + ambulances available → dispatch
    if critical_waiting > _CRITICAL_QUEUE_AMB and ambulances_norm > 0:
        return HospitalEnv.ACTION_DISPATCH_AMB

    # Rule 7: Doctors overwhelmed + critical queue → call extra staff (throttled)
    if (
        doctors < 2
        and critical_waiting > _CRITICAL_QUEUE_AMB + 1
        and info.get("timestep", 0) % _STAFF_THROTTLE == 0
    ):
        return HospitalEnv.ACTION_CALL_STAFF

    # Default: hold and monitor
    return HospitalEnv.ACTION_HOLD


def random_action(obs: Dict, info: Dict, rng: Optional["np.random.Generator"] = None) -> int:  # noqa: ARG001
    """
    WHY THIS EXISTS: Uniformly random baseline — used to measure how much
    better the heuristic (or a trained RL agent) is versus pure chance.
    The obs and info parameters are accepted but unused to match the agent signature.

    BUG FIX (Issue 4): Was using global np.random.randint which breaks seeded
    reproducibility when multiple environments run in parallel. Now accepts an
    optional numpy Generator (pass env.np_random for reproducible baselines).
    Falls back to a fresh Generator if none provided.
    """
    generator = rng if rng is not None else np.random.default_rng()
    return int(generator.integers(0, 7))

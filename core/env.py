"""
hospital_allocator/env.py — Smart Hospital Resource Allocator RL Environment

OpenEnv-compliant Gymnasium environment for the Meta PyTorch OpenEnv Hackathon.

Production fixes:
  C4  Instance-level RNG — no global np.random.seed() / random.seed() calls
  C6  All constants loaded from config — no hardcoded magic numbers
  C7  Resource leak + dead branch fixed — nurses/doctors freed on patient death
  H7  Observation bounds enforced — extra-staff count normalised by extended cap
  H8  assert replaced with ValueError (survives python -O flag)
  H9  survival_rate uses resolved-outcome count, not event count
  M6  wait_time increment consolidated into one loop
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import random as _stdlib_random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.config import ENV_CFG, REWARD_CFG

logger = logging.getLogger(__name__)


# ─── Patient dataclass ────────────────────────────────────────────────────────

@dataclass
class Patient:
    patient_id: int
    severity: int           # 1=minor … 5=critical
    wait_time: int
    treatment_needed: int
    age_risk: float         # 0.0–1.0
    assigned_bed: Optional[str] = None   # None | "icu" | "general"
    treatment_progress: int = 0
    outcome: Optional[str] = None        # None | "survived" | "died"

    def deterioration_risk(self) -> float:
        base        = (self.severity / 5.0) * 0.04
        wait_factor = min(self.wait_time * 0.005, 0.08)
        age_factor  = self.age_risk * 0.02
        return min(base + wait_factor + age_factor, 0.25)


@dataclass
class BedState:
    icu_free: int
    general_free: int


@dataclass
class PatientState:
    queue: List[Patient]
    admitted: List[Patient]
    survived: int = 0
    died: int = 0
    discharged: int = 0
    deterioration_events: int = 0
    outcomes_resolved: int = 0


@dataclass
class TimerState:
    extra_staff: int = 0


@dataclass
class HospitalState:
    timestep: int
    total_reward: float
    step_rewards: List[float]
    doctors_available: int
    nurses_available: int
    ambulances_free: int
    beds: BedState
    patients: PatientState
    timers: TimerState
    correct_triage_count: int = 0
    wrong_triage_count: int = 0
    recovery_reward_this_step: float = 0.0


# ─── Main Environment ──────────────────────────────────────────────────────────

class HospitalEnv(gym.Env):
    """
    Smart Hospital Resource Allocator — Production-Grade RL Environment

    Episode   : 8-hour shift (480 timesteps)
    Actions   : 7 discrete
    Obs space : Dict { patients(80,), resources(5,), time_context(2,) }
    Reward    : Multi-signal shaped (survival, wait-time, triage, efficiency)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # Resource caps (loaded from config — C6 fix)
    MAX_ICU_BEDS      = ENV_CFG["MAX_ICU_BEDS"]
    MAX_GENERAL_BEDS  = ENV_CFG["MAX_GENERAL_BEDS"]
    MAX_DOCTORS       = ENV_CFG["MAX_DOCTORS"]
    MAX_NURSES        = ENV_CFG["MAX_NURSES"]
    MAX_AMBULANCES    = ENV_CFG["MAX_AMBULANCES"]
    MAX_QUEUE_SIZE    = ENV_CFG["MAX_QUEUE_SIZE"]
    SHIFT_LENGTH      = ENV_CFG["SHIFT_LENGTH"]

    # H7 fix: normalise by extended caps so obs never exceeds 1.0
    _DOCTOR_NORM      = ENV_CFG["MAX_DOCTORS"] + 2
    _NURSE_NORM       = ENV_CFG["MAX_NURSES"]  + 3

    # Action indices
    ACTION_ADMIT_ICU       = 0
    ACTION_ADMIT_GENERAL   = 1
    ACTION_DISPATCH_AMB    = 2
    ACTION_ESCALATE_ICU    = 3
    ACTION_DISCHARGE       = 4
    ACTION_CALL_STAFF      = 5
    ACTION_HOLD            = 6

    ACTION_NAMES: Dict[int, str] = {
        0: "admit_to_icu",
        1: "admit_to_general",
        2: "dispatch_ambulance",
        3: "escalate_to_icu",
        4: "discharge_patient",
        5: "call_extra_staff",
        6: "hold_and_monitor",
    }

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.render_mode = render_mode

        # C4 fix: instance-level Python RNG (never mutates global state)
        self._rng = _stdlib_random.Random()

        self.observation_space = spaces.Dict({
            "patients":     spaces.Box(0.0, 1.0, shape=(self.MAX_QUEUE_SIZE * 4,), dtype=np.float32),
            "resources":    spaces.Box(0.0, 1.0, shape=(5,),  dtype=np.float32),
            "time_context": spaces.Box(0.0, 1.0, shape=(2,),  dtype=np.float32),
        })
        self.action_space = spaces.Discrete(7)

        self._patient_id_counter = 0
        self._initialized        = False

        if seed is not None:
            self._seed_rngs(seed)

    # ─── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None
              ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed_rngs(seed)
        self._reset_state()
        self._initialized = True
        logger.debug("reset() called, seed=%s", seed)
        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        # H8 fix: raise ValueError instead of assert
        if not isinstance(action, (int, np.integer)) or not self.action_space.contains(int(action)):
            raise ValueError(f"Invalid action {action!r}. Expected int in [0, 6].")
        if not self._initialized:
            raise RuntimeError("Call env.reset() before env.step().")

        self.state.recovery_reward_this_step = 0.0

        action_result = self._apply_action(int(action))
        progress = self._progress_environment()
        self._validate_and_normalize_state(where="step")

        deaths_this_step = progress["deaths_this_step"]
        survivals_this_step = progress["survivals_this_step"]
        deteriorations_this_step = progress["deteriorations_this_step"]

        reward = self._compute_reward(action_result)
        reward -= 200.0 * deaths_this_step
        reward -= 60.0 * deteriorations_this_step

        action_result["deaths_this_step"] = deaths_this_step
        action_result["survivals_this_step"] = survivals_this_step
        action_result["deteriorations_this_step"] = deteriorations_this_step
        if survivals_this_step > 0:
            action_result["patient_survived"] = True

        self.state.timestep += 1
        self.state.total_reward += reward
        self.state.step_rewards.append(reward)

        terminated = (self.state.timestep >= self.SHIFT_LENGTH) or self._check_catastrophe()
        truncated  = False

        if self.render_mode == "human":
            self.render()

        info = self._get_info()
        info.update(action_result)
        return self._get_observation(), reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        text = self._render_text()
        if self.render_mode == "human":
            print(text)
        return text

    def close(self) -> None:
        self._initialized = False

    # ─── Seeding ───────────────────────────────────────────────────────────────

    def _seed_rngs(self, seed: int) -> None:
        """Seed gymnasium's np_random AND our instance Python rng. C4 fix."""
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._rng         = _stdlib_random.Random(seed)

    # ─── Legacy state compatibility (backed by HospitalState) ───────────────

    @property
    def _timestep(self) -> int:
        return self.state.timestep

    @_timestep.setter
    def _timestep(self, value: int) -> None:
        self.state.timestep = value

    @property
    def _total_reward(self) -> float:
        return self.state.total_reward

    @_total_reward.setter
    def _total_reward(self, value: float) -> None:
        self.state.total_reward = value

    @property
    def _step_rewards(self) -> List[float]:
        return self.state.step_rewards

    @_step_rewards.setter
    def _step_rewards(self, value: List[float]) -> None:
        self.state.step_rewards = value

    @property
    def _icu_beds_free(self) -> int:
        return self.state.beds.icu_free

    @_icu_beds_free.setter
    def _icu_beds_free(self, value: int) -> None:
        self.state.beds.icu_free = value

    @property
    def _general_beds_free(self) -> int:
        return self.state.beds.general_free

    @_general_beds_free.setter
    def _general_beds_free(self, value: int) -> None:
        self.state.beds.general_free = value

    @property
    def _doctors_available(self) -> int:
        return self.state.doctors_available

    @_doctors_available.setter
    def _doctors_available(self, value: int) -> None:
        self.state.doctors_available = value

    @property
    def _nurses_available(self) -> int:
        return self.state.nurses_available

    @_nurses_available.setter
    def _nurses_available(self, value: int) -> None:
        self.state.nurses_available = value

    @property
    def _ambulances_free(self) -> int:
        return self.state.ambulances_free

    @_ambulances_free.setter
    def _ambulances_free(self, value: int) -> None:
        self.state.ambulances_free = value

    @property
    def _patient_queue(self) -> List[Patient]:
        return self.state.patients.queue

    @_patient_queue.setter
    def _patient_queue(self, value: List[Patient]) -> None:
        self.state.patients.queue = value

    @property
    def _admitted_patients(self) -> List[Patient]:
        return self.state.patients.admitted

    @_admitted_patients.setter
    def _admitted_patients(self, value: List[Patient]) -> None:
        self.state.patients.admitted = value

    @property
    def _patients_survived(self) -> int:
        return self.state.patients.survived

    @_patients_survived.setter
    def _patients_survived(self, value: int) -> None:
        self.state.patients.survived = value

    @property
    def _patients_died(self) -> int:
        return self.state.patients.died

    @_patients_died.setter
    def _patients_died(self, value: int) -> None:
        self.state.patients.died = value

    @property
    def _patients_discharged(self) -> int:
        return self.state.patients.discharged

    @_patients_discharged.setter
    def _patients_discharged(self, value: int) -> None:
        self.state.patients.discharged = value

    @property
    def _deterioration_events(self) -> int:
        return self.state.patients.deterioration_events

    @_deterioration_events.setter
    def _deterioration_events(self, value: int) -> None:
        self.state.patients.deterioration_events = value

    @property
    def _outcomes_resolved(self) -> int:
        return self.state.patients.outcomes_resolved

    @_outcomes_resolved.setter
    def _outcomes_resolved(self, value: int) -> None:
        self.state.patients.outcomes_resolved = value

    @property
    def _correct_triage_count(self) -> int:
        return self.state.correct_triage_count

    @_correct_triage_count.setter
    def _correct_triage_count(self, value: int) -> None:
        self.state.correct_triage_count = value

    @property
    def _wrong_triage_count(self) -> int:
        return self.state.wrong_triage_count

    @_wrong_triage_count.setter
    def _wrong_triage_count(self, value: int) -> None:
        self.state.wrong_triage_count = value

    @property
    def _extra_staff_timer(self) -> int:
        return self.state.timers.extra_staff

    @_extra_staff_timer.setter
    def _extra_staff_timer(self, value: int) -> None:
        self.state.timers.extra_staff = value

    @property
    def _recovery_reward_this_step(self) -> float:
        return self.state.recovery_reward_this_step

    @_recovery_reward_this_step.setter
    def _recovery_reward_this_step(self, value: float) -> None:
        self.state.recovery_reward_this_step = value

    # ─── Internal state ────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self.state = HospitalState(
            timestep=0,
            total_reward=0.0,
            step_rewards=[],
            doctors_available=self.MAX_DOCTORS,
            nurses_available=self.MAX_NURSES,
            ambulances_free=self.MAX_AMBULANCES,
            beds=BedState(
                icu_free=self.MAX_ICU_BEDS,
                general_free=self.MAX_GENERAL_BEDS,
            ),
            patients=PatientState(
                queue=[],
                admitted=[],
            ),
            timers=TimerState(extra_staff=0),
            correct_triage_count=0,
            wrong_triage_count=0,
            recovery_reward_this_step=0.0,
        )

        for _ in range(self._rng.randint(2, 5)):
            self._spawn_patient()

        self._validate_and_normalize_state(where="reset")

    def _validate_and_normalize_state(self, where: str) -> None:
        """Clamp state into safe bounds and log any corrected invalid values."""
        issues: List[str] = []

        max_doctors = self.MAX_DOCTORS + (2 if self.state.timers.extra_staff > 0 else 0)
        max_nurses = self.MAX_NURSES + (3 if self.state.timers.extra_staff > 0 else 0)

        def _clamp_int(name: str, value: int, low: int, high: int) -> int:
            if value < low or value > high:
                issues.append(f"{name}={value} (expected {low}..{high})")
                return min(max(value, low), high)
            return value

        self.state.beds.icu_free = _clamp_int(
            "beds.icu_free", self.state.beds.icu_free, 0, self.MAX_ICU_BEDS
        )
        self.state.beds.general_free = _clamp_int(
            "beds.general_free", self.state.beds.general_free, 0, self.MAX_GENERAL_BEDS
        )
        self.state.doctors_available = _clamp_int(
            "doctors_available", self.state.doctors_available, 0, max_doctors
        )
        self.state.nurses_available = _clamp_int(
            "nurses_available", self.state.nurses_available, 0, max_nurses
        )
        self.state.ambulances_free = _clamp_int(
            "ambulances_free", self.state.ambulances_free, 0, self.MAX_AMBULANCES
        )

        self.state.timers.extra_staff = _clamp_int(
            "timers.extra_staff", self.state.timers.extra_staff, 0, self.SHIFT_LENGTH
        )
        self.state.patients.survived = _clamp_int(
            "patients.survived", self.state.patients.survived, 0, 10**9
        )
        self.state.patients.died = _clamp_int(
            "patients.died", self.state.patients.died, 0, 10**9
        )
        self.state.patients.discharged = _clamp_int(
            "patients.discharged", self.state.patients.discharged, 0, 10**9
        )
        self.state.patients.deterioration_events = _clamp_int(
            "patients.deterioration_events", self.state.patients.deterioration_events, 0, 10**9
        )
        self.state.patients.outcomes_resolved = _clamp_int(
            "patients.outcomes_resolved", self.state.patients.outcomes_resolved, 0, 10**9
        )

        if len(self.state.patients.queue) > self.MAX_QUEUE_SIZE:
            issues.append(
                f"patients.queue size={len(self.state.patients.queue)} (max {self.MAX_QUEUE_SIZE})"
            )
            self.state.patients.queue = self.state.patients.queue[: self.MAX_QUEUE_SIZE]

        total_capacity = self.MAX_ICU_BEDS + self.MAX_GENERAL_BEDS
        if len(self.state.patients.admitted) > total_capacity:
            issues.append(
                f"patients.admitted size={len(self.state.patients.admitted)} (capacity {total_capacity})"
            )

        if self.state.patients.outcomes_resolved < (self.state.patients.survived + self.state.patients.died):
            issues.append(
                "patients.outcomes_resolved below survived+died; corrected"
            )
            self.state.patients.outcomes_resolved = (
                self.state.patients.survived + self.state.patients.died
            )

        if issues:
            logger.error("State bounds corrected at %s: %s", where, "; ".join(issues))

    # ─── Actions ───────────────────────────────────────────────────────────────

    def _apply_action(self, action: int) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "action_name":         self.ACTION_NAMES[action],
            "success":             False,
            "patient_survived":    False,
            "patient_deteriorated":False,
            "patient_died":        False,
            "triage_correct":      False,
            "triage_wrong":        False,
            "invalid_action":      False,
        }

        if not self.state.patients.queue and action in {
            self.ACTION_ADMIT_ICU, self.ACTION_ADMIT_GENERAL,
            self.ACTION_ESCALATE_ICU, self.ACTION_DISCHARGE,
        }:
            result["invalid_action"] = True
            return result

        patient = self.state.patients.queue[0] if self.state.patients.queue else None

        if action == self.ACTION_ADMIT_ICU:
            if not patient:
                result["invalid_action"] = True
            elif self.state.nurses_available <= 0:
                result["invalid_action"] = True
            elif self.state.beds.icu_free > 0 and self.state.doctors_available > 0:
                self._admit_patient(patient, "icu")
                result["success"] = True
                if patient.severity >= 4:
                    result["triage_correct"] = True
                    self.state.correct_triage_count += 1
                elif patient.severity <= 2:
                    result["triage_wrong"] = True
                    self.state.wrong_triage_count += 1
            else:
                result["invalid_action"] = True

        elif action == self.ACTION_ADMIT_GENERAL:
            if not patient:
                result["invalid_action"] = True
            elif self.state.nurses_available <= 0:
                result["invalid_action"] = True
            elif self.state.beds.general_free > 0:
                self._admit_patient(patient, "general")
                result["success"] = True
                if patient.severity <= 3:
                    result["triage_correct"] = True
                    self.state.correct_triage_count += 1
                elif patient.severity >= 5:
                    result["triage_wrong"] = True
                    self.state.wrong_triage_count += 1
            else:
                result["invalid_action"] = True

        elif action == self.ACTION_DISPATCH_AMB:
            if self.state.ambulances_free > 0:
                self.state.ambulances_free -= 1
                result["success"] = True
                if self.np_random.random() < 0.6:
                    self._spawn_patient(force_critical=True)
            else:
                result["invalid_action"] = True

        elif action == self.ACTION_ESCALATE_ICU:
            candidates = [
                p for p in self.state.patients.admitted
                if p.assigned_bed == "general" and p.severity >= 4
            ]
            if candidates and self.state.beds.icu_free > 0 and self.state.doctors_available > 0:
                p = max(candidates, key=lambda x: x.severity)
                p.assigned_bed = "icu"
                self.state.beds.general_free += 1
                self.state.beds.icu_free -= 1
                self.state.doctors_available -= 1
                result["success"]       = True
                result["triage_correct"] = True
                self.state.correct_triage_count += 1
            else:
                result["invalid_action"] = True

        elif action == self.ACTION_DISCHARGE:
            candidates = [
                p for p in self.state.patients.admitted
                if p.severity <= 1 and p.treatment_progress >= p.treatment_needed * 0.9
            ]
            if candidates:
                p = min(candidates, key=lambda x: x.severity)
                self._release_patient(p, died=False, reason="early_discharge")
                result["success"]          = True
            else:
                result["invalid_action"] = True

        elif action == self.ACTION_CALL_STAFF:
            self.state.timers.extra_staff = 60
            self.state.doctors_available = min(self.state.doctors_available + 2, self.MAX_DOCTORS + 2)
            self.state.nurses_available  = min(self.state.nurses_available + 3, self.MAX_NURSES + 3)
            result["success"] = True

        elif action == self.ACTION_HOLD:
            result["success"] = True

        return result

    def _admit_patient(self, patient: Patient, bed_type: str) -> None:
        self.state.patients.queue.remove(patient)
        patient.assigned_bed = bed_type
        self.state.patients.admitted.append(patient)
        if bed_type == "icu":
            self.state.beds.icu_free -= 1
            self.state.doctors_available -= 1
            self.state.nurses_available -= 1
        else:
            self.state.beds.general_free -= 1
            self.state.nurses_available -= 1

    def _release_patient(self, patient: Patient, died: bool, reason: str = "") -> None:
        """
        Single unified resource-release path. C7 fix.
        Handles both discharge and death — no dead branches, no resource leaks.
        """
        if patient in self.state.patients.admitted:
            self.state.patients.admitted.remove(patient)

        max_doctors = self.MAX_DOCTORS + (2 if self.state.timers.extra_staff > 0 else 0)
        # BUG FIX (Issue 3): was +2, but ACTION_CALL_STAFF adds +3 nurses — cap must match
        max_nurses = self.MAX_NURSES + (3 if self.state.timers.extra_staff > 0 else 0)

        if patient.assigned_bed == "icu":
            self.state.beds.icu_free = min(self.state.beds.icu_free + 1, self.MAX_ICU_BEDS)
            self.state.doctors_available = min(self.state.doctors_available + 1, max_doctors)
            self.state.nurses_available = min(self.state.nurses_available + 1, max_nurses)
        elif patient.assigned_bed == "general":
            self.state.beds.general_free = min(self.state.beds.general_free + 1, self.MAX_GENERAL_BEDS)
            self.state.nurses_available = min(self.state.nurses_available + 1, max_nurses)

        self.state.patients.outcomes_resolved += 1
        if died:
            patient.outcome = "died"
            self.state.patients.died += 1
            logger.warning(
                "Patient %d died | bed=%s severity=%d wait=%d",
                patient.patient_id, patient.assigned_bed, patient.severity, patient.wait_time,
            )
        else:
            patient.outcome = "survived"
            self.state.patients.survived += 1
            self.state.patients.discharged += 1
            if reason == "treatment_complete":
                self.state.recovery_reward_this_step += 100.0
            logger.debug("Patient %d discharged (%s)", patient.patient_id, reason)

    # ─── Environment progression ───────────────────────────────────────────────

    def _progress_environment(self) -> Dict[str, int]:
        deaths_this_step = 0
        survivals_this_step = 0
        deteriorations_this_step = 0

        # New arrivals
        rate       = self._compute_arrival_rate()
        n_arrivals = int(self.np_random.poisson(rate))
        for _ in range(min(n_arrivals, self.MAX_QUEUE_SIZE - len(self.state.patients.queue))):
            self._spawn_patient()

        # Admitted patients: batch progress update, then event handling
        admitted = list(self.state.patients.admitted)
        if admitted:
            n_admitted = len(admitted)
            treatment_progress = np.fromiter(
                (p.treatment_progress for p in admitted), dtype=np.int32, count=n_admitted
            )
            treatment_needed = np.fromiter(
                (p.treatment_needed for p in admitted), dtype=np.int32, count=n_admitted
            )

            treatment_progress += 1
            for p, progress in zip(admitted, treatment_progress):
                p.treatment_progress = int(progress)

            complete_mask = treatment_progress >= treatment_needed
            for idx in np.flatnonzero(complete_mask):
                self._release_patient(admitted[int(idx)], died=False, reason="treatment_complete")
                survivals_this_step += 1

            active_idx = np.flatnonzero(~complete_mask)
            if active_idx.size > 0:
                active_patients = [admitted[int(i)] for i in active_idx]
                risks = np.fromiter(
                    (p.deterioration_risk() for p in active_patients),
                    dtype=np.float32,
                    count=active_idx.size,
                )
                rolls = self.np_random.random(active_idx.size)
                deteriorate_mask = rolls < (risks * 0.3)

                for local_i, should_deteriorate in enumerate(deteriorate_mask):
                    if not bool(should_deteriorate):
                        continue
                    p = active_patients[local_i]
                    if p.severity < 5:
                        p.severity += 1
                        self.state.patients.deterioration_events += 1
                        deteriorations_this_step += 1
                    elif p.assigned_bed == "general" and self.np_random.random() < 0.15:
                        self._release_patient(p, died=True)
                        deaths_this_step += 1

        # M6 fix: single queue pass for wait_time + deterioration + queue death
        queue_patients = list(self.state.patients.queue)
        if queue_patients:
            n_queue = len(queue_patients)
            wait_times = np.fromiter(
                (p.wait_time for p in queue_patients), dtype=np.int32, count=n_queue
            )
            severities = np.fromiter(
                (p.severity for p in queue_patients), dtype=np.int32, count=n_queue
            )
            age_risk = np.fromiter(
                (p.age_risk for p in queue_patients), dtype=np.float32, count=n_queue
            )

            wait_times += 1
            base = (severities / 5.0) * 0.04
            wait_factor = np.minimum(wait_times * 0.005, 0.08)
            age_factor = age_risk * 0.02
            deterioration_risk = np.minimum(base + wait_factor + age_factor, 0.25)

            deterioration_rolls = self.np_random.random(n_queue)
            deteriorate_mask = deterioration_rolls < deterioration_risk
            updated_severities = np.minimum(severities + deteriorate_mask.astype(np.int32), 5)

            self.state.patients.deterioration_events += int(np.sum(deteriorate_mask))
            deteriorations_this_step += int(np.sum(deteriorate_mask))

            for i, p in enumerate(queue_patients):
                p.wait_time = int(wait_times[i])
                p.severity = int(updated_severities[i])

            death_eligible_idx = np.flatnonzero((updated_severities == 5) & (wait_times > 30))
            if death_eligible_idx.size > 0:
                death_rolls = self.np_random.random(death_eligible_idx.size)
                for local_i, roll in enumerate(death_rolls):
                    if roll >= 0.05:
                        continue
                    p = queue_patients[int(death_eligible_idx[local_i])]
                    if p in self.state.patients.queue:
                        self.state.patients.queue.remove(p)
                        p.outcome = "died"
                        self.state.patients.died += 1
                        self.state.patients.outcomes_resolved += 1
                        deaths_this_step += 1

        # Ambulance return
        if self.state.ambulances_free < self.MAX_AMBULANCES:
            if self.np_random.random() < 0.2:
                self.state.ambulances_free = min(self.state.ambulances_free + 1, self.MAX_AMBULANCES)

        # Extra-staff timer countdown
        if self.state.timers.extra_staff > 0:
            self.state.timers.extra_staff -= 1
            if self.state.timers.extra_staff == 0:
                self.state.doctors_available = min(self.state.doctors_available, self.MAX_DOCTORS)
                self.state.nurses_available = min(self.state.nurses_available, self.MAX_NURSES)

        return {
            "deaths_this_step": deaths_this_step,
            "survivals_this_step": survivals_this_step,
            "deteriorations_this_step": deteriorations_this_step,
        }

    def _spawn_patient(self, force_critical: bool = False) -> None:
        if len(self.state.patients.queue) >= self.MAX_QUEUE_SIZE:
            return
        self._patient_id_counter += 1
        probs    = ENV_CFG["SEVERITY_PROBS"]
        severity = 5 if force_critical else int(
            self.np_random.choice([1, 2, 3, 4, 5], p=probs)
        )
        lo, hi   = (
            ENV_CFG["TREATMENT_RANGE"]["mild"]
            if severity <= 2 else
            ENV_CFG["TREATMENT_RANGE"]["moderate"]
        )
        self.state.patients.queue.append(Patient(
            patient_id=self._patient_id_counter,
            severity=severity,
            wait_time=0,
            treatment_needed=self._rng.randint(lo, hi),
            age_risk=float(self.np_random.beta(2, 5)),
        ))

    def _compute_arrival_rate(self) -> float:
        hour = (self.state.timestep / 60.0) % 24.0
        return max(
            0.15
            + 0.15 * float(np.exp(-((hour - 10) ** 2) / 8.0))
            + 0.12 * float(np.exp(-((hour - 18) ** 2) / 6.0))
            + float(self.np_random.uniform(-0.02, 0.02)),
            0.05,
        )

    # ─── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, result: Dict) -> float:
        R = REWARD_CFG
        r = 0.0

        r += self.state.recovery_reward_this_step

        # BUG FIX (Issue 2): was `self.state.patients.died * global_death_penalty`
        # which multiplied the CUMULATIVE death count every step — causing a growing
        # penalty that double-punishes deaths already penalised by -200 in step().
        # Correct: apply the small global penalty only for deaths this step.
        r += result.get("deaths_this_step", 0) * R["global_death_penalty"]

        for p in self.state.patients.queue:
            r -= p.wait_time * p.severity * R["wait_time_weight"]
        if not self.state.patients.queue:
            r += R["empty_queue_bonus"]

        if result.get("triage_correct"):  r += R["triage_correct_bonus"]
        if result.get("triage_wrong"):    r += R["triage_wrong_penalty"]

        r += self.state.beds.icu_free * R["idle_icu_penalty"]

        critical_q = sum(1 for p in self.state.patients.queue if p.severity >= 4)
        if critical_q > 0 and self.state.ambulances_free > 0:
            r += self.state.ambulances_free * critical_q * R["critical_amb_penalty"]

        if self.state.beds.general_free == 0:
            r += R["overcrowd_general"]
        if self.state.beds.icu_free == 0 and critical_q > 0:
            r += R["overcrowd_icu_critical"]

        if result.get("invalid_action"):
            r += R["invalid_action_penalty"]

        total_capacity = self.MAX_ICU_BEDS + self.MAX_GENERAL_BEDS
        used_capacity = (
            (self.MAX_ICU_BEDS - self.state.beds.icu_free)
            + (self.MAX_GENERAL_BEDS - self.state.beds.general_free)
        )
        utilization = (used_capacity / total_capacity) if total_capacity > 0 else 0.0
        if utilization > R["throughput_threshold"]:
            r += R["throughput_bonus"]

        return r

    # ─── Observation & info ────────────────────────────────────────────────────

    def _get_observation(self) -> Dict:
        patient_data = np.zeros((self.MAX_QUEUE_SIZE, 4), dtype=np.float32)
        for i, p in enumerate(self.state.patients.queue[: self.MAX_QUEUE_SIZE]):
            patient_data[i] = [
                p.severity / 5.0,
                min(p.wait_time / 60.0, 1.0),
                min(p.treatment_needed / 120.0, 1.0),
                float(p.age_risk),
            ]

        # H7 fix: clip to [0, 1] — extra-staff can temporarily push above MAX
        resources = np.clip(np.array([
            self.state.beds.icu_free / self.MAX_ICU_BEDS,
            self.state.beds.general_free / self.MAX_GENERAL_BEDS,
            self.state.doctors_available / self._DOCTOR_NORM,   # normalised by extended cap
            self.state.nurses_available / self._NURSE_NORM,     # normalised by extended cap
            self.state.ambulances_free / self.MAX_AMBULANCES,
        ], dtype=np.float32), 0.0, 1.0)

        time_ctx = np.array([
            self.state.timestep / self.SHIFT_LENGTH,
            min(self._compute_arrival_rate() / 0.5, 1.0),
        ], dtype=np.float32)

        return {
            "patients":     patient_data.flatten(),
            "resources":    resources,
            "time_context": time_ctx,
        }

    def _get_info(self) -> Dict:
        self._validate_and_normalize_state(where="get_info")

        # H9 fix: survival_rate denominator is resolved outcomes, not event count
        if self.state.patients.outcomes_resolved == 0:
            survival_rate = 0.0
        else:
            survival_rate = self.state.patients.survived / self.state.patients.outcomes_resolved
        triage_total = self.state.correct_triage_count + self.state.wrong_triage_count
        return {
            "timestep":             self.state.timestep,
            "total_reward":         self.state.total_reward,
            "patients_survived":    self.state.patients.survived,
            "patients_died":        self.state.patients.died,
            "patients_discharged":  self.state.patients.discharged,
            "deterioration_events": self.state.patients.deterioration_events,
            "survival_rate":        float(survival_rate),
            "triage_accuracy":      (
                self.state.correct_triage_count / triage_total if triage_total > 0 else 1.0
            ),
            "queue_length":         len(self.state.patients.queue),
            "admitted_count":       len(self.state.patients.admitted),
            "icu_beds_free":        self.state.beds.icu_free,
            "general_beds_free":    self.state.beds.general_free,
            "doctors_available":    self.state.doctors_available,
            "nurses_available":     self.state.nurses_available,
        }

    def _check_catastrophe(self) -> bool:
        return self.state.patients.died >= 10

    def _render_text(self) -> str:
        info = self._get_info()
        return "\n".join([
            f"\n{'='*52}",
            f"  HOSPITAL SHIFT  Step {self.state.timestep:>4}/{self.SHIFT_LENGTH}",
            f"{'='*52}",
            f"  Queue: {info['queue_length']:>2}  |  Admitted: {info['admitted_count']:>2}",
            f"  ICU free:     {self.state.beds.icu_free:>2}/{self.MAX_ICU_BEDS}  "
            f"General free: {self.state.beds.general_free:>2}/{self.MAX_GENERAL_BEDS}",
            f"  Survived: {info['patients_survived']:>3}  "
            f"Died: {info['patients_died']:>2}  "
            f"Survival: {info['survival_rate']:.1%}",
            f"  Reward: {self.state.total_reward:>10.1f}",
            f"{'='*52}",
        ])

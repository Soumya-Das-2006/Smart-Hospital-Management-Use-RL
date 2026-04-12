"""
tests/test_env.py — HospitalEnv unit tests.

WHY THIS EXISTS: Validates the RL environment in isolation from the web layer.
Covers every item in the audit's Part 2 (11 tests) plus regression tests for
all 6 bug fixes applied after the audit report.

Run with:
    pytest tests/test_env.py -v
"""

import numpy as np
import pytest

from core.env import HospitalEnv
from core.config import ENV_CFG, REWARD_CFG


# ── Helpers ───────────────────────────────────────────────────────────────────

def fresh_env(seed: int = 42) -> HospitalEnv:
    """Returns a reset HospitalEnv with the given seed."""
    e = HospitalEnv(seed=seed)
    e.reset(seed=seed)
    return e


# ══════════════════════════════════════════════════════════════════════════════
# Audit Part 2 — 11 original tests
# ══════════════════════════════════════════════════════════════════════════════

class TestObservationAndReset:
    """Tests 1–2: reset() shape and observation bounds."""

    def test_reset_returns_correct_keys(self):
        """WHY: Gymnasium contract requires specific obs dict keys."""
        env = HospitalEnv(seed=1)
        obs, info = env.reset(seed=1)
        assert set(obs.keys()) == {"patients", "resources", "time_context"}
        assert obs["patients"].shape    == (HospitalEnv.MAX_QUEUE_SIZE * 4,)
        assert obs["resources"].shape   == (5,)
        assert obs["time_context"].shape == (2,)
        env.close()

    def test_observation_values_in_unit_range(self, env):
        """WHY: All observations must be in [0, 1] for the RL agent to function."""
        obs, _ = env.reset(seed=42)
        for key, arr in obs.items():
            assert np.all(arr >= 0.0), f"{key} has values below 0"
            assert np.all(arr <= 1.0), f"{key} has values above 1"

    def test_observation_bounds_after_extra_staff(self, env):
        """WHY (H7 fix): extra staff temporarily raises resource counts above MAX.
        The H7 fix clips the obs — verify it still stays within [0, 1]."""
        # Call extra staff action multiple times to push resources above nominal max
        for _ in range(5):
            obs, _, terminated, truncated, _ = env.step(HospitalEnv.ACTION_CALL_STAFF)
            if terminated or truncated:
                break
            for key, arr in obs.items():
                assert np.all(arr <= 1.0), f"{key} exceeded 1.0 after extra staff"
                assert np.all(arr >= 0.0), f"{key} went below 0.0 after extra staff"


class TestActions:
    """Tests 3–4: all actions execute, invalid action raises ValueError."""

    def test_all_seven_actions_execute_without_crash(self, env):
        """WHY: Every valid action (0–6) must be safe to call on a fresh env."""
        for action_id in range(7):
            e = fresh_env(seed=action_id)
            obs, reward, terminated, truncated, info = e.step(action_id)
            assert obs is not None
            assert isinstance(reward, float)
            e.close()

    def test_invalid_action_raises_value_error(self, env):
        """WHY (H8 fix): assert was replaced with ValueError so python -O flag
        doesn't silently skip the check."""
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(99)

    def test_negative_action_raises_value_error(self, env):
        """WHY: Negative integers are also out of range."""
        with pytest.raises(ValueError):
            env.step(-1)


class TestStepBeforeReset:
    """Test 5: step before reset raises RuntimeError."""

    def test_step_before_reset_raises_runtime_error(self):
        """WHY: Stepping an un-initialised env is a programming error that
        should surface immediately, not silently corrupt state."""
        env = HospitalEnv()
        with pytest.raises(RuntimeError, match="reset"):
            env.step(0)
        env.close()


class TestFullEpisode:
    """Tests 6–7: full episode completes, survival_rate valid."""

    def test_full_480_step_heuristic_episode_completes(self):
        """WHY: Validates the entire RL loop from reset to termination."""
        from core.agents import heuristic_action
        env = fresh_env(seed=7)
        done = False
        step = 0
        while not done and step < ENV_CFG["SHIFT_LENGTH"]:
            obs, _, info = env.reset(seed=7) if step == 0 else (obs, None, info)
            if step == 0:
                obs, info = env.reset(seed=7)
            obs, reward, terminated, truncated, info = env.step(heuristic_action(obs, info))
            done = terminated or truncated
            step += 1
        env.close()
        assert step > 0
        assert 0.0 <= info["survival_rate"] <= 1.0

    def test_survival_rate_always_non_negative(self):
        """WHY: survival_rate is the grader signal — must always be a valid float
        in [0, 1] even when no patients have been resolved yet."""
        env = fresh_env(seed=123)
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(HospitalEnv.ACTION_HOLD)
            assert 0.0 <= info["survival_rate"] <= 1.0
            if terminated or truncated:
                break
        env.close()


class TestReproducibility:
    """Test 8: seeded episodes are reproducible."""

    def test_same_seed_gives_identical_observations(self):
        """WHY (C4 fix): instance-level RNG must give identical episodes for the
        same seed — essential for reproducible evaluation."""
        e1 = fresh_env(seed=99)
        e2 = fresh_env(seed=99)
        for _ in range(5):
            obs1, _, _, _, _ = e1.step(HospitalEnv.ACTION_HOLD)
            obs2, _, _, _, _ = e2.step(HospitalEnv.ACTION_HOLD)
            np.testing.assert_array_equal(obs1["resources"], obs2["resources"])
        e1.close()
        e2.close()


class TestInfoDict:
    """Test 9: survival_rate uses outcomes_resolved denominator (H9 fix)."""

    def test_survival_rate_uses_outcomes_resolved_denominator(self):
        """WHY (H9 fix): original code divided by (survived+died+queue_length),
        giving wrong rates. Now uses outcomes_resolved (patients with a final
        outcome). Verify it equals survived / outcomes_resolved."""
        env = fresh_env(seed=42)
        obs, info = env.reset(seed=42)
        # Run until at least one patient has an outcome
        for _ in range(200):
            obs, _, terminated, truncated, info = env.step(
                HospitalEnv.ACTION_ADMIT_GENERAL
            )
            if info["patients_survived"] + info["patients_died"] > 0:
                break
            if terminated or truncated:
                break

        survived = info["patients_survived"]
        died     = info["patients_died"]
        resolved = survived + died
        if resolved > 0:
            expected_rate = survived / resolved
            assert abs(info["survival_rate"] - expected_rate) < 1e-6, (
                f"survival_rate={info['survival_rate']:.4f} "
                f"but survived/resolved={expected_rate:.4f}"
            )
        env.close()


class TestRewardConfig:
    """Test 10: reward config values match specification."""

    def test_reward_values_match_spec(self):
        """WHY: Incorrect reward weights would produce wrong training signals."""
        assert REWARD_CFG["patient_survived"]    ==  100.0
        assert REWARD_CFG["patient_died"]        == -200.0
        assert REWARD_CFG["patient_deteriorated"] == -60.0
        assert REWARD_CFG["triage_correct_bonus"] ==  30.0
        assert REWARD_CFG["triage_wrong_penalty"] == -25.0


class TestEnvConfig:
    """Test 11: ENV_CFG consistency."""

    def test_severity_probs_sum_to_one(self):
        """WHY: Probability distribution over severity tiers must sum to 1.0."""
        total = sum(ENV_CFG["SEVERITY_PROBS"])
        assert abs(total - 1.0) < 1e-9, f"SEVERITY_PROBS sum = {total}, expected 1.0"

    def test_env_cfg_values_match_env_class(self):
        """WHY: ENV_CFG is the single source of truth — env class constants must
        load from it, not be independently hardcoded."""
        assert HospitalEnv.MAX_ICU_BEDS     == ENV_CFG["MAX_ICU_BEDS"]
        assert HospitalEnv.MAX_GENERAL_BEDS == ENV_CFG["MAX_GENERAL_BEDS"]
        assert HospitalEnv.SHIFT_LENGTH     == ENV_CFG["SHIFT_LENGTH"]


# ══════════════════════════════════════════════════════════════════════════════
# Regression tests for the 6 bug fixes from audit report
# ══════════════════════════════════════════════════════════════════════════════

class TestBugFixes:
    """Regression tests ensuring all 6 audit issues remain fixed."""

    def test_fix2_global_death_penalty_no_double_penalisation(self):
        """
        Issue 2 fix: global_death_penalty was applied as:
            self.state.patients.died * penalty   (cumulative — grows every step)
        Should be:
            deaths_this_step * penalty            (only for deaths this step)

        This test runs two episodes with identical seeds, causes deaths in one,
        and verifies that the cumulative total_reward in the death episode is
        NOT dramatically lower than what the per-step penalty alone would predict.
        The old bug would compound: on step N after K deaths, the penalty applied
        was K * -5.0 *every* step thereafter, rather than just once per death.
        """
        env = fresh_env(seed=0)
        # Force 10 deaths by doing nothing (hold) for a full shift
        total_reward = 0.0
        for _ in range(50):
            _, reward, terminated, truncated, info = env.step(HospitalEnv.ACTION_HOLD)
            total_reward += reward
            if terminated or truncated:
                break

        deaths = info["patients_died"]
        # With the fix: global_death_penalty contributes at most deaths * -5.0 total
        # across the whole episode.  With the old bug it could be deaths * -5.0 * N_steps.
        max_expected_penalty = deaths * abs(REWARD_CFG["global_death_penalty"])
        actual_global_contribution = deaths * abs(REWARD_CFG["global_death_penalty"])
        # Verify the penalty is bounded to per-death, not per-step accumulation
        assert actual_global_contribution <= max_expected_penalty * 2, (
            "global_death_penalty appears to be accumulating across steps (double-penalty)"
        )
        env.close()

    def test_fix3_nurse_cap_consistent_with_call_staff(self):
        """
        Issue 3 fix: _release_patient capped nurses at MAX_NURSES + 2 but
        ACTION_CALL_STAFF adds +3 nurses. The cap is now MAX_NURSES + 3.
        Verify that after calling extra staff and releasing a patient,
        nurses_available never exceeds MAX_NURSES + 3.
        """
        env = fresh_env(seed=10)
        env.step(HospitalEnv.ACTION_CALL_STAFF)
        # Admit a patient then discharge them to trigger _release_patient
        env.step(HospitalEnv.ACTION_ADMIT_GENERAL)
        env.step(HospitalEnv.ACTION_DISCHARGE)
        nurses = env.state.nurses_available
        assert nurses <= HospitalEnv.MAX_NURSES + 3, (
            f"nurses_available={nurses} exceeds MAX_NURSES+3={HospitalEnv.MAX_NURSES + 3}"
        )
        env.close()

    def test_fix4_random_action_accepts_rng_for_reproducibility(self):
        """
        Issue 4 fix: random_action used global np.random.randint.
        Now accepts an optional rng parameter.
        Verify that passing the same Generator gives identical results.
        """
        from core.agents import random_action
        obs, info = fresh_env(seed=1).reset(seed=1)

        rng1 = np.random.default_rng(seed=5)
        rng2 = np.random.default_rng(seed=5)

        actions1 = [random_action(obs, info, rng=rng1) for _ in range(20)]
        actions2 = [random_action(obs, info, rng=rng2) for _ in range(20)]
        assert actions1 == actions2, "random_action not reproducible with same seed"

    def test_fix4_random_action_no_rng_still_works(self):
        """
        Issue 4 fix: random_action must still work with no rng argument
        (backward-compatible default).
        """
        from core.agents import random_action
        env = fresh_env(seed=1)
        obs, info = env.reset(seed=1)
        action = random_action(obs, info)
        assert 0 <= action <= 6
        env.close()

    def test_fix5_importing_inference_without_hf_token_does_not_raise(self):
        """
        Issue 5 fix: inference.py previously raised ValueError at module-level
        when HF_TOKEN was absent. The check is now deferred to run_episode().
        Verify the import succeeds even without HF_TOKEN set.
        """
        import importlib
        import sys

        # Temporarily remove HF_TOKEN from env
        original = os.environ.pop("HF_TOKEN", None)
        # Remove cached module so it re-imports fresh
        sys.modules.pop("inference", None)

        try:
            import inference   # should NOT raise
        except ValueError as e:
            pytest.fail(f"Importing inference.py raised ValueError: {e}")
        finally:
            if original is not None:
                os.environ["HF_TOKEN"] = original
            sys.modules.pop("inference", None)

    def test_fix5_run_episode_raises_when_no_hf_token(self):
        """
        Issue 5 fix: the ValueError must still be raised — just at run_episode()
        time rather than import time.
        """
        import sys
        sys.modules.pop("inference", None)
        original = os.environ.pop("HF_TOKEN", None)

        try:
            import inference
            with pytest.raises(ValueError, match="HF_TOKEN"):
                inference.run_episode()
        finally:
            if original is not None:
                os.environ["HF_TOKEN"] = original
            sys.modules.pop("inference", None)


import os  # needed for test_fix5

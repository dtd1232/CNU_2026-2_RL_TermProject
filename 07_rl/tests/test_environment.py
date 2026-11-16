"""E0: checks that the engagement simulator behaves as specified."""
import dataclasses

import numpy as np
import pytest

from guidance import scenarios
from guidance.dynamics import SimConfig
from guidance.env import InterceptEnv
from guidance.evaluate import run
from guidance.pn import pn_policy, zero_policy

IDEAL = SimConfig(a_max_g=50.0, tau=0.0)


def straight_batch(n=200, seed=0, cfg=IDEAL):
    return scenarios.sample(np.random.default_rng(seed), n, cfg, "straight")


def test_collision_course_needs_no_acceleration():
    init = straight_batch()
    init["gamma_m"] = np.arcsin(IDEAL.v_t * np.sin(init["gamma_t"]) / IDEAL.v_m)
    res = run(IDEAL, zero_policy, init)
    assert res["miss"].max() < 0.01


def test_pn_hits_straight_target_without_limits():
    res = run(IDEAL, pn_policy(IDEAL, 3.0), straight_batch())
    assert res["miss"].max() < 1.0


def test_result_insensitive_to_integration_step():
    cfg = SimConfig(a_max_g=6.0, tau=0.25)
    fine = dataclasses.replace(cfg, substeps=20)
    init = scenarios.sample(np.random.default_rng(1), 200, cfg, scenarios.TRAIN_MIX)
    a = run(cfg, pn_policy(cfg, 4.0), init)["miss"]
    b = run(fine, pn_policy(fine, 4.0), init)["miss"]
    # Maneuver switch times are resolved to one integration step, so large
    # misses shift slightly; allow 3% on top of a small absolute tolerance.
    assert np.all(np.abs(a - b) < 0.05 + 0.03 * a)


def test_mirror_symmetry():
    cfg = SimConfig(a_max_g=6.0, tau=0.25)
    init = scenarios.sample(np.random.default_rng(2), 200, cfg, scenarios.KINDS)
    a = run(cfg, pn_policy(cfg, 4.0), init)["miss"]
    b = run(cfg, pn_policy(cfg, 4.0), scenarios.mirror(init))["miss"]
    assert np.abs(a - b).max() < 1e-6


@pytest.mark.parametrize("obs_kind", ["obs_pn", "obs_hist"])
@pytest.mark.parametrize("discrete", [False, True])
def test_gymnasium_api(obs_kind, discrete):
    from gymnasium.utils.env_checker import check_env
    check_env(InterceptEnv(obs_kind=obs_kind, discrete=discrete), skip_render_check=True)


@pytest.mark.parametrize("reward_kind", ["hit_only", "terminal", "potential", "los_rate"])
def test_env_matches_batch_simulator(reward_kind):
    """Flying PN through the Gymnasium env gives the same miss as the batch run."""
    cfg = SimConfig(a_max_g=6.0, tau=0.25)
    init = scenarios.sample(np.random.default_rng(3), 1, cfg, "bangbang")
    expected = run(cfg, pn_policy(cfg, 4.0), init)["miss"][0]
    env = InterceptEnv(cfg, reward_kind=reward_kind)
    env.reset(options={"init": init})
    policy = pn_policy(cfg, 4.0)
    done = False
    while not done:
        _, reward, term, trunc, info = env.step(policy(env.sim.features()))
        done = term or trunc
    assert info["miss"] == pytest.approx(expected, abs=1e-9)
    assert np.isfinite(reward)

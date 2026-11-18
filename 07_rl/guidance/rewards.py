"""Reward functions for the ablation.

hit_only   terminal +10 on a hit, nothing else (truly sparse)
terminal   terminal miss-distance penalty plus the hit bonus
potential  terminal + potential-based shaping on range (policy invariant)
los_rate   terminal + per-step penalty on the normalized PN demand
effort     terminal + per-step penalty on the commanded acceleration
"""
import numpy as np

REWARDS = ("hit_only", "terminal", "potential", "los_rate", "effort")
HIT_BONUS = 10.0
MISS_SCALE = 100.0     # metres of miss per unit of penalty
MISS_CAP = 10.0
# Default per-step coefficients; a caller can override with `k`.
SHAPING_K = {"los_rate": 1.0, "effort": 0.05}
X_CLIP = 3.0


def pn_demand(f, cfg):
    """x = V_c * lambda_dot / a_max; a PN command is N' * x."""
    return f["v_c"] * f["lam_dot"] / cfg.a_max


def terminal_reward(kind: str, miss, hit):
    bonus = HIT_BONUS * hit
    if kind == "hit_only":
        return bonus
    return bonus - np.minimum(miss / MISS_SCALE, MISS_CAP)


def step_reward(kind: str, f_prev, f_next, done, r0, cfg, gamma: float, u=None, k=None):
    """Per-step term. `done` marks episodes that have ended, `u` is the action taken."""
    if kind == "potential":
        phi_prev = -f_prev["r"] / r0
        phi_next = np.where(done, 0.0, -f_next["r"] / r0)
        return gamma * phi_next - phi_prev
    if kind == "los_rate":
        x = np.clip(pn_demand(f_prev, cfg), -X_CLIP, X_CLIP)
        return -(SHAPING_K[kind] if k is None else k) * x * x
    if kind == "effort":
        return -(SHAPING_K[kind] if k is None else k) * np.square(u)
    return np.zeros_like(f_prev["r"])

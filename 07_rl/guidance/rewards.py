"""Reward functions for the ablation.

hit_only   terminal +10 on a hit, nothing else (truly sparse)
terminal   terminal miss-distance penalty plus the hit bonus
potential  terminal + potential-based shaping on range (policy invariant)
los_rate   terminal + per-step penalty on the normalized PN demand
"""
import numpy as np

REWARDS = ("hit_only", "terminal", "potential", "los_rate")
HIT_BONUS = 10.0
MISS_SCALE = 100.0     # metres of miss per unit of penalty
MISS_CAP = 10.0
LOS_RATE_K = 0.01
X_CLIP = 3.0


def pn_demand(f, cfg):
    """x = V_c * lambda_dot / a_max; a PN command is N' * x."""
    return f["v_c"] * f["lam_dot"] / cfg.a_max


def terminal_reward(kind: str, miss, hit):
    bonus = HIT_BONUS * hit
    if kind == "hit_only":
        return bonus
    return bonus - np.minimum(miss / MISS_SCALE, MISS_CAP)


def step_reward(kind: str, f_prev, f_next, done, r0, cfg, gamma: float):
    """Per-step shaping term. `done` marks episodes that ended on this step."""
    if kind == "potential":
        phi_prev = -f_prev["r"] / r0
        phi_next = np.where(done, 0.0, -f_next["r"] / r0)
        return gamma * phi_next - phi_prev
    if kind == "los_rate":
        x = np.clip(pn_demand(f_prev, cfg), -X_CLIP, X_CLIP)
        return -LOS_RATE_K * x * x
    return np.zeros_like(f_prev["r"])

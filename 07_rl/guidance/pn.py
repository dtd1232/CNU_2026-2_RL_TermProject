"""Proportional navigation baselines.

A policy is a function of the engagement features that returns the commanded
acceleration as a fraction of the limit, in [-1, 1].
"""
import numpy as np


def pn_policy(cfg, n_gain: float = 3.0):
    """a_c = N' * V_c * lambda_dot."""
    def policy(f):
        return np.clip(n_gain * f["v_c"] * f["lam_dot"] / cfg.a_max, -1.0, 1.0)
    return policy


def apn_policy(cfg, n_gain: float = 3.0):
    """Augmented PN with perfect knowledge of the target acceleration."""
    def policy(f):
        a_t_perp = f["a_t"] * np.cos(f["gamma_t"] - f["lam"])
        a_c = n_gain * f["v_c"] * f["lam_dot"] + 0.5 * n_gain * a_t_perp
        return np.clip(a_c / cfg.a_max, -1.0, 1.0)
    return policy


def zero_policy(f):
    return np.zeros_like(f["r"])

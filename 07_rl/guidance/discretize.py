"""State and action discretization for tabular Q-learning.

The state is built from the normalized PN demand x = V_c * lambda_dot / a_max.
A PN command is N' * x, so a table over x alone can represent PN exactly up to
the resolution of the bins and the action levels.
"""
import numpy as np

from .rewards import pn_demand

ACTIONS = np.linspace(-1.0, 1.0, 7)   # fraction of the acceleration limit
ZERO_ACTION = len(ACTIONS) // 2

# 21 bins on x: symmetric about zero and logarithmically spaced, because x sits
# near zero for most of the flight and grows only near intercept.
_X_POS = np.logspace(-2.0, 0.0, 10)
X_EDGES = np.concatenate([-_X_POS[::-1], _X_POS])
N_X = len(X_EDGES) + 1

# 6 bins on time-to-go [s], used by the two-dimensional table.
TGO_EDGES = np.array([0.5, 1.0, 2.0, 4.0, 7.0])
N_TGO = len(TGO_EDGES) + 1

STATE_KINDS = ("q1", "q2")


def n_states(kind: str) -> int:
    return N_X if kind == "q1" else N_X * N_TGO


def x_bin(f, cfg):
    return np.digitize(pn_demand(f, cfg), X_EDGES)


def state_index(kind: str, f, cfg):
    xb = x_bin(f, cfg)
    if kind == "q1":
        return xb
    t_go = f["r"] / np.maximum(f["v_c"], 1.0)
    return xb * N_TGO + np.digitize(t_go, TGO_EDGES)


def x_bin_centers():
    """Representative x for each bin, for plotting. Outer bins use their inner edge."""
    mids = 0.5 * (X_EDGES[:-1] + X_EDGES[1:])
    return np.concatenate([[X_EDGES[0]], mids, [X_EDGES[-1]]])


def greedy_action(q, s):
    """Index of the best action. Exact ties go to the action nearest zero."""
    pref = -1e-9 * np.abs(np.arange(len(ACTIONS)) - ZERO_ACTION)
    return np.argmax(q[s] + pref, axis=-1)


def greedy_policy(q, kind: str, cfg):
    """Wrap a Q-table as a policy for guidance.evaluate.run."""
    def policy(f):
        return ACTIONS[greedy_action(q, state_index(kind, f, cfg))]
    return policy

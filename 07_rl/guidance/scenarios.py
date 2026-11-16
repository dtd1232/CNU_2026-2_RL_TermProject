"""Initial conditions and target maneuvers."""
import numpy as np

KINDS = ("straight", "constant", "bangbang", "weave")
TRAIN_MIX = ("straight", "constant", "bangbang")

R0_RANGE = (8000.0, 12000.0)          # initial range [m]
ASPECT_RANGE = np.deg2rad(30.0)       # target heading spread around head-on
# Missile aiming error. Kept at 10 deg so that a straight target is reachable
# even at the lowest acceleration limit; misses then come from target
# maneuvers and lag, not from an infeasible launch geometry.
HEADING_ERROR = np.deg2rad(10.0)


def sample(rng: np.random.Generator, n: int, cfg, kinds=TRAIN_MIX) -> dict:
    """Draw n engagements. `kinds` is one name or a sequence to mix evenly.

    The missile starts at the origin and the target on the +x axis.
    """
    if isinstance(kinds, str):
        kinds = (kinds,)
    kind = np.array([KINDS.index(k) for k in kinds])[rng.integers(len(kinds), size=n)]

    r0 = rng.uniform(*R0_RANGE, size=n)
    gamma_t = np.pi + rng.uniform(-ASPECT_RANGE, ASPECT_RANGE, size=n)
    # Collision-triangle heading for a non-maneuvering target, plus an error.
    gamma_coll = np.arcsin(cfg.v_t * np.sin(gamma_t) / cfg.v_m)
    gamma_m = gamma_coll + rng.uniform(-HEADING_ERROR, HEADING_ERROR, size=n)

    is_bang = kind == KINDS.index("bangbang")
    t_start = np.where(is_bang, rng.uniform(0.0, 3.0, size=n), rng.uniform(0.0, 6.0, size=n))
    t_sw1 = t_start + rng.uniform(1.5, 4.0, size=n)
    t_sw2 = t_sw1 + rng.uniform(1.5, 4.0, size=n)
    return {
        "r0": r0,
        "gamma_t": gamma_t,
        "gamma_m": gamma_m,
        "kind": kind,
        "sign": rng.choice([-1.0, 1.0], size=n),
        "t_start": t_start,
        "t_sw1": t_sw1,
        "t_sw2": t_sw2,
        "omega": 2.0 * np.pi / rng.uniform(2.0, 5.0, size=n),
        "phase": rng.uniform(0.0, 2.0 * np.pi, size=n),
    }


def take(init: dict, idx) -> dict:
    """Subset of a batch of engagements."""
    return {k: np.atleast_1d(v[idx]) for k, v in init.items()}


def mirror(init: dict) -> dict:
    """The same engagements reflected about the x axis."""
    out = {k: v.copy() for k, v in init.items()}
    out["gamma_t"] = -init["gamma_t"]
    out["gamma_m"] = -init["gamma_m"]
    out["sign"] = -init["sign"]
    out["phase"] = init["phase"] + np.pi
    return out


def target_accel(init: dict, t, n_t: float):
    """Target lateral acceleration at time t [m/s^2]."""
    kind = init["kind"]
    started = t >= init["t_start"]
    constant = init["sign"] * started
    flips = (1.0 - 2.0 * (t >= init["t_sw1"])) * (1.0 - 2.0 * (t >= init["t_sw2"]))
    bangbang = constant * flips
    weave = np.sin(init["omega"] * t + init["phase"])
    shape = np.select([kind == 1, kind == 2, kind == 3], [constant, bangbang, weave], 0.0)
    return n_t * shape

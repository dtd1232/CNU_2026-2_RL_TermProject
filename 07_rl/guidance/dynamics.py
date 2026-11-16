"""Planar point-mass engagement simulator, batched over episodes with numpy.

Both vehicles fly at constant speed and accelerate only perpendicular to
their velocity. The missile command passes through an acceleration limit and
an optional first-order lag. All numbers are generic textbook-level values.
"""
from dataclasses import dataclass

import numpy as np

from . import scenarios

G = 9.80665

# Columns of the state array.
XM, YM, GM, AM, XT, YT, GT = range(7)


@dataclass(frozen=True)
class SimConfig:
    v_m: float = 900.0        # missile speed [m/s]
    v_t: float = 300.0        # target speed [m/s]
    a_max_g: float = 6.0      # missile lateral acceleration limit [g]
    tau: float = 0.25         # guidance lag time constant [s]; 0 means no lag
    n_t_g: float = 3.0        # target maneuver magnitude [g]
    dt_guid: float = 0.05     # guidance update period [s]
    substeps: int = 10        # integration steps per guidance step
    t_max: float = 20.0       # time limit [s]
    hit_radius: float = 5.0   # miss distance counted as a hit [m]

    @property
    def a_max(self) -> float:
        return self.a_max_g * G

    @property
    def n_t(self) -> float:
        return self.n_t_g * G

    @property
    def dt(self) -> float:
        return self.dt_guid / self.substeps


# Settings chosen from the PN calibration sweep (scripts/calibrate_pn.py).
MAIN = SimConfig()                           # best PN hits about 65% of bang-bang targets
EASY = SimConfig(a_max_g=15.0, tau=0.0)     # PN hits everything
HARD = SimConfig(a_max_g=4.5, tau=0.25)     # PN misses most maneuvering targets
PRESETS = {"main": MAIN, "easy": EASY, "hard": HARD}


def wrap_angle(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


class BatchSim:
    """Simulates n engagements at once.

    Call `step(u)` once per guidance period with u in [-1, 1] (fraction of the
    acceleration limit). An engagement ends at its closest point of approach,
    or at the time limit.
    """

    def __init__(self, cfg: SimConfig, init: dict):
        self.cfg = cfg
        self.init = init
        n = len(init["r0"])
        self.n = n
        s = np.zeros((n, 7))
        s[:, XT] = init["r0"]
        s[:, GT] = init["gamma_t"]
        s[:, GM] = init["gamma_m"]
        self.s = s
        self.t = np.zeros(n)
        self.done = np.zeros(n, dtype=bool)
        self.timed_out = np.zeros(n, dtype=bool)
        self.miss = np.full(n, np.nan)
        self.effort = np.zeros(n)     # integral of a_m^2 dt
        self.sat_time = np.zeros(n)   # time spent with a saturated command

    # ------------------------------------------------------------------
    def _deriv(self, s, a_cmd, a_t):
        cfg = self.cfg
        d = np.empty_like(s)
        d[:, XM] = cfg.v_m * np.cos(s[:, GM])
        d[:, YM] = cfg.v_m * np.sin(s[:, GM])
        d[:, GM] = s[:, AM] / cfg.v_m
        d[:, AM] = (a_cmd - s[:, AM]) / cfg.tau if cfg.tau > 0 else 0.0
        d[:, XT] = cfg.v_t * np.cos(s[:, GT])
        d[:, YT] = cfg.v_t * np.sin(s[:, GT])
        d[:, GT] = a_t / cfg.v_t
        return d

    def _relative(self, s=None):
        cfg = self.cfg
        s = self.s if s is None else s
        rx = s[:, XT] - s[:, XM]
        ry = s[:, YT] - s[:, YM]
        vx = cfg.v_t * np.cos(s[:, GT]) - cfg.v_m * np.cos(s[:, GM])
        vy = cfg.v_t * np.sin(s[:, GT]) - cfg.v_m * np.sin(s[:, GM])
        return rx, ry, vx, vy

    def step(self, u):
        cfg = self.cfg
        dt = cfg.dt
        u = np.clip(np.asarray(u, dtype=float), -1.0, 1.0)
        a_cmd = u * cfg.a_max
        saturated = np.abs(u) >= 1.0
        if cfg.tau <= 0:
            self.s[:, AM] = np.where(self.done, self.s[:, AM], a_cmd)

        for _ in range(cfg.substeps):
            active = ~self.done
            if not active.any():
                break
            # Closest point of approach, treating relative motion as a
            # straight line within one integration step.
            rx, ry, vx, vy = self._relative()
            vv = np.maximum(vx * vx + vy * vy, 1e-12)
            t_star = -(rx * vx + ry * vy) / vv
            passing = active & (t_star <= dt)
            if passing.any():
                tc = np.clip(t_star, 0.0, dt)
                miss = np.hypot(rx + vx * tc, ry + vy * tc)
                self.miss[passing] = miss[passing]
                self.t[passing] += tc[passing]
                self.done |= passing
                active = ~self.done

            a_t = scenarios.target_accel(self.init, self.t, cfg.n_t)
            s = self.s
            k1 = self._deriv(s, a_cmd, a_t)
            k2 = self._deriv(s + 0.5 * dt * k1, a_cmd, a_t)
            k3 = self._deriv(s + 0.5 * dt * k2, a_cmd, a_t)
            k4 = self._deriv(s + dt * k3, a_cmd, a_t)
            s_new = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.effort += np.where(active, s[:, AM] ** 2 * dt, 0.0)
            self.sat_time += np.where(active & saturated, dt, 0.0)
            self.s = np.where(active[:, None], s_new, s)
            self.t += np.where(active, dt, 0.0)

            late = ~self.done & (self.t >= cfg.t_max - 1e-9)
            if late.any():
                rx, ry, _, _ = self._relative()
                self.miss[late] = np.hypot(rx, ry)[late]
                self.timed_out |= late
                self.done |= late

    # ------------------------------------------------------------------
    def features(self) -> dict:
        """Engagement geometry that guidance laws and observations are built from."""
        cfg = self.cfg
        s = self.s
        rx, ry, vx, vy = self._relative()
        r = np.maximum(np.hypot(rx, ry), 1e-6)
        lam = np.arctan2(ry, rx)
        return {
            "t": self.t.copy(),
            "r": r,
            "lam": lam,
            "lam_dot": (rx * vy - ry * vx) / (r * r),
            "v_c": -(rx * vx + ry * vy) / r,
            "lead": wrap_angle(s[:, GM] - lam),
            "a_m": s[:, AM].copy(),
            "gamma_t": s[:, GT].copy(),
            "a_t": scenarios.target_accel(self.init, self.t, cfg.n_t),
        }

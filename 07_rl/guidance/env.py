"""Gymnasium environment for one engagement."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import rewards, scenarios
from .dynamics import BatchSim, SimConfig

OBS_KINDS = ("obs_pn", "obs_hist")
DISCRETE_LEVELS = np.linspace(-1.0, 1.0, 7)

LAM_DOT_SCALE = 0.05   # rad/s
V_C_SCALE = 1200.0     # m/s
R_SCALE = 12000.0      # m
LEAD_SCALE = np.pi / 4
OBS_CLIP = 5.0


def make_obs(kind: str, f, prev_lam_dot, prev_u):
    ld = np.clip(f["lam_dot"] / LAM_DOT_SCALE, -OBS_CLIP, OBS_CLIP)
    vc = np.clip(f["v_c"] / V_C_SCALE, -OBS_CLIP, OBS_CLIP)
    if kind == "obs_pn":
        return np.stack([ld, vc], axis=-1)
    pld = np.clip(prev_lam_dot / LAM_DOT_SCALE, -OBS_CLIP, OBS_CLIP)
    return np.stack([ld, vc, f["r"] / R_SCALE, f["lead"] / LEAD_SCALE, pld, prev_u], axis=-1)


class InterceptEnv(gym.Env):
    """Missile lateral-acceleration control against one target.

    Action: fraction of the acceleration limit, either continuous in [-1, 1]
    or one of seven discrete levels.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: SimConfig | None = None, obs_kind: str = "obs_pn",
                 reward_kind: str = "terminal", scenario=scenarios.TRAIN_MIX,
                 discrete: bool = False, gamma: float = 0.995):
        assert obs_kind in OBS_KINDS and reward_kind in rewards.REWARDS
        self.cfg = cfg or SimConfig()
        self.obs_kind = obs_kind
        self.reward_kind = reward_kind
        self.scenario = scenario
        self.discrete = discrete
        self.gamma = gamma
        dim = 2 if obs_kind == "obs_pn" else 6
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, shape=(dim,), dtype=np.float32)
        self.action_space = (spaces.Discrete(len(DISCRETE_LEVELS)) if discrete
                             else spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32))
        self.sim = None

    def _obs(self, f):
        return make_obs(self.obs_kind, f, self._prev_lam_dot, self._prev_u)[0].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        init = (options or {}).get("init")
        if init is None:
            init = scenarios.sample(self.np_random, 1, self.cfg, self.scenario)
        self.sim = BatchSim(self.cfg, init)
        self._f = self.sim.features()
        self._prev_lam_dot = self._f["lam_dot"]
        self._prev_u = np.zeros(1)
        return self._obs(self._f), {}

    def step(self, action):
        if self.discrete:
            u = np.array([DISCRETE_LEVELS[int(action)]])
        else:
            u = np.clip(np.asarray(action, dtype=float).reshape(1), -1.0, 1.0)
        f_prev = self._f
        self.sim.step(u)
        f_next = self.sim.features()
        done = self.sim.done
        reward = rewards.step_reward(self.reward_kind, f_prev, f_next, done,
                                     self.sim.init["r0"], self.cfg, self.gamma)[0]
        info = {}
        if done[0]:
            miss = self.sim.miss[0]
            hit = bool(miss < self.cfg.hit_radius)
            reward += float(rewards.terminal_reward(self.reward_kind, miss, hit))
            info = {"miss": float(miss), "hit": hit,
                    "effort": float(self.sim.effort[0])}
        self._prev_lam_dot = f_prev["lam_dot"]
        self._prev_u = u
        self._f = f_next
        timed_out = bool(self.sim.timed_out[0])
        return self._obs(f_next), float(reward), bool(done[0]) and not timed_out, timed_out, info

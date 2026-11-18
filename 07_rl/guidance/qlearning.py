"""Tabular Q-learning on the intercept problem.

Episodes are simulated in batches that share one Q-table. Every transition of
every episode produces the standard one-step update

    Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

The updates of one guidance step are applied together, so transitions in the
same step all see the table as it was before that step.

With `hold=True` a decision is made only when the state bin changes, and the
chosen action is held in between. The update then spans the whole stay in the
bin, with the reward discounted over its k steps:

    Q(s,a) += alpha * (R_k + gamma^k * max_a' Q(s',a') - Q(s,a))

A greedy policy already behaves this way (same bin, same action), so this
variant learns the value of what the greedy policy actually does. Without it,
one 0.05 s step rarely moves the state to another bin and the values of the
actions in a bin are nearly indistinguishable.
"""
from dataclasses import dataclass, asdict

import numpy as np

from . import rewards, scenarios
from .discretize import ACTIONS, greedy_action, greedy_policy, n_states, state_index
from .dynamics import BatchSim, SimConfig
from .evaluate import run, summarize


@dataclass(frozen=True)
class QConfig:
    state_kind: str = "q1"
    reward_kind: str = "los_rate"
    shaping_k: float | None = None  # per-step reward coefficient; None uses the default
    hold: bool = True               # hold each action until the state bin changes
    alpha: float = 0.5
    alpha_n0: float | None = 100.0    # if set, alpha decays as alpha * n0 / (n0 + visits(s,a))
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.8     # fraction of training over which epsilon decays
    episodes: int = 30000
    batch: int = 64
    eval_every: int = 512           # episodes between greedy evaluations
    eval_episodes: int = 300
    eval_seed: int = 9000
    seed: int = 0


def train(cfg: SimConfig, qc: QConfig, scenario=scenarios.TRAIN_MIX, snapshot_at=()):
    """Train one Q-table. Returns the table, the learning history and snapshots."""
    rng = np.random.default_rng(qc.seed)
    q = np.zeros((n_states(qc.state_kind), len(ACTIONS)))
    visits = np.zeros_like(q)
    eval_init = scenarios.sample(np.random.default_rng(qc.eval_seed), qc.eval_episodes, cfg, scenario)
    history = {k: [] for k in ("episodes", "epsilon", "hit_rate", "miss_median", "miss_mean", "train_hit_rate")}
    snapshots = {}
    pending = sorted(snapshot_at)

    def evaluate(done_eps, eps, train_hit):
        stats = summarize(run(cfg, greedy_policy(q, qc.state_kind, cfg), eval_init))
        history["episodes"].append(done_eps)
        history["epsilon"].append(eps)
        history["hit_rate"].append(stats["hit_rate"])
        history["miss_median"].append(stats["miss_median"])
        history["miss_mean"].append(stats["miss_mean"])
        history["train_hit_rate"].append(train_hit)

    evaluate(0, qc.eps_start, np.nan)
    done_eps, next_eval, recent_hits = 0, qc.eval_every, []
    while done_eps < qc.episodes:
        while pending and pending[0] <= done_eps:
            snapshots[pending.pop(0)] = q.copy()
        frac = min(done_eps / (qc.eps_decay_frac * qc.episodes), 1.0)
        eps = qc.eps_start + frac * (qc.eps_end - qc.eps_start)

        init = scenarios.sample(rng, qc.batch, cfg, scenario)
        sim = BatchSim(cfg, init)
        f = sim.features()
        s = state_index(qc.state_kind, f, cfg)

        def choose(states):
            greedy = greedy_action(q, states)
            explore = rng.random(qc.batch) < eps
            return np.where(explore, rng.integers(len(ACTIONS), size=qc.batch), greedy)

        s_dec, a_dec = s, choose(s)         # state and action of the open decision
        ret = np.zeros(qc.batch)            # discounted reward since that decision
        disc = np.ones(qc.batch)
        while not sim.done.all():
            active = ~sim.done
            sim.step(ACTIONS[a_dec])
            f2 = sim.features()
            s2 = state_index(qc.state_kind, f2, cfg)
            ended = active & sim.done
            r = rewards.step_reward(qc.reward_kind, f, f2, sim.done, init["r0"], cfg, qc.gamma,
                                    u=ACTIONS[a_dec], k=qc.shaping_k)
            miss = np.where(ended, sim.miss, 0.0)
            r = r + np.where(ended, rewards.terminal_reward(qc.reward_kind, miss, miss < cfg.hit_radius), 0.0)
            ret = ret + disc * r
            disc = disc * qc.gamma

            decide = active & (sim.done | (s2 != s_dec) | (not qc.hold))
            target = ret + disc * np.where(sim.done, 0.0, q[s2].max(axis=1))
            td = target - q[s_dec, a_dec]
            idx = (s_dec[decide], a_dec[decide])
            alpha = qc.alpha
            if qc.alpha_n0 is not None:
                alpha = qc.alpha * qc.alpha_n0 / (qc.alpha_n0 + visits[idx])
            np.add.at(q, idx, alpha * td[decide])
            np.add.at(visits, idx, 1.0)
            a_dec = np.where(decide, choose(s2), a_dec)
            s_dec = np.where(decide, s2, s_dec)
            ret = np.where(decide, 0.0, ret)
            disc = np.where(decide, 1.0, disc)
            f = f2

        recent_hits.append(np.mean(sim.miss < cfg.hit_radius))
        done_eps += qc.batch
        if done_eps >= next_eval or done_eps >= qc.episodes:
            evaluate(done_eps, eps, float(np.mean(recent_hits)))
            next_eval += qc.eval_every
            recent_hits = []

    for ep in pending:
        snapshots[ep] = q.copy()
    return {"q": q, "visits": visits, "history": {k: np.array(v) for k, v in history.items()},
            "snapshots": snapshots, "config": asdict(qc)}

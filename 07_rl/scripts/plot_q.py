"""Figures for the Q-learning experiments (E2, E3, Q-learning part of E5).

Reads results/models/q_runs.pkl and writes results/figures/q_*.png and
results/tables/q_summary.csv.
"""
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from guidance import scenarios
from guidance.discretize import (ACTIONS, N_TGO, N_X, TGO_EDGES, X_EDGES, greedy_action,
                                 greedy_policy, x_bin_centers)
from guidance.dynamics import PRESETS
from guidance.evaluate import run, summarize
from guidance.pn import pn_policy

ROOT = Path(__file__).resolve().parents[1] / "results"
FIG = ROOT / "figures"
LAST = 10   # evaluations averaged for the "final" hit rate


def load():
    with open(ROOT / "models" / "q_runs.pkl", "rb") as fh:
        runs = pickle.load(fh)
    by = defaultdict(list)
    for r in runs:
        by[(r["group"], r["label"])].append(r)
    return by


def curve(ax, runs, label, color=None, key="hit_rate"):
    """Mean over seeds with a band of one standard deviation."""
    x = runs[0]["history"]["episodes"]
    y = np.stack([r["history"][key] for r in runs])
    mean, std = y.mean(axis=0), y.std(axis=0)
    line, = ax.plot(x, mean, label=f"{label} (n={len(runs)})", color=color)
    ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2)


def pn_reference(preset="main", gains=(5, 20)):
    """PN hit rate on the same evaluation set the learning curves use."""
    cfg = PRESETS[preset]
    init = scenarios.sample(np.random.default_rng(9000), 300, cfg, scenarios.TRAIN_MIX)
    return {n: summarize(run(cfg, pn_policy(cfg, n), init))["hit_rate"] for n in gains}


def style(ax, title, ylabel="hit rate"):
    ax.set_title(title)
    ax.set_xlabel("training episodes")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def fig_learning_curve(by, ref):
    base = by[("main", "base")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    curve(axes[0], base, "Q-learning", "C0")
    for (n, hr), ls in zip(ref.items(), ("--", ":")):
        axes[0].axhline(hr, color="k", ls=ls, lw=1, label=f"PN N'={n}")
    style(axes[0], "Hit rate of the greedy policy")
    curve(axes[1], base, "Q-learning", "C0", key="miss_median")
    axes[1].set_yscale("log")
    axes[1].axhline(PRESETS["main"].hit_radius, color="r", lw=1, label="hit radius")
    axes[1].set_title("Median miss distance")
    axes[1].set_xlabel("training episodes")
    axes[1].set_ylabel("miss [m]")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    fig.suptitle("Q-learning on the main setting (state: 21 bins of PN demand, reward: los_rate)")
    fig.tight_layout()
    fig.savefig(FIG / "q_learning_curve.png", dpi=150)


def fig_groups(by, name, title, panels):
    """One panel per (panel title, [(group, label, shown label)])."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), squeeze=False)
    for ax, (ptitle, entries) in zip(axes[0], panels):
        for group, label, shown in entries:
            curve(ax, by[(group, label)], shown)
        style(ax, ptitle)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)


def fit_gain(actions, centers):
    """Least-squares slope through the origin over unsaturated bins."""
    use = (np.abs(actions) < 1.0) & (np.abs(centers) > 0)
    if use.sum() < 2:
        return np.nan, np.nan, int(use.sum())
    x, y = centers[use], actions[use]
    slope = float(x @ y / (x @ x))
    ss_res = float(np.sum((y - slope * x) ** 2))
    ss_tot = float(np.sum(y ** 2))
    return slope, 1.0 - ss_res / ss_tot, int(use.sum())


def fig_policy(by):
    """E3: greedy action of every Q1 table against the PN demand."""
    centers = x_bin_centers()
    entries = [("main", "base", "los_rate"), ("reward", "terminal", "terminal"),
               ("reward", "potential", "potential"), ("reward", "effort", "effort"),
               ("reward", "hit_only", "hit_only")]
    fig, axes = plt.subplots(1, len(entries), figsize=(4.2 * len(entries), 4), sharey=True)
    xs = np.linspace(-1, 1, 400)
    rows = []
    for ax, (group, label, shown) in zip(axes, entries):
        for n, ls in ((3, ":"), (5, "--")):
            ax.plot(xs, np.clip(n * xs, -1, 1), color="k", ls=ls, lw=1, label=f"PN N'={n}")
        for r in by[(group, label)]:
            act = ACTIONS[greedy_action(r["q"], np.arange(N_X))]
            seen = r["visits"].sum(axis=1) > 0
            ax.plot(np.arange(N_X)[seen], act[seen], "o-", ms=4, alpha=0.7, label=f"seed {r['seed']}")
            slope, r2, n_used = fit_gain(act[seen], centers[seen])
            sign_ok = float(np.mean(np.sign(act[seen]) == np.sign(centers[seen])))
            rows.append({"reward": shown, "seed": r["seed"], "sign_agreement": sign_ok,
                         "saturated_bins": int(np.sum(np.abs(act[seen]) >= 1.0)),
                         "fit_gain": slope, "fit_r2": r2, "fit_bins": n_used})
        ax.set_title(shown)
        ax.set_xlabel("state bin (PN demand x, log-spaced)")
        ax.set_xticks([0, 5, 10, 15, 20], ["-1", "-0.08", "0", "0.08", "1"])
        ax.grid(alpha=0.3)
    # PN lines are drawn against bin index, so redraw them on that axis.
    for ax in axes:
        for line in ax.lines[:2]:
            n = 3 if line.get_linestyle() == ":" else 5
            line.set_data(np.arange(N_X), np.clip(n * centers, -1, 1))
        ax.set_xlim(-0.5, N_X - 0.5)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("greedy action (fraction of limit)")
    fig.suptitle("Learned Q1 policy by reward: greedy action in each state bin")
    fig.tight_layout()
    fig.savefig(FIG / "q_policy.png", dpi=150)
    return rows


def fig_q2_policy(by):
    runs = by[("variant", "q2")]
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 3.6), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), runs):
        act = ACTIONS[greedy_action(r["q"], np.arange(N_X * N_TGO))].reshape(N_X, N_TGO)
        seen = (r["visits"].sum(axis=1) > 0).reshape(N_X, N_TGO)
        im = ax.imshow(np.where(seen, act, np.nan).T, origin="lower", aspect="auto",
                       cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(f"seed {r['seed']}")
        ax.set_xlabel("PN demand bin")
        ax.set_xticks([0, 5, 10, 15, 20], ["-1", "-0.08", "0", "0.08", "1"])
        ax.set_yticks(range(N_TGO), ["<0.5", "0.5-1", "1-2", "2-4", "4-7", ">7"])
    np.atleast_1d(axes)[0].set_ylabel("time to go [s]")
    fig.colorbar(im, ax=axes, label="greedy action")
    fig.suptitle("Q2 policy: greedy action by PN demand and time to go (blank: never visited)")
    fig.savefig(FIG / "q_policy_q2.png", dpi=150, bbox_inches="tight")


def fig_stages(by):
    """Trajectories of the same engagements at several points in training."""
    cfg = PRESETS["main"]
    r = next(r for r in by[("main", "base")] if r["seed"] == 0)
    stages = sorted(r["snapshots"].items()) + [(r["config"]["episodes"], r["q"])]
    init = scenarios.sample(np.random.default_rng(4000), 3, cfg, "bangbang")
    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 3.8), sharey=True)
    for ax, (ep, q) in zip(axes, stages):
        res = run(cfg, greedy_policy(q, "q1", cfg), init, record=True)
        tr = res["traj"]
        for j in range(3):
            k = tr["active"][:, j]
            ax.plot(tr["mx"][k, j] / 1e3, tr["my"][k, j] / 1e3, color=f"C{j}",
                    label=f"miss {res['miss'][j]:.1f} m")
            ax.plot(tr["tx"][k, j] / 1e3, tr["ty"][k, j] / 1e3, color=f"C{j}", ls="--")
        ax.set_title(f"after {ep} episodes")
        ax.set_xlabel("x [km]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("y [km]")
    fig.suptitle("Same three bang-bang engagements during training (solid: missile, dashed: target)")
    fig.tight_layout()
    fig.savefig(FIG / "q_training_stages.png", dpi=150)


def summary_table(by):
    rows = []
    for (group, label), runs in sorted(by.items()):
        final = np.array([r["history"]["hit_rate"][-LAST:].mean() for r in runs])
        best = np.array([r["history"]["hit_rate"].max() for r in runs])
        rows.append({"group": group, "label": label, "preset": runs[0]["preset"],
                     "seeds": len(runs), "final_hit_mean": final.mean(), "final_hit_std": final.std(),
                     "final_hit_min": final.min(), "final_hit_max": final.max(),
                     "best_eval_hit_mean": best.mean()})
    with open(ROOT / "tables" / "q_summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    by = load()
    ref = pn_reference()
    fig_learning_curve(by, ref)
    fig_groups(by, "q_hyperparameters.png", "Q-learning hyperparameters (main setting)", [
        ("Learning rate", [("main", "base", "decaying from 0.5")] +
         [("alpha", f"constant {a:g}", f"constant {a:g}") for a in (0.01, 0.05, 0.1, 0.5)]),
        ("Discount factor", [("gamma", "0.9", "0.9"), ("main", "base", "0.99"), ("gamma", "1", "1.0")]),
        ("Exploration schedule", [("epsilon", "decay over 40%", "decay over 40%"),
                                  ("main", "base", "decay over 80%"),
                                  ("epsilon", "decay over 100%", "decay over 100%")]),
    ])
    fig_groups(by, "q_reward_ablation.png", "Q-learning reward ablation (main setting, Q1)", [
        ("Reward", [("main", "base", "los_rate")] +
         [("reward", r, r) for r in ("terminal", "potential", "effort", "hit_only")]),
    ])
    fig_groups(by, "q_variants.png", "Q-learning variants", [
        ("Update and state (main setting)", [("main", "base", "Q1, hold"), ("variant", "no hold", "Q1, one-step"),
                                             ("variant", "q2", "Q2, hold"),
                                             ("variant", "q2 terminal", "Q2, hold, terminal reward")]),
        ("Setting", [("setting", "easy", "easy"), ("main", "base", "main"), ("setting", "hard", "hard")]),
    ])
    policy_rows = fig_policy(by)
    fig_q2_policy(by)
    fig_stages(by)
    with open(ROOT / "tables" / "q_policy_structure.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(policy_rows[0]))
        writer.writeheader()
        writer.writerows(policy_rows)

    print("PN reference on the learning-curve evaluation set:", ref)
    for row in summary_table(by):
        print(f"{row['group']:8s} {row['label']:16s} {row['preset']:5s} n={row['seeds']} "
              f"final {row['final_hit_mean']:.2f} +- {row['final_hit_std']:.2f} "
              f"[{row['final_hit_min']:.2f}, {row['final_hit_max']:.2f}]")
    for row in policy_rows:
        print(f"policy {row['reward']:9s} seed {row['seed']} sign {row['sign_agreement']:.2f} "
              f"saturated {row['saturated_bins']} gain {row['fit_gain']:.1f} r2 {row['fit_r2']:.2f} ({row['fit_bins']} bins)")


if __name__ == "__main__":
    main()

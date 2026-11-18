"""PN hit rate against the navigation constant, well beyond the usual 3 to 5.

Writes results/tables/pn_gain_sweep.csv and results/figures/pn_gain_sweep.png.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from guidance import scenarios
from guidance.dynamics import PRESETS
from guidance.evaluate import run, summarize
from guidance.pn import pn_policy

ROOT = Path(__file__).resolve().parents[1] / "results"
GAINS = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100)
N_EPISODES = 1000
SEED = 3000


def main():
    rows = []
    for preset in ("main", "hard"):
        cfg = PRESETS[preset]
        for i, sc in enumerate(scenarios.KINDS):
            init = scenarios.sample(np.random.default_rng(SEED + i), N_EPISODES, cfg, sc)
            for n in GAINS:
                stats = summarize(run(cfg, pn_policy(cfg, n), init))
                rows.append({"preset": preset, "scenario": sc, "gain": n, **stats})
    with open(ROOT / "tables" / "pn_gain_sweep.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, preset in zip(axes, ("main", "hard")):
        cfg = PRESETS[preset]
        for sc in scenarios.KINDS:
            sel = [r for r in rows if r["preset"] == preset and r["scenario"] == sc]
            ax.plot([r["gain"] for r in sel], [r["hit_rate"] for r in sel], marker="o", label=sc)
        ax.axvspan(3, 5, color="0.85", zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(GAINS, [str(g) for g in GAINS], fontsize=7)
        ax.set_xlabel("navigation constant N'")
        ax.set_title(f"{preset}: limit {cfg.a_max_g:g} g, lag {cfg.tau:g} s")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"hit rate (miss < {cfg.hit_radius:g} m)")
    axes[0].legend(title="target maneuver")
    fig.suptitle("PN hit rate vs navigation constant (shaded: conventional range 3 to 5)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "pn_gain_sweep.png", dpi=150)

    for preset in ("main", "hard"):
        print(preset)
        for sc in scenarios.KINDS:
            sel = [r for r in rows if r["preset"] == preset and r["scenario"] == sc]
            print(f"  {sc:9s} " + " ".join(f"{r['gain']}:{r['hit_rate']:.2f}" for r in sel))


if __name__ == "__main__":
    main()

"""Sample PN trajectories in the main setting, one panel per target maneuver.

Writes results/figures/pn_trajectories.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from guidance import scenarios
from guidance.dynamics import MAIN
from guidance.evaluate import run
from guidance.pn import pn_policy

ROOT = Path(__file__).resolve().parents[1] / "results"
N_GAIN = 5.0
N_SHOWN = 4


def main():
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for i, (ax, sc) in enumerate(zip(axes, scenarios.KINDS)):
        init = scenarios.sample(np.random.default_rng(2000 + i), N_SHOWN, MAIN, sc)
        res = run(MAIN, pn_policy(MAIN, N_GAIN), init, record=True)
        tr = res["traj"]
        for j in range(N_SHOWN):
            k = tr["active"][:, j]
            color = f"C{j}"
            ax.plot(tr["mx"][k, j] / 1e3, tr["my"][k, j] / 1e3, color=color,
                    label=f"miss {res['miss'][j]:.1f} m")
            ax.plot(tr["tx"][k, j] / 1e3, tr["ty"][k, j] / 1e3, color=color, ls="--")
        ax.set_title(sc)
        ax.set_xlabel("x [km]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="solid: missile, dashed: target", title_fontsize=7)
    axes[0].set_ylabel("y [km]")
    fig.suptitle(f"PN (N' = {N_GAIN:g}), limit {MAIN.a_max_g:g} g, lag {MAIN.tau:g} s, "
                 f"target maneuver {MAIN.n_t_g:g} g")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "pn_trajectories.png", dpi=150)


if __name__ == "__main__":
    main()

"""E1: sweep the acceleration limit and guidance lag to find where PN starts to miss.

Writes results/tables/pn_calibration.csv and results/figures/pn_calibration.png.
"""
import csv
import dataclasses
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from guidance import scenarios
from guidance.dynamics import SimConfig
from guidance.evaluate import run, summarize
from guidance.pn import apn_policy, pn_policy

ROOT = Path(__file__).resolve().parents[1] / "results"
A_MAX_G = (4.5, 6.0, 9.0, 15.0)
TAUS = (0.0, 0.25, 0.5)
GAINS = (3.0, 4.0, 5.0)
SCENARIOS = ("straight", "constant", "bangbang", "weave")
N_EPISODES = 500
SEED = 1000


def main():
    base = SimConfig()
    inits = {sc: scenarios.sample(np.random.default_rng(SEED + i), N_EPISODES, base, sc)
             for i, sc in enumerate(SCENARIOS)}
    rows = []
    for a_max_g in A_MAX_G:
        for tau in TAUS:
            cfg = dataclasses.replace(base, a_max_g=a_max_g, tau=tau)
            laws = {f"PN{int(n)}": pn_policy(cfg, n) for n in GAINS}
            laws["APN3"] = apn_policy(cfg, 3.0)
            for sc in SCENARIOS:
                for name, policy in laws.items():
                    stats = summarize(run(cfg, policy, inits[sc]))
                    rows.append({"a_max_g": a_max_g, "tau": tau, "scenario": sc, "law": name, **stats})

    (ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "tables" / "pn_calibration.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Best PN gain per setting: hit rate against the acceleration limit.
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(16, 3.8), sharey=True)
    for ax, sc in zip(axes, SCENARIOS):
        for tau in TAUS:
            best = [max(r["hit_rate"] for r in rows
                        if r["scenario"] == sc and r["tau"] == tau
                        and r["a_max_g"] == a and r["law"].startswith("PN"))
                    for a in A_MAX_G]
            ax.plot(A_MAX_G, best, marker="o", label=f"lag {tau:g} s")
        ax.set_title(sc)
        ax.set_xlabel("missile acceleration limit [g]")
        ax.set_xticks(A_MAX_G)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("hit rate, best PN gain")
    axes[0].set_ylim(-0.02, 1.02)
    axes[-1].legend()
    fig.suptitle(f"PN hit rate (miss < {base.hit_radius:g} m), target maneuver {base.n_t_g:g} g")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "pn_calibration.png", dpi=150)

    print(f"{'a_max':>5} {'tau':>5} {'scenario':>9} | " + " ".join(f"{k:>6}" for k in ("PN3", "PN4", "PN5", "APN3")))
    for a_max_g in A_MAX_G:
        for tau in TAUS:
            for sc in SCENARIOS:
                hr = {r["law"]: r["hit_rate"] for r in rows
                      if r["a_max_g"] == a_max_g and r["tau"] == tau and r["scenario"] == sc}
                print(f"{a_max_g:5.1f} {tau:5.2f} {sc:>9} | " + " ".join(f"{hr[k]:6.2f}" for k in ("PN3", "PN4", "PN5", "APN3")))


if __name__ == "__main__":
    main()

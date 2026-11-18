"""E2, E3 and the Q-learning part of E5: train every Q-learning run.

Writes results/models/q_runs.pkl, a list of runs. Each run holds its group,
label, seed, Q-table, visit counts, learning history and snapshots.

Usage: python scripts/train_q.py [group ...]     (no argument runs every group)

Runs are spread over all CPU cores. Set the WORKERS environment variable to
use fewer.
"""
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

from guidance.dynamics import PRESETS
from guidance.qlearning import QConfig, train

OUT = Path(__file__).resolve().parents[1] / "results" / "models" / "q_runs.pkl"
SNAPSHOTS = (0, 256, 1024, 4096)

# group -> list of (label, preset, QConfig overrides, seeds)
# The base configuration is QConfig(): los_rate reward, decaying alpha, hold.
THREE = (0, 1, 2)
GROUPS = {
    "main": [("base", "main", {}, (0, 1, 2, 3, 4))],
    # The first 2048 episodes of the base run, evaluated every 64 episodes.
    # eps_decay_frac is scaled so that epsilon follows the base schedule.
    "early": [("base", "main", {"episodes": 2048, "eval_every": 64,
                                "eps_decay_frac": 0.8 * 30000 / 2048}, (0, 1, 2, 3, 4))],
    "alpha": [(f"constant {a:g}", "main", {"alpha": a, "alpha_n0": None}, THREE)
              for a in (0.01, 0.05, 0.1, 0.5)],
    "gamma": [(f"{g:g}", "main", {"gamma": g}, THREE) for g in (0.9, 1.0)],
    "epsilon": [(f"decay over {int(f * 100)}%", "main", {"eps_decay_frac": f}, THREE)
                for f in (0.4, 1.0)],
    "reward": [(r, "main", {"reward_kind": r}, THREE)
               for r in ("hit_only", "terminal", "potential", "effort")],
    "variant": [("no hold", "main", {"hold": False}, THREE),
                ("q2", "main", {"state_kind": "q2"}, THREE),
                ("q2 terminal", "main", {"state_kind": "q2", "reward_kind": "terminal"}, THREE)],
    "setting": [("easy", "easy", {}, THREE), ("hard", "hard", {}, THREE)],
}


def job(args):
    group, label, preset, overrides, seed = args
    snaps = SNAPSHOTS if (group, seed) == ("main", 0) else ()
    out = train(PRESETS[preset], QConfig(seed=seed, **overrides), snapshot_at=snaps)
    out.update(group=group, label=label, preset=preset, seed=seed)
    return out


def main():
    wanted = sys.argv[1:] or list(GROUPS)
    jobs = [(g, label, preset, over, seed)
            for g in wanted for label, preset, over, seeds in GROUPS[g] for seed in seeds]
    runs = []
    if OUT.exists():
        with open(OUT, "rb") as fh:
            runs = [r for r in pickle.load(fh) if r["group"] not in wanted]
    workers = int(os.environ.get("WORKERS", os.cpu_count() or 1))
    with Pool(min(workers, len(jobs))) as pool:
        for i, out in enumerate(pool.imap_unordered(job, jobs), 1):
            runs.append(out)
            h = out["history"]
            print(f"[{i}/{len(jobs)}] {out['group']:8s} {out['label']:16s} seed {out['seed']} "
                  f"final hit {h['hit_rate'][-1]:.2f}  last-10 mean {h['hit_rate'][-10:].mean():.2f}",
                  flush=True)
    runs.sort(key=lambda r: (r["group"], r["label"], r["seed"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as fh:
        pickle.dump(runs, fh)


if __name__ == "__main__":
    main()

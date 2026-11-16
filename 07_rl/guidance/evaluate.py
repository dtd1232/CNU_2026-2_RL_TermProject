"""Monte Carlo evaluation of a guidance policy."""
import numpy as np

from .dynamics import BatchSim, XM, YM, XT, YT, GM, GT


def run(cfg, policy, init, record: bool = False) -> dict:
    """Fly every engagement in `init` with `policy` and return per-episode results."""
    sim = BatchSim(cfg, init)
    traj = [] if record else None
    max_steps = int(round(cfg.t_max / cfg.dt_guid)) + 1
    for _ in range(max_steps):
        if sim.done.all():
            break
        f = sim.features()
        u = policy(f)
        if record:
            traj.append({
                "t": sim.t.copy(), "active": ~sim.done,
                "mx": sim.s[:, XM].copy(), "my": sim.s[:, YM].copy(),
                "tx": sim.s[:, XT].copy(), "ty": sim.s[:, YT].copy(),
                "m_heading": sim.s[:, GM].copy(), "t_heading": sim.s[:, GT].copy(),
                "u": np.asarray(u, dtype=float).copy(), "a_m": f["a_m"],
                "lam_dot": f["lam_dot"], "v_c": f["v_c"], "r": f["r"],
            })
        sim.step(u)
    out = {
        "miss": sim.miss,
        "hit": sim.miss < cfg.hit_radius,
        "effort": sim.effort,
        "sat_frac": sim.sat_time / np.maximum(sim.t, 1e-9),
        "t_final": sim.t,
        "timed_out": sim.timed_out,
    }
    if record:
        out["traj"] = {k: np.stack([step[k] for step in traj]) for k in traj[0]}
    return out


def summarize(res: dict) -> dict:
    miss = res["miss"]
    return {
        "n": int(len(miss)),
        "hit_rate": float(np.mean(res["hit"])),
        "miss_mean": float(np.mean(miss)),
        "miss_median": float(np.median(miss)),
        "miss_p95": float(np.percentile(miss, 95)),
        "effort_mean": float(np.mean(res["effort"])),
        "sat_frac_mean": float(np.mean(res["sat_frac"])),
    }

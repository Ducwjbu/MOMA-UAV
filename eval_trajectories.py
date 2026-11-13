"""
eval_trajectories.py
====================
What the 3-UAV policies do, step by step, at the three corner weights (QoS, energy, fairness), on the same
held-out episodes as eval_checkpoint.py (episode k seeded by seed_base + k). Records every UAV's position after
reset and after each step, its commanded power and bandwidth shares, and the step's coverage and reward vector,
for the figures (trajectories) and the behaviour numbers (speed, distance to the formation, coverage) of the
report.

Writes <out>.csv, one row per (method, seed, corner, episode, t, uav); t = -1 is the position after reset.
Columns: x, y, h, power (action), bw_embb, bw_urllc, bw_mmtc (sum of the actor's bandwidth outputs over the
slice's 3 demand areas, before the env scales them; nan at t = -1), coverage, qos, energy, fair (the team's,
repeated for each UAV), dist_vertex (distance to the UAV's vertex of the r=600 triangle of the prior, in UAV id
order, also for the methods without the prior; that overstates their distance when they hold the triangle with
the vertices swapped, so plot_results.py also reports dist_formation_m, with the best vertex per UAV per episode).

Run (checkpoints are not in git; the default glob is the runs/ layout of the course-report runs):
  py eval_trajectories.py --out results-v1.1/eval_test/trajectories
Role test of the thesis pilot B1 (any method name maps to config/train/moma_ac_<name>.yaml):
  py eval_trajectories.py --methods ma3_prior ma3_prior_pa --roles --seed_base 100000 --out evals/roles
  (checkpoint_step_300000.pth for the pilots: --checkpoint_glob "runs/saved_models_moma_ac_{m}/seed_*/checkpoints/
  checkpoint_step_300000.pth")
"""

import argparse
import glob
import os
import re

os.environ.setdefault("MOMA_DEVICE", "cpu")

import numpy as np
import pandas as pd

import agents_moma_ac
from eval_checkpoint import load_trainer
from prior import formation_vertices

METHODS = {   # name -> train config
    "ma3": "config/train/moma_ac_ma3.yaml",
    "ma3_g90": "config/train/moma_ac_ma3_g90.yaml",
    "ma3_prior_g99": "config/train/moma_ac_ma3_prior_g99.yaml",
    "ma3_prior": "config/train/moma_ac_ma3_prior.yaml",
}
CORNERS = {"qos": [1, 0, 0], "energy": [0, 1, 0], "fair": [0, 0, 1]}
CHECKPOINT = "runs/saved_models_moma_ac_{m}/seed_*/checkpoints/checkpoint_step_1000000.pth"


def record_episodes(trainer, omega, n_episodes, seed_base, steps, radius):
    """Run eval_omega on each episode with env.reset and env.step wrapped to log the team."""
    env = trainer.eval_env
    uids = sorted(env.uavs)
    vertices = formation_vertices(env, radius, len(uids))
    rows, state = [], {"ep": -1, "t": -1}
    inner_step, inner_reset = env.step, env.reset

    def log(actions, info):
        for i, u in enumerate(uids):
            p = env.uavs[u].position
            a = None if actions is None else np.asarray(actions[u], dtype=float)
            rows.append({
                "ep": state["ep"], "t": state["t"], "uav": i, "x": p[0], "y": p[1], "h": p[2],
                "power": np.nan if a is None else a[3],
                "bw_embb": np.nan if a is None else a[4:7].sum(),
                "bw_urllc": np.nan if a is None else a[7:10].sum(),
                "bw_mmtc": np.nan if a is None else a[10:13].sum(),
                "coverage": np.nan if info is None else info["coverage"],
                "qos": np.nan if info is None else info["reward_vector"][0],
                "energy": np.nan if info is None else info["reward_vector"][1],
                "fair": np.nan if info is None else info["reward_vector"][2],
                "dist_vertex": float(np.linalg.norm(p - vertices[i])),
            })

    def reset(*args, **kwargs):
        out = inner_reset(*args, **kwargs)
        state["ep"] += 1
        state["t"] = -1
        log(None, None)
        return out

    def step(actions):
        out = inner_step(actions)
        state["t"] += 1
        log(actions, out[3])
        return out

    env.reset, env.step = reset, step
    default_seed = agents_moma_ac.EVAL_SEED
    try:
        for k in range(n_episodes):
            agents_moma_ac.EVAL_SEED = seed_base + k
            trainer.agent.eval_omega(env, np.array(omega, np.float32), 1, steps)
    finally:
        agents_moma_ac.EVAL_SEED = default_seed
        env.reset, env.step = inner_reset, inner_step
    return rows


def main():
    parser = argparse.ArgumentParser(description="Log 3-UAV trajectories at the corner weights")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--checkpoint_glob", default=CHECKPOINT, help="{m} is replaced by the method name")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--seed_base", type=int, default=200000)
    parser.add_argument("--radius", type=float, default=600.0, help="Triangle for dist_vertex")
    parser.add_argument("--roles", action="store_true",
                        help="Also the role test: corner u<i>_energy gives UAV i the energy corner and the others "
                             "the QoS corner (one omega per UAV), to see whether UAV i alone saves energy")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    corners = dict(CORNERS)
    if args.roles:
        for i in range(3):
            w = np.tile(np.array(CORNERS["qos"], np.float32), (3, 1))
            w[i] = CORNERS["energy"]
            corners[f"u{i}_energy"] = w

    frames = []
    for m in args.methods:
        files = sorted(glob.glob(args.checkpoint_glob.format(m=m)))
        if not files:
            parser.error(f"no checkpoint for {m}: {args.checkpoint_glob.format(m=m)}")
        for f in files:
            seed = int(re.search(r"seed_(\d+)", f.replace("\\", "/")).group(1))
            trainer = load_trainer(METHODS.get(m, f"config/train/moma_ac_{m}.yaml"), f)
            for cname, omega in corners.items():
                rows = record_episodes(trainer, omega, args.n_episodes, args.seed_base,
                                       trainer.cfg["eval_steps"], args.radius)
                df = pd.DataFrame(rows)
                df.insert(0, "corner", cname)
                df.insert(0, "seed", seed)
                df.insert(0, "method", m)
                frames.append(df)
            print(f"{m} seed {seed} done", flush=True)
    df = pd.concat(frames)
    df.to_csv(f"{args.out}.csv", index=False, float_format="%.4g")
    if args.roles:
        print(role_summary(df).to_string(float_format="%.3f"))


def role_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per method, corner and UAV, from step 3 on: mean commanded power and ground speed (m/s), and the team's
    qos and energy. In the role test a UAV with a role of its own should differ from its teammates"""
    df = df.sort_values(["method", "seed", "corner", "ep", "uav", "t"])
    key = ["method", "seed", "corner", "ep", "uav"]
    df["speed"] = np.hypot(df.groupby(key)["x"].diff(), df.groupby(key)["y"].diff()) / 60.0
    on = df[df.t >= 3]
    return on.groupby(["method", "corner", "uav"])[["power", "speed", "qos", "energy"]].mean().unstack("uav")


if __name__ == "__main__":
    main()

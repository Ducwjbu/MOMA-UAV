"""
eval_heuristics.py
==================
The no-training baseline family for 3 UAVs: the r-triangle of the formation prior (radius 600 m, centred on the
fly range, top altitude), flown in 10 fixed ways. The UAVs start from the env's random positions like the agents,
and fly to the triangle. Each configuration sets:
  - movement: hold the triangle (dtheta 0), rotate it by dtheta degrees per step about the centre, or oscillate
    by +-dtheta/2 (cruising costs less than hovering, Zeng et al. 2019);
  - power: 1 or 0.05 of the maximum;
  - bandwidth: even over the 9 demand areas, or "urllc", 3/15 to each URLLC area and 1/15 to each other.
The family's front is the set of these 10 points: its hv, and eu_front picks the best configuration per omega.

Same episodes as eval_checkpoint.py with the same --seed_base and --n_episodes, so the outputs pair with its
.npy files in aggregate_seeds.py (--family).

Writes:
  <out>.npy      reward vectors (n_episodes, 10, 3), configuration order as in CONFIGS
  <out>_sla.npy  share of UEs meeting their full SLA per slice (eMBB, URLLC, mMTC), same shape
  <out>.csv      one row per configuration: mean reward vector, speed, closest pair, URLLC SLA, hv and eum alone

v2 (an env with an objectives section, e.g. config/environment/v2.yaml; docs/v2/plan.md TIP 8): the family is the
k-means hover instead, 12 configurations. The serving positions hover at the demand prior's vertices (prior.attach,
mode demand: the weighted k-means of the episode's expected demand, recomputed when a position is left empty or
filled), with power 1, 0.5 or 0.2 and slice shares even, or 0.6 to one slice and 0.2 to each other. --n_serving
fixes the serving positions of the shift (the family is per N, like the policies), --embb_streams the load.

Run:
  py eval_heuristics.py --seed_base 200000 --out evals/heuristics
  py eval_heuristics.py --env_config config/environment/v2.yaml --n_serving 6 --out evals/kmeans_n6
"""

import argparse
import contextlib
import io

import numpy as np
import pandas as pd

import prior
from agents_moma_ac import SLA_KEYS, nan_mean, sla_of
from environment import NetworkSlicingEnv
from eval_checkpoint import env_config_for, eval_omegas, front_metrics
from train_moma_ac import HV_REF_POINT, V11_OBJECTIVES, _hv_3d, front_hv

URLLC_BW = np.array([1, 1, 1, 3, 3, 3, 1, 1, 1], float) / 15   # DA order: eMBB, URLLC, mMTC x Near, Medium, Far
EVEN_BW = np.full(9, 1 / 9)

# (radius m, dtheta deg per step, power, bandwidth, mode)
CONFIGS = [(600, 0, 1.0, "even", "rotate"), (600, 0, 1.0, "urllc", "rotate"),
           (600, 60, 1.0, "even", "rotate"), (600, 81, 1.0, "even", "rotate"),
           (600, 105, 1.0, "even", "rotate"), (600, 81, 1.0, "urllc", "rotate"),
           (600, 81, 1.0, "even", "osc"), (600, 81, 1.0, "urllc", "osc"),
           (600, 81, 0.05, "even", "rotate"), (600, 105, 0.05, "even", "rotate")]


# v2 k-means hover: (power, slice shares eMBB, URLLC, mMTC)
SHARES = {"even": (1 / 3, 1 / 3, 1 / 3), "embb": (0.6, 0.2, 0.2), "urllc": (0.2, 0.6, 0.2), "mmtc": (0.2, 0.2, 0.6)}
KMEANS_CONFIGS = [(p, sh) for p in (1.0, 0.5, 0.2) for sh in SHARES]


def triangle(env, r: float, theta: float) -> np.ndarray:
    """Vertices (3, 3) of the r-triangle rotated by theta about the centre of the fly range, at the top altitude."""
    cx, cy = sum(env.uav_fly_range_x) / 2, sum(env.uav_fly_range_y) / 2
    a = np.pi / 2 + theta + 2 * np.pi * np.arange(3) / 3
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.full(3, float(env.uav_fly_range_h[1]))], 1)


def run(env, r, dtheta_deg, power, bw, mode, n_episodes, seed_base, steps):
    uids = sorted(env.uavs)
    step = env.uavs[uids[0]].velocity_max * env.T_L
    vstep = env.velocity_max_vertical * env.T_L
    dth = np.deg2rad(dtheta_deg)
    out, sla, speed, dmin = [], [], [], []
    for ep in range(n_episodes):
        np.random.seed(seed_base + ep)
        env.reset()
        rv, ep_sla = [], []
        for t in range(steps):
            # rotate: theta = (t + 1) dtheta; osc: theta alternates +-dtheta / 2
            th = (t + 1) * dth if mode == "rotate" else (dth / 2 if t % 2 == 0 else -dth / 2)
            tg = triangle(env, r, th)
            acts = {}
            for i, u in enumerate(uids):
                pos = env.uavs[u].position
                a = np.zeros(13, np.float32)
                a[0:2] = (tg[i, :2] - pos[:2]) / step      # The env caps the horizontal move at velocity_max * T_L
                a[2] = np.clip((tg[i, 2] - pos[2]) / vstep, -1, 1)
                a[3] = power
                a[4:13] = bw
                acts[u] = a
            before = np.array([env.uavs[u].position.copy() for u in uids])
            _, _, done, info = env.step(acts)
            after = np.array([env.uavs[u].position for u in uids])
            speed.append(np.linalg.norm((after - before)[:, :2], axis=1).mean() / env.T_L)
            dmin.append(info["min_uav_distance"])
            rv.append(info["reward_vector"])
            ep_sla.append(sla_of(info))
            if done:
                break
        out.append(np.mean(rv, 0))
        sla.append(nan_mean(ep_sla))
    return np.array(out), np.array(sla), float(np.mean(speed)), float(np.mean(dmin))


def run_kmeans(env, power, shares, n_episodes, seed_base, steps):
    """v2: every position hovers at its demand-prior vertex (env.step is wrapped by prior.attach, so a zero movement
    action is the prior's move); one power level and fixed slice shares"""
    uids = sorted(env.uavs)
    action = np.concatenate([[0.0, 0.0, 0.0, power], shares]).astype(np.float32)
    out, sla, speed, dmin = [], [], [], []
    for ep in range(n_episodes):
        np.random.seed(seed_base + ep)
        env.reset()
        rv, ep_sla = [], []
        for t in range(steps):
            before = np.array([env.uavs[u].position.copy() for u in uids])
            _, _, done, info = env.step({u: action for u in uids})
            after = np.array([env.uavs[u].position for u in uids])
            act = [i for i, u in enumerate(uids) if env.uavs[u].active]
            if act:
                speed.append(np.linalg.norm((after - before)[act, :2], axis=1).mean() / env.T_L)
            if np.isfinite(info["min_uav_distance"]):
                dmin.append(info["min_uav_distance"])
            rv.append(info["reward_vector"])
            ep_sla.append(sla_of(info))
            if done:
                break
        out.append(np.mean(rv, 0))
        sla.append(nan_mean(ep_sla))
    return np.array(out), np.array(sla), float(np.mean(speed)), float(np.mean(dmin)) if dmin else np.nan


def main_v2(env, args):
    names = list(env.objective_names)
    omegas = eval_omegas(args.n_omega, len(names))
    prior.attach(env, sorted(env.uavs), 600.0, "demand")
    steps = args.steps if args.steps_given else int(env.shift["duration_steps"]) if env.shift else args.steps
    rows, results, slas = [], [], []
    for p, sh in KMEANS_CONFIGS:
        x, s, v, d = run_kmeans(env, p, np.array(SHARES[sh]), args.n_episodes, args.seed_base, steps)
        m = x.mean(0)
        alone = front_metrics(np.tile(m, (len(omegas), 1)), omegas, names)
        row = {"name": f"kmeans_p{p}_{sh}", **{n: float(v_) for n, v_ in zip(names, m)},
               "speed_mps": v, "min_uav_distance": d, "hv_alone": alone["hv"], "eum_alone": alone["eum"],
               "n_serving": args.n_serving, "embb_streams": args.embb_streams}
        row.update({f"{k}_share": float(val) for k, val in zip(SLA_KEYS, nan_mean(s))})
        rows.append(row)
        results.append(x)
        slas.append(s)
        print({k: round(val, 3) if isinstance(val, float) else val for k, val in row.items()}, flush=True)
    results, slas = np.stack(results, 1), np.stack(slas, 1)
    np.save(f"{args.out}.npy", results)
    np.save(f"{args.out}_sla.npy", slas)
    pd.DataFrame(rows).to_csv(f"{args.out}.csv", index=False)
    front = results.mean(0)
    print(f"family hv {front_hv(front, names):.4f}, eu_front {(omegas @ front.T).max(axis=1).mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the 3-UAV heuristic family on held-out episodes")
    parser.add_argument("--env_config", default="config/environment/ma3.yaml")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--seed_base", type=int, default=100000, help="Episode k uses seed_base + k")
    parser.add_argument("--steps", type=int, default=None,
                        help="Episode length; default 18 (eval_steps of the ma3 configs), a whole shift in v2")
    parser.add_argument("--n_omega", type=int, default=30)
    parser.add_argument("--n_serving", type=int, default=None, help="v2: serving positions of the shift fixed at N")
    parser.add_argument("--embb_streams", type=float, default=None, help="v2: mean eMBB streams per UE (load level)")
    parser.add_argument("--out", required=True, help="Output prefix")
    args = parser.parse_args()
    args.steps_given = args.steps is not None
    args.steps = 18 if args.steps is None else args.steps

    with contextlib.redirect_stdout(io.StringIO()):
        env = NetworkSlicingEnv(config_path=env_config_for(None, args.env_config, args.n_serving, args.embb_streams))
    if list(getattr(env, "objective_names", V11_OBJECTIVES)) != V11_OBJECTIVES:
        return main_v2(env, args)
    omegas = eval_omegas(args.n_omega)
    rows, results, slas = [], [], []
    for r, dth, p, bwn, mode in CONFIGS:
        x, s, v, d = run(env, r, dth, p, URLLC_BW if bwn == "urllc" else EVEN_BW, mode,
                         args.n_episodes, args.seed_base, args.steps)
        m = x.mean(0)
        alone = front_metrics(np.tile(m, (len(omegas), 1)), omegas)
        row = {"name": f"r{r}_d{dth}_p{p}_{bwn}_{mode}", "qos": m[0], "energy": m[1], "fair": m[2],
               "speed_mps": v, "min_uav_distance": d, "hv_alone": alone["hv"], "eum_alone": alone["eum"]}
        row.update({k: float(val) for k, val in zip(SLA_KEYS, nan_mean(s))})
        rows.append(row)
        results.append(x)
        slas.append(s)
        print({k: round(val, 3) if isinstance(val, float) else val for k, val in row.items()}, flush=True)
    results, slas = np.stack(results, 1), np.stack(slas, 1)
    np.save(f"{args.out}.npy", results)
    np.save(f"{args.out}_sla.npy", slas)
    pd.DataFrame(rows).to_csv(f"{args.out}.csv", index=False)
    front = results.mean(0)
    print(f"family hv {_hv_3d(front, HV_REF_POINT):.4f}, eu_front {(omegas @ front.T).max(axis=1).mean():.4f}")


if __name__ == "__main__":
    main()

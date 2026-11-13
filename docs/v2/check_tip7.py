"""
check_tip7.py (TIP 7 of docs/v2/plan.md)
========================================
The formation prior with no residual (every movement action 0, power 1, even slice shares) on
config/environment/v2.yaml, with the shift's serving positions fixed at S = 2, 3, 6 and 10, 3 validation episodes
(100000+) of 60 steps each. Two prior modes:
  ngon     the regular S-gon of radius 600 m around the middle (the v1.1 prior, made for S = 3)
  demand   the weighted k-means of the episode's expected demand for the positions serving (prior.demand_vertices)

Reported per S and mode, from step 6 on (the UAVs reach their vertices within 3 steps): coverage, the three SLA
objectives (sla_embb, sla_urllc, sla_mmtc, mean over steps) and their mean, and the time to compute the k-means.
PASS (set before the run): over the four S, demand has a higher mean coverage than ngon, and a mean SLA objective
at least as high.

Run from the repo root: py docs/v2/check_tip7.py
"""
import contextlib
import io
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import yaml

with contextlib.redirect_stdout(io.StringIO()):
    from environment import NetworkSlicingEnv
import prior

CONFIG = "config/environment/v2.yaml"
SLICES = ("embb", "urllc", "mmtc")


def config_with_s(S):
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["shift"]["n_serving"] = [S, S]
    path = os.path.join(tempfile.mkdtemp(prefix="check_tip7_"), f"v2_s{S}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def run(S, mode, n_ep=3, steps=60, seed_base=100000, skip=5):
    with contextlib.redirect_stdout(io.StringIO()):
        env = NetworkSlicingEnv(config_path=config_with_s(S))
    uids = sorted(env.uavs)
    prior.attach(env, uids, 600.0, mode)
    n_share = env.act_dim - 4
    action = np.concatenate([[0.0, 0.0, 0.0, 1.0], np.full(n_share, 1.0 / n_share)])
    names = env.objective_names
    cov, sla, t_kmeans = [], [], []
    for ep in range(n_ep):
        np.random.seed(seed_base + ep)
        with contextlib.redirect_stdout(io.StringIO()):
            env.reset()
        for t in range(steps):
            t0 = time.perf_counter()
            env.prior_targets()   # The wrapped step calls it again; here only to time a fresh k-means
            if t == 0:
                t_kmeans.append(time.perf_counter() - t0)
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, done, info = env.step({u: action for u in uids})
            if t >= skip:
                cov.append(info["coverage"])
                rv = info["reward_vector"]
                sla.append([rv[names.index(f"sla_{s}")] for s in SLICES])
            if done:
                break
    sla = np.array(sla)
    return float(np.mean(cov)), sla.mean(0), float(np.mean(t_kmeans))


def main():
    rows = {}
    print("  S  mode     coverage   sla_embb  sla_urllc  sla_mmtc   SLA mean   k-means s")
    for S in (2, 3, 6, 10):
        for mode in ("ngon", "demand"):
            cov, sla, tk = run(S, mode)
            rows[(S, mode)] = (cov, sla.mean())
            print(f"{S:3d}  {mode:7s}  {cov:8.3f}   {sla[0]:8.3f}  {sla[1]:9.3f}  {sla[2]:8.3f}   {sla.mean():8.3f}"
                  f"   {tk:9.3f}", flush=True)
    cov_d = np.mean([rows[(S, 'demand')][0] for S in (2, 3, 6, 10)])
    cov_n = np.mean([rows[(S, 'ngon')][0] for S in (2, 3, 6, 10)])
    sla_d = np.mean([rows[(S, 'demand')][1] for S in (2, 3, 6, 10)])
    sla_n = np.mean([rows[(S, 'ngon')][1] for S in (2, 3, 6, 10)])
    print(f"\nmean over S: coverage demand {cov_d:.3f} / ngon {cov_n:.3f}; SLA mean demand {sla_d:.3f} / ngon {sla_n:.3f}")
    ok = cov_d > cov_n and sla_d >= sla_n
    print("TIP 7 check:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

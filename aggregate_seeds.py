"""
aggregate_seeds.py
==================
Final numbers over training seeds, from the per-episode reward vectors that eval_checkpoint.py saves
(<out>_actor.npy, shape (n_episodes, n_omega, 3)). Every file must come from the same eval episodes
(same --seed_base and --n_episodes) and the same omegas.

A method's score is the mean over seeds of each seed's front metric (hv, eum, eu_front), where a seed's front
is its reward vectors averaged over the eval episodes. The 95% CI is a two-level bootstrap: resample the seeds,
then the eval episodes, so it covers both the variance between training runs and the noise of the eval
(Agarwal et al., "Deep RL at the Edge of the Statistical Precipice", NeurIPS 2021, use the same idea of a
stratified bootstrap over runs). The episodes are resampled with the same indices for every method, so
differences between methods are paired on episodes; seeds are resampled independently per method, since
seed i of two methods are unrelated training runs.

Also reported against each --baseline: the difference of the means with its CI, and the probability of
improvement P(X > Y) over pairs of seeds (Agarwal et al. 2021, Sec. 5), which is 1 when every seed of the
method beats every seed of the baseline.

A --family is a fixed set of non-learning policies, one file (n_episodes, n_policies, 3) with no seeds. Its
hv is that of the set of points and its eu_front picks the best policy at each eval omega; eum is undefined.

Run:
  py aggregate_seeds.py --method ma3 "evals/ma3_s*_actor.npy" --method ma3_prior "evals/ma3_prior_s*_actor.npy" \
      --family heuristics evals/heur.npy --baseline ma3 --baseline heuristics --out agg
"""

import argparse
import glob

import numpy as np
import pandas as pd

from eval_checkpoint import eval_omegas
from train_moma_ac import HV_REF_POINT, _hv_3d

METRICS = ("hv", "eum", "eu_front")
CORNERS = ("qos", "energy", "fair")


def seed_metrics(front: np.ndarray, omegas: np.ndarray, family: bool) -> dict:
    """front: mean reward vectors (n_points, 3). For an agent point i is its policy at omegas[i]."""
    return {
        "hv": float(_hv_3d(front, HV_REF_POINT)),
        "eum": np.nan if family else float((front * omegas).sum(axis=1).mean()),
        "eu_front": float((omegas @ front.T).max(axis=1).mean()),
    }


def method_score(runs: list, omegas: np.ndarray, family: bool, seed_idx=None, ep_idx=None) -> dict:
    """Mean over the (resampled) seeds of each seed's metrics on the (resampled) episodes."""
    seed_idx = range(len(runs)) if seed_idx is None else seed_idx
    rows = []
    for s in seed_idx:
        r = runs[s] if ep_idx is None else runs[s][ep_idx]
        rows.append(seed_metrics(r.mean(axis=0), omegas, family))
    return {m: float(np.mean([row[m] for row in rows])) for m in METRICS}


def main():
    parser = argparse.ArgumentParser(description="Aggregate eval_checkpoint.py outputs over training seeds")
    parser.add_argument("--method", nargs=2, action="append", default=[], metavar=("NAME", "GLOB"),
                        help="A trained method: one .npy per seed (quote the glob)")
    parser.add_argument("--family", nargs=2, action="append", default=[], metavar=("NAME", "NPY"),
                        help="A set of non-learning policies in one .npy, shape (n_episodes, n_policies, 3)")
    parser.add_argument("--baseline", action="append", default=[], help="Name to report differences against")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output prefix: <out>.csv and <out>_seeds.csv")
    args = parser.parse_args()

    methods = {}   # name -> (list of arrays (n_ep, n_points, 3), is_family, files)
    for name, pattern in args.method:
        files = sorted(glob.glob(pattern))
        if not files:
            parser.error(f"--method {name}: no file matches {pattern}")
        methods[name] = ([np.load(f) for f in files], False, files)
    for name, path in args.family:
        methods[name] = ([np.load(path)], True, [path])
    for b in args.baseline:
        if b not in methods:
            parser.error(f"--baseline {b} is not a --method or --family")

    n_ep = {runs[0].shape[0] for runs, _, _ in methods.values()}
    n_omega = {runs[0].shape[1] for runs, fam, _ in methods.values() if not fam}
    if len(n_ep) != 1 or len(n_omega) > 1 or any(r.shape[0] != next(iter(n_ep)) for runs, _, _ in methods.values() for r in runs):
        parser.error(f"All files need the same eval episodes and omegas (episodes {n_ep}, omegas {n_omega})")
    n_ep = n_ep.pop()
    omegas = eval_omegas(n_omega.pop() if n_omega else 30)

    # Point estimates and per-seed values
    seed_rows, point = [], {}
    for name, (runs, fam, files) in methods.items():
        point[name] = method_score(runs, omegas, fam)
        for f, r in zip(files, runs):
            row = {"method": name, "file": f, **seed_metrics(r.mean(axis=0), omegas, fam)}
            if not fam:
                for c, cname in enumerate(CORNERS):
                    for o, oname in enumerate(CORNERS):
                        row[f"w_{cname}_{oname}"] = float(r[:, c, o].mean())   # Corner weight c, objective o
            seed_rows.append(row)

    # Two-level bootstrap with episode indices shared by all methods
    rng = np.random.default_rng(args.seed)
    boot = {name: {m: [] for m in METRICS} for name in methods}
    diff = {(name, b): {m: [] for m in METRICS} for name in methods for b in args.baseline if b != name}
    for _ in range(args.n_boot):
        ep_idx = rng.integers(0, n_ep, n_ep)
        scores = {}
        for name, (runs, fam, _) in methods.items():
            seed_idx = rng.integers(0, len(runs), len(runs))
            scores[name] = method_score(runs, omegas, fam, seed_idx, ep_idx)
            for m in METRICS:
                boot[name][m].append(scores[name][m])
        for (name, b), d in diff.items():
            for m in METRICS:
                d[m].append(scores[name][m] - scores[b][m])

    seeds_df = pd.DataFrame(seed_rows)
    rows = []
    for name, (runs, fam, _) in methods.items():
        row = {"method": name, "n_seeds": len(runs), "family": fam}
        for m in METRICS:
            row[m] = point[name][m]
            row[f"{m}_lo"], row[f"{m}_hi"] = np.nanpercentile(boot[name][m], [2.5, 97.5]) if not (fam and m == "eum") else (np.nan, np.nan)
            row[f"{m}_seed_std"] = float(seeds_df.loc[seeds_df.method == name, m].std(ddof=1)) if len(runs) > 1 else np.nan
        for b in args.baseline:
            if b == name:
                continue
            for m in METRICS:
                d = np.array(diff[(name, b)][m])
                row[f"d_{m}_vs_{b}"] = point[name][m] - point[b][m]
                row[f"d_{m}_vs_{b}_lo"], row[f"d_{m}_vs_{b}_hi"] = (np.nanpercentile(d, [2.5, 97.5])
                                                                    if np.isfinite(d).any() else (np.nan, np.nan))
            x = seeds_df.loc[seeds_df.method == name, "hv"].to_numpy()
            y = seeds_df.loc[seeds_df.method == b, "hv"].to_numpy()
            row[f"p_improve_hv_vs_{b}"] = float((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(f"{args.out}.csv", index=False)
    seeds_df.to_csv(f"{args.out}_seeds.csv", index=False)
    show = ["method", "n_seeds", "hv", "hv_lo", "hv_hi", "hv_seed_std", "eum", "eum_lo", "eum_hi", "eu_front"]
    show += [c for c in out.columns if c.startswith("d_hv_vs_") or c.startswith("p_improve")]
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.4f}".format):
        print(out[show].to_string(index=False))


if __name__ == "__main__":
    main()

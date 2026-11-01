"""
compare_runs.py
===============
So sanh cac variant MOMA-AC tu eval_metrics.csv: mean +- std qua cac seed.
Chay:
  py compare_runs.py saved_models_moma_ac_random saved_models_moma_ac_gpi_ls saved_models_moma_ac_gpi_pd

Moi thu muc la save_dir cua mot variant, chua seed_*/eval_metrics.csv
(hoac eval_metrics.csv truc tiep neu chay khong co --seed).

In bang ket qua tai moc eval cuoi ma moi seed deu da chay toi, va luu bang
day du theo tung moc eval vao --out.
"""

import argparse
import glob
import os
import pandas as pd

METRICS = ["hv", "eum", "eu_front", "n_pareto",
           "hv_actor", "eum_actor", "eu_front_actor", "n_pareto_actor"]


def load_variant(run_dir: str) -> list:
    paths = glob.glob(os.path.join(run_dir, "seed_*", "eval_metrics.csv"))
    if not paths:
        paths = glob.glob(os.path.join(run_dir, "eval_metrics.csv"))
    return [pd.read_csv(p) for p in paths]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare MOMA-AC runs")
    parser.add_argument("run_dirs", nargs="+", help="save_dir of each variant")
    parser.add_argument("--out", default="comparison.csv", help="Output table with every eval step")
    args = parser.parse_args()

    final_rows, all_rows = [], []
    for run_dir in args.run_dirs:
        runs = load_variant(run_dir)
        if not runs:
            print(f"[WARN] No eval_metrics.csv under {run_dir}")
            continue
        label = os.path.basename(os.path.normpath(run_dir)).replace("saved_models_moma_ac_", "")
        combined = pd.concat(runs)
        keys = [k for k in METRICS if k in combined.columns]
        stats = combined.groupby("step")[keys].agg(["mean", "std"])
        seeds = combined.groupby("step").size()

        for step in stats.index:
            row = {"variant": label, "step": int(step), "seeds": int(seeds[step])}
            for k in keys:
                row[f"{k}_mean"] = stats.loc[step, (k, "mean")]
                row[f"{k}_std"]  = stats.loc[step, (k, "std")]
            all_rows.append(row)

        # Last step reached by every seed, so all seeds are compared at the same point
        step = max(s for s in seeds.index if seeds[s] == len(runs)) if (seeds == len(runs)).any() else stats.index[-1]
        row = {"variant": label, "seeds": int(seeds[step]), "step": int(step)}
        for k in keys:
            row[k] = f"{stats.loc[step, (k, 'mean')]:.4f} ± {stats.loc[step, (k, 'std')]:.4f}"
        final_rows.append(row)

    if not final_rows:
        print("[WARN] Nothing to compare")
    else:
        pd.DataFrame(all_rows).to_csv(args.out, index=False)
        print(pd.DataFrame(final_rows).fillna("-").T.to_string(header=False))
        print(f"Full table: {args.out}")

"""
check_tip8.py (TIP 8 of docs/v2/plan.md)
========================================
Checks of the n-objective evaluation (criteria set before the run):

  (a) HV in 3D: the trainer's HV (_hv_3d, a z-sweep) equals an independent exact algorithm, WFG (While, Bradstreet
      & Barone, "A Fast Way of Calculating Exact Hypervolumes", IEEE Trans. Evol. Comput. 16(1):86-95, 2012), to
      1e-9 on every front of results-v1.1 (every pareto_points.csv: each run, eval step and policy).
  (b) HV in 4D: _hv_nd (used for the v2 vector) equals WFG to 1e-9 on 300 random fronts of 1-30 points in the
      v2 ranges (SLA objectives in [0, 1], energy in [-1, 0]) at the v2 reference point; equals inclusion-exclusion
      (exact, a third algorithm) to 1e-9 on the fronts of at most 10 points; and lies within the 99.9% interval of a
      Monte Carlo estimate (200k points) on 5 fronts.
  (c) A controller holding a fixed omega (eval_controller.FixedOmega) reproduces eval_checkpoint.py: the episode
      mean reward vectors are equal (to 1e-12) at the corner weights, on 2 validation episodes (100000+) of
      (the first runs differed by 3e-8 to 6e-8 with equal per-step rewards: eval_omega averages float32 rows, and
      numpy's float32 sum rounds differently on a column-major array; the controller's rows are now averaged the
      same way, row-major float32)
        - the v1.1 ma3_prior checkpoint (seed 0, 1M steps; runs/ is not in git, skipped when absent), and
        - a v2 smoke checkpoint at N = 3 serving positions, 10 steps (trained by this script in about 2 min if
          runs/smoke_v2_tip8 is absent).

WFG here: hv(S) = sum_i exclhv(p_i, {p_i+1, ...}), exclhv(p, S) = inclhv(p) - hv(nondominated(limit(S, p))), with
limit(q, p) = min(q, p) componentwise (maximization), points sorted on the last objective. No slicing or bounding
tricks: exact and independent of the sweep code.

Run from the repo root: py docs/v2/check_tip8.py
"""
import contextlib
import glob
import io
import itertools
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    from train_moma_ac import HV_REF_POINT, V2_OBJECTIVES, _hv_3d, _hv_nd, hv_ref_point

TOL = 1e-9


def nondominated(pts: np.ndarray) -> np.ndarray:
    keep = []
    for i, p in enumerate(pts):
        dom = False
        for j, q in enumerate(pts):
            if j != i and np.all(q >= p) and (np.any(q > p) or j < i):   # Duplicates: keep the first
                dom = True
                break
        if not dom:
            keep.append(i)
    return pts[keep]


def wfg(pts: np.ndarray, ref: np.ndarray) -> float:
    pts = np.asarray(pts, float)
    pts = pts[np.all(pts > ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = nondominated(pts)
    pts = pts[np.argsort(-pts[:, -1], kind="stable")]
    total = 0.0
    for i, p in enumerate(pts):
        incl = float(np.prod(p - ref))
        rest = pts[i + 1:]
        if len(rest):
            incl -= wfg(np.minimum(rest, p), ref)
        total += incl
    return total


def inclusion_exclusion(pts: np.ndarray, ref: np.ndarray) -> float:
    pts = np.asarray(pts, float)
    pts = pts[np.all(pts > ref, axis=1)]
    total = 0.0
    for k in range(1, len(pts) + 1):
        for sub in itertools.combinations(range(len(pts)), k):
            total += (-1) ** (k + 1) * float(np.prod(pts[list(sub)].min(axis=0) - ref))
    return total


def monte_carlo(pts: np.ndarray, ref: np.ndarray, n: int = 200_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    hi = pts.max(axis=0)
    box = float(np.prod(hi - ref))
    x = ref + rng.random((n, len(ref))) * (hi - ref)
    inside = np.zeros(n, bool)
    for p in pts:
        inside |= np.all(x <= p, axis=1)
    m = inside.mean()
    half = 3.29 * np.sqrt(m * (1 - m) / n)   # 99.9%
    return box * m, box * (m - half), box * (m + half)


def check_a() -> bool:
    files = sorted(glob.glob("results-v1.1/**/pareto_points.csv", recursive=True))
    worst, n = 0.0, 0
    for f in files:
        df = pd.read_csv(f)
        for _, g in df.groupby(["step", "policy"]):
            pts = g[["qos", "energy", "fair"]].to_numpy(float)
            worst = max(worst, abs(_hv_3d(pts, HV_REF_POINT) - wfg(pts, HV_REF_POINT)))
            n += 1
    ok = n > 0 and worst <= TOL
    print(f"(a) 3D: {n} fronts in {len(files)} files, max |_hv_3d - WFG| = {worst:.2e}: {'ok' if ok else 'FAIL'}")
    return ok


def check_b() -> bool:
    ref = hv_ref_point(V2_OBJECTIVES)
    rng = np.random.default_rng(8)
    worst_w, worst_ie, n_ie = 0.0, 0.0, 0
    fronts = []
    for _ in range(300):
        k = int(rng.integers(1, 31))
        pts = np.column_stack([rng.random((k, 3)), -rng.random(k)])
        if rng.random() < 0.3:   # Ties and duplicates, as a front of mean rewards can have
            pts = np.round(pts, 1)
        fronts.append(pts)
        worst_w = max(worst_w, abs(_hv_nd(pts, ref) - wfg(pts, ref)))
        if k <= 10:
            worst_ie = max(worst_ie, abs(_hv_nd(pts, ref) - inclusion_exclusion(pts, ref)))
            n_ie += 1
    mc_ok = True
    for pts in fronts[:5]:
        est, lo, hi = monte_carlo(pts, ref)
        mc_ok &= lo <= _hv_nd(pts, ref) <= hi
    ok = worst_w <= TOL and worst_ie <= TOL and mc_ok
    print(f"(b) 4D: 300 fronts, max |_hv_nd - WFG| = {worst_w:.2e}; {n_ie} small fronts, max |_hv_nd - "
          f"inclusion-exclusion| = {worst_ie:.2e}; Monte Carlo 99.9% interval holds on 5: {mc_ok}: "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def compare_fixed(config: str, checkpoint: str, steps: int, n_serving=None) -> float:
    import agents_moma_ac  # noqa: F401
    from eval_checkpoint import env_config_for, eval_omegas, load_trainer, run_agent
    from eval_controller import FixedOmega, run_episodes
    trainer = load_trainer(config, checkpoint, env_config_for(config, None, n_serving))
    n_obj = len(trainer.obj_names)
    omegas = eval_omegas(n_obj, n_obj)   # The corners only
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = run_agent(trainer.agent, trainer.eval_env, omegas, 2, 100000, steps, False)
    worst = 0.0
    for i, w in enumerate(omegas):
        rows = pd.DataFrame(run_episodes(trainer, FixedOmega(w), [100000, 100001], steps, False))
        # eval_omega keeps each step's reward vector in float32 and averages in float32; do the same
        # (row-major, as eval_omega's stacked list: numpy's float32 sum rounds differently on a column-major array)
        ctrl = np.stack([np.mean(np.ascontiguousarray(g[list(trainer.obj_names)].to_numpy(np.float32)), axis=0)
                         for _, g in rows.groupby("ep")])
        worst = max(worst, float(np.abs(ctrl - res[:, i]).max()))
    return worst


def check_c() -> bool:
    ok = True
    v11 = "runs/saved_models_moma_ac_ma3_prior/seed_0/checkpoints/checkpoint_step_1000000.pth"
    if os.path.exists(v11):
        d = compare_fixed("config/train/moma_ac_ma3_prior.yaml", v11, 18)
        ok &= d <= 1e-12
        print(f"(c) v1.1 ma3_prior seed 0: max |controller - eval_checkpoint| = {d:.2e}")
    else:
        print("(c) v1.1 checkpoint not found, skipped")
    ckpt = "runs/smoke_v2_tip8/checkpoints/checkpoint_step_1500.pth"
    if not os.path.exists(ckpt):
        cfg = "runs/smoke_v2_tip8.yaml"
        with open("config/train/moma_ac_v2_prior.yaml", encoding="utf-8") as f:
            text = f.read()
        for k, v in {"total_training_steps": 1500, "min_buffer_size": 100, "batch_size": 32,
                     "max_steps_per_episode": 20, "save_interval": 1500, "eval_interval": 750, "eval_n_omega": 6,
                     "eval_n_episodes": 1, "eval_steps": 5, "buffer_size": 5000}.items():
            text = "\n".join(f"{k}: {v}" if line.split(":")[0].strip() == k else line for line in text.splitlines())
        text = text.replace('save_dir: "saved_models_moma_ac_v2_prior"', 'save_dir: "runs/smoke_v2_tip8"')
        with open(cfg, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        subprocess.run([sys.executable, "train_moma_ac.py", "--config", cfg], check=True, capture_output=True)
    d = compare_fixed("runs/smoke_v2_tip8.yaml", ckpt, 10, n_serving=3)
    ok &= d <= 1e-12
    print(f"(c) v2 smoke, N = 3: max |controller - eval_checkpoint| = {d:.2e}: {'ok' if ok else 'FAIL'}")
    return ok


def main():
    results = [check_a(), check_b(), check_c()]
    print("TIP 8 check:", "PASS" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

"""
run_kaggle.py
=============
Chay nhieu run MOMA-AC song song trong mot session Kaggle, va tiep tuc duoc qua
nhieu session. Notebook kaggle/moma_ac_kaggle.ipynb goi file nay.

Moi session:
  1. Khoi phuc cac run dang do tu output cua session truoc (neu da gan lam input)
  2. Chay cac run chua xong, moi run tu dung sau --max_hours va luu trang thai
  3. In tinh trang va bang so sanh, luu comparison.csv
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import torch
import yaml

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SMOKE_OVERRIDES = {
    "total_training_steps": 1500, "min_buffer_size": 100, "batch_size": 32,
    "max_steps_per_episode": 20, "save_interval": 1500, "eval_interval": 750,
    "eval_n_omega": 5, "eval_n_episodes": 1, "eval_steps": 5,
    "gpi_ls_interval": 300, "gpi_ls_episodes": 1,
    "dyna_interval": 100, "dyna_rollouts": 100, "model_fit_steps": 30,
}


def run_dir(runs_root: str, variant: str, seed: int) -> str:
    return os.path.join(runs_root, f"saved_models_moma_ac_{variant}", f"seed_{seed}")


def progress(path: str) -> int:
    """Number of evaluations a run has logged, used to pick the most advanced copy."""
    csv = os.path.join(path, "eval_metrics.csv")
    if not os.path.exists(csv):
        return 0
    with open(csv) as f:
        return sum(1 for _ in f) - 1


def restore_runs(input_root: str, runs_root: str):
    """Copy runs from attached outputs of earlier sessions into the working directory."""
    found = {}
    pattern = os.path.join(input_root, "**", "saved_models_moma_ac_*", "seed_*")
    for path in glob.glob(pattern, recursive=True):
        key = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
        rank = (os.path.exists(os.path.join(path, "DONE")), progress(path))
        if key not in found or rank > found[key][0]:
            found[key] = (rank, path)

    if not found:
        print("[Restore] No earlier runs attached. Starting fresh.")
    for key, (_, path) in sorted(found.items()):
        dst = os.path.join(runs_root, key)
        if not os.path.exists(dst):
            shutil.copytree(path, dst)
        print(f"[Restore] {key}: {status(dst)}  (from {path})")


def status(path: str) -> str:
    if os.path.exists(os.path.join(path, "DONE")):
        return "DONE"
    if os.path.exists(os.path.join(path, "resume", "state.pth")):
        return f"in progress, {progress(path)} evals"
    if os.path.exists(path):
        return f"NO RESUME STATE, restarts from 0 ({progress(path)} evals logged)"
    return "not started"


def write_config(variant: str, seed_root: str, config_dir: str, smoke: bool) -> str:
    with open(os.path.join(CODE_DIR, "config", "train", f"moma_ac_{variant}.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["save_dir"] = seed_root
    if smoke:
        cfg.update(SMOKE_OVERRIDES)
    path = os.path.join(config_dir, f"{variant}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def warm_up_numba():
    """Compile the numba functions once, so parallel runs don't all compile them at the same time."""
    code = ("import numpy as np; from environment import NetworkSlicingEnv;"
            "env = NetworkSlicingEnv(config_path='./config/environment/default.yaml');"
            "a = np.zeros(13, dtype=np.float32); a[3] = 1.0; a[4:] = 0.1;"
            "env.step({0: a})")
    subprocess.run([sys.executable, "-c", code], cwd=CODE_DIR, check=True, stdout=subprocess.DEVNULL)


def least_busy_gpu(active: list, n_gpus: int) -> int:
    """GPU running the fewest of the active runs. The last field of each active item is its GPU."""
    busy = [sum(item[-1] == gpu for item in active) for gpu in range(n_gpus)]
    return busy.index(min(busy))


def tail(path: str, n: int = 25) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-n:])


def main():
    parser = argparse.ArgumentParser(description="Run MOMA-AC variants in a Kaggle session")
    parser.add_argument("--variants", nargs="+", default=["random", "gpi_ls", "gpi_pd"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max_hours", type=float, default=10.5, help="Training time budget of this session")
    parser.add_argument("--parallel", type=int, default=None, help="Runs at once (default: CPU count)")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("--input_root", default="/kaggle/input")
    parser.add_argument("--work_root", default="/kaggle/working")
    parser.add_argument("--smoke", action="store_true", help="Short test run in a throwaway folder")
    parser.add_argument("--monitor_minutes", type=float, default=15.0)
    args = parser.parse_args()

    start = time.time()
    n_gpus = torch.cuda.device_count() if args.device == "cuda" else 0
    if args.device == "cuda" and n_gpus == 0:
        print("[Kaggle] WARNING: no GPU found, training on CPU. Check Accelerator in Session options.")
        args.device = "cpu"
    runs_root = os.path.join(args.work_root, "runs_smoke" if args.smoke else "runs")
    logs_dir  = os.path.join(runs_root, "logs")
    if args.smoke:
        shutil.rmtree(runs_root, ignore_errors=True)
    os.makedirs(logs_dir, exist_ok=True)
    if not args.smoke:
        restore_runs(args.input_root, runs_root)

    # One BLAS/numba thread per run: several small runs make better use of few CPUs
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
           "MOMA_DEVICE": args.device, "PYTHONUTF8": "1"}
    os.environ.update(env)
    warm_up_numba()

    queue = [(v, s) for s in args.seeds for v in args.variants
             if not os.path.exists(os.path.join(run_dir(runs_root, v, s), "DONE"))]
    n_parallel = args.parallel or os.cpu_count() or 1
    print(f"[Kaggle] {len(queue)} runs to do, {n_parallel} at once, device={args.device}, "
          f"budget={args.max_hours} h, CPUs={os.cpu_count()}, GPUs={n_gpus}")

    active, failed = [], []
    last_report = time.time()
    while queue or active:
        # A run started late gets only what is left of the session's budget
        left = args.max_hours - (time.time() - start) / 3600
        while queue and len(active) < n_parallel and left > 0.25:
            variant, seed = queue.pop(0)
            seed_root = os.path.join(runs_root, f"saved_models_moma_ac_{variant}")
            config_dir = os.path.join(logs_dir, f"config_seed_{seed}")
            os.makedirs(config_dir, exist_ok=True)
            config = write_config(variant, seed_root, config_dir, args.smoke)
            log_path = os.path.join(logs_dir, f"{variant}_seed_{seed}.log")
            log = open(log_path, "a", encoding="utf-8")
            gpu = least_busy_gpu(active, n_gpus) if n_gpus else None
            run_env = {**env, "CUDA_VISIBLE_DEVICES": str(gpu)} if n_gpus else env
            proc = subprocess.Popen(
                [sys.executable, "train_moma_ac.py", "--config", config, "--seed", str(seed),
                 "--resume", "--max_hours", f"{left:.3f}"],
                cwd=CODE_DIR, env=run_env, stdout=log, stderr=subprocess.STDOUT)
            active.append((proc, variant, seed, log, log_path, gpu))
            where = f"GPU {gpu}" if n_gpus else "CPU"
            print(f"[Kaggle] Started {variant} seed {seed} on {where} with {left:.2f} h "
                  f"({status(run_dir(runs_root, variant, seed))})", flush=True)
        if left <= 0.25:
            queue.clear()

        time.sleep(5 if args.smoke else 30)

        for item in list(active):
            proc, variant, seed, log, log_path, _ = item
            if proc.poll() is None:
                continue
            log.close()
            active.remove(item)
            if proc.returncode != 0:
                failed.append((variant, seed))
                print(f"[Kaggle] FAILED {variant} seed {seed}, exit {proc.returncode}:\n{tail(log_path)}", flush=True)
            else:
                print(f"[Kaggle] Finished session of {variant} seed {seed}: "
                      f"{status(run_dir(runs_root, variant, seed))}", flush=True)

        if time.time() - last_report > args.monitor_minutes * 60:
            last_report = time.time()
            hours = (time.time() - start) / 3600
            running = ", ".join(f"{v}/s{s}: {progress(run_dir(runs_root, v, s))} evals"
                                for _, v, s, _, _, _ in active)
            print(f"[Kaggle] {hours:.1f} h elapsed | {running}", flush=True)

    # ── Summary ──────────────────────────────────────────────────
    print("\n[Kaggle] Status after this session:")
    all_done = True
    for seed in args.seeds:
        for variant in args.variants:
            state = status(run_dir(runs_root, variant, seed))
            all_done = all_done and state == "DONE"
            print(f"  {variant:8s} seed {seed}: {state}")
    print("[Kaggle] ALL RUNS DONE" if all_done
          else "[Kaggle] Not finished: attach this version's output and run the notebook again.")
    if failed:
        print(f"[Kaggle] Failed runs: {failed}. See {logs_dir}")

    variant_dirs = [os.path.join(runs_root, f"saved_models_moma_ac_{v}") for v in args.variants]
    variant_dirs = [d for d in variant_dirs if os.path.exists(d)]
    if variant_dirs:
        subprocess.run([sys.executable, "compare_runs.py", *variant_dirs,
                        "--out", os.path.join(args.work_root, "comparison.csv")], cwd=CODE_DIR, env=env)

    if args.smoke:
        ok = all_done and not failed
        shutil.rmtree(runs_root, ignore_errors=True)
        print("[Kaggle] SMOKE TEST " + ("PASSED" if ok else "FAILED"))


if __name__ == "__main__":
    main()

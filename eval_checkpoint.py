"""
eval_checkpoint.py
==================
Eval rieng mot checkpoint (thuong la 1M) de lay so lieu cuoi, thay cho eval luc train.
Eval luc train chi co 3 episode voi seed lap lai, va vong lap kin rat nhay (sai so float
giua CPU va GPU da doi ket qua), nen khong du de xep hang cac bien the.

Dung cung 30 omega voi eval_pareto_front, nhung n_episodes episode voi seed rieng
(seed_base + k, mac dinh 100000), chua dung luc train. Moi policy chay tren cung cac episode,
nen so duoc tung cap. Khoang tin cay 95% la bootstrap tren episode: no chi phan anh nhieu
cua eval, khong phan anh khac biet giua cac seed train.

Ghi ra:
  <out>.csv  mot dong moi policy (actor, va gpi voi bien the GPI): hv, eum, eu_front, n_pareto,
             CI, reward vector o 3 goc, va delta so voi --baseline neu co
  <out>_<policy>.npy  reward vector tung episode, shape (n_episodes, n_omega, 3)
  <out>_<policy>_sla.npy  ty le UE dat du SLA theo slice (eMBB, URLLC, mMTC), cung shape, nan neu slice khong co tai

Chay:
  py eval_checkpoint.py --config config/train/moma_ac_random.yaml --checkpoint <dir>/checkpoints/checkpoint_step_1000000.pth --out eval_random
  py eval_checkpoint.py --config config/train/moma_ac_ols.yaml --checkpoint <...> --out eval_ols --baseline eval_random_actor.npy
  py eval_checkpoint.py --formation --out eval_triangle        # 3 UAV dung yen thanh tam giac (eval_formation.py)
"""

import argparse
import contextlib
import io
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("MOMA_DEVICE", "cpu")   # Truoc khi import agent: CPU cho ket qua lap lai tren moi may

import agents_moma_ac
from agents_moma_ac import EVAL_SEED, SLA_KEYS, nan_mean
from train_moma_ac import HV_REF_POINT, MOMA_AC_Trainer, _hv_3d


def eval_omegas(n_omega: int) -> np.ndarray:
    """The weights of eval_pareto_front: the corners, then seeded Dirichlet samples."""
    interior = np.random.RandomState(EVAL_SEED).dirichlet(np.ones(3), n_omega - 3).astype(np.float32)
    return np.vstack([np.eye(3, dtype=np.float32), interior])


def count_pareto(pts: np.ndarray) -> int:
    return int(sum(not any(np.all(q >= p) and np.any(q > p) for j, q in enumerate(pts) if j != i)
                   for i, p in enumerate(pts)))


def front_metrics(results: np.ndarray, omegas: np.ndarray) -> dict:
    """Same metrics as the trainer's _sweep_metrics, from mean reward vectors (n_omega, 3)."""
    return {
        "hv":       float(_hv_3d(results, HV_REF_POINT)),
        "eum":      float((results * omegas).sum(axis=1).mean()),
        "eu_front": float((omegas @ results.T).max(axis=1).mean()),
        "n_pareto": count_pareto(results),
    }


def load_trainer(config: str, checkpoint: str, env_config: str = None) -> MOMA_AC_Trainer:
    """Build the agent through the trainer, with save_dir in a temp folder and a small buffer."""
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tmp = tempfile.mkdtemp(prefix="eval_checkpoint_")
    cfg["save_dir"] = tmp
    cfg["buffer_size"] = 1000   # Eval never samples it; the full buffer of ma3 is several GB
    path = os.path.join(tmp, "train.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = MOMA_AC_Trainer(train_config_path=path, env_config_path=env_config, checkpoint_path=checkpoint)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)   # Eval writes nothing there
    return trainer


def run_agent(agent, env, omegas: np.ndarray, n_episodes: int, seed_base: int, steps: int, use_gpi: bool):
    """Reward vector and per-slice SLA share of every (episode, omega), with the episode seeded by seed_base + k."""
    results = np.zeros((n_episodes, len(omegas), 3))
    sla = np.zeros((n_episodes, len(omegas), len(SLA_KEYS)))
    default_seed = agents_moma_ac.EVAL_SEED
    try:
        for k in range(n_episodes):
            # eval_omega seeds episode 0 with the module's EVAL_SEED, so moving it moves the episode
            agents_moma_ac.EVAL_SEED = seed_base + k
            for i, w in enumerate(omegas):
                results[k, i] = agent.eval_omega(env, w, 1, steps, use_gpi=use_gpi)
                sla[k, i] = agent.last_sla
    finally:
        agents_moma_ac.EVAL_SEED = default_seed
    return results, sla


def run_formation(env_config: str, n_omega: int, n_episodes: int, seed_base: int, steps: int,
                  radius: float, power: float):
    import eval_formation
    from environment import NetworkSlicingEnv
    with contextlib.redirect_stdout(io.StringIO()):
        env = NetworkSlicingEnv(config_path=env_config)
    positions = eval_formation.formation_positions(env, radius)
    results = np.zeros((n_episodes, n_omega, 3))
    sla = np.zeros((n_episodes, n_omega, len(SLA_KEYS)))
    for k in range(n_episodes):
        row = eval_formation.evaluate(env, positions, power, 1, steps, n_omega, seed_base=seed_base + k)
        results[k, :] = [row["qos"], row["energy"], row["fair"]]   # The same action for every omega
        sla[k, :] = [row[key] for key in SLA_KEYS]
    return results, sla


def summarize(results: np.ndarray, omegas: np.ndarray, baseline: np.ndarray = None,
              n_boot: int = 1000, seed: int = 0, sla: np.ndarray = None) -> dict:
    """Metrics of the mean front, bootstrap CIs over episodes, and the paired difference to a baseline."""
    n = len(results)
    row = front_metrics(results.mean(axis=0), omegas)
    rng = np.random.default_rng(seed)
    boot, diff = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = front_metrics(results[idx].mean(axis=0), omegas)
        boot.append((m["hv"], m["eum"]))
        if baseline is not None:
            b = front_metrics(baseline[idx].mean(axis=0), omegas)
            diff.append((m["hv"] - b["hv"], m["eum"] - b["eum"]))
    boot = np.array(boot)
    row["hv_lo"], row["hv_hi"] = np.percentile(boot[:, 0], [2.5, 97.5])
    row["eum_lo"], row["eum_hi"] = np.percentile(boot[:, 1], [2.5, 97.5])
    if baseline is not None:
        diff = np.array(diff)
        base = front_metrics(baseline.mean(axis=0), omegas)
        row["d_hv"], row["d_eum"] = row["hv"] - base["hv"], row["eum"] - base["eum"]
        row["d_hv_lo"], row["d_hv_hi"] = np.percentile(diff[:, 0], [2.5, 97.5])
        row["d_eum_lo"], row["d_eum_hi"] = np.percentile(diff[:, 1], [2.5, 97.5])
    for c, name in enumerate(("qos", "energy", "fair")):
        for o, obj in enumerate(("qos", "energy", "fair")):
            row[f"w_{name}_{obj}"] = float(results[:, c, o].mean())   # Corner weight c, objective o
    if sla is not None:
        # Share of UEs meeting their full SLA: over all weights, and at each corner weight
        row.update({f"avg_{k}": float(v) for k, v in zip(SLA_KEYS, nan_mean(sla.reshape(-1, len(SLA_KEYS))))})
        for c, name in enumerate(("qos", "energy", "fair")):
            row.update({f"w_{name}_{k}": float(v) for k, v in zip(SLA_KEYS, nan_mean(sla[:, c]))})
    return {k: float(v) if isinstance(v, np.floating) else v for k, v in row.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on held-out episodes")
    parser.add_argument("--config", help="Train config the checkpoint was trained with")
    parser.add_argument("--checkpoint", help="Path to a .pth checkpoint")
    parser.add_argument("--env_config", default=None, help="Overrides the train config's env_config")
    parser.add_argument("--formation", action="store_true", help="Evaluate the fixed 3-UAV triangle instead")
    parser.add_argument("--radius", type=float, default=600.0)
    parser.add_argument("--power", type=float, default=0.5)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--seed_base", type=int, default=100000, help="Episode k uses seed_base + k")
    parser.add_argument("--steps", type=int, default=None,
                        help="Episode length; default eval_steps of the train config, 18 for --formation (moma_ac_ma3.yaml)")
    parser.add_argument("--n_omega", type=int, default=30)
    parser.add_argument("--baseline", default=None, help=".npy of another policy on the same episodes, for paired deltas")
    parser.add_argument("--out", required=True, help="Output prefix")
    args = parser.parse_args()

    omegas = eval_omegas(args.n_omega)
    if args.formation:
        steps = args.steps or 18
        policies = {"formation": run_formation(args.env_config or "config/environment/ma3.yaml", args.n_omega,
                                               args.n_episodes, args.seed_base, steps, args.radius, args.power)}
    else:
        if not (args.config and args.checkpoint and os.path.exists(args.checkpoint)):
            parser.error("--config and an existing --checkpoint are required without --formation")
        trainer = load_trainer(args.config, args.checkpoint, args.env_config)
        steps = args.steps or trainer.cfg["eval_steps"]
        modes = [("actor", False)] + ([("gpi", True)] if trainer.agent.use_gpi else [])
        policies = {label: run_agent(trainer.agent, trainer.eval_env, omegas, args.n_episodes,
                                     args.seed_base, steps, use_gpi)
                    for label, use_gpi in modes}

    baseline = np.load(args.baseline) if args.baseline else None
    if baseline is not None and baseline.shape != next(iter(policies.values()))[0].shape:
        parser.error(f"--baseline has shape {baseline.shape}, expected the same episodes and omegas")
    rows = []
    for label, (results, sla) in policies.items():
        np.save(f"{args.out}_{label}.npy", results)
        np.save(f"{args.out}_{label}_sla.npy", sla)
        rows.append({"policy": label, "checkpoint": args.checkpoint or "formation",
                     "n_episodes": args.n_episodes, "seed_base": args.seed_base,
                     **summarize(results, omegas, baseline, sla=sla)})
    pd.DataFrame(rows).to_csv(f"{args.out}.csv", index=False)
    for row in rows:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

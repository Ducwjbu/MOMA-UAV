"""
check_v11.py (TIP 0 of docs/v2/plan.md)
=======================================
Bit-for-bit check that the v1.1 configs still give the v1.1 output while v2 features are added (each new feature
is off when its config key is absent). Two parts, each run in its own process:

  env    config/environment/ma3.yaml and default.yaml, 3 episodes each (np.random.seed(SEED + ep) before reset,
         as eval_omega does), actions drawn from a separate RandomState so they do not touch the env's RNG. Each
         step stores the observation of every UAV, the scalar reward, done, every scalar of info and the reward
         vector, each UAV's position, power, battery, energy and queue, and every active UE's position, slice,
         serving UAV, throughput, delay, reliability and the three satisfactions.
  train  a smoke run (the smoke settings of CLAUDE.md) of config/train/moma_ac_ma3_prior.yaml and of
         moma_ac_gpi_pd.yaml (one UAV, GPI-LS, prioritized replay, Dyna), seed 0, CPU, one torch thread, then a
         SHA-256 of every tensor of every network of the agent.

Arrays compare exactly (NaN equal to NaN). Keys added by v2 (new info entries, new attributes) are ignored; keys of
the reference that are missing count as a difference. The reference is machine-specific (float results can differ
between CPUs and library builds), so it lives in runs/ (not tracked) and is recorded on the machine that checks.

Run from the repo root:
  py docs/v2/check_v11.py record            # on the v1.1 code: writes runs/check_v11_ref/{env,train}.npz
  py docs/v2/check_v11.py check             # after a change: compares, exit code 1 on any difference
  py docs/v2/check_v11.py check --parts env # env only (seconds; the train part takes a few minutes)
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REF = os.path.join(ROOT, "runs", "check_v11_ref")
ENV_CONFIGS = {"ma3": "config/environment/ma3.yaml", "one": "config/environment/default.yaml"}
TRAIN_CONFIGS = {"ma3_prior": "config/train/moma_ac_ma3_prior.yaml", "gpi_pd": "config/train/moma_ac_gpi_pd.yaml"}
SEED, N_EP, STEPS = 100000, 3, 19
SLICES = {"embb": 0, "urllc": 1, "mmtc": 2}
UAV_FIELDS = ("position", "current_power", "current_battery", "energy_used", "avg_queuing_delay_ms",
              "avg_drop_rate", "movement_energy", "communication_energy")
UE_FIELDS = ("throughput", "latency_ms", "reliability", "throughput_satisfaction", "delay_satisfaction",
             "reliability_satisfaction", "demand_bps")
SMOKE = {"total_training_steps": 1500, "min_buffer_size": 100, "batch_size": 32, "max_steps_per_episode": 20,
         "save_interval": 1500, "eval_interval": 750, "eval_n_omega": 5, "eval_n_episodes": 1, "eval_steps": 5,
         "gpi_ls_interval": 300, "gpi_ls_episodes": 1, "dyna_interval": 100, "dyna_rollouts": 100,
         "model_fit_steps": 30, "buffer_size": 5000}


def _num(x):
    return np.nan if x is None else float(x)


def _flat(x):
    """A field as a flat float array: dicts by sorted key (e.g. per-slice values), None as nan"""
    if isinstance(x, dict):
        return np.concatenate([_flat(x[k]) for k in sorted(x, key=str)]) if x else np.zeros(0)
    if x is None:
        return np.array([np.nan])
    return np.asarray(x, dtype=np.float64).ravel()


def env_part(out):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        from environment import NetworkSlicingEnv
    rec = {}
    for name, path in ENV_CONFIGS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            env = NetworkSlicingEnv(config_path=path)
        uids = sorted(env.uavs)
        n_da = 9
        for ep in range(N_EP):
            act_rng = np.random.RandomState(1000 + ep)
            np.random.seed(SEED + ep)
            with contextlib.redirect_stdout(io.StringIO()):
                obs = env.reset()
            rec[f"{name}/ep{ep}/reset_obs"] = np.stack([obs[u] for u in uids])
            for t in range(STEPS):
                acts = {}
                for u in uids:
                    acts[u] = np.concatenate([act_rng.uniform(-1, 1, 3), act_rng.uniform(0.05, 1, 1),
                                              act_rng.dirichlet(np.ones(n_da))])
                with contextlib.redirect_stdout(io.StringIO()):
                    obs, r, done, info = env.step(acts)
                k = f"{name}/ep{ep}/t{t:02d}"
                rec[f"{k}/obs"] = np.stack([obs[u] for u in uids])
                rec[f"{k}/reward"] = np.array([float(r), float(done)])
                rec[f"{k}/reward_vector"] = np.asarray(info["reward_vector"], dtype=np.float64)
                for key, v in info.items():
                    if np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0):
                        rec[f"{k}/info/{key}"] = np.array([_num(v)])
                for f in UAV_FIELDS:
                    rec[f"{k}/uav/{f}"] = np.concatenate([_flat(getattr(env.uavs[u], f)) for u in uids])
                ues = sorted((ue for ue in env.ues.values() if ue.is_active), key=lambda ue: ue.id)
                rec[f"{k}/ue/id"] = np.array([ue.id for ue in ues], dtype=np.int64)
                rec[f"{k}/ue/position"] = np.array([ue.position for ue in ues], dtype=np.float64).reshape(-1, 3)
                rec[f"{k}/ue/slice"] = np.array([SLICES.get(ue.slice_type, -1) for ue in ues], dtype=np.int64)
                rec[f"{k}/ue/assigned_uav"] = np.array([-1 if ue.assigned_uav is None else ue.assigned_uav
                                                        for ue in ues], dtype=np.int64)
                for f in UE_FIELDS:
                    rec[f"{k}/ue/{f}"] = (np.concatenate([_flat(getattr(ue, f, None)) for ue in ues]) if ues
                                          else np.zeros(0))
                if done:
                    break
    np.savez_compressed(out, **rec)


def train_part(out):
    import contextlib
    import io
    import torch
    import yaml
    torch.set_num_threads(1)
    from train_moma_ac import MOMA_AC_Trainer
    rec = {}
    for name, path in TRAIN_CONFIGS.items():
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg.update(SMOKE)
        tmp = tempfile.mkdtemp(prefix=f"check_v11_{name}_")
        cfg["save_dir"] = tmp
        cfg_path = os.path.join(tmp, "train.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = MOMA_AC_Trainer(train_config_path=cfg_path, seed=0)
            trainer.train()
        agent = trainer.agent
        for attr, mod in sorted(vars(agent).items()):
            if isinstance(mod, torch.nn.Module):
                for pname, tensor in sorted(mod.state_dict().items()):
                    digest = hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                    rec[f"{name}/{attr}/{pname}"] = np.array(digest)
        print(f"{name}: {sum(k.startswith(name + '/') for k in rec)} tensors", flush=True)
    np.savez_compressed(out, **rec)


def compare(ref_path, new_path, label):
    ref, new = np.load(ref_path), np.load(new_path)
    missing = [k for k in ref.files if k not in new.files]
    diff = []
    for k in ref.files:
        if k in new.files:
            a, b = ref[k], new[k]
            same = a.shape == b.shape and (np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f"
                                           else np.array_equal(a, b))
            if not same:
                diff.append(k)
    print(f"[{label}] {len(ref.files)} reference arrays: {len(diff)} differ, {len(missing)} missing, "
          f"{len([k for k in new.files if k not in ref.files])} new (ignored)")
    for k in (missing[:10]):
        print(f"  missing {k}")
    for k in diff[:15]:
        a, b = ref[k], new[k]
        if a.dtype.kind == "f" and a.shape == b.shape:
            print(f"  differs {k}: max |diff| {np.nanmax(np.abs(a - b)):.3g}")
        else:
            print(f"  differs {k}: shapes {a.shape} -> {b.shape}" if a.shape != b.shape else f"  differs {k}")
    if diff:
        first = sorted(diff)[0]
        print(f"  first differing key (sorted): {first}")
    return not diff and not missing


def run_worker(part, out):
    env = dict(os.environ, MOMA_DEVICE="cpu", PYTHONHASHSEED="0")
    subprocess.run([sys.executable, os.path.abspath(__file__), "_worker", part, out], cwd=ROOT, env=env, check=True)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "_worker":
        sys.path.insert(0, ROOT)
        os.chdir(ROOT)
        {"env": env_part, "train": train_part}[sys.argv[2]](sys.argv[3])
        return
    p = argparse.ArgumentParser(description="Bit-for-bit check of the v1.1 output")
    p.add_argument("mode", choices=["record", "check"])
    p.add_argument("--parts", nargs="+", default=["env", "train"], choices=["env", "train"])
    p.add_argument("--ref", default=REF)
    args = p.parse_args()
    if args.mode == "record":
        os.makedirs(args.ref, exist_ok=True)
        for part in args.parts:
            run_worker(part, os.path.join(args.ref, f"{part}.npz"))
            print(f"recorded {part} -> {os.path.join(args.ref, part + '.npz')}")
        return
    ok = True
    tmp = tempfile.mkdtemp(prefix="check_v11_new_")
    for part in args.parts:
        new = os.path.join(tmp, f"{part}.npz")
        run_worker(part, new)
        ok &= compare(os.path.join(args.ref, f"{part}.npz"), new, part)
    print("v1.1 output unchanged" if ok else "v1.1 output CHANGED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

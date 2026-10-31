"""
train_moma_ac.py
================
Training loop cho MOMA-AC (Random va OLS sampling variants).
Chay:
  py train_moma_ac.py --config config/train/moma_ac_random.yaml
  py train_moma_ac.py --config config/train/moma_ac_ols.yaml
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import Configuration
from environment import NetworkSlicingEnv
from agents_moma_ac import MOMA_AC_Agent


# Shared HV reference point — same as GPI-LS (train_morl.py)
# Must be worse than any observed reward on all objectives:
#   QoS    in [0,  ~0.5]  -> ref = -0.1
#   Energy in [-1,  0.0]  -> ref = -1.1
#   Fair   in [0,  ~0.8]  -> ref = -0.1
HV_REF_POINT = np.array([-0.1, -1.1, -0.1])


def _pareto_filter(pts: np.ndarray) -> np.ndarray:
    """Return non-dominated subset (maximization)."""
    if len(pts) == 0:
        return pts
    dominated = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        if dominated[i]:
            continue
        for j in range(len(pts)):
            if i == j or dominated[j]:
                continue
            if np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                dominated[i] = True
                break
    return pts[~dominated]


def _hv_2d(pts: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2D hypervolume (staircase, maximization)."""
    mask = (pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])
    pts = pts[mask]
    if len(pts) == 0:
        return 0.0
    pts = _pareto_filter(pts)
    pts = pts[np.argsort(-pts[:, 1])]
    hv = 0.0
    for i, p in enumerate(pts):
        y_next = pts[i + 1, 1] if i + 1 < len(pts) else ref[1]
        hv += (p[0] - ref[0]) * (p[1] - y_next)
    return max(0.0, hv)


def _hv_3d(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact 3D hypervolume via z-sweep (same algorithm as GPI-LS)."""
    if len(points) == 0:
        return 0.0
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(pts > ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = _pareto_filter(pts)
    pts = pts[np.argsort(-pts[:, 2])]
    hv, active_xy = 0.0, []
    for i in range(len(pts)):
        active_xy.append(pts[i, :2])
        z_curr = pts[i, 2]
        z_next = pts[i + 1, 2] if i + 1 < len(pts) else ref[2]
        z_width = z_curr - z_next
        if z_width > 0:
            hv += z_width * _hv_2d(np.array(active_xy), ref[:2])
    return max(0.0, hv)



def _get(cfg_obj, key: str, default):
    """Safe config getter supporting dotted keys."""
    try:
        val = cfg_obj
        for k in key.split("."):
            val = getattr(val, k) if hasattr(val, k) else val[k]
        return val
    except Exception:
        return default


class MOMA_AC_Trainer:
    def __init__(
        self,
        train_config_path: str,
        env_config_path:   str = "./config/environment/default.yaml",
        checkpoint_path:   str = None,
    ):
        tcfg = Configuration(train_config_path)
        ecfg = Configuration(env_config_path)

        self.cfg = {
            "total_steps":            int(_get(tcfg, "total_training_steps", 1_000_000)),
            "min_buffer_size":        int(_get(tcfg, "min_buffer_size",       5_000)),
            "train_frequency":        int(_get(tcfg, "train_frequency",       1)),
            "train_iterations":       int(_get(tcfg, "train_iterations",      1)),
            "batch_size":             int(_get(tcfg, "batch_size",            256)),
            "max_steps_per_episode":  int(_get(tcfg, "max_steps_per_episode", 200)),
            "omega_sampling":         str(_get(tcfg, "omega_sampling",        "random")),
            "corner_warmup_episodes": int(_get(tcfg, "corner_warmup_episodes", 30)),
            # TD3
            "actor_lr":         float(_get(tcfg, "actor_lr",         3e-4)),
            "critic_lr":        float(_get(tcfg, "critic_lr",         3e-4)),
            "gamma":            float(_get(tcfg, "gamma",             0.99)),
            "tau":              float(_get(tcfg, "tau",               0.005)),
            "policy_delay":     int(  _get(tcfg, "policy_delay",      2)),
            "policy_noise":     float(_get(tcfg, "policy_noise",      0.2)),
            "noise_clip":       float(_get(tcfg, "noise_clip",        0.5)),
            "expl_noise":       float(_get(tcfg, "expl_noise",        0.3)),
            "expl_noise_min":   float(_get(tcfg, "expl_noise_min",    0.05)),
            "expl_noise_decay": float(_get(tcfg, "expl_noise_decay",  0.9999)),
            "buffer_size":      int(  _get(tcfg, "buffer_size",       500_000)),
            # Logging
            "log_interval":    int(_get(tcfg, "log_interval",    5_000)),
            "save_interval":   int(_get(tcfg, "save_interval",  100_000)),
            "eval_interval":   int(_get(tcfg, "eval_interval",  100_000)),
            "eval_n_omega":    int(_get(tcfg, "eval_n_omega",    30)),
            "eval_n_episodes": int(_get(tcfg, "eval_n_episodes", 5)),
            "eval_steps":      int(_get(tcfg, "eval_steps",      150)),
            "save_dir":        str(_get(tcfg, "save_dir",        "saved_models_moma_ac")),
        }

        self.model_dir = self.cfg["save_dir"]
        os.makedirs(os.path.join(self.model_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "plots"),       exist_ok=True)

        # ── Environment ──────────────────────────────────────────
        print("[MOMA-AC] Initializing environment...")
        self.env    = NetworkSlicingEnv(config_path=env_config_path)
        obs_dict    = self.env.reset()
        self.uav_id = sorted(obs_dict.keys())[0]
        self.obs_dim = len(obs_dict[self.uav_id])

        # Infer act_dim from env config
        n_das_per_slice = int(_get(ecfg, "system.num_das_per_slice", 3))
        n_slices        = 3
        self.act_dim    = 4 + n_das_per_slice * n_slices   # 4 pose+power + N BW alloc
        print(f"[MOMA-AC] obs_dim={self.obs_dim}, act_dim={self.act_dim}")

        # ── Agent ────────────────────────────────────────────────
        self.agent = MOMA_AC_Agent(
            obs_dim   = self.obs_dim,
            act_dim   = self.act_dim,
            n_objectives = 3,
            actor_lr  = self.cfg["actor_lr"],
            critic_lr = self.cfg["critic_lr"],
            gamma     = self.cfg["gamma"],
            tau       = self.cfg["tau"],
            policy_delay      = self.cfg["policy_delay"],
            policy_noise      = self.cfg["policy_noise"],
            noise_clip        = self.cfg["noise_clip"],
            expl_noise        = self.cfg["expl_noise"],
            expl_noise_min    = self.cfg["expl_noise_min"],
            expl_noise_decay  = self.cfg["expl_noise_decay"],
            buffer_size       = self.cfg["buffer_size"],
            omega_sampling    = self.cfg["omega_sampling"],
            corner_warmup_episodes = self.cfg["corner_warmup_episodes"],
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.agent.load(checkpoint_path)

        # ── Metrics ──────────────────────────────────────────────
        self.step_log: list = []
        self.eval_log: list = []

    # ── Main Training Loop ──────────────────────────────────────

    def train(self):
        total    = self.cfg["total_steps"]
        pbar     = tqdm(total=total, desc="MOMA-AC", dynamic_ncols=True)

        step          = 0
        obs_dict      = self.env.reset()
        obs           = obs_dict[self.uav_id]
        ep_rvec       = []
        ep_steps      = 0
        last_ep_val   = None
        omega         = self.agent.sample_omega(None)
        t0            = time.time()

        while step < total:
            # ── Environment step ─────────────────────────────────────
            action = self.agent.select_action(obs, omega, explore=True)
            # Actor output: dims[0:4]=tanh[-1,1] (pos+power),
            #               dims[4:] =sigmoid[0,1] (BW) — no clipping needed
            obs_dict_new, _, done, info = self.env.step({self.uav_id: action})
            obs_next = obs_dict_new.get(self.uav_id, obs)
            r_vec    = np.array(info.get("reward_vector", [0., 0., 0.]),
                                dtype=np.float32)

            # Use raw reward vector — same as GPI-LS (no normalization)
            # Env reward ranges: QoS[0,~0.5], Energy[-1,0], Fair[0,~0.8]
            # all already in similar scale, no normalization needed
            ep_end = done or (ep_steps + 1 >= self.cfg["max_steps_per_episode"])
            self.agent.buffer.add(obs, action, r_vec, obs_next, omega, ep_end)

            ep_rvec.append(r_vec)
            obs      = obs_next
            ep_steps += 1
            step     += 1
            pbar.update(1)

            # ── Network update ───────────────────────────────────
            if (step % self.cfg["train_frequency"] == 0 and
                    len(self.agent.buffer) >= self.cfg["min_buffer_size"]):
                for _ in range(self.cfg["train_iterations"]):
                    self.agent.update(self.cfg["batch_size"])

            # ── Episode reset ────────────────────────────────────
            if ep_end:
                last_ep_val = np.mean(ep_rvec, axis=0) if ep_rvec else None
                omega       = self.agent.sample_omega(last_ep_val)
                obs_dict    = self.env.reset()
                obs         = obs_dict[self.uav_id]
                ep_rvec     = []
                ep_steps    = 0

            # ── Logging ─────────────────────────────────────────
            if step % self.cfg["log_interval"] == 0 and last_ep_val is not None:
                elapsed = time.time() - t0
                self.step_log.append({
                    "step":    step,
                    "qos":     float(last_ep_val[0]),
                    "energy":  float(last_ep_val[1]),
                    "fair":    float(last_ep_val[2]),
                    "noise":   self.agent.expl_noise,
                    "elapsed": elapsed,
                })
                pbar.set_postfix({
                    "QoS":  f"{last_ep_val[0]:.3f}",
                    "Eng":  f"{last_ep_val[1]:.3f}",
                    "Fair": f"{last_ep_val[2]:.3f}",
                    "w":    str(np.round(omega, 2)),
                    "eps":  f"{self.agent.expl_noise:.3f}",
                })
                self._save_step_csv()

            # ── Periodic Pareto eval ─────────────────────────────
            if step % self.cfg["eval_interval"] == 0:
                print(f"\n[Eval] step={step} — Pareto front sweep...")
                self._eval_and_log(step)

            # ── Checkpoint ──────────────────────────────────────
            if step % self.cfg["save_interval"] == 0:
                self._save_checkpoint(step)

        pbar.close()

        # Final save
        self._save_checkpoint(total)
        self._save_step_csv()
        self._eval_and_log(total)
        print("\n[MOMA-AC] Training complete!")

    # ── Helpers ─────────────────────────────────────────────────

    def _eval_and_log(self, step: int):
        use_gpi = (self.cfg["omega_sampling"] == "gpi_ols")
        results, omegas = self.agent.eval_pareto_front(
            self.env,
            n_omega    = self.cfg["eval_n_omega"],
            n_episodes = self.cfg["eval_n_episodes"],
            steps      = self.cfg["eval_steps"],
            use_gpi    = use_gpi,
        )
        hv       = self._compute_hv(results)
        n_pareto = self._count_pareto(results)
        self.eval_log.append({
            "step":     step,
            "hv":       hv,
            "n_pareto": n_pareto,
            "avg_qos":  float(results[:, 0].mean()),
            "avg_eng":  float(results[:, 1].mean()),
            "avg_fair": float(results[:, 2].mean()),
            "best_qos": float(results[:, 0].max()),
            "best_fair":float(results[:, 2].max()),
        })
        self._save_eval_csv()
        self._plot_pareto(results, omegas, step)
        print(f"  HV={hv:.4f}  n_pareto={n_pareto}  "
              f"best_QoS={results[:,0].max():.3f}  best_Fair={results[:,2].max():.3f}")

    def _compute_hv(self, pts: np.ndarray) -> float:
        """Exact 3D HV via sweep-line, fixed ref point (same as GPI-LS)."""
        return _hv_3d(pts, HV_REF_POINT)

    def _count_pareto(self, pts: np.ndarray) -> int:
        """Count non-dominated points (3D strict Pareto)."""
        mask = np.ones(len(pts), dtype=bool)
        for i, p in enumerate(pts):
            for j, q in enumerate(pts):
                if i != j and np.all(q >= p) and np.any(q > p):
                    mask[i] = False
                    break
        return int(mask.sum())

    def _plot_pareto(self, results: np.ndarray, omegas: np.ndarray, step: int):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"MOMA-AC ({self.cfg['omega_sampling']}) Pareto — step {step}")

        # QoS vs Fairness
        sc = axes[0].scatter(results[:, 0], results[:, 2],
                             c=omegas[:, 0], cmap="RdYlGn", s=60, alpha=0.8,
                             edgecolors="black", lw=0.4)
        plt.colorbar(sc, ax=axes[0], label="ω_QoS")
        axes[0].set_xlabel("QoS Reward")
        axes[0].set_ylabel("Fairness Reward")
        axes[0].set_title("QoS vs Fairness")
        axes[0].grid(True, alpha=0.3)

        # QoS vs Energy
        sc2 = axes[1].scatter(results[:, 0], results[:, 1],
                              c=omegas[:, 1], cmap="coolwarm", s=60, alpha=0.8,
                              edgecolors="black", lw=0.4)
        plt.colorbar(sc2, ax=axes[1], label="ω_Energy")
        axes[1].set_xlabel("QoS Reward")
        axes[1].set_ylabel("Energy Reward")
        axes[1].set_title("QoS vs Energy")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.model_dir, "plots", f"pareto_step_{step}.png")
        plt.savefig(path, dpi=100)
        plt.close()

    def _save_checkpoint(self, step: int):
        path = os.path.join(self.model_dir, "checkpoints",
                            f"checkpoint_step_{step}.pth")
        self.agent.save(path)

    def _save_step_csv(self):
        pd.DataFrame(self.step_log).to_csv(
            os.path.join(self.model_dir, "step_metrics.csv"), index=False)

    def _save_eval_csv(self):
        pd.DataFrame(self.eval_log).to_csv(
            os.path.join(self.model_dir, "eval_metrics.csv"), index=False)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MOMA-AC")
    parser.add_argument("--config",
                        default="config/train/moma_ac_random.yaml",
                        help="Training config yaml")
    parser.add_argument("--env_config",
                        default="config/environment/default.yaml",
                        help="Environment config yaml")
    parser.add_argument("--checkpoint",
                        default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    trainer = MOMA_AC_Trainer(
        train_config_path = args.config,
        env_config_path   = args.env_config,
        checkpoint_path   = args.checkpoint,
    )
    trainer.train()

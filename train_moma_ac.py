"""
train_moma_ac.py
================
Training loop cho MOMA-AC (Random, OLS, GPI-LS va GPI-PD sampling variants).
Chay:
  py train_moma_ac.py --config config/train/moma_ac_random.yaml
  py train_moma_ac.py --config config/train/moma_ac_ols.yaml
  py train_moma_ac.py --config config/train/moma_ac_gpi_ls.yaml
  py train_moma_ac.py --config config/train/moma_ac_gpi_pd.yaml
  py train_moma_ac.py --config config/train/moma_ac_ma3.yaml      # 3 UAV, CTDE
"""

import os
import sys
import argparse
import shutil
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import Configuration
from environment import NetworkSlicingEnv
from guide import FormationGuide
import prior
from agents_moma_ac import SLA_KEYS, MOMA_AC_Agent, MOMA_CTDE_Agent, episode_diagnostics, nan_mean


# Shared HV reference point
# Must be worse than any observed reward on all objectives:
#   QoS    in [0,  1]  -> ref = -0.1
#   Energy in [-1, 0]  -> ref = -1.1
#   Fair   in [0,  1]  -> ref = -0.1
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


def _hv_nd(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact hypervolume in any dimension (maximization): the sweep of _hv_3d on the last objective, recursing down
    to _hv_3d. Fine for the tens of points of an eval sweep (cost grows as n_points^(d-1))"""
    pts = np.asarray(points, dtype=float)
    d = pts.shape[1] if pts.ndim == 2 else len(ref)
    if d == 2:
        return _hv_2d(pts, ref) if len(pts) else 0.0
    if d == 3:
        return _hv_3d(pts, ref)
    if len(pts) == 0:
        return 0.0
    pts = pts[np.all(pts > ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = _pareto_filter(pts)
    pts = pts[np.argsort(-pts[:, -1])]
    hv = 0.0
    for i in range(len(pts)):
        lo = pts[i + 1, -1] if i + 1 < len(pts) else ref[-1]
        width = pts[i, -1] - lo
        if width > 0:
            hv += width * _hv_nd(pts[:i + 1, :-1], ref[:-1])
    return max(0.0, hv)


def hv_ref_point(names) -> np.ndarray:
    """HV reference point for the named objectives: -1.1 for energy (in [-1, 0]), -0.1 for the others (in [0, 1])"""
    return np.array([-1.1 if n == "energy" else -0.1 for n in names])


V11_OBJECTIVES = ["qos", "energy", "fair"]


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
        env_config_path:   str = None,
        checkpoint_path:   str = None,
        seed:              int = None,
        resume:            bool = False,
        max_hours:         float = None,
    ):
        tcfg = Configuration(train_config_path)
        # An explicit path wins, then the train config's env_config key
        env_config_path = env_config_path or str(_get(tcfg, "env_config", "./config/environment/default.yaml"))
        ecfg = Configuration(env_config_path)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.cfg = {
            "total_steps":            int(_get(tcfg, "total_training_steps", 1_000_000)),
            "min_buffer_size":        int(_get(tcfg, "min_buffer_size",       5_000)),
            "train_frequency":        int(_get(tcfg, "train_frequency",       1)),
            "train_iterations":       int(_get(tcfg, "train_iterations",      1)),
            "batch_size":             int(_get(tcfg, "batch_size",            256)),
            "max_steps_per_episode":  int(_get(tcfg, "max_steps_per_episode", 60)),
            "omega_sampling":         str(_get(tcfg, "omega_sampling",        "random")),
            "corner_warmup_episodes": int(_get(tcfg, "corner_warmup_episodes", 0)),
            # Formation guide (guide.py): share of episodes whose movement it steers, falling linearly
            # to 0 at guide_until steps (0: constant). guide_prob 0 turns it off
            "guide_prob":   float(_get(tcfg, "guide_prob",  0.0)),
            "guide_until":  int(  _get(tcfg, "guide_until", 0)),
            # Formation prior (prior.py): the actor learns a residual on top of a fixed formation
            "formation_prior": bool( _get(tcfg, "formation_prior", False)),
            "prior_radius":    float(_get(tcfg, "prior_radius",    600.0)),
            # Several UAVs: share of episodes with one omega per UAV (0: one team omega, as in the paper),
            # and the per-step probability of a new omega within an episode (MOMA_CTDE_Agent)
            "omega_per_agent":   float(_get(tcfg, "omega_per_agent",   0.0)),
            "omega_switch_prob": float(_get(tcfg, "omega_switch_prob", 0.0)),
            # Several UAVs: "mlp" (the paper's MultiHeadActor and flat critics) or "attention" (one actor for every
            # UAV and an attention critic, independent of the number of UAVs)
            "network":           str(  _get(tcfg, "network",           "mlp")),
            # GPI-LS
            "gpi_ls_interval":  int(_get(tcfg, "gpi_ls_interval",  50_000)),
            "gpi_ls_episodes":  int(_get(tcfg, "gpi_ls_episodes",  3)),
            "gpi_top_k":        int(_get(tcfg, "gpi_top_k",        2)),
            "gpi_max_support":  int(_get(tcfg, "gpi_max_support",  20)),
            "gpi_max_corners":  int(_get(tcfg, "gpi_max_corners",  20)),
            # GPI-PD
            "per_alpha":         float(_get(tcfg, "per_alpha",         0.6)),
            "per_kappa":         float(_get(tcfg, "per_kappa",         0.1)),
            "dyna_beta":         float(_get(tcfg, "dyna_beta",         0.5)),
            "dyna_interval":     int(  _get(tcfg, "dyna_interval",     250)),
            "dyna_rollouts":     int(  _get(tcfg, "dyna_rollouts",     2000)),
            "model_fit_steps":   int(  _get(tcfg, "model_fit_steps",   200)),
            "model_buffer_size": int(  _get(tcfg, "model_buffer_size", 100_000)),
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
            "eval_interval":   int(_get(tcfg, "eval_interval",   50_000)),
            "eval_n_omega":    int(_get(tcfg, "eval_n_omega",    30)),
            "eval_n_episodes": int(_get(tcfg, "eval_n_episodes", 3)),
            "eval_steps":      int(_get(tcfg, "eval_steps",      60)),
            "save_dir":        str(_get(tcfg, "save_dir",        "saved_models_moma_ac")),
        }

        self.model_dir = self.cfg["save_dir"]
        if seed is not None:
            self.model_dir = os.path.join(self.model_dir, f"seed_{seed}")
        os.makedirs(os.path.join(self.model_dir, "checkpoints"), exist_ok=True)

        # ── Environment ──────────────────────────────────────────
        print("[MOMA-AC] Initializing environment...")
        self.env    = NetworkSlicingEnv(config_path=env_config_path)
        self.eval_env = NetworkSlicingEnv(config_path=env_config_path)
        obs_dict    = self.env.reset()
        self.uav_ids = sorted(obs_dict.keys())
        self.uav_id = self.uav_ids[0]
        self.multi_uav = len(self.uav_ids) > 1
        self.obs_dim = len(obs_dict[self.uav_id])
        # Objectives of the env's reward vector; the v1.1 vector keeps the v1.1 CSV columns and the 3D HV
        self.obj_names = list(getattr(self.env, "objective_names", V11_OBJECTIVES))
        self.legacy_objectives = self.obj_names == V11_OBJECTIVES
        self.hv_ref = HV_REF_POINT if self.legacy_objectives else hv_ref_point(self.obj_names)

        # Infer act_dim from env config
        n_das_per_slice = int(_get(ecfg, "system.num_das_per_slice", 3))
        n_slices        = 3
        self.act_dim    = 4 + n_das_per_slice * n_slices   # 4 pose+power + N BW alloc
        print(f"[MOMA-AC] obs_dim={self.obs_dim}, act_dim={self.act_dim}")

        # ── Agent ────────────────────────────────────────────────
        agent_cls  = MOMA_CTDE_Agent if self.multi_uav else MOMA_AC_Agent
        agent_args = ({"n_agents": len(self.uav_ids), "omega_per_agent": self.cfg["omega_per_agent"],
                       "omega_switch_prob": self.cfg["omega_switch_prob"], "network": self.cfg["network"]}
                      if self.multi_uav else {})
        self.agent = agent_cls(
            obs_dim   = self.obs_dim,
            act_dim   = self.act_dim,
            n_objectives = len(self.obj_names),
            **agent_args,
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
            gpi_top_k         = self.cfg["gpi_top_k"],
            gpi_max_support   = self.cfg["gpi_max_support"],
            gpi_max_corners   = self.cfg["gpi_max_corners"],
            per_alpha         = self.cfg["per_alpha"],
            per_kappa         = self.cfg["per_kappa"],
            dyna_beta         = self.cfg["dyna_beta"],
            dyna_rollouts     = self.cfg["dyna_rollouts"],
            model_fit_steps   = self.cfg["model_fit_steps"],
            model_buffer_size = self.cfg["model_buffer_size"],
        )

        if self.cfg["formation_prior"]:
            if self.cfg["guide_prob"] > 0:
                raise ValueError("formation_prior and the formation guide (guide_prob > 0) can't be combined")
            # Both envs, so that evaluation adds the same prior. The zero start is overwritten by a checkpoint
            for env in (self.env, self.eval_env):
                vertices = prior.attach(env, self.uav_ids, self.cfg["prior_radius"])
            prior.zero_movement_output(self.agent)
            print(f"[MOMA-AC] Formation prior, r={self.cfg['prior_radius']:.0f} m: {np.round(vertices, 1).tolist()}")

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.agent.load(checkpoint_path)

        self.guide = FormationGuide(self.env) if self.cfg["guide_prob"] > 0 else None
        self.guide_p = 0.0

        # ── Metrics ──────────────────────────────────────────────
        self.step_log: list = []
        self.eval_log: list = []
        self.gpi_log:  list = []
        self.pareto_log: list = []

        # ── Resume across sessions ───────────────────────────────
        self.max_hours  = max_hours
        self.start_step = 0
        self.resume_dir = os.path.join(self.model_dir, "resume")
        if resume and os.path.exists(os.path.join(self.resume_dir, "state.pth")):
            self._load_resume()

    # ── Main Training Loop ──────────────────────────────────────

    def _guide_episode(self, step: int) -> int:
        """Steps the formation guide drives in the next episode (0: the actor alone)"""
        if self.guide is None:
            return 0
        self.guide_p = self.cfg["guide_prob"]
        if self.cfg["guide_until"] > 0:
            self.guide_p *= max(0.0, 1.0 - step / self.cfg["guide_until"])
        if np.random.rand() >= self.guide_p:
            return 0
        return self.guide.new_episode(self.cfg["max_steps_per_episode"])

    def _joint_obs(self, obs_dict: dict) -> np.ndarray:
        """Observations of all UAVs in id order, shape (n_uavs, obs_dim)."""
        return np.stack([obs_dict[uid] for uid in self.uav_ids])

    def train(self):
        total    = self.cfg["total_steps"]
        pbar     = tqdm(total=total, initial=self.start_step, desc="MOMA-AC",
                        dynamic_ncols=True, mininterval=30)

        step          = self.start_step
        obs_dict      = self.env.reset()
        obs           = self._joint_obs(obs_dict) if self.multi_uav else obs_dict[self.uav_id]
        ep_rvec       = []
        ep_diag       = []
        ep_steps      = 0
        last_ep_val   = None
        last_ep_diag  = None
        omega         = self.agent.sample_omega()
        guide_steps   = self._guide_episode(step)
        use_gpi       = self.agent.use_gpi               # Follow GPI policy (GPI-PD Alg. 2, line 12)
        t0            = time.time()

        while step < total:
            # ── Environment step ─────────────────────────────────────
            # Actor output: dims[0:3]=tanh[-1,1] (pos),
            #               dims[3:] =sigmoid[0,1] (power, BW) — no clipping needed
            if self.multi_uav:
                action = self.agent.select_joint_action(obs, omega, explore=True)
                if ep_steps < guide_steps:   # The guide steers movement, the actor keeps power and bandwidth
                    action[:, 0:3] = self.guide.movement(self.uav_ids, self.agent.expl_noise)
                obs_dict_new, _, done, info = self.env.step(dict(zip(self.uav_ids, action)))
                obs_next = self._joint_obs(obs_dict_new)
                ep_diag.append((info["coverage"], info["handovers"], info["min_uav_distance"], info["backhaul_limited"]))
            else:
                action = self.agent.select_action(obs, omega, explore=True, use_gpi=use_gpi)
                if ep_steps < guide_steps:
                    action[0:3] = self.guide.movement([self.uav_id], self.agent.expl_noise)[0]
                obs_dict_new, _, done, info = self.env.step({self.uav_id: action})
                obs_next = obs_dict_new.get(self.uav_id, obs)
            r_vec    = np.array(info.get("reward_vector", [0., 0., 0.]),
                                dtype=np.float32)

            # Use raw reward vector (no normalization)
            # Env reward ranges: QoS[0,1], Energy[-1,0], Fair[0,1]
            # all already in similar scale, no normalization needed
            ep_end = done or (ep_steps + 1 >= self.cfg["max_steps_per_episode"])
            # Store the env's done only: the time limit truncates, it isn't terminal
            self.agent.buffer.add(np.ravel(obs), np.ravel(action), r_vec, np.ravel(obs_next), np.ravel(omega), done)

            ep_rvec.append(r_vec)
            obs      = obs_next
            if self.multi_uav:
                omega = self.agent.maybe_switch_omega(omega)
            ep_steps += 1
            step     += 1
            pbar.update(1)

            # ── Network update ───────────────────────────────────
            if (step % self.cfg["train_frequency"] == 0 and
                    len(self.agent.buffer) >= self.cfg["min_buffer_size"]):
                for _ in range(self.cfg["train_iterations"]):
                    self.agent.update(self.cfg["batch_size"])

            # ── Dyna: fit model, add simulated transitions (GPI-PD) ──
            if (self.agent.model is not None and
                    step % self.cfg["dyna_interval"] == 0 and
                    len(self.agent.buffer) >= self.cfg["min_buffer_size"]):
                self.last_dyna_info = self.agent.dyna_step()

            # ── OLS / GPI-LS: grow the weight support ────────────
            if (self.agent.support is not None and step % self.cfg["gpi_ls_interval"] == 0 and
                    len(self.agent.buffer) >= self.cfg["min_buffer_size"]):
                self._gpi_ls_iteration(step)

            # ── Episode reset ────────────────────────────────────
            if ep_end:
                last_ep_val = np.mean(ep_rvec, axis=0) if ep_rvec else None
                last_ep_diag = episode_diagnostics(ep_diag) if ep_diag else None
                omega       = self.agent.sample_omega()
                guide_steps = self._guide_episode(step)
                obs_dict   = self.env.reset()
                obs         = self._joint_obs(obs_dict) if self.multi_uav else obs_dict[self.uav_id]
                ep_rvec     = []
                ep_diag     = []
                ep_steps    = 0

            # ── Logging ─────────────────────────────────────────
            if step % self.cfg["log_interval"] == 0 and last_ep_val is not None:
                elapsed = time.time() - t0
                self.step_log.append({
                    "step":    step,
                    **{n: float(v) for n, v in zip(self.obj_names, last_ep_val)},
                    "noise":   self.agent.expl_noise,
                    "elapsed": elapsed,
                })
                if last_ep_diag is not None:
                    self.step_log[-1].update(last_ep_diag)
                if self.guide is not None:
                    self.step_log[-1]["guide_prob"] = self.guide_p
                pbar.set_postfix({
                    **({"QoS": f"{last_ep_val[0]:.3f}", "Eng": f"{last_ep_val[1]:.3f}",
                        "Fair": f"{last_ep_val[2]:.3f}"} if self.legacy_objectives else
                       {n: f"{v:.3f}" for n, v in zip(self.obj_names, last_ep_val)}),
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
                if step < total:
                    self._save_resume(step)

            # ── Time budget: stop cleanly before the session limit ──
            if (self.max_hours is not None and step < total
                    and time.time() - t0 > self.max_hours * 3600):
                self._save_resume(step)
                pbar.close()
                print(f"\n[MOMA-AC] Time budget reached at step {step}/{total}. "
                      f"Continue with --resume.")
                return

        pbar.close()

        # Final save
        self._save_checkpoint(total)
        self._save_step_csv()
        if total % self.cfg["eval_interval"] != 0:
            self._eval_and_log(total)
        # A finished run needs no replay buffer: free the space and mark it done
        shutil.rmtree(self.resume_dir, ignore_errors=True)
        open(os.path.join(self.model_dir, "DONE"), "w").close()
        print("\n[MOMA-AC] Training complete!")

    # ── Resume ──────────────────────────────────────────────────

    def _save_resume(self, step: int):
        """
        Save everything training needs to continue. The env isn't saved, so the
        run continues from a fresh episode. The swap at the end keeps the last
        complete copy if the session dies mid-save.
        """
        tmp = self.resume_dir + "_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        self.agent.save(os.path.join(tmp, "agent.pth"))
        self.agent.buffer.save(os.path.join(tmp, "buffer.npz"))
        if self.agent.model_buffer is not None:
            self.agent.model_buffer.save(os.path.join(tmp, "model_buffer.npz"))
        torch.save({
            "step":      step,
            "step_log":  self.step_log,
            "eval_log":  self.eval_log,
            "gpi_log":   self.gpi_log,
            "pareto_log": self.pareto_log,
            "last_dyna_info": getattr(self, "last_dyna_info", {}),
            "np_rng":    np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }, os.path.join(tmp, "state.pth"))
        shutil.rmtree(self.resume_dir, ignore_errors=True)
        os.replace(tmp, self.resume_dir)
        print(f"[MOMA-AC] Resume state saved at step {step}")

    def _load_resume(self):
        self.agent.load(os.path.join(self.resume_dir, "agent.pth"))
        self.agent.buffer.load(os.path.join(self.resume_dir, "buffer.npz"))
        if self.agent.model_buffer is not None:
            self.agent.model_buffer.load(os.path.join(self.resume_dir, "model_buffer.npz"))
        st = torch.load(os.path.join(self.resume_dir, "state.pth"), weights_only=False)
        self.start_step = st["step"]
        self.step_log   = st["step_log"]
        self.eval_log   = st["eval_log"]
        self.gpi_log    = st["gpi_log"]
        self.pareto_log = st["pareto_log"]
        self.last_dyna_info = st["last_dyna_info"]
        np.random.set_state(st["np_rng"])
        torch.set_rng_state(st["torch_rng"])
        print(f"[MOMA-AC] Resumed at step {self.start_step} (buffer={len(self.agent.buffer)})")

    # ── Helpers ─────────────────────────────────────────────────

    def _gpi_ls_iteration(self, step: int):
        info = self.agent.update_weight_support(
            self.eval_env,
            n_episodes = self.cfg["gpi_ls_episodes"],
            steps      = self.cfg["eval_steps"],
        )
        self.gpi_log.append({"step": step, **info, **getattr(self, "last_dyna_info", {})})
        pd.DataFrame(self.gpi_log).to_csv(
            os.path.join(self.model_dir, "gpi_ls_metrics.csv"), index=False)
        print(f"\n[{self.cfg['omega_sampling']}] step={step}  {info}")

    def _sweep_metrics(self, use_gpi: bool):
        results, omegas = self.agent.eval_pareto_front(
            self.eval_env,
            n_omega    = self.cfg["eval_n_omega"],
            n_episodes = self.cfg["eval_n_episodes"],
            steps      = self.cfg["eval_steps"],
            use_gpi    = use_gpi,
        )
        metrics = {
            "hv":       self._compute_hv(results),
            # EUM of MOMA-AC: utility of each omega's own policy
            "eum":      float((results * omegas).sum(axis=1).mean()),
            # EU of GPI-PD: best utility over the whole front for each omega
            "eu_front": float((omegas @ results.T).max(axis=1).mean()),
            "n_pareto": self._count_pareto(results),
        }
        sla = np.array(self.agent.sweep_sla)   # (n_omega, 3): share of UEs meeting their full SLA
        # Mean over the sweep, like avg_qos, and at the QoS corner (omega index 0)
        if self.legacy_objectives:
            metrics.update({f"avg_{k}": float(v) for k, v in zip(SLA_KEYS, nan_mean(sla))})
            metrics.update({f"wqos_{k}": float(v) for k, v in zip(SLA_KEYS, sla[0])})
        else:   # The binary shares, beside the sla_* objectives; corner<j> is omega e_j
            metrics.update({f"avg_{k}_share": float(v) for k, v in zip(SLA_KEYS, nan_mean(sla))})
            for j, n in enumerate(self.obj_names):
                metrics.update({f"at_{n}_{k}_share": float(v) for k, v in zip(SLA_KEYS, sla[j])})
        return metrics, results, omegas, sla

    def _eval_and_log(self, step: int):
        # GPI variants report the GPI policy, and the plain actor as *_actor for
        # a like-for-like comparison with the random baseline
        use_gpi = self.agent.use_gpi
        metrics, results, omegas, sla = self._sweep_metrics(use_gpi)
        if self.legacy_objectives:
            row = {
                "step":     step,
                **metrics,
                "avg_qos":  float(results[:, 0].mean()),
                "avg_eng":  float(results[:, 1].mean()),
                "avg_fair": float(results[:, 2].mean()),
                "best_qos": float(results[:, 0].max()),
                "best_fair":float(results[:, 2].max()),
            }
        else:
            row = {"step": step, **metrics,
                   **{f"avg_{n}": float(results[:, j].mean()) for j, n in enumerate(self.obj_names)},
                   **{f"best_{n}": float(results[:, j].max()) for j, n in enumerate(self.obj_names)}}
        if self.agent.support is not None:
            row["support_size"] = len(self.agent.support.weights)
        if self.multi_uav:
            row.update(self.agent.pop_eval_diagnostics())
        if use_gpi:
            actor_metrics, actor_results, _, actor_sla = self._sweep_metrics(False)
            row.update({f"{k}_actor": v for k, v in actor_metrics.items()})
            self._log_pareto(step, "actor", actor_results, omegas, actor_sla)
        self._log_pareto(step, "gpi" if use_gpi else "actor", results, omegas, sla)
        self.eval_log.append(row)
        self._save_eval_csv()
        best = (f"best_QoS={results[:,0].max():.3f}  best_Fair={results[:,2].max():.3f}" if self.legacy_objectives
                else "  ".join(f"best_{n}={results[:, j].max():.3f}" for j, n in enumerate(self.obj_names)))
        print(f"  HV={metrics['hv']:.4f}  EUM={metrics['eum']:.4f}  n_pareto={metrics['n_pareto']}  {best}")

    def _log_pareto(self, step: int, policy: str, results: np.ndarray, omegas: np.ndarray, sla: np.ndarray):
        """Keep each evaluated point, so the front can be analyzed without rerunning."""
        for w, rv, sl in zip(omegas, results, sla):
            self.pareto_log.append({
                "step":   step,
                "policy": policy,
                **{f"w_{n}": float(x) for n, x in zip(self.obj_names, w)},
                **{n: float(x) for n, x in zip(self.obj_names, rv)},
                # In v2 the binary SLA shares are reported next to the sla_* objectives (their smooth stand-ins)
                **{(k if self.legacy_objectives else f"{k}_share"): float(v) for k, v in zip(SLA_KEYS, sl)},
            })
        pd.DataFrame(self.pareto_log).to_csv(
            os.path.join(self.model_dir, "pareto_points.csv"), index=False)

    def _compute_hv(self, pts: np.ndarray) -> float:
        """Exact HV, fixed ref point (3D sweep for the v1.1 objectives, the n-dimensional sweep otherwise)"""
        return _hv_3d(pts, HV_REF_POINT) if self.legacy_objectives else _hv_nd(pts, self.hv_ref)

    def _count_pareto(self, pts: np.ndarray) -> int:
        """Count non-dominated points (3D strict Pareto)."""
        mask = np.ones(len(pts), dtype=bool)
        for i, p in enumerate(pts):
            for j, q in enumerate(pts):
                if i != j and np.all(q >= p) and np.any(q > p):
                    mask[i] = False
                    break
        return int(mask.sum())

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
                        default=None,
                        help="Environment config yaml (default: the train config's env_config, else default.yaml)")
    parser.add_argument("--checkpoint",
                        default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed; results go to <save_dir>/seed_<seed>")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from <save_dir>/resume if it exists")
    parser.add_argument("--max_hours", type=float, default=None,
                        help="Save resume state and stop after this many hours")
    args = parser.parse_args()

    trainer = MOMA_AC_Trainer(
        train_config_path = args.config,
        env_config_path   = args.env_config,
        checkpoint_path   = args.checkpoint,
        seed              = args.seed,
        resume            = args.resume,
        max_hours         = args.max_hours,
    )
    trainer.train()

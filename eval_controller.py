"""
eval_controller.py
==================
Runtime preference control on a trained multi-UAV policy, with no retraining (thesis item A1; v2 TIP 8).

The trained policy pi(o_i, omega) is conditioned on omega. Here omega is not fixed for the episode: a controller
picks it at every step from KPIs the operator measures, so that the team meets KPI targets while it optimizes a
primary objective. This tests the idea before any env change.

Problem: maximize the primary objective V_0 subject to time-averaged constraints KPI_k >= c_k.
Its Lagrangian is <e_0 + sum_k lambda_k d_k, V> - sum_k lambda_k c_k, a weighted sum with
omega = (e_0 + sum_k lambda_k d_k) / ||.||_1, so a policy conditioned on omega answers every lambda
(Paternain et al., NeurIPS 2019, zero duality gap for one agent; CAPQL Prop. 4). d_k is the objective that KPI k
belongs to. The Lagrangian argument is exact only when KPI_k IS that objective (e.g. the qos reward, or a v2
sla_<slice> objective); for a binary SLA share (sla_urllc in v1.1, sla_urllc_share in v2) the objective is a proxy
and the argument is heuristic.

Objectives are the env's (env.objective_names): qos, energy, fair in v1.1; sla_embb, sla_urllc, sla_mmtc, energy
in config/environment/v2.yaml. KPIs are every objective, coverage, and the binary SLA share of each slice (named
sla_<slice> in v1.1, sla_<slice>_share in v2). A constraint's objective defaults to the KPI itself, or for
X_share to X.

Controllers (one call to observe per env step):
  pid       lambda_k = [K_P e_k + K_I sum e_k + K_D (e_k - e_k')]_+ with e_k = c_k - KPI_k and the integral kept
            >= 0, the PID Lagrangian of Stooke, Achiam & Abbeel (ICML 2020, Alg. 2). With K_P = K_D = 0 it is the
            virtual queue of drift-plus-penalty (Neely 2010): Q_k <- [Q_k + K_I (c_k - KPI_k)]_+.
  i_only    K_P = K_D = 0 (virtual queue).
  p_only    K_I = K_D = 0 (acts on the current deficit only).
  local     v2, one omega per UAV: the virtual queue of i_only kept by each UAV on its own KPI, the mean of the
            sla_<slice> score (or the binary SLA, for X_share) over the UEs of that slice it serves this step;
            omega_i = (e_0 + sum_k lambda_ik d_k) / ||.||_1. A KPI with no per-UAV value (energy, coverage) uses
            the team's. A UAV serving no UE of the slice keeps its lambda. Uncovered UEs count in the team KPI (near
            0) but in no UAV's, so every local queue can be satisfied while the team constraint fails. The team
            policy was trained with one omega for all UAVs (unless omega_per_agent), so this is out of its training
            distribution. The steps file logs the mean omega over the positions serving when it was applied.
  bang_bang a threshold rule: omega is the corner of the constraints whose running mean (since the episode
            started, or the shift with --carry) is below its threshold (their mean if several), else e_0.
  rcpo      lambda held for a whole episode and updated after it on the episode's mean KPI,
            lambda_k <- [lambda_k + eta (c_k - mean KPI_k)]_+ (--rcpo_lr): the multiplier update of RCPO on the
            slow timescale (Tessler, Mankowitz & Mannor, "Reward Constrained Policy Optimization", ICLR 2019,
            arXiv:1805.11074), here without retraining the policy: projected dual ascent on the episode mean,
            none of RCPO's penalized reward or policy update. The update after an episode uses the thresholds in
            force during it. lambda carries across the episodes of a seed whether or not --carry is set.
  dpi_lite  omega = E[w] under p(w) proportional to exp(beta <w, G>) on the simplex, G the critic's predicted return
            of the current action (times 1 - gamma, per step). This is DPI's evidence (Cao et al., ICLR 2026: the
            agent's own predicted return), without its learned inference network, so it keeps the property that
            matters: dE[w_k]/dG_k = beta Var(w_k) >= 0, weight moves toward objectives that are already high.
            It ignores the KPI targets.
  fixed     the constant-lambda policies. One constraint: omega = (1 - a) e_0 + a d on --fixed_grid points
            (fixed_a<a>); several: a grid of step 1 / (fixed_grid - 1) on the face spanned by e_0 and the
            constraints' objectives (fixed_<weights>). The best of them in hindsight is the "oracle" fixed omega,
            by two criteria: oracle, meets every constraint on average over the evaluated episodes (as before);
            oracle_viol, also with at most --oracle_viol of the units (seed, episode) violating any constraint
            (viol_any), the criterion the
            controllers are judged by (an independent audit, 2026-11-13, found a fine grid's best fixed omega met
            the mean while 31% of its episodes violated).
Every change of omega is capped at delta in L1 (--delta): PCMA Thm 4.6 bounds the distance to equilibrium by a
term proportional to the per-step change of preferences.

KPIs over time (summary): tviol_<kpi>, the share of steps whose KPI is below the threshold in force; viol_any, the
share of units (seed, episode) violating any constraint on the episode mean (viol_<kpi> is nan for an episode with
no measurement of that KPI); and the recovery after events, rec_<kpi>_steps (mean steps until the trailing mean
over --window steps, counting only steps from the event on, is at the threshold, over events that recover) and
rec_<kpi>_share (share of events that recover before the episode, or the shift with --carry, ends). Events: a
threshold change (--change; with --carry, as the episodes of a seed then form one sequence) and, in a v2 shift, a
serving position left empty (info n_serving_step falls within an episode). Events count in the phase they start.

Episodes are those of eval_checkpoint.py: episode k is seeded by seed_base + k (100000+ validation, used to tune the
gains; 200000+ test). With --carry the controller state (lambda, integral, omega) carries over consecutive episodes
of a seed, as one long shift; otherwise it restarts at every episode (18 steps for ma3, a 240-step shift in v2).

Writes:
  <out>_steps.csv    one row per (method, seed, ep, t): omega (the mean over UAVs for local), lambda per
                     constraint (per UAV and constraint for local), KPIs, reward vector, and in v2 n_serving, gaps
  <out>_summary.csv  one row per method: mean primary objective, mean KPIs, constraint gaps, share of episodes
                     that violate, KPIs over time, with 95% bootstrap CIs over (seed, episode) units

Run:
  py eval_controller.py --out evals/ctrl_val --seed_base 100000
  py eval_controller.py --out evals/ctrl --seed_base 200000 --k_i 0.5 --k_p 0 --lambda_init 0.5
  py eval_controller.py --out evals/ctrl_shift --seed_base 200000 --k_i 0.5 --k_p 2 --lambda_init 0.5 --carry
  py eval_controller.py --out evals/ctrl_urllc --constraint sla_urllc:0.30:qos
  py eval_controller.py --config config/train/moma_ac_v2_prior.yaml
      --checkpoint_glob "<run>/seed_*/checkpoints/checkpoint_step_1000000.pth" --n_serving 6 --primary energy
      --constraint sla_urllc:0.8 --constraint sla_embb:0.6 --out evals/ctrl_v2
  (fixed omegas: --fixed_grid points per axis, so (g + 1) g / 2 of them with two constraints on a 3-axis face:
  66 at the default 11; 41 gives 861, about 96 h per checkpoint at 20 s a shift and 20 episodes. Use a fine grid
  with one constraint)

Gains (tuned on 3 seeds x 10 validation episodes, qos >= 0.30, primary energy): per episode K_I 0.5, K_P 0,
lambda_init 0.5 is the only setting with at most 10% of episodes violating; with --carry the time-averaged qos
settles at the threshold (0.296-0.302) for every gain, and a P term adds oscillation.
"""

import argparse
import glob
import itertools
import os
import re

os.environ.setdefault("MOMA_DEVICE", "cpu")

import numpy as np
import pandas as pd
import torch

from agents_moma_ac import DEVICE, SLA_KEYS, sla_of
from eval_checkpoint import env_config_for, load_trainer
from train_moma_ac import V11_OBJECTIVES

CHECKPOINT = "runs/saved_models_moma_ac_ma3_prior/seed_*/checkpoints/checkpoint_step_1000000.pth"
SLICES = ("embb", "urllc", "mmtc")


def share_keys(names) -> list:
    """Names of the binary SLA shares: sla_<slice> in v1.1, sla_<slice>_share in v2 (beside the objectives)"""
    return list(SLA_KEYS) if list(names) == V11_OBJECTIVES else [f"{k}_share" for k in SLA_KEYS]


def kpis_of(info: dict, names) -> dict:
    """KPIs measured after one step: the reward vector, coverage, the per-slice SLA shares (nan without demand)"""
    rv = info["reward_vector"]
    out = {n: float(rv[j]) for j, n in enumerate(names)}
    out["coverage"] = float(info["coverage"])
    out.update({k: float(v) for k, v in zip(share_keys(names), sla_of(info))})
    return out


def local_kpis(env, uids) -> dict:
    """v2: per-UAV KPIs, arrays over uids (nan where the UAV serves no UE of the slice with demand this step):
    sla_<slice>, the mean of the UEs' SLA scores (the objective's own per-UE term), and sla_<slice>_share, the share
    of them meeting their SLA"""
    idx = {u: i for i, u in enumerate(uids)}
    score = {s: [[] for _ in uids] for s in SLICES}
    ok = {s: [[] for _ in uids] for s in SLICES}
    for ue in env.ues.values():
        if not ue.is_active or getattr(ue, "demand_bps", 0.0) <= 0 or getattr(ue, "assigned_uav", None) not in idx:
            continue
        if not hasattr(ue, "sla_score"):
            continue
        i = idx[ue.assigned_uav]
        score[ue.slice_type][i].append(ue.sla_score)
        ok[ue.slice_type][i].append(float(ue.sla_ok))
    out = {}
    for s in SLICES:
        out[f"sla_{s}"] = np.array([np.mean(v) if v else np.nan for v in score[s]])
        out[f"sla_{s}_share"] = np.array([np.mean(v) if v else np.nan for v in ok[s]])
    return out


def parse_constraint(text: str, obj: dict) -> dict:
    """kpi:threshold[:objective]; the objective defaults to the KPI itself, or X for X_share"""
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(f"constraint must be kpi:threshold[:objective], got {text}")
    kpi, c = parts[0], float(parts[1])
    if len(parts) == 3:
        o = parts[2]
    elif kpi in obj:
        o = kpi
    elif kpi.endswith("_share") and kpi[:-len("_share")] in obj:
        o = kpi[:-len("_share")]
    else:
        o = kpi
    if o not in obj:
        raise argparse.ArgumentTypeError(f"objective of {text} must be one of {list(obj)}")
    return {"kpi": kpi, "c": c, "obj": obj[o]}


def limit_step(old: np.ndarray, target: np.ndarray, delta: float) -> np.ndarray:
    """Move from old toward target by at most delta in L1; stays on the simplex (convex combination)"""
    d = target - old
    n = np.abs(d).sum()
    return target if n <= delta else old + d * (delta / n)


def lagrangian_omega(n_obj: int, primary: int, lam, constraints) -> np.ndarray:
    v = np.zeros(n_obj)
    v[primary] = 1.0
    for lam_k, con in zip(lam, constraints):
        v[con["obj"]] += lam_k
    return v / v.sum()


class OmegaController:
    """PID Lagrangian on KPI deficits; omega = (e_0 + sum_k lambda_k d_k) / ||.||_1, rate-limited"""

    def __init__(self, n_obj: int, primary: int, constraints: list, k_p: float, k_i: float, k_d: float,
                 delta: float = 2.0, lambda_init: float = 0.0):
        self.n_obj, self.primary, self.constraints = n_obj, primary, constraints
        self.k_p, self.k_i, self.k_d, self.delta = k_p, k_i, k_d, delta
        self.lambda_init = lambda_init
        self.reset()

    def reset(self):
        n = len(self.constraints)
        self.integral = np.full(n, self.lambda_init / self.k_i if self.k_i > 0 else 0.0)
        self.prev_e = np.full(n, np.nan)
        self.lam = np.full(n, self.lambda_init)
        self.w = self._target()

    def new_episode(self):
        self.reset()

    def _target(self) -> np.ndarray:
        return lagrangian_omega(self.n_obj, self.primary, self.lam, self.constraints)

    def omega(self, **_) -> np.ndarray:
        return self.w.astype(np.float32)

    def observe(self, kpis: dict, local=None):
        for k, con in enumerate(self.constraints):
            x = kpis.get(con["kpi"], np.nan)
            if np.isnan(x):   # Slice without demand this step: no measurement, keep lambda
                continue
            e = con["c"] - x
            self.integral[k] = max(0.0, self.integral[k] + e)
            de = 0.0 if np.isnan(self.prev_e[k]) else e - self.prev_e[k]
            self.prev_e[k] = e
            self.lam[k] = max(0.0, self.k_p * e + self.k_i * self.integral[k] + self.k_d * de)
        self.w = limit_step(self.w, self._target(), self.delta)


class LocalQueue:
    """One virtual queue per UAV and constraint on the UAV's own KPI; one omega per UAV, rate-limited"""

    def __init__(self, n_agents: int, n_obj: int, primary: int, constraints: list, k_i: float,
                 delta: float = 2.0, lambda_init: float = 0.0):
        self.n_agents, self.n_obj, self.primary, self.constraints = n_agents, n_obj, primary, constraints
        self.k_i, self.delta, self.lambda_init = k_i, delta, lambda_init
        self.reset()

    def reset(self):
        self.lam = np.full((self.n_agents, len(self.constraints)), self.lambda_init)
        self.w = np.stack([lagrangian_omega(self.n_obj, self.primary, l, self.constraints) for l in self.lam])

    def new_episode(self):
        self.reset()

    def omega(self, **_) -> np.ndarray:
        return self.w.astype(np.float32)

    def observe(self, kpis: dict, local=None):
        for k, con in enumerate(self.constraints):
            x = local[con["kpi"]] if local is not None and con["kpi"] in local else \
                np.full(self.n_agents, kpis.get(con["kpi"], np.nan))
            ok = ~np.isnan(x)
            self.lam[ok, k] = np.maximum(0.0, self.lam[ok, k] + self.k_i * (con["c"] - x[ok]))
        for i in range(self.n_agents):
            self.w[i] = limit_step(self.w[i], lagrangian_omega(self.n_obj, self.primary, self.lam[i],
                                                               self.constraints), self.delta)


class BangBang:
    """omega = the corner(s) of the constraints whose running mean is below threshold, else e_0"""

    def __init__(self, n_obj: int, primary: int, constraints: list, delta: float = 2.0):
        self.n_obj, self.primary, self.constraints, self.delta = n_obj, primary, constraints, delta
        self.reset()

    def reset(self):
        n = len(self.constraints)
        self.sum, self.count = np.zeros(n), np.zeros(n)
        self.lam = np.zeros(n)   # 1 while the constraint is active (recorded like a multiplier)
        self.w = np.eye(self.n_obj)[self.primary]

    def new_episode(self):
        self.reset()

    def omega(self, **_) -> np.ndarray:
        return self.w.astype(np.float32)

    def observe(self, kpis: dict, local=None):
        for k, con in enumerate(self.constraints):
            x = kpis.get(con["kpi"], np.nan)
            if not np.isnan(x):
                self.sum[k] += x
                self.count[k] += 1
            mean = self.sum[k] / self.count[k] if self.count[k] else np.inf
            self.lam[k] = float(mean < con["c"])
        if self.lam.any():
            target = np.zeros(self.n_obj)
            for lam_k, con in zip(self.lam, self.constraints):
                target[con["obj"]] += lam_k
            target /= target.sum()
        else:
            target = np.eye(self.n_obj)[self.primary]
        self.w = limit_step(self.w, target, self.delta)


class RCPOLambda:
    """lambda fixed within an episode, updated after it on the episode's mean KPI (slow timescale)"""

    def __init__(self, n_obj: int, primary: int, constraints: list, lr: float, lambda_init: float = 0.0):
        self.n_obj, self.primary, self.constraints, self.lr = n_obj, primary, constraints, lr
        self.lambda_init = lambda_init
        self.reset()

    def reset(self):
        n = len(self.constraints)
        self.lam = np.full(n, self.lambda_init)
        self.sum, self.count = np.zeros(n), np.zeros(n)   # Sum of the deficits c_k - KPI_k over the episode
        self.w = lagrangian_omega(self.n_obj, self.primary, self.lam, self.constraints)

    def new_episode(self):
        """Apply the multiplier update for the episode just ended (with the thresholds that were in force during
        it, even when the next episode changes them); lambda carries over"""
        for k in range(len(self.constraints)):
            if self.count[k]:
                self.lam[k] = max(0.0, self.lam[k] + self.lr * self.sum[k] / self.count[k])
        self.sum[:], self.count[:] = 0.0, 0.0
        self.w = lagrangian_omega(self.n_obj, self.primary, self.lam, self.constraints)

    def omega(self, **_) -> np.ndarray:
        return self.w.astype(np.float32)

    def observe(self, kpis: dict, local=None):
        for k, con in enumerate(self.constraints):
            x = kpis.get(con["kpi"], np.nan)
            if not np.isnan(x):
                self.sum[k] += con["c"] - x
                self.count[k] += 1


class FixedOmega:
    def __init__(self, w):
        self.w = np.asarray(w, dtype=np.float64)
        self.lam = np.zeros(0)

    def reset(self):
        pass

    def new_episode(self):
        pass

    def omega(self, **_) -> np.ndarray:
        return self.w.astype(np.float32)

    def observe(self, kpis: dict, local=None):
        pass


class DPILite:
    """omega = E[w] under p(w) ~ exp(beta <w, G>) over a fixed Dirichlet(1) sample of the simplex"""

    def __init__(self, agent, n_obj: int, beta: float, delta: float, w0, n_samples: int = 4000):
        self.agent, self.beta, self.delta = agent, beta, delta
        self.samples = np.random.RandomState(0).dirichlet(np.ones(n_obj), n_samples)
        self.w0 = np.asarray(w0, dtype=np.float64)
        self.lam = np.zeros(0)
        self.reset()

    def reset(self):
        self.w = self.w0.copy()

    def new_episode(self):
        self.reset()

    def omega(self, **_) -> np.ndarray:
        return self.w.astype(np.float32)

    @torch.no_grad()
    def evidence(self, obs: np.ndarray, action: np.ndarray, w: np.ndarray) -> np.ndarray:
        a = self.agent
        cw = a.critic_omega(w) if hasattr(a, "critic_omega") else w
        q = a.critic1(torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=DEVICE),
                      torch.as_tensor(action.reshape(1, -1), dtype=torch.float32, device=DEVICE),
                      torch.as_tensor(np.reshape(cw, (1, -1)), dtype=torch.float32, device=DEVICE))
        return q.cpu().numpy()[0] * (1.0 - a.gamma)

    def update(self, g: np.ndarray):
        logits = self.beta * (self.samples @ g)
        p = np.exp(logits - logits.max())
        self.w = limit_step(self.w, (p[:, None] * self.samples).sum(0) / p.sum(), self.delta)

    def observe(self, kpis: dict, local=None):
        pass


def run_episodes(trainer, controller, seeds, steps: int, carry: bool, constraints: list = (), changes=()):
    """Roll the team with omega from the controller; returns one row per step.

    changes: (episode, threshold) pairs. From that episode of the sequence on, the first constraint's threshold
    is the new value (the operator changes the target during the shift); the controller reads it from the
    shared constraint dict. Each row records the threshold in force (c_<kpi>) and the phase (changes applied).
    """
    agent, env = trainer.agent, trainer.eval_env
    names = trainer.obj_names
    uids = sorted(env.uavs)
    rows = []
    rng_state = np.random.get_state()
    c0 = constraints[0]["c"] if constraints else None
    is_local = isinstance(controller, LocalQueue)
    controller.reset()
    try:
        for ep, s in enumerate(seeds):
            phase = sum(ep >= e for e, _ in changes)
            if constraints:
                constraints[0]["c"] = c0 if phase == 0 else sorted(changes)[phase - 1][1]
            np.random.seed(s)
            if ep > 0 and (not carry or isinstance(controller, RCPOLambda)):
                controller.new_episode()
            obs_dict = env.reset()
            for t in range(steps):
                w = controller.omega()
                serving = [i for i, u in enumerate(uids) if getattr(env.uavs[u], "active", True)]
                obs = np.stack([obs_dict[u] for u in uids])
                action = agent.select_joint_action(obs, w, explore=False)
                if isinstance(controller, DPILite):
                    g = controller.evidence(obs, action, w)
                obs_dict, _, done, info = env.step({u: action[i] for i, u in enumerate(uids)})
                k = kpis_of(info, names)
                controller.observe(k, local_kpis(env, uids) if is_local else None)
                if isinstance(controller, DPILite):
                    controller.update(g)
                w_team = np.asarray(w, dtype=np.float64)
                if w_team.ndim == 2:   # One omega per UAV: the mean over the serving ones
                    w_team = w_team[serving or list(range(len(uids)))].mean(0)   # Serving when omega was applied
                row = {"ep": ep, "t": t, "phase": phase, **{f"w_{n}": w_team[j] for j, n in enumerate(names)}, **k}
                if "n_serving_step" in info:
                    row["n_serving"], row["gaps"] = info["n_serving_step"], info["gaps"]
                row.update({f"c_{con['kpi']}": con["c"] for con in constraints})
                lam = np.asarray(controller.lam)
                if lam.ndim == 2:
                    row.update({f"lambda_u{i}_{j}": v for i, r in enumerate(lam) for j, v in enumerate(r)})
                else:
                    row.update({f"lambda_{j}": v for j, v in enumerate(np.atleast_1d(lam))})
                rows.append(row)
                if done:
                    break
    finally:
        np.random.set_state(rng_state)
        if constraints:
            constraints[0]["c"] = c0
    return rows


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def recovery(steps: pd.DataFrame, constraints: list, carry: bool, window: int) -> pd.DataFrame:
    """Per method and constraint: events (threshold changes; a serving position left empty) and the steps until
    the trailing mean of the KPI over `window` steps, counting only steps from the event on, is at the threshold in
    force. Pass the whole run (every phase): a threshold change is the step where the phase changes, and each
    event is reported with the phase it starts"""
    out = []
    keys = ["method", "seed"] if carry else ["method", "seed", "ep"]
    for key, g in steps.groupby(keys, sort=False):
        g = g.sort_values(["ep", "t"])
        events = list(np.flatnonzero(np.diff(g["phase"].to_numpy()) > 0) + 1)
        if "n_serving" in g:   # Within an episode: a new episode may start with another fleet size
            same_ep = np.diff(g["ep"].to_numpy()) == 0
            events += list(np.flatnonzero((np.diff(g["n_serving"].to_numpy()) < 0) & same_ep) + 1)
        for con in constraints:
            x = g[con["kpi"]].to_numpy(dtype=float)
            c = g[f"c_{con['kpi']}"].to_numpy(dtype=float) if f"c_{con['kpi']}" in g else np.full(len(x), con["c"])
            phase = g["phase"].to_numpy()
            for e in sorted(set(events)):
                # Pre-event steps never count: a KPI that was fine before the event is not "recovered" at once
                trail = pd.Series(x[e:]).rolling(window, min_periods=1).mean().to_numpy()
                after = np.flatnonzero(trail >= c[e:])
                out.append({"method": key[0], "phase": int(phase[e]), "kpi": con["kpi"], "recovered": len(after) > 0,
                            "steps": float(after[0]) if len(after) else np.nan})
    return pd.DataFrame(out, columns=["method", "phase", "kpi", "recovered", "steps"])


def summarize(steps: pd.DataFrame, primary: str, constraints: list, names=V11_OBJECTIVES,
              carry: bool = False, window: int = 5, rec: pd.DataFrame = None) -> pd.DataFrame:
    """Per (method, seed, ep): episode means; then mean and CI over those units, per method. rec: the events of
    recovery() on the whole run, restricted to this phase (computed from steps when None)"""
    ep = steps.groupby(["method", "seed", "ep"]).mean(numeric_only=True).reset_index()
    for con in constraints:
        c = ep[f"c_{con['kpi']}"] if f"c_{con['kpi']}" in ep else con["c"]   # The threshold in force
        ep[f"gap_{con['kpi']}"] = np.maximum(0.0, c - ep[con["kpi"]])
        # nan when the KPI had no measurement in the whole episode (e.g. a slice with no demand)
        ep[f"viol_{con['kpi']}"] = np.where(ep[con["kpi"]].isna(), np.nan, (ep[con["kpi"]] < c).astype(float))
    vk = [f"viol_{con['kpi']}" for con in constraints]
    # A unit violates if any constraint does (nan only if no constraint was measured)
    ep["viol_any"] = ep[vk].max(axis=1, skipna=True)
    # KPIs over time: share of a unit's steps (with a measurement) below the threshold in force
    tv = steps.copy()
    for con in constraints:
        c = tv[f"c_{con['kpi']}"] if f"c_{con['kpi']}" in tv else con["c"]
        tv[f"tviol_{con['kpi']}"] = np.where(tv[con["kpi"]].isna(), np.nan, (tv[con["kpi"]] < c).astype(float))
    tcols = [f"tviol_{con['kpi']}" for con in constraints]
    ep = ep.merge(tv.groupby(["method", "seed", "ep"])[tcols].mean().reset_index(), on=["method", "seed", "ep"])
    cols = [primary] + list(names) + ["coverage"] + share_keys(names)
    cols = list(dict.fromkeys(cols)) + [c for c in ep.columns if c.startswith(("gap_", "viol_"))]
    if rec is None:
        rec = recovery(steps, constraints, carry, window)
    out = []
    for m, g in ep.groupby("method", sort=False):
        row = {"method": m, "n_units": len(g)}
        for c in cols:
            row[c] = float(g[c].mean())
            if c == primary or c.startswith(("gap_", "viol_")) or any(c == con["kpi"] for con in constraints):
                row[f"{c}_lo"], row[f"{c}_hi"] = bootstrap_ci(g[c].to_numpy())
        for c in tcols:
            row[c] = float(g[c].mean())
        for con in constraints:
            r = rec[(rec.method == m) & (rec.kpi == con["kpi"])]
            row[f"n_events_{con['kpi']}"] = len(r)
            row[f"rec_{con['kpi']}_share"] = float(r.recovered.mean()) if len(r) else np.nan
            row[f"rec_{con['kpi']}_steps"] = float(r.steps.mean()) if r.recovered.any() else np.nan
        out.append(row)
    return pd.DataFrame(out)


def oracle_fixed(summary: pd.DataFrame, primary: str, constraints: list, max_viol: float = None):
    """The fixed omega with the best primary objective among those meeting every constraint on average, and with
    max_viol also with at most that share of the units violating any constraint (viol_any)"""
    fixed = summary[summary.method.str.startswith("fixed_")]
    ok = np.ones(len(fixed), bool)
    for con in constraints:
        ok &= fixed[con["kpi"]].to_numpy() >= con["c"]
    if max_viol is not None:
        ok &= fixed["viol_any"].to_numpy() <= max_viol
    if not ok.any():
        return None
    return fixed[ok].sort_values(primary, ascending=False).iloc[0]["method"]


def fixed_omegas(n_obj: int, primary: int, constraints: list, grid: int) -> dict:
    """One constraint: the line e_0 -> d; several: a grid on the face spanned by e_0 and the constraints' objectives"""
    e0 = np.eye(n_obj)[primary]
    axes = list(dict.fromkeys([primary] + [con["obj"] for con in constraints]))
    if len(axes) <= 2:
        d = np.eye(n_obj)[constraints[0]["obj"]]
        return {f"fixed_a{a:.2f}": (1 - a) * e0 + a * d for a in np.linspace(0, 1, grid)}
    out = {}
    n = grid - 1
    for counts in itertools.product(range(n + 1), repeat=len(axes) - 1):
        if sum(counts) > n:
            continue
        w = np.zeros(n_obj)
        w[axes[0]] = (n - sum(counts)) / n
        for ax, k in zip(axes[1:], counts):
            w[ax] = k / n
        out["fixed_" + "_".join(f"{x:.2f}" for x in w)] = w
    return out


def main():
    p = argparse.ArgumentParser(description="Runtime omega control on a trained multi-UAV policy")
    p.add_argument("--config", default="config/train/moma_ac_ma3_prior.yaml")
    p.add_argument("--checkpoint_glob", default=CHECKPOINT)
    p.add_argument("--env_config", default=None, help="Overrides the train config's env_config")
    p.add_argument("--n_serving", type=int, default=None, help="v2: serving positions of the shift fixed at N")
    p.add_argument("--embb_streams", type=float, default=None, help="v2: mean eMBB streams per UE (load level)")
    p.add_argument("--primary", default="energy")
    p.add_argument("--constraint", action="append",
                   help="kpi:threshold[:objective], repeatable; default qos:0.30 (v1.1 only)")
    p.add_argument("--methods", nargs="+", default=["pid", "i_only", "p_only", "dpi_lite", "fixed"],
                   help="pid i_only p_only local bang_bang rcpo dpi_lite fixed")
    p.add_argument("--k_p", type=float, default=2.0)
    p.add_argument("--k_i", type=float, default=0.5)
    p.add_argument("--k_d", type=float, default=0.0)
    p.add_argument("--lambda_init", type=float, default=0.0, help="lambda at the start (of each episode, or of "
                   "the shift with --carry)")
    p.add_argument("--delta", type=float, default=2.0, help="Max L1 change of omega per step (2 = no limit)")
    p.add_argument("--beta", type=float, default=20.0, help="dpi_lite temperature")
    p.add_argument("--rcpo_lr", type=float, default=1.0, help="rcpo multiplier step per episode")
    p.add_argument("--fixed_grid", type=int, default=11, help="Points per axis for the fixed omegas (41: step 0.025)")
    p.add_argument("--oracle_viol", type=float, default=0.1,
                   help="oracle_viol: the best fixed omega violating in at most this share of the units")
    p.add_argument("--window", type=int, default=5, help="Trailing window (steps) for the recovery time")
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=None, help="Episode length; default eval_steps of the train config")
    p.add_argument("--seed_base", type=int, default=100000, help="100000+ validation (tuning), 200000+ test")
    p.add_argument("--carry", action="store_true", help="Keep the controller state across episodes of a seed")
    p.add_argument("--n_seeds", type=int, default=None, help="Only the first n checkpoints")
    p.add_argument("--change", action="append", default=[],
                   help="EPISODE:THRESHOLD, repeatable: the first constraint's threshold from that episode of the "
                        "sequence on (use with --carry for one shift); summaries are then per phase")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    files = sorted(glob.glob(args.checkpoint_glob))[:args.n_seeds]
    if not files:
        p.error(f"no checkpoint matches {args.checkpoint_glob}")
    seeds = [args.seed_base + k for k in range(args.n_episodes)]
    changes = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in args.change]
    env_config = env_config_for(args.config, args.env_config, args.n_serving, args.embb_streams)

    frames = []
    names = constraints = None
    for f in files:
        m_seed = re.search(r"seed_(\d+)", f.replace("\\", "/"))
        seed = int(m_seed.group(1)) if m_seed else len(frames)
        trainer = load_trainer(args.config, f, env_config)
        if names is None:
            names = list(trainer.obj_names)
            obj = {n: j for j, n in enumerate(names)}
            if args.primary not in obj:
                p.error(f"--primary must be one of {names}")
            primary = obj[args.primary]
            if not args.constraint and names != V11_OBJECTIVES:
                p.error("give --constraint for this env (the default qos:0.30 is v1.1's)")
            try:
                constraints = [parse_constraint(c, obj) for c in (args.constraint or ["qos:0.30"])]
            except argparse.ArgumentTypeError as e:
                p.error(str(e))
            e0 = np.eye(len(names))[primary]
            d = np.eye(len(names))[constraints[0]["obj"]]
        n_obj = len(names)
        steps = args.steps or trainer.cfg["eval_steps"]
        ctrls = {}
        for m in args.methods:
            if m == "pid":
                ctrls[m] = OmegaController(n_obj, primary, constraints, args.k_p, args.k_i, args.k_d, args.delta,
                                           args.lambda_init)
            elif m == "i_only":
                ctrls[m] = OmegaController(n_obj, primary, constraints, 0.0, args.k_i, 0.0, args.delta,
                                           args.lambda_init)
            elif m == "p_only":
                ctrls[m] = OmegaController(n_obj, primary, constraints, args.k_p, 0.0, 0.0, args.delta,
                                           args.lambda_init)
            elif m == "local":
                if names == V11_OBJECTIVES:
                    p.error("local needs the v2 objectives (per-UE SLA scores)")
                ctrls[m] = LocalQueue(len(trainer.uav_ids), n_obj, primary, constraints, args.k_i, args.delta,
                                      args.lambda_init)
            elif m == "bang_bang":
                ctrls[m] = BangBang(n_obj, primary, constraints, args.delta)
            elif m == "rcpo":
                ctrls[m] = RCPOLambda(n_obj, primary, constraints, args.rcpo_lr, args.lambda_init)
            elif m == "dpi_lite":
                ctrls[m] = DPILite(trainer.agent, n_obj, args.beta, args.delta, (e0 + d) / 2)
            elif m == "fixed":
                ctrls.update({k: FixedOmega(w) for k, w in
                              fixed_omegas(n_obj, primary, constraints, args.fixed_grid).items()})
            else:
                p.error(f"unknown method {m}")
        for name, c in ctrls.items():
            df = pd.DataFrame(run_episodes(trainer, c, seeds, steps, args.carry, constraints, changes))
            df.insert(0, "seed", seed)
            df.insert(0, "method", name)
            frames.append(df)
        print(f"seed {seed} done", flush=True)

    steps_df = pd.concat(frames, ignore_index=True)
    steps_df.to_csv(f"{args.out}_steps.csv", index=False, float_format="%.5g")
    show = ["method", args.primary] + [con["kpi"] for con in constraints] + \
           [f"viol_{con['kpi']}" for con in constraints] + [f"tviol_{con['kpi']}" for con in constraints]
    parts = []
    rec_all = recovery(steps_df, constraints, args.carry, args.window)   # Whole run: sees the threshold changes
    for phase, g in steps_df.groupby("phase"):
        cons = [dict(con) for con in constraints]
        cons[0]["c"] = float(g[f"c_{cons[0]['kpi']}"].iloc[0])
        summary = summarize(g, args.primary, cons, names, args.carry, args.window, rec_all[rec_all.phase == phase])
        best = oracle_fixed(summary, args.primary, cons)
        best_viol = oracle_fixed(summary, args.primary, cons, args.oracle_viol)
        summary["oracle"] = summary.method == best
        summary["oracle_viol"] = summary.method == best_viol
        summary.insert(1, "phase", phase)
        summary.insert(2, "threshold", cons[0]["c"])
        parts.append(summary)
        print(f"phase {phase}: " + ", ".join(f"{c['kpi']} >= {c['c']}" for c in cons))
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(summary[list(dict.fromkeys(show))].to_string(index=False, float_format="%.3f"))
        print(f"oracle fixed omega (hindsight): {best}; violating in at most {args.oracle_viol:.0%} of units: "
              f"{best_viol}")
    pd.concat(parts).to_csv(f"{args.out}_summary.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()

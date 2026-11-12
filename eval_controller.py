"""
eval_controller.py
==================
Runtime preference control on a trained 3-UAV policy, with no retraining (thesis item A1).

The trained policy pi(o_i, omega) is conditioned on omega. Here omega is not fixed for the episode: a controller
picks it at every step from KPIs the operator measures, so that the team meets KPI targets while it optimizes a
primary objective. This tests the idea before any env change.

Problem: maximize the primary objective V_0 subject to time-averaged constraints KPI_k >= c_k.
Its Lagrangian is <e_0 + sum_k lambda_k d_k, V> - sum_k lambda_k c_k, a weighted sum with
omega = (e_0 + sum_k lambda_k d_k) / ||.||_1, so a policy conditioned on omega answers every lambda
(Paternain et al., NeurIPS 2019, zero duality gap for one agent; CAPQL Prop. 4). d_k is the objective that KPI k
belongs to. The Lagrangian argument is exact only when KPI_k IS that objective (e.g. the qos reward); for an SLA
share (sla_urllc) the qos objective is a proxy and the argument is heuristic.

Controllers (one call to OmegaController.observe per env step):
  pid      lambda_k = [K_P e_k + K_I sum e_k + K_D (e_k - e_k')]_+ with e_k = c_k - KPI_k and the integral kept
           >= 0, the PID Lagrangian of Stooke, Achiam & Abbeel (ICML 2020, Alg. 2). With K_P = K_D = 0 it is the
           virtual queue of drift-plus-penalty (Neely 2010): Q_k <- [Q_k + K_I (c_k - KPI_k)]_+.
  i_only   K_P = K_D = 0 (virtual queue).
  p_only   K_I = K_D = 0 (acts on the current deficit only).
  dpi_lite omega = E[w] under p(w) proportional to exp(beta <w, G>) on the simplex, G the critic's predicted return of
           the current action (times 1 - gamma, per step). This is DPI's evidence (Cao et al., ICLR 2026: the
           agent's own predicted return), without its learned inference network, so it keeps the property that
           matters: dE[w_k]/dG_k = beta Var(w_k) >= 0, weight moves toward objectives that are already high.
           It ignores the KPI targets.
  fixed_aX omega = (1 - X) e_0 + X d (one constraint): the constant-lambda policies. The best of them that meets
           the constraints on average over the evaluated episodes, picked in hindsight, is the "oracle" fixed omega.
Every change of omega is capped at delta in L1 (--delta): PCMA Thm 4.6 bounds the distance to equilibrium by a
term proportional to the per-step change of preferences.

Episodes are those of eval_checkpoint.py: episode k is seeded by seed_base + k (100000+ validation, used to tune the
gains; 200000+ test). With --carry the controller state (lambda, integral, omega) carries over consecutive episodes
of a seed, as one long shift; otherwise it restarts at every episode (18 steps for ma3).

Writes:
  <out>_steps.csv    one row per (method, seed, ep, t): omega, lambda per constraint, KPIs, reward vector
  <out>_summary.csv  one row per method: mean primary objective, mean KPIs, constraint gaps, share of episodes
                     that violate, with 95% bootstrap CIs over (seed, episode) units

Run:
  py eval_controller.py --out evals/ctrl_val --seed_base 100000
  py eval_controller.py --out evals/ctrl --seed_base 200000 --k_i 0.5 --k_p 0 --lambda_init 0.5
  py eval_controller.py --out evals/ctrl_shift --seed_base 200000 --k_i 0.5 --k_p 2 --lambda_init 0.5 --carry

Gains (tuned on 3 seeds x 10 validation episodes, qos >= 0.30, primary energy): per episode K_I 0.5, K_P 0,
lambda_init 0.5 is the only setting with at most 10% of episodes violating; with --carry the time-averaged qos
settles at the threshold (0.296-0.302) for every gain, and a P term adds oscillation.
  py eval_controller.py --out evals/ctrl_urllc --constraint sla_urllc:0.30:qos
"""

import argparse
import glob
import os
import re

os.environ.setdefault("MOMA_DEVICE", "cpu")

import numpy as np
import pandas as pd
import torch

from agents_moma_ac import DEVICE, sla_of
from eval_checkpoint import load_trainer

OBJ = {"qos": 0, "energy": 1, "fair": 2}
CHECKPOINT = "runs/saved_models_moma_ac_ma3_prior/seed_*/checkpoints/checkpoint_step_1000000.pth"


def kpis_of(info: dict) -> dict:
    """KPIs measured after one step: the reward vector, the per-slice SLA shares (nan without demand), coverage"""
    rv = info["reward_vector"]
    out = {"qos": float(rv[0]), "energy": float(rv[1]), "fair": float(rv[2]), "coverage": float(info["coverage"])}
    out.update({k: float(v) for k, v in zip(("sla_embb", "sla_urllc", "sla_mmtc"), sla_of(info))})
    return out


def parse_constraint(text: str) -> dict:
    """kpi:threshold[:objective]; the objective defaults to the KPI itself when it is one"""
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(f"constraint must be kpi:threshold[:objective], got {text}")
    kpi, c = parts[0], float(parts[1])
    obj = parts[2] if len(parts) == 3 else kpi
    if obj not in OBJ:
        raise argparse.ArgumentTypeError(f"objective of {text} must be one of {list(OBJ)}")
    return {"kpi": kpi, "c": c, "obj": OBJ[obj]}


def limit_step(old: np.ndarray, target: np.ndarray, delta: float) -> np.ndarray:
    """Move from old toward target by at most delta in L1; stays on the simplex (convex combination)"""
    d = target - old
    n = np.abs(d).sum()
    return target if n <= delta else old + d * (delta / n)


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

    def _target(self) -> np.ndarray:
        v = np.zeros(self.n_obj)
        v[self.primary] = 1.0
        for lam, con in zip(self.lam, self.constraints):
            v[con["obj"]] += lam
        return v / v.sum()

    def omega(self, **_) -> np.ndarray:
        return self.w.astype(np.float32)

    def observe(self, kpis: dict):
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


class FixedOmega:
    def __init__(self, w):
        self.w = np.asarray(w, dtype=np.float64)
        self.lam = np.zeros(0)

    def reset(self):
        pass

    def omega(self, **_) -> np.ndarray:
        return self.w.astype(np.float32)

    def observe(self, kpis: dict):
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

    def observe(self, kpis: dict):
        pass


def run_episodes(trainer, controller, seeds, steps: int, carry: bool, constraints: list = (), changes=()):
    """Roll the team with omega from the controller; returns one row per step.

    changes: (episode, threshold) pairs. From that episode of the sequence on, the first constraint's threshold
    is the new value (the operator changes the target during the shift); the controller reads it from the
    shared constraint dict. Each row records the threshold in force (c_<kpi>) and the phase (changes applied).
    """
    agent, env = trainer.agent, trainer.eval_env
    uids = sorted(env.uavs)
    rows = []
    rng_state = np.random.get_state()
    c0 = constraints[0]["c"] if constraints else None
    controller.reset()
    try:
        for ep, s in enumerate(seeds):
            phase = sum(ep >= e for e, _ in changes)
            if constraints:
                constraints[0]["c"] = c0 if phase == 0 else sorted(changes)[phase - 1][1]
            np.random.seed(s)
            if not carry:
                controller.reset()
            obs_dict = env.reset()
            for t in range(steps):
                w = controller.omega()
                obs = np.stack([obs_dict[u] for u in uids])
                action = agent.select_joint_action(obs, w, explore=False)
                if isinstance(controller, DPILite):
                    g = controller.evidence(obs, action, w)
                obs_dict, _, done, info = env.step({u: action[i] for i, u in enumerate(uids)})
                k = kpis_of(info)
                controller.observe(k)
                if isinstance(controller, DPILite):
                    controller.update(g)
                row = {"ep": ep, "t": t, "phase": phase, "w_qos": w[0], "w_energy": w[1], "w_fair": w[2], **k}
                row.update({f"c_{con['kpi']}": con["c"] for con in constraints})
                row.update({f"lambda_{j}": v for j, v in enumerate(np.atleast_1d(controller.lam))})
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


def summarize(steps: pd.DataFrame, primary: str, constraints: list) -> pd.DataFrame:
    """Per (method, seed, ep): episode means; then mean and CI over those units, per method"""
    ep = steps.groupby(["method", "seed", "ep"]).mean(numeric_only=True).reset_index()
    for con in constraints:
        c = ep[f"c_{con['kpi']}"] if f"c_{con['kpi']}" in ep else con["c"]   # The threshold in force
        ep[f"gap_{con['kpi']}"] = np.maximum(0.0, c - ep[con["kpi"]])
        ep[f"viol_{con['kpi']}"] = (ep[con["kpi"]] < c).astype(float)
    cols = [primary, "qos", "energy", "fair", "coverage", "sla_embb", "sla_urllc", "sla_mmtc"]
    cols = list(dict.fromkeys(cols)) + [c for c in ep.columns if c.startswith(("gap_", "viol_"))]
    out = []
    for m, g in ep.groupby("method", sort=False):
        row = {"method": m, "n_units": len(g)}
        for c in cols:
            row[c] = float(g[c].mean())
            if c == primary or c.startswith(("gap_", "viol_")) or any(c == con["kpi"] for con in constraints):
                row[f"{c}_lo"], row[f"{c}_hi"] = bootstrap_ci(g[c].to_numpy())
        out.append(row)
    return pd.DataFrame(out)


def oracle_fixed(summary: pd.DataFrame, primary: str, constraints: list):
    """The fixed omega with the best primary objective among those meeting every constraint on average"""
    fixed = summary[summary.method.str.startswith("fixed_")]
    ok = np.ones(len(fixed), bool)
    for con in constraints:
        ok &= fixed[con["kpi"]].to_numpy() >= con["c"]
    if not ok.any():
        return None
    return fixed[ok].sort_values(primary, ascending=False).iloc[0]["method"]


def main():
    p = argparse.ArgumentParser(description="Runtime omega control on a trained 3-UAV policy")
    p.add_argument("--config", default="config/train/moma_ac_ma3_prior.yaml")
    p.add_argument("--checkpoint_glob", default=CHECKPOINT)
    p.add_argument("--primary", default="energy", choices=list(OBJ))
    p.add_argument("--constraint", type=parse_constraint, action="append",
                   help="kpi:threshold[:objective], repeatable; default qos:0.30")
    p.add_argument("--methods", nargs="+", default=["pid", "i_only", "p_only", "dpi_lite", "fixed"])
    p.add_argument("--k_p", type=float, default=2.0)
    p.add_argument("--k_i", type=float, default=0.5)
    p.add_argument("--k_d", type=float, default=0.0)
    p.add_argument("--lambda_init", type=float, default=0.0, help="lambda at the start (of each episode, or of "
                   "the shift with --carry)")
    p.add_argument("--delta", type=float, default=2.0, help="Max L1 change of omega per step (2 = no limit)")
    p.add_argument("--beta", type=float, default=20.0, help="dpi_lite temperature")
    p.add_argument("--fixed_grid", type=int, default=11, help="Points on the e_0 -> d line for the fixed omegas")
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument("--seed_base", type=int, default=100000, help="100000+ validation (tuning), 200000+ test")
    p.add_argument("--carry", action="store_true", help="Keep the controller state across episodes of a seed")
    p.add_argument("--n_seeds", type=int, default=None, help="Only the first n checkpoints")
    p.add_argument("--change", action="append", default=[],
                   help="EPISODE:THRESHOLD, repeatable: the first constraint's threshold from that episode of the "
                        "sequence on (use with --carry for one shift); summaries are then per phase")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    constraints = args.constraint or [parse_constraint("qos:0.30")]
    primary = OBJ[args.primary]
    files = sorted(glob.glob(args.checkpoint_glob))[:args.n_seeds]
    if not files:
        p.error(f"no checkpoint matches {args.checkpoint_glob}")
    seeds = [args.seed_base + k for k in range(args.n_episodes)]
    changes = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in args.change]
    e0 = np.eye(3)[primary]
    d = np.eye(3)[constraints[0]["obj"]]

    frames = []
    for f in files:
        seed = int(re.search(r"seed_(\d+)", f.replace("\\", "/")).group(1))
        trainer = load_trainer(args.config, f)
        steps = trainer.cfg["eval_steps"]
        ctrls = {}
        for m in args.methods:
            if m == "pid":
                ctrls[m] = OmegaController(3, primary, constraints, args.k_p, args.k_i, args.k_d, args.delta,
                                           args.lambda_init)
            elif m == "i_only":
                ctrls[m] = OmegaController(3, primary, constraints, 0.0, args.k_i, 0.0, args.delta, args.lambda_init)
            elif m == "p_only":
                ctrls[m] = OmegaController(3, primary, constraints, args.k_p, 0.0, 0.0, args.delta, args.lambda_init)
            elif m == "dpi_lite":
                ctrls[m] = DPILite(trainer.agent, 3, args.beta, args.delta, (e0 + d) / 2)
            elif m == "fixed":
                for a in np.linspace(0, 1, args.fixed_grid):
                    ctrls[f"fixed_a{a:.2f}"] = FixedOmega((1 - a) * e0 + a * d)
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
           [f"viol_{con['kpi']}" for con in constraints]
    parts = []
    for phase, g in steps_df.groupby("phase"):
        cons = [dict(con) for con in constraints]
        cons[0]["c"] = float(g[f"c_{cons[0]['kpi']}"].iloc[0])
        summary = summarize(g, args.primary, cons)
        best = oracle_fixed(summary, args.primary, cons)
        summary["oracle"] = summary.method == best
        summary.insert(1, "phase", phase)
        summary.insert(2, "threshold", cons[0]["c"])
        parts.append(summary)
        print(f"phase {phase}: {cons[0]['kpi']} >= {cons[0]['c']}")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(summary[list(dict.fromkeys(show))].to_string(index=False, float_format="%.3f"))
        print(f"oracle fixed omega (hindsight): {best}")
    pd.concat(parts).to_csv(f"{args.out}_summary.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()

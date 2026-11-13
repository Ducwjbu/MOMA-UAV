"""Does the vector critic drift below the true values with gamma 0.99? (CLAUDE.md, formation prior, "Why gamma 0.9")

At every state the actor visits (validation episodes, seed_base 100000), compare critic1's Q(s, pi(s), omega), one
value per objective, with the discounted Monte Carlo return of the rest of the episode under the critic's own gamma.

The time limit (18 steps) is stored as non-terminal, so the critic bootstraps past it and estimates an infinite
horizon. The env stops at step 18, so that value has no measured counterpart; it lies between two bounds for every
possible continuation: QoS and fairness are in [0, 1], so their value is in [MC, MC + gamma^(T-t) / (1 - gamma)];
energy is in [-1, 0], so in [MC - gamma^(T-t) / (1 - gamma), MC]. A Q below the lower bound (QoS, fairness) or above
the upper one (energy) is wrong under any continuation. The script reports the share of states where that happens.

This also shows the confound that the audit of 2026-11-13 named: with gamma 0.99 the tail term is large, so the
critic must guess what happens after step 18, which the observation (no time feature) does not tell it.

Run from the repo root (checkpoints in runs/, as for eval_checkpoint.py):
  py report/scripts/critic_bias.py --out evals/diag/critic_bias
  ... at earlier checkpoints: --steps 100000 1000000
Writes <out>.csv (one row per method, checkpoint, seed, corner, episode, step) and prints the summary.
"""
import argparse
import contextlib
import io
import os
import sys

os.environ.setdefault("MOMA_DEVICE", "cpu")
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import torch

from eval_checkpoint import load_trainer

METHODS = ["ma3", "ma3_g90", "ma3_prior_g99", "ma3_prior"]
CORNERS = {"qos": [1, 0, 0], "energy": [0, 1, 0], "fair": [0, 0, 1]}
CK = "runs/saved_models_moma_ac_{m}/seed_{s}/checkpoints/checkpoint_step_{step}.pth"
OBJ = ("qos", "energy", "fair")


def episode(trainer, omega, seed, steps):
    agent, env = trainer.agent, trainer.eval_env
    uids = sorted(env.uavs)
    np.random.seed(seed)
    obs = env.reset()
    O, A, R = [], [], []
    for _ in range(steps):
        o = np.stack([obs[u] for u in uids])
        a = agent.select_joint_action(o, omega, explore=False)
        obs, _, done, info = env.step({u: a[i] for i, u in enumerate(uids)})
        O.append(o.ravel()); A.append(a.ravel()); R.append(info["reward_vector"])
        if done:
            break
    with torch.no_grad():
        q = agent.critic1(torch.as_tensor(np.array(O), dtype=torch.float32),
                          torch.as_tensor(np.array(A), dtype=torch.float32),
                          torch.as_tensor(np.tile(omega, (len(O), 1)), dtype=torch.float32)).numpy()
    return q, np.array(R, dtype=np.float64), done


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed_base", type=int, default=100000)
    p.add_argument("--steps", nargs="+", type=int, default=[1000000], help="Checkpoint training steps")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []
    for m, step, s in [(m, st, s) for m in args.methods for st in args.steps for s in args.seeds]:
            with contextlib.redirect_stdout(io.StringIO()):
                tr = load_trainer(f"config/train/moma_ac_{m}.yaml", CK.format(m=m, s=s, step=step))
            g, T = tr.cfg["gamma"], tr.cfg["eval_steps"]
            for cname, w in CORNERS.items():
                omega = np.array(w, np.float32)
                for k in range(args.n_episodes):
                    q, r, terminal = episode(tr, omega, args.seed_base + k, T)
                    n = len(r)
                    mc = np.zeros_like(r)
                    acc = np.zeros(3)
                    for t in reversed(range(n)):
                        acc = r[t] + g * acc
                        mc[t] = acc
                    for t in range(n):
                        tail = 0.0 if terminal else g ** (n - t) / (1 - g)
                        row = {"method": m, "step": step, "gamma": g, "seed": s, "corner": cname, "ep": k, "t": t,
                               "tail": tail}
                        for j, o in enumerate(OBJ):
                            row[f"q_{o}"], row[f"mc_{o}"] = float(q[t, j]), float(mc[t, j])
                            # Stationary continuation: the episode's mean reward kept forever after step T
                            row[f"stat_{o}"] = float(mc[t, j] + tail * r[:, j].mean())
                        rows.append(row)
            print(f"{m} step {step} seed {s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}.csv", index=False, float_format="%.5g")

    # Wrong under any continuation: below MC for the [0,1] objectives, above MC for energy (in [-1,0])
    df["wrong_qos"] = df.q_qos < df.mc_qos
    df["wrong_fair"] = df.q_fair < df.mc_fair
    df["wrong_energy"] = df.q_energy > df.mc_energy
    df["neg_qos"] = df.q_qos < 0
    df["neg_fair"] = df.q_fair < 0
    cols = ["q_qos", "mc_qos", "stat_qos", "q_fair", "stat_fair", "q_energy", "stat_energy", "tail",
            "wrong_qos", "wrong_fair", "wrong_energy", "neg_qos", "neg_fair"]
    order = {m: i for i, m in enumerate(args.methods)}
    summ = df.groupby(["method", "step", "corner"])[cols].mean().sort_index(
        key=lambda x: x.map(order) if x.name == "method" else x)
    with pd.option_context("display.width", 250):
        print("\nMeans over seeds x episodes x steps. q_*: critic1; mc_*: discounted return to the episode's end; "
              "tail: gamma^(T-t)/(1-gamma),\nthe most the continuation past step T can add; stat_*: MC + tail x "
              "the episode's mean reward (the value if the step reward stayed at that mean after step T);\n"
              "wrong_*: share of states where Q is outside [MC, MC + tail] (energy: [MC - tail, MC]); neg_*: share with Q < 0")
        print(summ.to_string(float_format="%.3f"))


if __name__ == "__main__":
    main()

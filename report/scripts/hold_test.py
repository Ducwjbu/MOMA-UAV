"""Does the back-and-forth flight help? (docs/thesis_brainstorm.md, "phép thử đứng yên", 2026-11-13)
Same policy (power and bandwidth its own), movement replaced from step S0 on:
  policy    : unchanged
  hold_now  : hover where the UAV is at step S0 (one end of its shuttle)
  hold_mid  : fly to the mean position of the policy's own steps S0+ in that episode, then hover
Validation episodes (seed_base 100000), corners qos/energy/fair, checkpoints 1M. Metrics over steps S0+.
Run from the repo root: py report/scripts/hold_test.py evals/diag/hold_test.csv  (about 1 h on CPU)
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
os.environ.setdefault("MOMA_DEVICE", "cpu")
import numpy as np, pandas as pd
import agents_moma_ac, prior
from eval_checkpoint import load_trainer
from visualize.record import _inner_step_cell

S0 = 3
N_EP = 10
SEED_BASE = 100000
TRAIN_SEEDS = [0, 1, 2]
METHODS = ["ma3_prior", "ma3"]
CORNERS = {"qos": [1, 0, 0], "energy": [0, 1, 0], "fair": [0, 0, 1]}
SL = ("embb", "urllc", "mmtc")
CK = "runs/saved_models_moma_ac_{m}/seed_{s}/checkpoints/checkpoint_step_1000000.pth"


def run(trainer, omega, seed, mode, targets=None):
    env = trainer.eval_env
    cell = _inner_step_cell(env.step)
    base = cell.cell_contents if cell is not None else env.step
    st = {"t": -1, "hold": None}
    log = []

    def recv(actions):
        st["t"] += 1
        t = st["t"]
        if mode != "policy" and t >= S0:
            if st["hold"] is None:
                st["hold"] = ({u: env.uavs[u].position.copy() for u in env.uavs} if mode == "hold_now" else targets)
            actions = {u: np.array(a, dtype=np.float64) for u, a in actions.items()}
            for u, a in actions.items():
                a[0:3] = prior.move_to(env, env.uavs[u].position, st["hold"][u])
        pre = {u: env.uavs[u].position.copy() for u in env.uavs}
        out = base(actions)
        info = out[3]
        speed = np.mean([np.hypot(*(env.uavs[u].position[:2] - pre[u][:2])) / env.T_L for u in env.uavs])
        log.append({"t": t, "qos": info["reward_vector"][0], "energy": info["reward_vector"][1],
                    "fair": info["reward_vector"][2], "coverage": info["coverage"], "speed": speed,
                    **{f"sla_{s}": info[f"sla_{s}"] for s in SL},
                    "pos": {u: env.uavs[u].position.copy() for u in env.uavs}})
        return out

    if cell is not None:
        cell.cell_contents = recv
    else:
        env.step = recv
    default = agents_moma_ac.EVAL_SEED
    try:
        agents_moma_ac.EVAL_SEED = seed
        trainer.agent.eval_omega(env, np.array(omega, np.float32), 1, trainer.cfg["eval_steps"])
    finally:
        agents_moma_ac.EVAL_SEED = default
        if cell is not None:
            cell.cell_contents = base
        else:
            env.step = base
    return log


rows = []
t0 = time.time()
for m in METHODS:
    for s in TRAIN_SEEDS:
        tr = load_trainer(f"config/train/moma_ac_{m}.yaml", CK.format(m=m, s=s))
        for cname, w in CORNERS.items():
            for k in range(N_EP):
                seed = SEED_BASE + k
                logs = {"policy": run(tr, w, seed, "policy")}
                mid = {u: np.mean([r["pos"][u] for r in logs["policy"] if r["t"] >= S0], axis=0)
                       for u in logs["policy"][0]["pos"]}
                logs["hold_now"] = run(tr, w, seed, "hold_now")
                logs["hold_mid"] = run(tr, w, seed, "hold_mid", mid)
                for v, lg in logs.items():
                    df = pd.DataFrame([{k2: v2 for k2, v2 in r.items() if k2 != "pos"} for r in lg])
                    on = df[df.t >= S0]
                    rows.append({"method": m, "train_seed": s, "corner": cname, "ep": k, "variant": v,
                                 **on.drop(columns="t").mean(skipna=True).to_dict()})
        print(f"{m} seed {s} done, {time.time() - t0:.0f} s", flush=True)

out = pd.DataFrame(rows)
out.to_csv(sys.argv[1], index=False)

cols = ["qos", "energy", "fair", "coverage", "speed", "sla_urllc", "sla_mmtc", "sla_embb"]
print("\nMeans over train seeds x episodes, steps 3+")
print(out.groupby(["method", "corner", "variant"])[cols].mean().to_string(float_format="%.3f"))
print("\nPaired difference variant - policy (mean [95% CI], n = seeds x episodes)")
key = ["method", "train_seed", "corner", "ep"]
base = out[out.variant == "policy"].set_index(key)
for v in ("hold_now", "hold_mid"):
    d = out[out.variant == v].set_index(key)[cols] - base[cols]
    for (m, c), g in d.groupby(level=["method", "corner"]):
        parts = []
        for col in ("qos", "energy", "fair", "coverage"):
            x = g[col].dropna().values
            se = x.std(ddof=1) / np.sqrt(len(x))
            parts.append(f"{col} {x.mean():+.3f} [{x.mean() - 1.96 * se:+.3f}, {x.mean() + 1.96 * se:+.3f}]")
        print(f"{v:9s} {m:10s} {c:7s} " + "  ".join(parts))

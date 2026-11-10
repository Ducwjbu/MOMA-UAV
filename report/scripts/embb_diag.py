"""Why do so few eMBB UEs meet their SLA? Log every eMBB UE-step with demand: which KPI fails and why.
Policies: ma3_prior seed 0 at omega (1,0,0), and the best QoS heuristic (hold triangle, URLLC bw, power 1).
Run from the repo root: py <this> <out.csv>
"""
import contextlib, io, os, sys
os.environ.setdefault("MOMA_DEVICE", "cpu")
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import agents_moma_ac
from eval_checkpoint import load_trainer
import eval_heuristics as eh

N_EP, SEED_BASE, STEPS = 5, 200000, 18
rows = []


def record(env, policy, ep, t):
    for ue in env.ues.values():
        if not ue.is_active or getattr(ue, "demand_bps", 0) <= 0 or ue.slice_type not in ("embb", "urllc"):
            continue
        covered = bool(ue.assigned_rb) and ue.assigned_uav is not None
        r = {"policy": policy, "ep": ep, "t": t, "slice": ue.slice_type, "covered": covered,
             "indoor": bool(np.any(getattr(ue, "indoor", False))),
             "demand_mbps": ue.demand_bps / 1e6, "n_rb": len(ue.assigned_rb) if ue.assigned_rb else 0}
        if covered:
            ts, thr = env._calculate_throughput_satisfaction(ue)
            ds, dly = env._calculate_delay_satisfaction(ue)
            rs, rel = env._calculate_reliability_satisfaction(ue)
            bd = env.delay_cache[ue.id]["breakdown"]
            uav = env.uavs[ue.assigned_uav]
            r.update(thr_mbps=thr / 1e6, thr_sat=ts, delay_ms=dly, delay_sat=ds, rel=rel, rel_sat=rs,
                     d_queue=bd["queuing"], d_trans=bd["transmission"], d_retx=bd["retransmission"],
                     d_proc=bd["processing"], drop=uav.avg_drop_rate, packet_bytes=ue.packet_bytes,
                     sinr_db=float(env._get_cached_sinr(ue, uav)))
        rows.append(r)


def wrap(env, policy, counter):
    inner = env.step
    def step(actions):
        out = inner(actions)
        record(env, policy, counter["ep"], counter["t"])
        counter["t"] += 1
        return out
    env.step = step
    inner_reset = env.reset
    def reset(*a, **k):
        counter["ep"] += 1; counter["t"] = 0
        return inner_reset(*a, **k)
    env.reset = reset


# Trained policy
trainer = load_trainer("config/train/moma_ac_ma3_prior.yaml",
                       "runs/saved_models_moma_ac_ma3_prior/seed_0/checkpoints/checkpoint_step_1000000.pth")
c = {"ep": -1, "t": 0}
wrap(trainer.eval_env, "ma3_prior", c)
for k in range(N_EP):
    agents_moma_ac.EVAL_SEED = SEED_BASE + k
    trainer.agent.eval_omega(trainer.eval_env, np.array([1, 0, 0], np.float32), 1, STEPS)

# Best QoS heuristic
from environment import NetworkSlicingEnv
with contextlib.redirect_stdout(io.StringIO()):
    env = NetworkSlicingEnv(config_path="config/environment/ma3.yaml")
c = {"ep": -1, "t": 0}
wrap(env, "heur_hold_urllc", c)
eh.run(env, 600, 0, 1.0, eh.URLLC_BW, "rotate", N_EP, SEED_BASE, STEPS)

df = pd.DataFrame(rows)
df.to_csv(sys.argv[1], index=False)
print(len(df), "rows")

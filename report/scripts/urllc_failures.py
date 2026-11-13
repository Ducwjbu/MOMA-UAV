"""Why do URLLC UEs miss their SLA? (report/05_han_che.md; audit of 2026-11-13)

With link_adaptation, URLLC's residual error is bler_target ** max_transmissions = 1e-5 one-shot, exactly the
target 1 - min_reliability. Reliability is then (1 - 1e-5) (1 - drop) (1 - handover loss), so a UE meets the
reliability part of its SLA only when drop and handover loss are exactly 0. This counts, for every active URLLC
UE-step at omega (1, 0, 0), the KPI that fails: not covered, no RBs, delay, throughput, or reliability alone
(and of those, how many are in a step with a handover).

Run from the repo root (checkpoints in runs/):
  py report/scripts/urllc_failures.py              # ma3_prior seed 0, 10 validation episodes (100000+)
  py report/scripts/urllc_failures.py --method ma3 --seed 1
"""
import argparse
import contextlib
import io
import os
import sys

os.environ.setdefault("MOMA_DEVICE", "cpu")
sys.path.insert(0, os.getcwd())
import numpy as np

import agents_moma_ac
from eval_checkpoint import load_trainer

EPS = 1e-9


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--method", default="ma3_prior")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--seed_base", type=int, default=100000)
    args = p.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        tr = load_trainer(f"config/train/moma_ac_{args.method}.yaml",
                          f"runs/saved_models_moma_ac_{args.method}/seed_{args.seed}/checkpoints/"
                          "checkpoint_step_1000000.pth")
    env = tr.eval_env
    inner = env.step
    cnt = dict.fromkeys(["ue_steps", "met", "not_covered", "no_rbs", "delay", "throughput", "reliability_only",
                         "reliability_only_in_handover"], 0)

    def step(a):
        out = inner(a)
        for ue in env.ues.values():
            if ue.slice_type != "urllc" or not ue.is_active:
                continue
            cnt["ue_steps"] += 1
            if ue.assigned_uav is None:
                cnt["not_covered"] += 1
                continue
            if not ue.assigned_rb:
                cnt["no_rbs"] += 1
                continue
            t, d, r = ue.throughput_satisfaction, ue.delay_satisfaction, ue.reliability_satisfaction
            if min(t, d, r) >= 1 - EPS:
                cnt["met"] += 1
            elif t >= 1 - EPS and d >= 1 - EPS:
                cnt["reliability_only"] += 1
                if getattr(ue, "handover_penalty_ms", 0) > 0:
                    cnt["reliability_only_in_handover"] += 1
            elif d < 1 - EPS:
                cnt["delay"] += 1
            else:
                cnt["throughput"] += 1
        return out

    env.step = step
    default = agents_moma_ac.EVAL_SEED
    try:
        for k in range(args.n_episodes):
            agents_moma_ac.EVAL_SEED = args.seed_base + k
            tr.agent.eval_omega(env, np.array([1, 0, 0], np.float32), 1, tr.cfg["eval_steps"])
    finally:
        agents_moma_ac.EVAL_SEED = default
        env.step = inner
    print(cnt)
    served = cnt["met"] + cnt["delay"] + cnt["throughput"] + cnt["reliability_only"]
    print(f"of covered UE-steps with RBs ({served}): met {cnt['met'] / served:.2f}, reliability alone "
          f"{cnt['reliability_only'] / served:.2f} ({cnt['reliability_only_in_handover']} of those in a handover "
          f"step), delay {cnt['delay'] / served:.2f}, throughput {cnt['throughput'] / served:.2f}")
    print(f"met if reliability alone counted as met: {(cnt['met'] + cnt['reliability_only']) / served:.2f}")


if __name__ == "__main__":
    main()

"""How much of a UAV's shared queue is served by mMTC capacity? (report/05_han_che.md, item 4; audit of 2026-11-13)

Without per_slice_queues each UAV has one queue for all slices, served at the sum of its UEs' rates. mMTC UEs
offer almost no load (about 0.06 bps) but get a rate when they have RBs, so their rate adds service capacity that
eMBB and URLLC traffic uses. This measures, on the hovering r=600 triangle with power 1 and the URLLC-first
bandwidth split of eval_heuristics.py, 10 validation episodes (100000+): the mean capacity and load per slice per
UAV-step, the share of the service rate that comes from mMTC UEs, and how many UAV-steps are stable (load below
service rate) only thanks to it.

Run from the repo root: py report/scripts/queue_pooling.py
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np

import eval_formation as ef
import eval_heuristics as eh

SL = ("embb", "urllc", "mmtc")


def main(n_ep=10, seed_base=100000):
    with contextlib.redirect_stdout(io.StringIO()):
        env = eh.NetworkSlicingEnv("config/environment/ma3.yaml")
    pos = ef.formation_positions(env, 600)
    a = np.concatenate([[0, 0, 0, 1], eh.URLLC_BW]).astype(np.float32)
    rows = []
    for ep in range(n_ep):
        np.random.seed(seed_base + ep)
        env.reset()
        for u, p in zip(env.uavs.values(), pos):
            u.position = p.copy()
            env.charge_outbound_flight(u)
        for _ in range(18):
            _, _, done, _ = env.step({u: a for u in env.uavs})
            for uav in env.uavs.values():
                at = [ue for ue in env.ues.values() if ue.assigned_uav == uav.id and ue.is_active]
                cap = [sum(env.throughput_cache[ue.id] for ue in at if ue.slice_type == s) for s in SL]
                load = [sum(getattr(ue, "demand_bps", 0) for ue in at if ue.slice_type == s) for s in SL]
                rows.append(cap + load)
            if done:
                break
    r = np.array(rows) / 1e6
    print("mean per UAV-step (Mbps): capacity eMBB %.1f URLLC %.1f mMTC %.1f | load eMBB %.1f URLLC %.1f "
          "mMTC %.6f" % tuple(r.mean(0)))
    cap, load = r[:, :3].sum(1), r[:, 3:].sum(1)
    print("share of the total service rate from mMTC UEs: %.2f" % (r[:, 2].sum() / cap.sum()))
    print("UAV-steps with load < service rate: %.2f; that would be so without the mMTC capacity: %.2f"
          % ((load < cap).mean(), (load < r[:, 0] + r[:, 1]).mean()))


if __name__ == "__main__":
    main()

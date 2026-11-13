"""
check_tip2.py (TIP 2 of docs/v2/plan.md)
========================================
The hovering r=600 triangle at power 1 on config/environment/v2.yaml (3 UAVs, without the shift section of TIP 4),
10 validation episodes (100000+), with the
bandwidth split evenly in two ways:
  demand_areas   the v1.1 action: 9 fractions of 1/9, round robin inside each demand area (v2.yaml without the
                 action section)
  slice_share    the v2 action: 3 shares of 1/3 and the scheduler (_allocate_rbs_slices)
Both use v2's per-slice queues and SLA thresholds, so only the action and the scheduler differ.

Reports per mode: the binary SLA share per slice (mean over steps, skipping nan), and idle RBs: the share of
UAV-steps where a UAV with a schedulable UE (traffic, serving UAV, non-zero rate) left RBs unused, and the mean
share of RBs unused then. TIP 2 passes if slice_share has no idle RB and a SLA share per slice at least that of
demand_areas.

Run from the repo root: py docs/v2/check_tip2.py
"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import yaml

with contextlib.redirect_stdout(io.StringIO()):
    from environment import NetworkSlicingEnv
from eval_formation import formation_positions

CONFIG = "config/environment/v2.yaml"
SLICES = ("embb", "urllc", "mmtc")


def v2_config(action=True, rb_sharing=None):
    """v2.yaml as it was at TIP 2: 3 UAVs, one sortie (no shift section); without the action section if asked,
    with slicing.rb_sharing set if given"""
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.pop("shift", None)
    cfg["system"]["num_uavs"] = 3
    if not action:
        cfg.pop("action", None)
    if rb_sharing is not None:
        cfg["slicing"]["rb_sharing"] = rb_sharing
    path = os.path.join(tempfile.mkdtemp(prefix="check_tip2_"), "v2_tip2.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def run(config, n_ep=10, seed_base=100000):
    with contextlib.redirect_stdout(io.StringIO()):
        env = NetworkSlicingEnv(config_path=config)
    pos = formation_positions(env, 600.0)
    n_bw = env.act_dim - 4
    action = np.concatenate([[0.0, 0.0, 0.0, 1.0], np.full(n_bw, 1.0 / n_bw)])
    sla, idle_steps, idle_share, busy_steps = [], 0, [], 0
    for ep in range(n_ep):
        np.random.seed(seed_base + ep)
        with contextlib.redirect_stdout(io.StringIO()):
            env.reset()
        for u, p in zip(env.uavs.values(), pos):
            u.position = p.copy()
            env.charge_outbound_flight(u)
        for _ in range(18):
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, done, info = env.step({u: action for u in env.uavs})
            sla.append([info[f"sla_{s}"] for s in SLICES])
            for uav in env.uavs.values():
                schedulable = [ue for ue in env.ues.values() if ue.is_active and ue.assigned_uav == uav.id
                               and getattr(ue, "demand_bps", 0) > 0 and env.sinr_cache.get((ue.id, uav.id)) is not None
                               and env._get_cached_throughput(ue) > 0]
                if not schedulable:
                    continue
                busy_steps += 1
                used = sum(rb.allocated_ue_id != -1 for rb in uav.RBs)
                if used < len(uav.RBs):
                    idle_steps += 1
                    idle_share.append(1.0 - used / len(uav.RBs))
            if done:
                break
    return np.nanmean(np.array(sla, dtype=float), axis=0), idle_steps / max(busy_steps, 1), \
        (float(np.mean(idle_share)) if idle_share else 0.0)


def main():
    res = {"demand_areas": run(v2_config(action=False)), "slice_share": run(v2_config())}
    # Reported only (added after the independent audit of 2026-11-14): the 9 even fractions with the unreserved
    # and idle RBs shared out (rb_sharing all), to separate what the scheduler adds from what sharing RBs adds
    res["demand_areas+all"] = run(v2_config(action=False, rb_sharing="all"))
    print("mode              SLA share eMBB / URLLC / mMTC     UAV-steps with idle RBs   idle share then")
    for mode, (sla, idle, share) in res.items():
        print(f"{mode:17s} {sla[0]:.3f} / {sla[1]:.3f} / {sla[2]:.3f}            {idle:.2f}                      "
              f"{share:.2f}")
    sla_new, idle_new, _ = res["slice_share"]
    sla_old = res["demand_areas"][0]
    ok = idle_new == 0 and bool(np.all(sla_new >= sla_old - 1e-12))
    print("\nTIP 2 check:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

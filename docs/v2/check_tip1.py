"""
check_tip1.py (TIP 1 of docs/v2/plan.md)
========================================
1. With sla_sharpness very large, each sla_<slice> objective of the reward vector must equal the binary share of the
   slice's UEs meeting their SLA (info['sla_<slice>']; a slice without demand counts 1 in the objective, nan in info)
   at every step. Run on 5 validation episodes (100000+) of config/environment/v2.yaml (3 UAVs, without the shift
   section of TIP 4) with random actions.
   In v2 the binary share itself uses the margin (margin >= -sla_tolerance), so this only checks the sigmoid limit.
2. Independently of the margin: for every served UE with demand, the v1.1 rule (the smallest of the rate, delay and
   reliability satisfactions is 1 within 1e-9, qos_aggregation min) must agree with margin >= -sla_tolerance.
   Added after an independent audit (2026-11-14) noted that check 1 alone is tautological.
3. At the default sharpness, prints the mean objective next to the mean binary share, and the share of UE-steps in
   each slice that fail on each KPI (rate, delay, reliability), to see where the v2 thresholds bind. The objective
   is not a smoothed share: at sharpness 4 a UE exactly at its SLA scores 0.5 and margin beyond it still counts.

Run from the repo root: py docs/v2/check_tip1.py
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    from environment import NetworkSlicingEnv

CONFIG = "config/environment/v2.yaml"
SLICES = ("embb", "urllc", "mmtc")


def v2_config():
    """v2.yaml as it was at TIP 1: 3 UAVs, one sortie (no shift section)"""
    import tempfile
    import yaml
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.pop("shift", None)
    cfg["system"]["num_uavs"] = 3
    path = os.path.join(tempfile.mkdtemp(prefix="check_tip1_"), "v2_tip1.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def run(sharpness, n_ep=5, seed_base=100000, kpi_log=None, agree=None):
    with contextlib.redirect_stdout(io.StringIO()):
        env = NetworkSlicingEnv(config_path=v2_config())
    if sharpness is not None:
        env.sla_sharpness = sharpness
    names = env.objective_names
    rows = []
    for ep in range(n_ep):
        act_rng = np.random.RandomState(ep)
        np.random.seed(seed_base + ep)
        with contextlib.redirect_stdout(io.StringIO()):
            env.reset()
        for _ in range(18):
            acts = {u: np.concatenate([act_rng.uniform(-1, 1, 3), act_rng.uniform(0.05, 1, 1),
                                       act_rng.dirichlet(np.ones(env.act_dim - 4))]) for u in env.uavs}
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, done, info = env.step(acts)
            rv = info["reward_vector"]
            rows.append([rv[names.index(f"sla_{s}")] for s in SLICES] +
                        [info[f"sla_{s}"] for s in SLICES])
            if kpi_log is not None:
                for ue in env.ues.values():
                    if not ue.is_active or getattr(ue, "demand_bps", 0) <= 0:
                        continue
                    p = env.qos_profiles[ue.slice_type]
                    if not ue.assigned_rb:
                        kpi_log.append((ue.slice_type, "not served"))
                        continue
                    if p.min_rate > 0 and ue.throughput < env._required_rate(ue) * (1 - 1e-9):
                        kpi_log.append((ue.slice_type, "rate"))
                    if ue.latency_ms > p.max_latency * (1 + 1e-9):
                        kpi_log.append((ue.slice_type, "delay"))
                    if ue.failure_rate > (1 - p.min_reliability) * (1 + 1e-9):
                        kpi_log.append((ue.slice_type, "reliability"))
                    kpi_log.append((ue.slice_type, "ue-steps"))
                    if agree is not None:
                        thr_sat, thr = env._calculate_throughput_satisfaction(ue)
                        delay_sat, delay = env._calculate_delay_satisfaction(ue)
                        rel_sat = env._calculate_reliability_satisfaction(ue)
                        rel_sat = rel_sat[0] if isinstance(rel_sat, tuple) else rel_sat
                        old = min(thr_sat, delay_sat, rel_sat) >= 1.0 - 1e-9
                        new = env._sla_margin(ue, thr, delay) >= -env.sla_tolerance
                        agree.append(old == new)
            if done:
                break
    return np.array(rows, dtype=float), names


def main():
    r, names = run(1e12)
    obj, share = r[:, :3], r[:, 3:]
    expect = np.where(np.isnan(share), 1.0, share)
    diff = np.abs(obj - expect)
    print(f"objectives {names}; {len(r)} steps")
    print("sharpness 1e12: max |objective - binary share| per slice:", np.round(diff.max(0), 12))
    ok = bool(np.all(diff < 1e-6))

    log, agree = [], []
    r, _ = run(None, kpi_log=log, agree=agree)
    print(f"\nv1.1 rule against the margin, served UE-steps: {len(agree)}, disagreements {len(agree) - sum(agree)}")
    ok = ok and len(agree) > 0 and all(agree)
    print("\ndefault sharpness: mean objective / mean binary share (nan skipped)")
    for j, s in enumerate(SLICES):
        print(f"  {s:6s} {np.mean(r[:, j]):.3f} / {np.nanmean(r[:, 3 + j]):.3f}")
    from collections import Counter
    c = Counter(log)
    print("\nshare of UE-steps with demand failing each KPI (a UE can fail several)")
    for s in SLICES:
        n = c[(s, "ue-steps")] + c[(s, "not served")]
        if n:
            print(f"  {s:6s} n={n:5d}  not served {c[(s, 'not served')] / n:.2f}  rate {c[(s, 'rate')] / n:.2f}  "
                  f"delay {c[(s, 'delay')] / n:.2f}  reliability {c[(s, 'reliability')] / n:.2f}")
    print("\nTIP 1 sharpness check:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

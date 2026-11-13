"""
check_tip6.py (TIP 6 of docs/v2/plan.md)
========================================
Load calibration for a variable fleet. The load is the scenario's, not the fleet's. eMBB UEs carry
streams_per_ue co-located video streams on average (a team's cameras), and it is the largest value on a 0.1 grid
for which a reference fleet of S_REF = 6 serving positions (the middle of the trained 2-10) still meets the SLA of
at least 80% of the covered eMBB UE-steps (the plan's 0.8-0.9: hard but reachable). Smaller fleets are then short of capacity as well as coverage,
larger ones have room, and the slice shares of the action decide who gets the RBs.

A first version set the mean offered load equal to the mean capacity of S_REF (as v1.1 did for its 3 UAVs): 17
streams per UE. It passed the then criteria but left eMBB unreachable (SLA share 0.07 at S = 6, 0.17 at S = 10):
the load sits on single UEs that need 34 Mbps each, which cell-edge UEs can't get whatever the share. Capacity
at the mean rate per RB ignores where the load is.

All runs: config/environment/v2.yaml with n_serving fixed, the k-means prior with no residual (prior.attach, mode
demand), power 1, validation episodes 100000+, 40 steps, counted from step 6.

  py docs/v2/check_tip6.py calibrate
      At S_REF, streams_per_ue = 1, 2, ... while the covered eMBB SLA share stays at least 0.80, then on a 0.1
      grid above the last such whole number: the covered SLA share, the share failing on rate, and offered load
      against capacity (capacity = sum over serving UAVs of total_rbs x the mean rate of one RB over the UEs it
      serves, env._provisional_rb_rates). Prints the largest value with a covered SLA share of at least 0.80.
      A first sweep with threshold 0.75 (not the plan's 0.8) found no whole number meeting the criteria below
      (2: 0.899 met, 5.4% rate failures; 3: 0.722), hence the finer grid.

  py docs/v2/check_tip6.py check
      With v2.yaml as it is. PASS (criteria set before the first run) if
      (a) the arriving population averages 91 within 2%;
      (b) at S_REF the covered eMBB SLA share is between 0.80 and 0.90 (reachable, not easy);
      (c) at S_REF at least 10% of the covered eMBB UE-steps fail on rate, and fewer at S = 10 than at S = 2;
      (d) at S_REF, slice shares (0.6, 0.2, 0.2) give a higher eMBB SLA share than (0.2, 0.6, 0.2), and the latter
          a higher URLLC SLA share, each by at least 0.02.
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
import prior

CONFIG = "config/environment/v2.yaml"
SLICES = ("embb", "urllc", "mmtc")
S_REF = 6


def config(S, streams=None):
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["shift"]["n_serving"] = [S, S]
    if streams is not None:
        cfg["traffic_patterns"]["embb"]["streams_per_ue"] = streams
    path = os.path.join(tempfile.mkdtemp(prefix="check_tip6_"), f"v2_s{S}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def run(S, streams=None, shares=None, n_ep=3, steps=40, skip=5, seed_base=100000):
    with contextlib.redirect_stdout(io.StringIO()):
        env = NetworkSlicingEnv(config_path=config(S, streams))
    uids = sorted(env.uavs)
    prior.attach(env, uids, 600.0, "demand")
    shares = np.full(env.act_dim - 4, 1.0 / (env.act_dim - 4)) if shares is None else np.asarray(shares)
    action = np.concatenate([[0.0, 0.0, 0.0, 1.0], shares])
    out = {"pop": [], "capacity": [], "offered": {s: [] for s in SLICES}, "sla": {s: [] for s in SLICES},
           "embb_covered": 0, "embb_rate_fail": 0, "embb_ok": 0}
    for ep in range(n_ep):
        np.random.seed(seed_base + ep)
        with contextlib.redirect_stdout(io.StringIO()):
            env.reset()
        for t in range(steps):
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, done, info = env.step({u: action for u in uids})
            out["pop"].append(sum(ue.is_active and not getattr(ue, "fixed", False) for ue in env.ues.values()))
            if t >= skip:
                with_demand = [ue for ue in env.ues.values() if ue.is_active and getattr(ue, "demand_bps", 0) > 0]
                for s in SLICES:
                    out["offered"][s].append(sum(ue.demand_bps for ue in with_demand if ue.slice_type == s))
                    if not np.isnan(info[f"sla_{s}"]):
                        out["sla"][s].append(info[f"sla_{s}"])
                served = [ue for ue in with_demand if ue.assigned_uav is not None]
                rates = env._provisional_rb_rates(served)
                cap = 0.0
                for uav in env.uavs.values():
                    r = [rates[ue.id] for ue in served if ue.assigned_uav == uav.id]
                    if uav.active and r:
                        cap += env.total_rbs * float(np.mean(r))
                out["capacity"].append(cap)
                for ue in served:
                    if ue.slice_type == "embb":
                        out["embb_covered"] += 1
                        thr = env._get_cached_throughput(ue)
                        out["embb_rate_fail"] += thr < env._required_rate(ue) * (1 - 1e-9)
                        out["embb_ok"] += env._sla_margin(ue, thr, ue.latency_ms) >= -env.sla_tolerance
            if done:
                break
    return out


def calibrate(threshold=0.80):
    def measure(w):
        o = run(S_REF, streams=w)
        ok = o["embb_ok"] / max(o["embb_covered"], 1)
        rf = o["embb_rate_fail"] / max(o["embb_covered"], 1)
        ratio = sum(np.mean(o["offered"][s]) for s in SLICES) / np.mean(o["capacity"])
        print(f"{w:7.1f}  {ok:16.3f}  {rf:17.3f}  {ratio:18.3f}", flush=True)
        return ok

    print("streams  covered eMBB SLA  covered rate-fail  offered / capacity")
    best, w = None, 1
    while measure(float(w)) >= threshold:
        best, w = float(w), w + 1
    for k in range(1, 10):
        if best is None or measure(round(best + 0.1, 1)) < threshold:
            break
        best = round(best + 0.1, 1)
    print(f"streams_per_ue = {best}")


def check():
    rows = {}
    print("  S   capacity Mbps  offered Mbps  covered eMBB: SLA  rate-fail   SLA share eMBB / URLLC / mMTC")
    for S in (2, 4, 6, 8, 10):
        o = run(S)
        rows[S] = o
        off = sum(np.mean(o["offered"][s]) for s in SLICES)
        rf = o["embb_rate_fail"] / max(o["embb_covered"], 1)
        okc = o["embb_ok"] / max(o["embb_covered"], 1)
        print(f"{S:3d}   {np.mean(o['capacity']) / 1e6:12.1f}  {off / 1e6:12.1f}  {okc:17.3f}  {rf:9.3f}   "
              + " / ".join(f"{np.mean(o['sla'][s]):.3f}" for s in SLICES), flush=True)
    pop = np.mean(np.concatenate([rows[S]["pop"] for S in rows]))
    ref = rows[S_REF]
    ratio = sum(np.mean(ref["offered"][s]) for s in SLICES) / np.mean(ref["capacity"])
    rf = {S: rows[S]["embb_rate_fail"] / max(rows[S]["embb_covered"], 1) for S in rows}
    a_embb = run(S_REF, shares=[0.6, 0.2, 0.2])
    a_urllc = run(S_REF, shares=[0.2, 0.6, 0.2])
    sla = lambda o, s: float(np.mean(o["sla"][s]))
    print(f"\npopulation {pop:.1f} (91); offered / capacity at S={S_REF}: {ratio:.3f}")
    print(f"shares (0.6,0.2,0.2): SLA eMBB {sla(a_embb, 'embb'):.3f}, URLLC {sla(a_embb, 'urllc'):.3f}; "
          f"(0.2,0.6,0.2): eMBB {sla(a_urllc, 'embb'):.3f}, URLLC {sla(a_urllc, 'urllc'):.3f}")
    checks = {
        "a population": abs(pop - 91) <= 0.02 * 91,
        "b reachable at S_REF": 0.80 <= ref["embb_ok"] / max(ref["embb_covered"], 1) <= 0.90,
        "c capacity binds": rf[S_REF] >= 0.10 and rf[10] < rf[2],
        "d shares matter": sla(a_embb, "embb") - sla(a_urllc, "embb") >= 0.02
                           and sla(a_urllc, "urllc") - sla(a_embb, "urllc") >= 0.02,
    }
    for k, v in checks.items():
        print(f"  {k}: {'ok' if v else 'FAIL'}")
    ok = all(checks.values())
    print("TIP 6 check:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    {"calibrate": calibrate, "check": check}[sys.argv[1] if len(sys.argv) > 1 else "check"]()

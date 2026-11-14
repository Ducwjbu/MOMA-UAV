"""
check_tip6.py (TIP 6 of docs/v2/plan.md)
========================================
Load calibration for a variable fleet. The load is the scenario's, not the fleet's. Each eMBB UE carries
streams_per_ue co-located 2 Mbps video streams on average (a team's cameras; with streams_dist poisson each UE
draws its number once). The mean is set so that a reference fleet of S_REF = 6 serving positions (the middle of
the trained 2-10) meets the SLA of about 80% of the covered eMBB UE-steps: hard but reachable. Smaller fleets are
then short of capacity as well as coverage, larger ones have room, and the slice shares of the action decide who
gets the RBs.

Deviation from the plan: the plan asks for "about 0.8-0.9 per emergency slice"; this targets covered eMBB UEs
only, since coverage (not load) sets the uncovered share and the overall share would mix the two. Overall shares
per slice are printed too.

All runs: config/environment/v2.yaml with n_serving fixed, the k-means prior with no residual (prior.attach, mode
demand), power 1, 40 steps per episode, counted from step 6.

  py docs/v2/check_tip6.py calibrate
      On CALIB_SEEDS (30 validation episodes, 100000+): the covered eMBB SLA share at S_REF for means 1, 2, ...
      until one falls below 0.80, then in steps of 0.1 above the last whole number at 0.80 or more until a step
      falls below. Prints the last mean before that first failure (a step search: the share is not monotone in
      the mean, on these episodes 2.2 gives 0.823 and 2.3 0.833, and values above the first failure are not tried).

  py docs/v2/check_tip6.py check
      On CHECK_SEEDS (30 other validation episodes, 100100+), with v2.yaml as it is. PASS if
      (a) the arriving population averages 91 within 2%;
      (b) the covered eMBB SLA share at S_REF, with a 90% bootstrap CI over episodes, has its CI within
          [0.70, 0.90] and contains 0.80, i.e. the calibration carries over to unseen episodes;
      (c) at S_REF at least 5% of the covered eMBB UE-steps fail on rate, and fewer at S = 10 than at S = 2;
      (d) at S_REF, slice shares (0.6, 0.2, 0.2) give a higher eMBB SLA share than (0.2, 0.6, 0.2), and the latter
          a higher URLLC SLA share, each by at least 0.02.

History (independent audit of 2026-11-14): the first version calibrated and checked on the same 3 episodes, so
(b) held by construction, while other batches of 3 episodes moved the covered share by about 0.13; its (c) asked
for 10% rate failures and was missed at 0.0997. Before that, a calibration to offered load = capacity of S_REF
(17 streams per UE) passed the then criteria but left eMBB unreachable (SLA share 0.07 at S = 6). The criteria
above were set after the audit and before running this version: separate seed sets, a CI, and a (c) threshold
that only asks capacity to bind at all (5%). With 10 episodes per set the calibration gave 2.4 and the check
0.756 [0.656, 0.844]: (b) failed on the CI width, since episodes differ a lot (random site and traffic map).
Both sets were then raised to 30 episodes, criteria unchanged.
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
CALIB_SEEDS = 100000
CHECK_SEEDS = 100100
N_EP = 30


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


def run(S, streams=None, shares=None, n_ep=N_EP, steps=40, skip=5, seed_base=CHECK_SEEDS):
    """Per-episode counts: covered eMBB UE-steps, of which meeting the SLA and failing on rate; SLA shares per
    slice (mean over steps, nan skipped); offered load and capacity (capacity = sum over serving UAVs of total_rbs
    x the mean rate of one RB over the UEs it serves, env._provisional_rb_rates)"""
    with contextlib.redirect_stdout(io.StringIO()):
        env = NetworkSlicingEnv(config_path=config(S, streams))
    uids = sorted(env.uavs)
    prior.attach(env, uids, 600.0, "demand")
    shares = np.full(env.act_dim - 4, 1.0 / (env.act_dim - 4)) if shares is None else np.asarray(shares)
    action = np.concatenate([[0.0, 0.0, 0.0, 1.0], shares])
    eps = []
    for ep in range(n_ep):
        np.random.seed(seed_base + ep)
        with contextlib.redirect_stdout(io.StringIO()):
            env.reset()
        e = {"pop": [], "capacity": [], "offered": [], "sla": {s: [] for s in SLICES},
             "covered": 0, "ok": 0, "rate_fail": 0}
        for t in range(steps):
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, done, info = env.step({u: action for u in uids})
            e["pop"].append(sum(ue.is_active and not getattr(ue, "fixed", False) for ue in env.ues.values()))
            if t >= skip:
                with_demand = [ue for ue in env.ues.values() if ue.is_active and getattr(ue, "demand_bps", 0) > 0]
                e["offered"].append(sum(ue.demand_bps for ue in with_demand))
                for s in SLICES:
                    if not np.isnan(info[f"sla_{s}"]):
                        e["sla"][s].append(info[f"sla_{s}"])
                served = [ue for ue in with_demand if ue.assigned_uav is not None]
                rates = env._provisional_rb_rates(served)
                cap = 0.0
                for uav in env.uavs.values():
                    r = [rates[ue.id] for ue in served if ue.assigned_uav == uav.id]
                    if uav.active and r:
                        cap += env.total_rbs * float(np.mean(r))
                e["capacity"].append(cap)
                for ue in served:
                    if ue.slice_type == "embb":
                        thr = env._get_cached_throughput(ue)
                        e["covered"] += 1
                        e["rate_fail"] += thr < env._required_rate(ue) * (1 - 1e-9)
                        e["ok"] += env._sla_margin(ue, thr, ue.latency_ms) >= -env.sla_tolerance
            if done:
                break
        eps.append(e)
    return eps


def covered_share(eps, key="ok"):
    return sum(e[key] for e in eps) / max(sum(e["covered"] for e in eps), 1)


def bootstrap_ci(eps, key="ok", n=2000, level=0.90, seed=0):
    """Percentile CI of the pooled share, resampling episodes"""
    rng = np.random.RandomState(seed)
    num = np.array([e[key] for e in eps], dtype=float)
    den = np.array([e["covered"] for e in eps], dtype=float)
    idx = rng.randint(0, len(eps), (n, len(eps)))
    boot = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return tuple(np.quantile(boot, [(1 - level) / 2, (1 + level) / 2]))


def sla_mean(eps, s):
    vals = [np.mean(e["sla"][s]) for e in eps if e["sla"][s]]
    return float(np.mean(vals)) if vals else float("nan")


def calibrate(threshold=0.80):
    def measure(w):
        eps = run(S_REF, streams=w, seed_base=CALIB_SEEDS)
        ok, rf = covered_share(eps), covered_share(eps, "rate_fail")
        ratio = np.mean([np.mean(e["offered"]) for e in eps]) / np.mean([np.mean(e["capacity"]) for e in eps])
        print(f"{w:7.1f}  {ok:16.3f}  {rf:17.3f}  {ratio:18.3f}", flush=True)
        return ok

    print(f"calibration episodes {CALIB_SEEDS}..{CALIB_SEEDS + N_EP - 1}, S = {S_REF}")
    print("streams  covered eMBB SLA  covered rate-fail  offered / capacity")
    best, w = None, 1
    while measure(float(w)) >= threshold:
        best, w = float(w), w + 1
    for _ in range(9):
        if best is None or measure(round(best + 0.1, 1)) < threshold:
            break
        best = round(best + 0.1, 1)
    print(f"streams_per_ue = {best}")


def check():
    rows = {}
    print(f"check episodes {CHECK_SEEDS}..{CHECK_SEEDS + N_EP - 1}")
    print("  S   capacity Mbps  offered Mbps  covered eMBB: SLA [90% CI]   rate-fail   SLA share eMBB / URLLC / mMTC")
    for S in (2, 4, 6, 8, 10):
        eps = run(S)
        rows[S] = eps
        lo, hi = bootstrap_ci(eps)
        print(f"{S:3d}   {np.mean([np.mean(e['capacity']) for e in eps]) / 1e6:12.1f}  "
              f"{np.mean([np.mean(e['offered']) for e in eps]) / 1e6:12.1f}  "
              f"{covered_share(eps):8.3f} [{lo:.3f}, {hi:.3f}]  {covered_share(eps, 'rate_fail'):9.3f}   "
              + " / ".join(f"{sla_mean(eps, s):.3f}" for s in SLICES), flush=True)
    pop = np.mean([p for S in rows for e in rows[S] for p in e["pop"]])
    lo, hi = bootstrap_ci(rows[S_REF])
    rf = {S: covered_share(rows[S], "rate_fail") for S in rows}
    a_embb = run(S_REF, shares=[0.6, 0.2, 0.2])
    a_urllc = run(S_REF, shares=[0.2, 0.6, 0.2])
    print(f"\npopulation {pop:.1f} (91); covered eMBB SLA at S={S_REF}: {covered_share(rows[S_REF]):.3f} "
          f"[{lo:.3f}, {hi:.3f}]")
    print(f"shares (0.6,0.2,0.2): SLA eMBB {sla_mean(a_embb, 'embb'):.3f}, URLLC {sla_mean(a_embb, 'urllc'):.3f}; "
          f"(0.2,0.6,0.2): eMBB {sla_mean(a_urllc, 'embb'):.3f}, URLLC {sla_mean(a_urllc, 'urllc'):.3f}")
    checks = {
        "a population": abs(pop - 91) <= 0.02 * 91,
        "b calibration carries over": 0.70 <= lo and hi <= 0.90 and lo <= 0.80 <= hi,
        "c capacity binds": rf[S_REF] >= 0.05 and rf[10] < rf[2],
        "d shares matter": sla_mean(a_embb, "embb") - sla_mean(a_urllc, "embb") >= 0.02
                           and sla_mean(a_urllc, "urllc") - sla_mean(a_embb, "urllc") >= 0.02,
    }
    for k, v in checks.items():
        print(f"  {k}: {'ok' if v else 'FAIL'}")
    ok = all(checks.values())
    print("TIP 6 check:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    {"calibrate": calibrate, "check": check}[sys.argv[1] if len(sys.argv) > 1 else "check"]()

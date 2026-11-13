"""
check_tip4.py (TIP 4 of docs/v2/plan.md)
========================================
Whole shifts of config/environment/v2.yaml (240 steps), with the serving positions hovering where they start at full
power and even slice shares, on validation seeds (100000+). Two set-ups, each with n_serving fixed:
  A  S = 3 with 6 BS60 and 3 spares: enough charged pairs and airframes, so relays should leave no position empty
  B  S = 10 with 3 BS60 for the team: the depot can't keep up, so airframes wait for a charged pair
and, for the record, the v2 defaults (2 BS60 and 0.5 spare per serving position) and an under-equipped team
(1 BS60 per position), each at S = 3 and S = 10.

Checked at every step: the episode ends with done exactly at step 240; no serving battery below 0 and no airframe
short of energy for its return leg; the number of battery pairs is conserved (depot + airframes holding a pair).
Over the shift: every serving position is handed over at least once (A, B). Reported: relays, the share of
position-steps left empty, airframe-steps waiting for a pair, battery swaps per airframe, ms per step.
PASS: all checks hold in every set-up, A has no empty position, and B has at least one airframe waiting.

Run from the repo root: py docs/v2/check_tip4.py
"""
import contextlib
import io
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import yaml

with contextlib.redirect_stdout(io.StringIO()):
    from environment import NetworkSlicingEnv

CONFIG = "config/environment/v2.yaml"


def config_with(**shift_keys):
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["shift"].update(shift_keys)
    path = os.path.join(tempfile.mkdtemp(prefix="check_tip4_"), "v2_shift.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def pairs_held(env):
    st = env.shift_state
    return len(st["pool"]) + sum(a["state"] != "waiting" for a in st["airframes"])


def run(config, n_ep=2, seed_base=100000):
    with contextlib.redirect_stdout(io.StringIO()):
        env = NetworkSlicingEnv(config_path=config)
    n_share = env.act_dim - 4
    action = np.concatenate([[0.0, 0.0, 0.0, 1.0], np.full(n_share, 1.0 / n_share)])
    res = {"ok": True, "relays": 0, "empty": 0, "pos_steps": 0, "waiting": 0, "swaps": [], "ms": [],
           "msgs": []}
    for ep in range(n_ep):
        np.random.seed(seed_base + ep)
        with contextlib.redirect_stdout(io.StringIO()):
            env.reset()
        st = env.shift_state
        n_pairs = pairs_held(env)
        serving_frame = {r: next(a["id"] for a in st["airframes"] if a["role"] == r) for r in st["roles"]}
        handed = {r: False for r in st["roles"]}
        for t in range(1, 1000):
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, done, info = env.step({u: action for u in env.uavs})
            res["ms"].append(1000 * (time.perf_counter() - t0))
            for r in st["roles"]:
                uav = env.uavs[r]
                if uav.active and uav.current_battery < 0:
                    res["ok"] = False
                    res["msgs"].append(f"ep {ep} step {t}: position {r} battery {uav.current_battery:.0f} J")
                now = next((a["id"] for a in st["airframes"] if a["state"] == "serving" and a["role"] == r), None)
                if now is not None and now != serving_frame[r]:
                    handed[r] = True
                    serving_frame[r] = now
            if pairs_held(env) != n_pairs:
                res["ok"] = False
                res["msgs"].append(f"ep {ep} step {t}: {pairs_held(env)} pairs, {n_pairs} at the start")
            res["relays"] += info["relays"]
            res["empty"] += info["gaps"]
            res["pos_steps"] += info["n_serving_target"]
            res["waiting"] += info["waiting_battery"]
            if done:
                break
        if t != env.shift["duration_steps"]:
            res["ok"] = False
            res["msgs"].append(f"ep {ep}: done at step {t}, not {env.shift['duration_steps']}")
        if st["battery_short"]:
            res["ok"] = False
            res["msgs"].append(f"ep {ep}: {st['battery_short']} return legs short of energy")
        if not all(handed.values()):
            res["ok"] = False
            res["msgs"].append(f"ep {ep}: positions never handed over: {[r for r, h in handed.items() if not h]}")
        res["swaps"] += [a["swaps"] for a in st["airframes"]]
    return res


def main():
    setups = {"A  S=3,  6 BS60, 3 spares": config_with(n_serving=[3, 3], station_count=6, spares_per_uav=1.0),
              "B  S=10, 3 BS60": config_with(n_serving=[10, 10], station_count=3),
              "v2 S=3": config_with(n_serving=[3, 3]),
              "v2 S=10": config_with(n_serving=[10, 10]),
              "S=3,  1 BS60 per position": config_with(n_serving=[3, 3], stations_per_uav=1.0),
              "S=10, 1 BS60 per position": config_with(n_serving=[10, 10], stations_per_uav=1.0)}
    out = {name: run(path) for name, path in setups.items()}
    print("set-up                         relays  empty share  airframe-steps waiting  swaps/airframe  ms/step")
    for name, r in out.items():
        print(f"{name:30s} {r['relays']:6d}  {r['empty'] / max(r['pos_steps'], 1):11.3f}  {r['waiting']:22d}  "
              f"{np.mean(r['swaps']):14.1f}  {np.median(r['ms']):7.1f}")
        for m in r["msgs"][:5]:
            print("   ", m)
    a, b = list(out.values())[:2]
    ok = all(r["ok"] for r in out.values()) and a["empty"] == 0 and b["waiting"] > 0
    print("\nTIP 4 check:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

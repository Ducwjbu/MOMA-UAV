"""How many arriving UEs are active by step? (report/05_han_che.md; audit of 2026-11-13)

ma3.yaml calibrates the load for 91 arriving UEs (plus 26 RSUs), the steady state of arrivals and departures.
This runs a fixed action (hover in place, power 1, even bandwidth) on 30 validation episodes
(100000+) and prints the mean number of active arriving UEs at each step, to compare with that calibration.

Run from the repo root: py report/scripts/ue_population.py
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    from environment import NetworkSlicingEnv


def main(n_ep=30, seed_base=100000, steps=18):
    with contextlib.redirect_stdout(io.StringIO()):
        env = NetworkSlicingEnv("config/environment/ma3.yaml")
    a = np.concatenate([[0, 0, 0, 1], np.full(9, 1 / 9)])
    pops = np.full((n_ep, steps + 1), np.nan)
    for ep in range(n_ep):
        np.random.seed(seed_base + ep)
        env.reset()
        pops[ep, 0] = sum(1 for u in env.ues.values() if u.is_active and not getattr(u, "fixed", False))
        for t in range(steps):
            _, _, done, _ = env.step({u: a for u in env.uavs})
            pops[ep, t + 1] = sum(1 for u in env.ues.values() if u.is_active and not getattr(u, "fixed", False))
            if done:
                break
    print("mean active arriving UEs after reset and after each step:", np.round(np.nanmean(pops, 0), 1))
    print(f"mean over steps 1-{steps}: {np.nanmean(pops[:, 1:]):.1f} (calibration: 91)")


if __name__ == "__main__":
    main()

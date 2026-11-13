"""Does the omega controller match the best fixed omega on a fine grid? (docs/thesis_brainstorm.md, A1 results)

eval_controller.py picks the hindsight-best fixed omega on a 0.1 grid (np.linspace(0, 1, 11)). This repeats the
test-seed comparison (constraint QoS >= 0.30, primary energy, 5 checkpoints x 20 episodes 200000+, reset every
episode) with fixed omegas every 0.025 around the best grid point, omega = (1 - a) e_energy + a e_qos, against
the I-only controller of the notes (K_I 0.5, lambda from 0.5). It is a check of an already-reported comparison,
not a tuning run: the controller's gains stay the ones chosen on validation seeds.

Run from the repo root (checkpoints in runs/):
  py report/scripts/controller_fine_grid.py evals/diag/ctrl_fine.csv
"""
import contextlib
import glob
import io
import os
import re
import sys

os.environ.setdefault("MOMA_DEVICE", "cpu")
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

import eval_controller as ec

GRID = [0.20, 0.225, 0.25, 0.275, 0.30, 0.325]


def main(out):
    cons = [ec.parse_constraint("qos:0.30")]
    seeds = [200000 + k for k in range(20)]
    e_energy, e_qos = np.eye(3)[1], np.eye(3)[0]
    frames = []
    for f in sorted(glob.glob(ec.CHECKPOINT)):
        s = int(re.search(r"seed_(\d+)", f.replace("\\", "/")).group(1))
        with contextlib.redirect_stdout(io.StringIO()):
            tr = ec.load_trainer("config/train/moma_ac_ma3_prior.yaml", f)
        ctrls = {f"fixed_a{a:.3f}": ec.FixedOmega((1 - a) * e_energy + a * e_qos) for a in GRID}
        # n_obj, primary (energy), constraints, k_p, k_i, k_d, delta, lambda_init
        ctrls["pid_i_only"] = ec.OmegaController(3, 1, cons, 0.0, 0.5, 0.0, 2.0, 0.5)
        for name, c in ctrls.items():
            df = pd.DataFrame(ec.run_episodes(tr, c, seeds, 18, False, cons, ()))
            df.insert(0, "seed", s)
            df.insert(0, "method", name)
            frames.append(df)
        print("seed", s, flush=True)
    steps = pd.concat(frames)
    steps.to_csv(out, index=False)
    summ = ec.summarize(steps, "energy", cons)
    print(summ[["method", "energy", "energy_lo", "energy_hi", "qos", "viol_qos"]].to_string(index=False,
                                                                                          float_format="%.4f"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

import contextlib, io, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import eval_heuristics as eh
from environment import NetworkSlicingEnv
from agents_moma_ac import nan_mean
with contextlib.redirect_stdout(io.StringIO()):
    env = NetworkSlicingEnv(config_path="config/environment/ma3.yaml")
splits = {"even": eh.EVEN_BW, "urllc": eh.URLLC_BW,
          "embb_only": np.array([1,1,1,0,0,0,0,0,0], float)/3,
          "embb+urllc": np.array([1,1,1,1,1,1,0,0,0], float)/6,
          "embb_heavy": np.array([4,4,4,2,2,2,.1,.1,.1], float)/18.3}
for name, bw in splits.items():
    x, s, v, d = eh.run(env, 600, 0, 1.0, bw, "rotate", 5, 200000, 18)
    m, sl = x.mean(0), nan_mean(s)
    print(f"{name:11s} qos {m[0]:.3f} energy {m[1]:.3f} fair {m[2]:.3f} | SLA embb {sl[0]:.3f} urllc {sl[1]:.3f} mmtc {sl[2]:.3f}", flush=True)

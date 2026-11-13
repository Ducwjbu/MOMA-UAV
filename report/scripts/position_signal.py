"""How much of a step's reward do the UAV positions explain? (CLAUDE.md, formation prior, "Why": "the UAV positions
explain only about 17% of a step's QoS variance (supervised check)")

Two subcommands, run from the repo root:
  gen  <out.npz> <n_episodes> <seed> [--random_radio]
       Episodes of ma3 with diverse movement: per episode one mode (random moves 40%, a regular triangle of the
       formation guide, held or rotating, 40%, hover 20%). Power 1 and the even bandwidth split, so only the
       positions vary (--random_radio draws power and bandwidth per step instead). Stores the joint observation,
       the joint action, the reward vector and the UAV positions after the step.
  fit  <train.npz> [<train.npz> ...] <test.npz>
       Fits the critic's architecture (2 x 256 ReLU MLP, Adam 3e-4, batch 256) to the step's reward vector by
       supervised regression, and reports the R2 on the held-out file at several training lengths (it overfits
       with this much data, so the best is the fair number). Inputs: "obs+delta" (what the critic sees: joint
       observation and joint action), "obs+target" (the movement replaced by the positions after the step) and
       "target only" (positions after the step, power and bandwidth). Then QoS by the closest pair's distance.

The check behind the 17%: three gen runs of 800 episodes (seeds 11, 12, 13; 28.8k train steps, 14.4k test):
  py report/scripts/position_signal.py gen evals/diag/pos_11.npz 800 11
  py report/scripts/position_signal.py gen evals/diag/pos_12.npz 800 12
  py report/scripts/position_signal.py gen evals/diag/pos_13.npz 800 13
  py report/scripts/position_signal.py fit evals/diag/pos_11.npz evals/diag/pos_12.npz evals/diag/pos_13.npz
The training seeds are not eval seeds (100000+, 200000+); the env draws its scenario from them as in training.
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np


def gen(out, n_ep, seed, random_radio):
    from environment import NetworkSlicingEnv
    from guide import FormationGuide
    with contextlib.redirect_stdout(io.StringIO()):
        env = NetworkSlicingEnv(config_path="config/environment/ma3.yaml")
    np.random.seed(seed)
    guide = FormationGuide(env)
    uids = sorted(env.uavs)
    O, A, R, P = [], [], [], []
    for ep in range(n_ep):
        obs = env.reset()
        mode = np.random.choice(["random", "guide", "hold"], p=[0.4, 0.4, 0.2])
        guide.new_episode(18)
        for _ in range(18):
            o = np.stack([obs[u] for u in uids])
            a = np.zeros((3, 13))
            if mode == "random":
                a[:, :3] = np.random.uniform(-1, 1, (3, 3))
            elif mode == "guide":
                a[:, :3] = guide.movement(uids, 0.1)
            if random_radio:
                a[:, 3] = np.random.uniform(0.05, 1.0, 3) if np.random.rand() < 0.5 else 1.0
                a[:, 4:] = np.random.dirichlet(np.ones(9), 3) * np.random.uniform(0.5, 1.0, (3, 1))
            else:
                a[:, 3] = 1.0
                a[:, 4:] = 1.0 / 9.0
            obs, _, done, info = env.step({u: a[i] for i, u in enumerate(uids)})
            O.append(o.ravel()); A.append(a.ravel()); R.append(info["reward_vector"])
            P.append(np.array([env.uavs[u].position for u in uids]).ravel())
            if done:
                break
        if ep % 100 == 0:
            print(ep, flush=True)
    np.savez(out, O=np.array(O, np.float32), A=np.array(A, np.float32), R=np.array(R, np.float32),
             P=np.array(P, np.float32))
    print("saved", len(O))


def fit(files):
    import torch
    import torch.nn as nn

    def load(fs):
        return [np.concatenate([np.load(f)[k] for f in fs]) for k in ("O", "A", "R", "P")]

    Otr, Atr, Rtr, Ptr = load(files[:-1])
    Ote, Ate, Rte, Pte = load(files[-1:])
    print(f"train {len(Otr)}, test {len(Ote)}; qos std test {Rte[:, 0].std():.3f}, mean {Rte[:, 0].mean():.3f}")

    def target_feats(P, A):
        p = P.reshape(-1, 3, 3)
        a = A.reshape(-1, 3, 13)
        return np.concatenate([(p[..., :2] / 2000.0).reshape(len(P), -1), ((p[..., 2:] - 50.0) / 70.0).reshape(len(P), -1),
                               a[..., 3:].reshape(len(P), -1)], 1)

    def obs_target(O, P, A):
        a = A.reshape(-1, 3, 13).copy()
        p = P.reshape(-1, 3, 3)
        a[..., 0:2] = p[..., :2] / 2000.0
        a[..., 2] = (p[..., 2] - 50.0) / 70.0
        return np.concatenate([O, a.reshape(len(O), -1)], 1)

    def run(Xtr, Ytr, Xte, Yte, label, checks=(2000, 5000, 10000, 20000)):
        torch.manual_seed(0)
        Xtr_t, Ytr_t, Xte_t = torch.tensor(Xtr), torch.tensor(Ytr), torch.tensor(Xte)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 3))
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)
        best, best_pred, out = None, None, []
        for it in range(1, max(checks) + 1):
            idx = torch.randint(0, len(Xtr_t), (256,))
            loss = ((net(Xtr_t[idx]) - Ytr_t[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if it in checks:
                with torch.no_grad():
                    pte = net(Xte_t).numpy()
                r2 = 1 - ((pte - Yte) ** 2).mean(0) / Yte.var(0)
                out.append(f"{it // 1000}k {np.round(r2, 2)}")
                if best is None or r2[0] > best:
                    best, best_pred = r2[0], pte
        print(f"{label:12s} test R2 (qos, -energy, fair) by steps: " + " | ".join(out))
        return best_pred

    p_delta = run(np.concatenate([Otr, Atr], 1), Rtr, np.concatenate([Ote, Ate], 1), Rte, "obs+delta")
    p_obst = run(obs_target(Otr, Ptr, Atr), Rtr, obs_target(Ote, Pte, Ate), Rte, "obs+target")
    p_targ = run(target_feats(Ptr, Atr), Rtr, target_feats(Pte, Ate), Rte, "target only")

    p = Pte.reshape(-1, 3, 3)
    d = np.min([np.linalg.norm(p[:, i, :2] - p[:, j, :2], axis=1) for i, j in ((0, 1), (0, 2), (1, 2))], axis=0)
    bins = [0, 200, 400, 600, 800, 1000, 1400, 3000]
    print("closest pair after step |    n | measured qos | pred obs+delta | pred obs+target | pred target only")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() > 20:
            print(f"{lo:5d}-{hi:<5d} m | {m.sum():5d} | {Rte[m, 0].mean():.3f} | {p_delta[m, 0].mean():.3f} | "
                  f"{p_obst[m, 0].mean():.3f} | {p_targ[m, 0].mean():.3f}")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "gen":
        gen(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), "--random_radio" in sys.argv[5:])
    elif len(sys.argv) >= 4 and sys.argv[1] == "fit":
        fit(sys.argv[2:])
    else:
        sys.exit(__doc__)

"""
eval_formation.py
=================
Baseline khong can train: 3 UAV dung yen thanh tam giac deu o do cao toi da, action co dinh.
Dung cung seed eval voi eval_omega (common random numbers) nen so duoc voi eval_metrics.csv.
Action khong phu thuoc omega, nen moi omega cho cung mot reward vector; eum lay tren
cung bo eval_n_omega omega ma eval_pareto_front dung.
Chay:
  py eval_formation.py
"""

import argparse
import numpy as np
import pandas as pd

from agents_moma_ac import EVAL_SEED, average_diagnostics, episode_diagnostics
from environment import NetworkSlicingEnv
from train_moma_ac import HV_REF_POINT, _hv_3d


def formation_positions(env, radius: float) -> np.ndarray:
    """Equilateral triangle around the middle of the fly range, at the top altitude."""
    x = sum(env.uav_fly_range_x) / 2.0
    y = sum(env.uav_fly_range_y) / 2.0
    angles = np.pi / 2.0 + 2.0 * np.pi * np.arange(3) / 3.0
    return np.stack([x + radius * np.cos(angles),
                     y + radius * np.sin(angles),
                     np.full(3, float(env.uav_fly_range_h[1]))], axis=1)


def evaluate(env, positions: np.ndarray, power: float, n_episodes: int, steps: int, n_omega: int,
             seed_base: int = EVAL_SEED) -> dict:
    # No movement, one power level, bandwidth split evenly across the 9 DAs
    action = np.concatenate([np.zeros(3), [power], np.full(9, 1.0 / 9.0)]).astype(np.float32)

    rng_state = np.random.get_state()
    rewards, episodes = [], []
    for ep in range(n_episodes):
        np.random.seed(seed_base + ep)
        env.reset()
        for uav, pos in zip(env.uavs.values(), positions):
            uav.position = pos.copy()
        # Placing the UAVs is setup: forget the assignments, so it isn't counted as handovers
        for ue in env.ues.values():
            ue.assigned_uav = None
        env._associate_ues_to_uavs()
        env._form_demand_areas()
        ep_rv, ep_diag = [], []
        done = False
        t = 0
        while not done and t < steps:
            _, _, done, info = env.step({uid: action for uid in env.uavs})
            ep_rv.append(np.array(info["reward_vector"], dtype=np.float32))
            ep_diag.append((info["coverage"], info["handovers"], info["min_uav_distance"], info["backhaul_limited"]))
            t += 1
        rewards.append(np.mean(ep_rv, axis=0))
        episodes.append(episode_diagnostics(ep_diag))
    np.random.set_state(rng_state)

    reward = np.mean(rewards, axis=0)
    corners = np.eye(3, dtype=np.float32)
    interior = np.random.RandomState(EVAL_SEED).dirichlet(np.ones(3), n_omega - 3).astype(np.float32)
    omegas = np.vstack([corners, interior])
    return {
        "qos": float(reward[0]), "energy": float(reward[1]), "fair": float(reward[2]),
        **average_diagnostics(episodes),
        "eum": float((omegas @ reward).mean()),
        "hv": float(_hv_3d(reward[np.newaxis], HV_REF_POINT)),
        "power": power,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fixed 3-UAV formation")
    parser.add_argument("--env_config", default="config/environment/ma3.yaml")
    parser.add_argument("--radius", type=float, default=600.0, help="Distance from the center to each UAV (m)")
    parser.add_argument("--power", type=float, default=0.5, help="Transmit power as a fraction of max_power")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20, help="Episode length, eval_steps of moma_ac_ma3.yaml")
    parser.add_argument("--n_omega", type=int, default=30)
    parser.add_argument("--out", default="formation_baseline.csv")
    args = parser.parse_args()

    env = NetworkSlicingEnv(config_path=args.env_config)
    assert env.num_uavs == 3, "The formation needs 3 UAVs"
    row = evaluate(env, formation_positions(env, args.radius), args.power,
                   args.n_episodes, args.steps, args.n_omega)
    pd.DataFrame([row]).to_csv(args.out, index=False)
    print({k: round(v, 4) for k, v in row.items()})

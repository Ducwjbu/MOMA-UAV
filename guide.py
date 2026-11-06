"""
guide.py
========
Formation guide for exploration (MOMA-TD3 on several UAVs). A trained team keeps its UAVs about 430 m apart and
never visits a spread formation held still, so its critic cannot learn that such states are good, while a fixed
triangle beats it at QoS-heavy weights. The guide puts those states into the replay buffer.

In a guided episode the guide steers the movement of every UAV for the first k steps, k uniform in
1..max_steps (Jump-Start RL, Uchendu et al., ICML 2023: a guide policy rolls in, the learner takes over), and the
actor keeps power and bandwidth with its own exploration noise. The transitions go to the replay buffer like the
others, the off-policy use of a heuristic's data of DDPGfD (Vecerik et al., arXiv:1707.08817). Each guided
episode draws a regular N-gon: center near the middle of the fly range, radius, orientation, height, and a
rotation per step (none with probability hover_prob, so the formation is held still). Evaluation never uses it.
"""

import itertools

import numpy as np


class FormationGuide:
    def __init__(self, env, radius=(300.0, 900.0), center_offset=300.0, hover_prob=0.5,
                 rotation_deg=(20.0, 110.0)):
        self.env = env
        self.radius = radius
        self.center_offset = center_offset
        self.hover_prob = hover_prob
        self.rotation_deg = rotation_deg
        self.order = None

    def new_episode(self, max_steps: int) -> int:
        """Draw this episode's formation; returns how many steps the guide drives"""
        env = self.env
        self.center = (np.array([sum(env.uav_fly_range_x), sum(env.uav_fly_range_y)]) / 2.0
                       + np.random.uniform(-self.center_offset, self.center_offset, 2))
        self.r = np.random.uniform(*self.radius)
        self.theta = np.random.uniform(0.0, 2.0 * np.pi)
        if np.random.rand() < self.hover_prob:
            self.dtheta = 0.0
        else:
            self.dtheta = np.deg2rad(np.random.uniform(*self.rotation_deg)) * np.random.choice([-1.0, 1.0])
        self.height = np.random.uniform(*env.uav_fly_range_h)
        self.order = None   # Vertices are assigned to UAVs at the first guided step
        return int(np.random.randint(1, max_steps + 1))

    def movement(self, uav_ids, noise: float) -> np.ndarray:
        """Movement actions, shape (n_uavs, 3) in [-1, 1], toward this step's formation, plus exploration noise"""
        env = self.env
        n = len(uav_ids)
        self.theta += self.dtheta
        angles = self.theta + 2.0 * np.pi * np.arange(n) / n
        vertices = self.center + self.r * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        vertices[:, 0] = np.clip(vertices[:, 0], *env.uav_fly_range_x)
        vertices[:, 1] = np.clip(vertices[:, 1], *env.uav_fly_range_y)
        positions = np.array([env.uavs[uid].position for uid in uav_ids])
        if self.order is None:
            # The assignment of UAVs to vertices with the shortest total flight
            self.order = min(itertools.permutations(range(n)),
                             key=lambda p: np.linalg.norm(positions[:, :2] - vertices[list(p)], axis=1).sum())
        targets = vertices[list(self.order)]
        step_xy = env.uav_params.velocity_max * env.T_L
        out = np.zeros((n, 3))
        for i in range(n):
            xy = (targets[i] - positions[i, :2]) / step_xy
            out[i, :2] = xy / max(1.0, np.linalg.norm(xy))
            dh = self.height - positions[i, 2]
            rate = env.velocity_max_vertical if dh > 0 else env.velocity_max_descent
            out[i, 2] = dh / (rate * env.T_L)
        return np.clip(out + np.random.normal(0.0, noise, out.shape), -1.0, 1.0)

"""
prior.py
========
Formation prior for a residual policy (formation_prior in the train config).

Each UAV i has a fixed vertex of a regular N-gon of radius prior_radius around the middle of the fly range, at
the top altitude (N = 1: the middle). UAVs are assigned to vertices in UAV id order. The movement the env
receives is clip(prior_i + a_i[0:3], -1, 1): prior_i steers UAV i to its vertex (one step reaches it when the
wind allows) and a_i is the actor's output with its exploration noise. The buffer, the critics and the actor all
work on a_i, the residual: Residual Policy Learning (Silver, Allen, Tenenbaum, Kaelbling, arXiv:1812.06298,
Sec. IV, pi_theta(s) = pi(s) + f_theta(s)); Residual RL for robot control (Johannink et al., arXiv:1812.03201,
Sec. III-A, Eq. 5). The movement rows of the actor's last layer start at zero, so the initial policy is the
formation with the actor's own power and bandwidth (Silver et al., Sec. IV-A). Power and bandwidth have no prior.

Why a prior: on ma3 the UAV positions explain only about 17% of the variance of a step's QoS (energy about 90%),
so MOMA-TD3 trained from scratch never finds spread formations and cruises at every weight. Why this formation:
for 3 UAVs on the 2 x 2 km area, the 3-means (Lloyd) placement for uniform coverage is a near-equilateral
triangle, and the r = 600 m triangle is within 0.4% of its RMS distance to the nearest UAV (519 m against 517 m).

Execution stays decentralized: prior_i needs only UAV i's own position (in o_i) and its vertex.
"""

import numpy as np
import torch


def formation_vertices(env, radius: float, n: int) -> np.ndarray:
    """Regular n-gon around the middle of the fly range at the top altitude, first vertex at 90 degrees"""
    x = sum(env.uav_fly_range_x) / 2.0
    y = sum(env.uav_fly_range_y) / 2.0
    r = radius if n > 1 else 0.0
    angles = np.pi / 2.0 + 2.0 * np.pi * np.arange(n) / n
    return np.stack([x + r * np.cos(angles), y + r * np.sin(angles),
                     np.full(n, float(env.uav_fly_range_h[1]))], axis=1)


def move_to(env, position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Movement action (3,) in [-1, 1] that flies from position toward target in one step"""
    step_xy = env.uav_params.velocity_max * env.T_L
    xy = (target[:2] - position[:2]) / step_xy
    norm = np.linalg.norm(xy)
    if norm > 1.0:
        xy = xy / norm
    dh = target[2] - position[2]
    rate = env.velocity_max_vertical if dh > 0 else env.velocity_max_descent
    return np.array([xy[0], xy[1], np.clip(dh / (rate * env.T_L), -1.0, 1.0)])


def assign_vertices(env, uav_ids, radius: float) -> dict:
    """With a fleet section (active UAVs change): the regular n-gon of the n active UAVs, each vertex given to
    one of them so that the total distance is smallest (Hungarian algorithm, scipy linear_sum_assignment)"""
    from scipy.optimize import linear_sum_assignment
    active = [uid for uid in uav_ids if env.uavs[uid].active]
    vertices = formation_vertices(env, radius, len(active))
    positions = np.array([env.uavs[uid].position for uid in active])
    cost = np.linalg.norm(positions[:, None, :2] - vertices[None, :, :2], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return {active[r]: vertices[c] for r, c in zip(rows, cols)}


def attach(env, uav_ids, radius: float) -> np.ndarray:
    """Make env.step add the prior to the movement of every UAV; returns the vertices.

    Without a fleet section UAV i takes vertex i of the N-gon, for the whole run. With one, the n-gon of the
    active UAVs is reassigned whenever the active set changes (a reset, a withdrawal)."""
    vertices = formation_vertices(env, radius, len(uav_ids))
    fixed = {uid: vertices[i] for i, uid in enumerate(uav_ids)}
    cache = {"active": None, "targets": fixed}
    inner_step = env.step

    def targets():
        if getattr(env, "fleet", None) is None:
            return fixed
        active = tuple(uid for uid in uav_ids if env.uavs[uid].active)
        if active != cache["active"] or env.current_time == 0.0:   # Reassign at every reset too
            cache["active"], cache["targets"] = active, assign_vertices(env, uav_ids, radius)
        return cache["targets"]

    def step(actions):
        combined = {}
        tgt = targets()
        for uid in uav_ids:
            if uid in actions:
                a = np.array(actions[uid], dtype=np.float64)
                if uid in tgt:
                    a[0:3] = np.clip(a[0:3] + move_to(env, env.uavs[uid].position, tgt[uid]), -1.0, 1.0)
                combined[uid] = a
        return inner_step(combined)

    env.step = step
    env.prior_vertices = vertices
    return vertices


@torch.no_grad()
def zero_movement_output(agent, n_pose_dims: int = 3):
    """Zero the movement rows of the actor's last layer (and its target), so the residual starts at 0"""
    for net in (agent.actor, agent.actor_target):
        layers = net.heads if hasattr(net, "heads") else [net.out]
        for layer in layers:
            layer.weight[:n_pose_dims].zero_()
            layer.bias[:n_pose_dims].zero_()

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

Why a prior: on ma3 the UAV positions explain only about 15-20% of the variance of a step's QoS (energy over 90%;
report/scripts/position_signal.py), so MOMA-TD3 trained from scratch finds only loose formations and cruises at
every weight. Why this formation:
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


SLICES = ("embb", "urllc", "mmtc")


def _pool(density: np.ndarray, f: int) -> np.ndarray:
    """Sum f x f blocks of a 2D array (the last blocks may be smaller)"""
    rows = np.add.reduceat(density, np.arange(0, density.shape[0], f), axis=0)
    return np.add.reduceat(rows, np.arange(0, density.shape[1], f), axis=1)


def demand_points(env, cell: float = 40.0):
    """
    Where demand is expected this episode, as weighted points (n, 2) and weights (n,): each slice with a
    positive slice weight and UEs gets mass equal to its slice weight (the QoS and SLA objectives average within
    a slice, then over slices), spread as the env places its UEs. Fixed UEs (the RSUs) at their positions; a
    slice of traffic_map.slices on the episode's density map, summed into cells of `cell` m; any other slice
    uniform over the area on the same cells. Only the slices and the map of the episode are read, not where the
    UEs happen to be now
    """
    width, height = env.service_area
    map_slices = set(env.traffic_map_params["slices"]) if env.traffic_map is not None else set()
    if env.traffic_map is not None:
        f = max(1, int(round(cell / env.traffic_map.pixel)))
        dens = _pool(env.traffic_map.density, f)
        step = f * env.traffic_map.pixel
    else:
        step = cell
        dens = np.ones((int(np.ceil(width / step)), int(np.ceil(height / step))))
    gx, gy = np.meshgrid(np.minimum((np.arange(dens.shape[0]) + 0.5) * step, width),
                         np.minimum((np.arange(dens.shape[1]) + 0.5) * step, height), indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    points, weights = [], []
    for sl in SLICES:
        w_slice = float(env.slice_weights[sl]) if sl in env.slice_weights else 0.0
        ues = [ue for ue in env.ues.values() if ue.is_active and ue.slice_type == sl]
        if w_slice <= 0 or not ues:
            continue
        fixed = np.array([ue.position[:2] for ue in ues if getattr(ue, "fixed", False)])
        if len(fixed) == len(ues):
            points.append(fixed)
            weights.append(np.full(len(fixed), w_slice / len(fixed)))
        else:
            w = dens.ravel() if sl in map_slices else np.ones(grid.shape[0])
            points.append(grid)
            weights.append(w_slice * w / w.sum())
    return np.concatenate(points), np.concatenate(weights)


def weighted_kmeans(points: np.ndarray, weights: np.ndarray, k: int, n_init: int = 8, iters: int = 100,
                    seed: int = 0) -> np.ndarray:
    """Lloyd's algorithm on weighted points, k-means++ seeding (Arthur & Vassilvitskii 2007) with weights, best
    of n_init by weighted inertia. Its own RandomState, so the env's random stream is not touched"""
    rng = np.random.RandomState(seed)
    p = weights / weights.sum()
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        centres = [points[rng.choice(len(points), p=p)]]
        for _ in range(1, k):
            d2 = np.min(((points[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
            q = p * d2
            q = q / q.sum() if q.sum() > 0 else p
            centres.append(points[rng.choice(len(points), p=q)])
        c = np.array(centres, dtype=np.float64)
        for _ in range(iters):
            label = np.argmin(((points[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
            new = c.copy()
            for j in range(k):
                m = label == j
                if weights[m].sum() > 0:
                    new[j] = (weights[m, None] * points[m]).sum(0) / weights[m].sum()
            converged = np.allclose(new, c, atol=1e-6)
            c = new
            if converged:
                break
        inertia = float((weights * np.min(((points[:, None, :] - c[None]) ** 2).sum(-1), axis=1)).sum())
        if inertia < best_inertia:
            best, best_inertia = c, inertia
    return best


def uniform_vertices(env, n: int, cell: float = 40.0) -> np.ndarray:
    """k-means (k = n) of the service area itself, every cell weighted the same, at the top altitude: a placement
    that knows the area but not the demand (a baseline for demand_vertices)"""
    width, height = env.service_area
    gx, gy = np.meshgrid((np.arange(int(np.ceil(width / cell))) + 0.5) * cell,
                         (np.arange(int(np.ceil(height / cell))) + 0.5) * cell, indexing="ij")
    grid = np.stack([np.minimum(gx.ravel(), width), np.minimum(gy.ravel(), height)], axis=1)
    c = weighted_kmeans(grid, np.ones(len(grid)), n)
    return np.concatenate([c, np.full((n, 1), float(env.uav_fly_range_h[1]))], axis=1)


def demand_vertices(env, n: int) -> np.ndarray:
    """k-means (k = n) of the episode's expected demand (demand_points), at the top altitude"""
    points, weights = demand_points(env)
    c = weighted_kmeans(points, weights, n)
    return np.concatenate([c, np.full((n, 1), float(env.uav_fly_range_h[1]))], axis=1)


def assign_vertices(env, uav_ids, radius: float, mode: str = "ngon", cache: dict = None) -> dict:
    """The formation of the n active UAVs (mode ngon: the regular n-gon; demand: demand_vertices, kept in `cache`
    per n), each vertex given to one of them so that the total distance is smallest (Hungarian algorithm, scipy
    linear_sum_assignment)"""
    from scipy.optimize import linear_sum_assignment
    active = [uid for uid in uav_ids if env.uavs[uid].active]
    if not active:
        return {}
    if mode in ("demand", "uniform"):
        cache = {} if cache is None else cache
        if len(active) not in cache:
            place = demand_vertices if mode == "demand" else uniform_vertices
            cache[len(active)] = place(env, len(active))
        vertices = cache[len(active)]
    else:
        vertices = formation_vertices(env, radius, len(active))
    positions = np.array([env.uavs[uid].position for uid in active])
    cost = np.linalg.norm(positions[:, None, :2] - vertices[None, :, :2], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return {active[r]: vertices[c] for r, c in zip(rows, cols)}


def attach(env, uav_ids, radius: float, mode: str = "ngon"):
    """Make env.step add the prior to the movement of every UAV; returns the fixed vertices (None for demand).

    mode ngon, without a fleet or shift section: UAV i takes vertex i of the N-gon, for the whole run. With one,
    the n-gon of the active UAVs is reassigned whenever the active set changes (a reset, a withdrawal, a position
    left empty or filled again in a shift). mode demand (v2, TIP 7): the vertices are the weighted k-means of the
    episode's expected demand for the number of active UAVs (demand_vertices), recomputed at each reset and
    reassigned whenever the active set changes. mode uniform: the same with the k-means of the area itself
    (uniform_vertices), a baseline that ignores the demand."""
    if mode not in ("ngon", "demand", "uniform"):
        raise ValueError(f"prior mode must be ngon, demand or uniform, got {mode}")
    vertices = formation_vertices(env, radius, len(uav_ids)) if mode == "ngon" else None
    fixed = {uid: vertices[i] for i, uid in enumerate(uav_ids)} if mode == "ngon" else None
    cache = {"active": None, "targets": fixed, "kmeans": {}}
    inner_step = env.step

    def targets():
        if mode == "ngon" and getattr(env, "fleet", None) is None and getattr(env, "shift", None) is None:
            return fixed
        active = tuple(uid for uid in uav_ids if env.uavs[uid].active)
        if env.current_time == 0.0:   # A new episode: new map and RSUs, so a new k-means
            cache["kmeans"] = {}
        if active != cache["active"] or env.current_time == 0.0:   # Reassign at every reset too
            cache["active"] = active
            cache["targets"] = assign_vertices(env, uav_ids, radius, mode, cache["kmeans"])
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
    env.prior_targets = targets
    env.prior_mode = mode
    return vertices


@torch.no_grad()
def zero_movement_output(agent, n_pose_dims: int = 3):
    """Zero the movement rows of the actor's last layer (and its target), so the residual starts at 0"""
    for net in (agent.actor, agent.actor_target):
        layers = net.heads if hasattr(net, "heads") else [net.out]
        for layer in layers:
            layer.weight[:n_pose_dims].zero_()
            layer.bias[:n_pose_dims].zero_()

"""
visualize/record.py
===================
Records evaluation episodes of the 3-UAV policies for the viewer (visualize/viewer.html): every UAV's and UE's
state after reset and after each step, on the same held-out episodes as eval_checkpoint.py (episode k seeded by
seed_base + k). Design in visualize/architecture.md.

It only reads the env: env.reset and env.step are wrapped from outside, and with the formation prior the action
the env receives (residual + prior) is read by wrapping the inner step that prior.attach closes over. Nothing in
the env, the agent or the RNG changes, so the recorded episode is the one eval_checkpoint.py scores (checked
bit-for-bit, see plan.md P0.4).

v2 (docs/v2/plan.md TIP 8): the corners and the reward are the env's objectives (meta.objectives), bw is the
action's slice share (7-dim action), the vertex is the UAV's current prior target (env.prior_targets, read with no
side effect on the run: the prior's cache gets what its next step computes anyway), and the team block has the
serving positions and empty ones of a shift. --n_serving fixes the serving positions. --controller records a run of
eval_controller.py (omega chosen at every step from KPI targets, --constraint): each frame then has ctrl (omega,
the mean over UAVs for local, and lambda per constraint, the mean over UAVs for local), drawn by the viewer.

Run from the repo root (environment.py loads its config by relative path):
  py -m visualize.record --methods ma3_prior ma3 --episodes 0 1 --out visualize/out/run1 --html
  py -m visualize.record --methods v2_prior --n_serving 6 --controller pid --constraint sla_urllc:0.8 --out o --html
"""

import argparse
import base64
import glob
import hashlib
import json
import os
import re
import subprocess

os.environ.setdefault("MOMA_DEVICE", "cpu")

import numpy as np

import agents_moma_ac
import prior
import eval_controller
from eval_checkpoint import env_config_for, load_trainer
from eval_trajectories import CORNERS, METHODS, corners_of
from train_moma_ac import V11_OBJECTIVES

SLICES = ("embb", "urllc", "mmtc")
# UE status codes in the trace (the viewer has the same table)
NO_DEMAND, SLA_MET, UNCOVERED, NO_RB, FAIL_RATE, FAIL_DELAY, FAIL_REL = range(7)
CHECKPOINT = "runs/saved_models_moma_ac_{m}/seed_{s}/checkpoints/checkpoint_step_{step}.pth"
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")


def _round(a, nd=1):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


# ============================================
# Reading the env: every attribute the recorder uses is read here
# ============================================

def scene_info(env) -> dict:
    """What stays fixed during an episode: area, fly range, city, traffic density, gateway, wind"""
    scene = {
        "area": _round(env.service_area),
        "fly_range": {"x": _round(env.uav_fly_range_x), "y": _round(env.uav_fly_range_y),
                      "h": _round(env.uav_fly_range_h)},
        "city": None, "density": None, "gateway": None,
        "wind": _round(env.wind_ref, 2),
        "v_max": float(next(iter(env.uavs.values())).velocity_max),   # m/s; a move action of 1 is v_max * T_L
    }
    if env.city is not None:
        c = env.city
        scene["city"] = {"pitch": round(float(c.pitch), 3), "width": round(float(c.width), 3),
                         "offset": round(float(c.offset), 3), "heights": _round(c.heights),
                         # v2 city.random_offset: the grid moved by -shift (city.py); [0, 0] in v1.1
                         "shift": _round(c.shift if getattr(c, "shift", None) is not None else np.zeros(2), 3)}
    if env.traffic_map is not None:
        # log10 of the density, 8 bits between its min and max (the viewer only shades it)
        d = np.log10(np.maximum(env.traffic_map.density, 1e-12))
        lo, hi = float(d.min()), float(d.max())
        q = np.round((d - lo) / max(hi - lo, 1e-12) * 255).astype(np.uint8)
        scene["density"] = {"pixel": float(env.traffic_map.pixel), "shape": list(q.shape), "log10_range": [lo, hi],
                            "u8_b64": base64.b64encode(q.tobytes(order="C")).decode("ascii")}
    gateway = getattr(env, "backhaul_gateway", None)
    if gateway is not None:
        scene["gateway"] = _round(gateway)
    return scene


def ue_status(env, ue):
    """(status code, [thr, delay, rel] satisfaction or None). Same rule as _calculate_qos_reward: a UE without
    demand is left out; with qos_aggregation min the SLA holds when the worst satisfaction is 1"""
    if getattr(ue, "demand_bps", 0.0) <= 0:
        return NO_DEMAND, None
    if ue.assigned_uav is None:
        return UNCOVERED, [0.0, 0.0, 0.0]
    if not ue.assigned_rb:
        return NO_RB, [0.0, 0.0, 0.0]   # The satisfaction helpers return 0 without touching the UE's fields
    sat = [float(ue.throughput_satisfaction), float(ue.delay_satisfaction), float(ue.reliability_satisfaction)]
    if min(sat) >= 1.0 - 1e-9:
        return SLA_MET, sat
    return FAIL_RATE + int(np.argmin(sat)), sat


def snapshot(env, uids, tracks, residual=None, combined=None, pre_pos=None, info=None, radius=None) -> dict:
    """One frame: the state after reset (no action) or after a step"""
    # UEs, column-wise
    ue_cols = {k: [] for k in ("track", "id", "x", "y", "slice", "indoor", "rsu", "uav", "status",
                               "thr", "delay", "rel", "rate_mbps", "demand_bps", "delay_ms")}
    served = {u: 0 for u in uids}
    for ue in env.ues.values():
        if not ue.is_active:
            continue
        status, sat = ue_status(env, ue)
        ue_cols["track"].append(tracks.track_of(ue))
        ue_cols["id"].append(int(ue.id))
        ue_cols["x"].append(round(float(ue.position[0]), 1))
        ue_cols["y"].append(round(float(ue.position[1]), 1))
        ue_cols["slice"].append(SLICES.index(ue.slice_type))
        ue_cols["indoor"].append(int(bool(getattr(ue, "indoor", False))))
        ue_cols["rsu"].append(int(bool(getattr(ue, "fixed", False))))
        ue_cols["uav"].append(None if ue.assigned_uav is None else int(ue.assigned_uav))
        ue_cols["status"].append(status)
        for k, v in zip(("thr", "delay", "rel"), sat if sat is not None else (None,) * 3):
            ue_cols[k].append(None if v is None else round(v, 3))
        has_rb = status not in (NO_DEMAND, UNCOVERED, NO_RB)
        ue_cols["rate_mbps"].append(round(float(ue.throughput) / 1e6, 3) if has_rb else None)
        ue_cols["demand_bps"].append(float(f"{float(getattr(ue, 'demand_bps', 0.0)):.3g}"))   # mMTC: ~0.06 bps
        ue_cols["delay_ms"].append(round(float(ue.latency_ms), 2) if has_rb else None)
        if ue.assigned_uav is not None and ue.assigned_uav in served:
            served[ue.assigned_uav] += 1

    # RBs given to each UAV's demand areas, per slice
    rbs = {u: [0, 0, 0] for u in uids}
    for da in env.demand_areas.values():
        if da.uav_id in rbs:
            rbs[da.uav_id][SLICES.index(da.slice_type)] += len(da.RB_ids_list)

    vertices = None
    if radius is not None:
        if getattr(env, "shift", None) is not None or getattr(env, "prior_mode", "ngon") != "ngon":
            vertices = env.prior_targets() if hasattr(env, "prior_targets") else None
        elif getattr(env, "fleet", None) is not None:
            # The n-gon of the active UAVs (the prior switches to it from the step after a change)
            vertices = prior.assign_vertices(env, uids, radius)
        else:
            vertices = dict(zip(uids, prior.formation_vertices(env, radius, len(uids))))

    uavs = []
    for u in uids:
        uav = env.uavs[u]
        row = {"id": int(u), "active": bool(uav.active), "pos": _round(uav.position),
               "power": round(float(uav.current_power / uav.max_power), 3),
               "battery": round(float(uav.current_battery / uav.battery_capacity), 4),
               "served": served[u], "rbs": rbs[u],
               "vertex": None if vertices is None or u not in vertices else _round(vertices[u])}
        if residual is not None and u in residual:
            a = combined.get(u, residual[u])
            row.update({
                "residual": _round(residual[u][0:3], 4),
                "move": _round(a[0:3], 4),            # residual + prior, clipped: what the env received
                "power_cmd": round(float(a[3]), 3),
                "bw": ([round(float(a[4 + s]), 3) for s in range(3)] if len(a) == 7    # slice_share
                       else [round(float(np.sum(a[4 + 3 * s:7 + 3 * s])), 3) for s in range(3)]),
                "cmd_target": _round(commanded_target(env, uav, pre_pos[u], a)),
                "pre": _round(pre_pos[u]),
            })
        uavs.append(row)

    frame = {"uavs": uavs, "ues": ue_cols}
    if info is not None:
        frame["team"] = {
            "reward": _round(info["reward_vector"], 4),
            "coverage": round(float(info["coverage"]), 4),
            "handovers": int(info["handovers"]),
            "n_active": int(info.get("n_active", len(uids))),
            "sla": [None if np.isnan(info[f"sla_{s}"]) else round(float(info[f"sla_{s}"]), 4) for s in SLICES],
        }
        if "n_serving_step" in info:   # v2 shift
            frame["team"].update({"n_serving": int(info["n_serving_step"]), "gaps": int(info["gaps"]),
                                  "relays": int(info["relays"])})
    return frame


def commanded_target(env, uav, pre, action) -> np.ndarray:
    """Where the movement action points before the env's limits (airspeed, fly range, separation push, roof
    clearance): the first lines of NetworkSlicingEnv.step"""
    delta_xy = np.asarray(action[:2], dtype=np.float64) * uav.velocity_max * env.T_L
    cap = uav.velocity_max * env.T_L
    delta_xy = delta_xy * min(1.0, cap / max(np.linalg.norm(delta_xy), 1e-9))
    rate = env.velocity_max_vertical if action[2] > 0 else env.velocity_max_descent
    return np.asarray(pre, dtype=np.float64) + np.array([delta_xy[0], delta_xy[1], action[2] * rate * env.T_L])


class Tracks:
    """Track ids for UEs. The env reuses the id of a departed UE (ue_id_pool), even within one step, so a track is
    one UE object: same id and same object as in the previous frame, else a new track"""

    def __init__(self):
        self.prev, self.ids, self.next = {}, {}, 0

    def reset(self):
        self.prev, self.ids = {}, {}

    def track_of(self, ue) -> int:
        if self.prev.get(ue.id) is not ue:
            self.ids[ue.id] = self.next
            self.next += 1
        self.prev[ue.id] = ue
        return self.ids[ue.id]


# ============================================
# Recording
# ============================================

def _inner_step_cell(step_fn):
    """The closure cell of prior.attach's step that holds the env's own step, or None without the prior"""
    code = getattr(step_fn, "__code__", None)
    if code is None or "inner_step" not in code.co_freevars or step_fn.__closure__ is None:
        return None
    return step_fn.__closure__[code.co_freevars.index("inner_step")]


def record_episode(trainer, omega, env_seed: int, steps: int):
    """Run eval_omega on one episode with reset and step wrapped; returns (scene, frames, mean reward)"""
    def run():
        default_seed = agents_moma_ac.EVAL_SEED
        try:
            agents_moma_ac.EVAL_SEED = env_seed
            return trainer.agent.eval_omega(trainer.eval_env, np.array(omega, np.float32), 1, steps)
        finally:
            agents_moma_ac.EVAL_SEED = default_seed
    return record_run(trainer, run)


def record_controlled(trainer, controller, constraints, env_seed: int, steps: int):
    """One episode of eval_controller.run_episodes (omega from the controller at every step), recorded; each frame
    after a step gets ctrl: the omega used for that step and lambda after it"""
    names = list(trainer.obj_names)
    scene, frames, rows = record_run(
        trainer, lambda: eval_controller.run_episodes(trainer, controller, [env_seed], steps, False, constraints))
    for r in rows:
        lam = [v for k, v in r.items() if k.startswith("lambda_")]
        if any(k.startswith("lambda_u") for k in r):   # local: mean over the UAVs per constraint
            n_c = len(constraints)
            lam = [float(np.nanmean([r[k] for k in r if k.startswith("lambda_u") and k.endswith(f"_{c}")]))
                   for c in range(n_c)]
        frames[r["t"] + 1]["ctrl"] = {"omega": [round(float(r[f"w_{n}"]), 4) for n in names],
                                      "lambda": [round(float(v), 4) for v in lam]}
    mean = np.mean([[r[n] for n in names] for r in rows], axis=0)
    return scene, frames, mean


def record_run(trainer, run):
    """Wrap the env's reset and step, call run(), restore; returns (scene, frames, run's result)"""
    env = trainer.eval_env
    uids = sorted(env.uavs)
    radius = trainer.cfg["prior_radius"] if trainer.cfg.get("formation_prior") else None
    tracks = Tracks()
    frames, state = [], {"scene": None, "combined": None}
    outer_step, outer_reset = env.step, env.reset
    cell = _inner_step_cell(outer_step)
    env_step = cell.cell_contents if cell is not None else None

    def inner(actions):
        state["combined"] = {u: np.array(a, dtype=np.float64) for u, a in actions.items()}
        return env_step(actions)

    def reset(*args, **kwargs):
        out = outer_reset(*args, **kwargs)
        tracks.reset()
        state["scene"] = scene_info(env)
        frames.append(dict(snapshot(env, uids, tracks, radius=radius), t=-1))
        return out

    def step(actions):
        residual = {u: np.array(a, dtype=np.float64) for u, a in actions.items()}
        pre = {u: env.uavs[u].position.copy() for u in uids}
        state["combined"] = None
        out = outer_step(actions)
        combined = state["combined"] if state["combined"] is not None else residual
        frames.append(dict(snapshot(env, uids, tracks, residual, combined, pre, out[3], radius), t=len(frames) - 1))
        return out

    env.reset, env.step = reset, step
    if cell is not None:
        cell.cell_contents = inner
    try:
        result = run()
    finally:
        env.reset, env.step = outer_reset, outer_step
        if cell is not None:
            cell.cell_contents = env_step
    return state["scene"], frames, result


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def write_html(trace_json: str, path: str):
    """Copy the viewer template with the trace in its <script id="trace"> block"""
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    marker = '<script type="application/json" id="trace"></script>'
    if marker not in html:
        raise RuntimeError(f"{TEMPLATE} has no empty trace block")
    payload = trace_json.replace("</", "<\\/")   # A JSON string can't close the script block
    with open(path, "w", encoding="utf-8") as f:
        f.write(html.replace(marker, f'<script type="application/json" id="trace">{payload}</script>'))


def make_controller(args, trainer, constraints, names):
    """A fresh controller of eval_controller.py for one recorded episode"""
    n_obj, primary = len(names), names.index(args.primary)
    kind = args.controller
    if kind in ("pid", "i_only", "p_only"):
        k_p = 0.0 if kind == "i_only" else args.k_p
        k_i = 0.0 if kind == "p_only" else args.k_i
        return eval_controller.OmegaController(n_obj, primary, constraints, k_p, k_i, 0.0, 2.0, args.lambda_init)
    if kind == "local":
        return eval_controller.LocalQueue(len(trainer.uav_ids), n_obj, primary, constraints, args.k_i, 2.0,
                                          args.lambda_init)
    if kind == "bang_bang":
        return eval_controller.BangBang(n_obj, primary, constraints)
    if kind == "rcpo":
        return eval_controller.RCPOLambda(n_obj, primary, constraints, args.rcpo_lr, args.lambda_init)
    raise SystemExit(f"unknown --controller {kind}")


def main():
    parser = argparse.ArgumentParser(description="Record 3-UAV eval episodes for visualize/viewer.html")
    parser.add_argument("--methods", nargs="+", default=["ma3_prior"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Training seeds (checkpoint folders)")
    parser.add_argument("--step", type=int, default=1000000, help="Checkpoint step")
    parser.add_argument("--checkpoint_glob", default=CHECKPOINT, help="{m} method, {s} seed, {step} step")
    parser.add_argument("--seed_base", type=int, default=200000)
    parser.add_argument("--episodes", nargs="+", type=int, default=[0], help="Episode k is seeded seed_base + k")
    parser.add_argument("--corners", nargs="+", default=list(CORNERS),
                        help="qos energy fair, and with --roles u0_energy u1_energy u2_energy")
    parser.add_argument("--roles", action="store_true", help="Add the corners u<i>_energy (one omega per UAV)")
    parser.add_argument("--omega", action="append", default=[], metavar="Q,E,F",
                        help="Any team weight, e.g. --omega 0.6,0.2,0.2 (normalized to sum 1); repeatable")
    parser.add_argument("--sweep", type=int, default=0, metavar="N",
                        help="Also the interior weights of the training eval's sweep of N weights (eval_pareto_front: "
                             "corners plus N - 3 seeded Dirichlet draws)")
    parser.add_argument("--config", default=None, help="Train config for every method (default: by method name)")
    parser.add_argument("--n_serving", type=int, default=None, help="v2: serving positions of the shift fixed at N")
    parser.add_argument("--controller", default=None,
                        help="Record a runtime controller of eval_controller.py instead of fixed weights: pid, i_only, "
                             "p_only, local, bang_bang or rcpo (with --constraint, --primary and its gains)")
    parser.add_argument("--constraint", action="append", default=[], help="kpi:threshold[:objective], repeatable")
    parser.add_argument("--primary", default="energy")
    parser.add_argument("--k_p", type=float, default=2.0)
    parser.add_argument("--k_i", type=float, default=0.5)
    parser.add_argument("--lambda_init", type=float, default=0.0)
    parser.add_argument("--rcpo_lr", type=float, default=1.0)
    parser.add_argument("--out", required=True, help="Writes <out>.json (and <out>.html with --html)")
    parser.add_argument("--html", action="store_true", help="Also a self-contained viewer page")
    args = parser.parse_args()

    if args.controller is None and args.config is None and args.n_serving is None and \
            all(METHODS.get(m, "").find("v2") < 0 for m in args.methods):
        names = V11_OBJECTIVES
    else:   # The objectives come from the first trainer: load it once to know them
        m0 = args.methods[0]
        f0 = sorted(glob.glob(args.checkpoint_glob.format(m=m0, s=args.seeds[0], step=args.step)))
        if not f0:
            parser.error(f"no checkpoint for {m0} seed {args.seeds[0]}")
        cfg0 = args.config or METHODS.get(m0, f"config/train/moma_ac_{m0}.yaml")
        names = list(load_trainer(cfg0, f0[0], env_config_for(cfg0, None, args.n_serving)).obj_names)
    n_obj = len(names)
    corners = {k: np.array(v, np.float32) for k, v in corners_of(names).items()}
    if args.corners == list(CORNERS) and names != V11_OBJECTIVES:
        args.corners = list(corners)
    if args.roles:
        for i in range(3):
            w = np.tile(corners["qos"], (3, 1))
            w[i] = corners["energy"]
            corners[f"u{i}_energy"] = w
    unknown = [c for c in args.corners if c not in corners]
    if unknown:
        parser.error(f"unknown corners {unknown} (u<i>_energy needs --roles)")
    if args.roles:
        args.corners += [c for c in corners if c.startswith("u") and c not in args.corners]
    extra = []
    for text in args.omega:
        w = np.array([float(v) for v in text.split(",")], np.float32)
        if w.shape != (n_obj,) or np.any(w < 0) or w.sum() <= 0:
            parser.error(f"--omega needs {n_obj} non-negative numbers: {text}")
        extra.append(w / w.sum())
    if args.sweep > n_obj:
        # Same weights as MOMA_AC_Agent.eval_pareto_front
        extra += list(np.random.RandomState(agents_moma_ac.EVAL_SEED).dirichlet(np.ones(n_obj), args.sweep - n_obj)
                      .astype(np.float32))
    for w in extra:
        name = "w(" + ",".join(f"{v:.2f}" for v in w) + ")"
        corners[name] = w
        args.corners.append(name)

    if args.controller is not None:
        if not args.constraint:
            parser.error("--controller needs --constraint")
        obj = {n: j for j, n in enumerate(names)}
        constraints = [eval_controller.parse_constraint(c, obj) for c in args.constraint]
        args.corners = [f"ctrl:{args.controller}"]
    scenes, episodes = {}, []
    for m in args.methods:
        config = args.config or METHODS.get(m, f"config/train/moma_ac_{m}.yaml")
        for s in args.seeds:
            pattern = args.checkpoint_glob.format(m=m, s=s, step=args.step)
            files = sorted(glob.glob(pattern))
            if not files:
                parser.error(f"no checkpoint for {m} seed {s}: {pattern}")
            trainer = load_trainer(config, files[0], env_config_for(config, None, args.n_serving))
            steps = trainer.cfg["eval_steps"]
            for c in args.corners:
                for k in args.episodes:
                    extra = {}
                    if args.controller is not None:
                        ctrl = make_controller(args, trainer, constraints, names)
                        scene, frames, rv = record_controlled(trainer, ctrl, constraints, args.seed_base + k, steps)
                        omega = np.atleast_2d(frames[1]["ctrl"]["omega"]) if len(frames) > 1 else np.zeros((1, n_obj))
                        extra["controller"] = {"method": args.controller, "primary": args.primary, "constraints": [
                            {"kpi": con["kpi"], "c": con["c"], "objective": names[con["obj"]]} for con in constraints]}
                    else:
                        scene, frames, rv = record_episode(trainer, corners[c], args.seed_base + k, steps)
                        omega = np.atleast_2d(corners[c])
                    key = hashlib.sha1(json.dumps(scene, sort_keys=True).encode()).hexdigest()[:12]
                    scenes[key] = scene
                    episodes.append({"method": m, "train_seed": s, "corner": c,
                                     "omega": _round(omega, 3), "episode": k,
                                     "env_seed": args.seed_base + k, "scene": key, "mean_reward": _round(rv, 4),
                                     "frames": frames, **extra})
            print(f"{m} seed {s} done ({files[0]})", flush=True)

    trace = {"meta": {"git_commit": _git_commit(), "device": os.environ.get("MOMA_DEVICE", ""),
                      "seed_base": args.seed_base, "step": args.step, "T_L": float(trainer.eval_env.T_L),
                      "slices": list(SLICES), **({} if names == V11_OBJECTIVES else {"objectives": names}),
                      "status": ["no_demand", "sla_met", "uncovered", "no_rb", "fail_rate", "fail_delay",
                                 "fail_reliability"]},
             "scenes": scenes, "episodes": episodes}
    text = json.dumps(trace, separators=(",", ":"), allow_nan=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {args.out}.json ({len(text) / 1e6:.1f} MB, {len(episodes)} episodes)")
    if args.html:
        write_html(text, f"{args.out}.html")
        print(f"Wrote {args.out}.html")


if __name__ == "__main__":
    main()

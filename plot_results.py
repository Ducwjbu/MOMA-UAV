"""
plot_results.py
===============
Figures of the 3-UAV course-report results (env v1.1) from the tracked files in results-v1.1/, as PNG and PDF:

  fig_learning_curves   HV of the training eval against training steps, mean and min-max over the 5 seeds.
                        Curves only: the training eval has 3 episodes on fixed seeds (see CLAUDE.md, Evaluation)
  fig_hv                HV on the test episodes with the 95% CI of aggregate_seeds.py and every seed as a dot
  fig_ablation          the 2x2 of prior x gamma (interaction plot)
  fig_fronts            the fronts projected on QoS-energy and QoS-fairness: each trained point is the policy at
                        one of the 30 eval weights, averaged over episodes and seeds
  fig_trajectories      top view of the 3 UAVs at the QoS and energy corners, ma3 against ma3_prior, one episode
                        (needs eval_trajectories.py output)
  behaviour.csv         per method and corner: speed, distance to the prior's triangle (dist_vertex_m in UAV id
                        order, the prior's assignment; dist_formation_m with the best vertex per UAV for each
                        episode, the fair number for methods without the prior), coverage, power (from the
                        trajectories)

Each method keeps one colour and one marker in every figure, and the lines are labelled directly, so the
figures stay readable in greyscale.

Run:
  py plot_results.py --results results-v1.1 --out figures
"""

import argparse
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Reference categorical palette of the dataviz method, fixed order; the non-learning baselines are neutral
STYLE = {
    "ma3_prior":     {"color": "#2a78d6", "marker": "o", "ls": "-",  "label": "MOMA-TD3 + prior, γ 0.9 (đề xuất)"},
    "ma3":           {"color": "#eb6834", "marker": "s", "ls": "--", "label": "MOMA-TD3, γ 0.99 (paper)"},
    "ma3_g90":       {"color": "#1baf7a", "marker": "^", "ls": "-.", "label": "MOMA-TD3, γ 0.9"},
    "ma3_prior_g99": {"color": "#eda100", "marker": "D", "ls": ":",  "label": "MOMA-TD3 + prior, γ 0.99"},
    "heuristics":    {"color": "#52514e", "marker": "x", "ls": "-",  "label": "Họ heuristic (10 cấu hình)"},
    "triangle":      {"color": "#9a9890", "marker": "+", "ls": "-",  "label": "Tam giác đứng yên"},
}
TRAINED = ["ma3", "ma3_g90", "ma3_prior_g99", "ma3_prior"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
UAV_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
    "savefig.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 1.5,
})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


def learning_curves(res, out):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for m in TRAINED:
        runs = [pd.read_csv(os.path.join(res, f"saved_models_moma_ac_{m}", f"seed_{s}", "eval_metrics.csv"))
                for s in range(5)]
        hv = np.stack([r.hv.to_numpy() for r in runs])
        step = runs[0].step.to_numpy() / 1e6
        st = STYLE[m]
        ax.fill_between(step, hv.min(0), hv.max(0), color=st["color"], alpha=0.12, lw=0)
        ax.plot(step, hv.mean(0), color=st["color"], ls=st["ls"], label=st["label"])
        ax.annotate(st["label"], (step[-1], hv.mean(0)[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)
    ax.set_xlabel("Số bước huấn luyện (triệu)")
    ax.set_ylabel("Hypervolume (eval lúc train)")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("Đường học: trung bình 5 seed, vùng tô là min–max", fontsize=9, color=INK, loc="left")
    save(fig, out, "fig_learning_curves")


def hv_dots(res, out):
    agg = pd.read_csv(os.path.join(res, "eval_test", "agg.csv")).set_index("method")
    seeds = pd.read_csv(os.path.join(res, "eval_test", "agg_seeds.csv"))
    order = ["triangle", "ma3", "ma3_g90", "heuristics", "ma3_prior_g99", "ma3_prior"]
    order = [m for m in order if m in agg.index]
    fig, ax = plt.subplots(figsize=(6, 2.9))
    for y, m in enumerate(order):
        st, r = STYLE[m], agg.loc[m]
        ax.plot([r.hv_lo, r.hv_hi], [y, y], color=st["color"], lw=2, solid_capstyle="round")
        line_marker = st["marker"] in ("x", "+")   # Line markers take the colour; filled ones a white ring
        ax.plot(r.hv, y, marker=st["marker"], color=st["color"], ms=9 if line_marker else 8,
                mec=st["color"] if line_marker else "white", mew=2 if line_marker else 1.2, ls="none")
        s = seeds.loc[seeds.method == m, "hv"]
        if len(s) > 1:
            ax.plot(s, np.full(len(s), y + 0.25), marker="|", color=st["color"], ms=7, ls="none", alpha=0.8)
        ax.annotate(f"{r.hv:.3f}", (r.hv_hi, y), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=INK)
    ax.set_yticks(range(len(order)), [STYLE[m]["label"] for m in order], fontsize=8)
    ax.set_xlabel("Hypervolume trên 20 episode test (CI 95%; vạch nhỏ: từng seed)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig_hv")


def ablation(res, out):
    agg = pd.read_csv(os.path.join(res, "eval_test", "agg.csv")).set_index("method")
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    x = [0, 1]
    for prior, (m99, m90) in {True: ("ma3_prior_g99", "ma3_prior"), False: ("ma3", "ma3_g90")}.items():
        st = STYLE["ma3_prior" if prior else "ma3"]
        hv = [agg.loc[m99, "hv"], agg.loc[m90, "hv"]]
        lo = [agg.loc[m99, "hv"] - agg.loc[m99, "hv_lo"], agg.loc[m90, "hv"] - agg.loc[m90, "hv_lo"]]
        hi = [agg.loc[m99, "hv_hi"] - agg.loc[m99, "hv"], agg.loc[m90, "hv_hi"] - agg.loc[m90, "hv"]]
        ax.errorbar(x, hv, yerr=[lo, hi], color=st["color"], marker=st["marker"], ms=8, mec="white", mew=1.2,
                    ls=st["ls"], capsize=3, elinewidth=1)
        ax.annotate("có prior" if prior else "không prior", (1, hv[1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    if "heuristics" in agg.index:
        ax.axhline(agg.loc["heuristics", "hv"], color=STYLE["heuristics"]["color"], lw=1, ls=(0, (4, 3)))
        ax.annotate("họ heuristic", (0.5, agg.loc["heuristics", "hv"]), xytext=(0, -10), textcoords="offset points",
                    ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(x, ["γ 0.99", "γ 0.9"])
    ax.set_xlim(-0.3, 1.5)
    ax.set_ylabel("Hypervolume (test, CI 95%)")
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig_ablation")


def fronts(res, out):
    ev = os.path.join(res, "eval_test")
    pts = {m: np.mean([np.load(os.path.join(ev, f"{m}_s{s}_actor.npy")).mean(0) for s in range(5)], 0)
           for m in ("ma3", "ma3_prior")}
    heur = np.load(os.path.join(ev, "heuristics.npy")).mean(0) if os.path.exists(os.path.join(ev, "heuristics.npy")) else None
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, (j, name) in zip(axes, [(1, "Năng lượng (−, cao hơn là tốt)"), (2, "Công bằng (Jain)")]):
        for m, p in pts.items():
            st = STYLE[m]
            ax.plot(p[:, j], p[:, 0], marker=st["marker"], color=st["color"], ms=6, mec="white", mew=0.8,
                    ls="none", label=st["label"])
        if heur is not None:
            st = STYLE["heuristics"]
            ax.plot(heur[:, j], heur[:, 0], marker=st["marker"], color=st["color"], ms=7, ls="none",
                    mew=1.5, label=st["label"])
        ax.set_xlabel(name)
        ax.set_ylabel("QoS")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Front Pareto chiếu 2 chiều (mỗi điểm: một trọng số ω, trung bình 20 episode × 5 seed)",
                 fontsize=9, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, out, "fig_fronts")


def trajectories(traj, out, env_config, seed=0, ep=0, r=600.0):
    with open(env_config, encoding="utf-8") as f:
        system = yaml.safe_load(f)["system"]
    fx, fy = system["uav_fly_range_x"], system["uav_fly_range_y"]
    centre = np.array([sum(fx) / 2, sum(fy) / 2])   # The prior's triangle (prior.formation_vertices)
    d = traj[(traj.seed == seed) & (traj.ep == ep)]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.4), sharex=True, sharey=True)
    for row, m in enumerate(("ma3", "ma3_prior")):
        for col, corner in enumerate(("qos", "energy")):
            ax = axes[row, col]
            a = np.pi / 2 + 2 * np.pi * np.arange(3) / 3
            ax.plot(centre[0] + r * np.cos(a), centre[1] + r * np.sin(a), ls="none", marker="*", ms=11,
                    color="#c3c2b7", zorder=1)
            g = d[(d.method == m) & (d.corner == corner)]
            if g.empty:
                ax.set_title(f"{m} · {corner}: không có dữ liệu", fontsize=8.5, loc="left")
                continue
            for u in range(3):
                p = g[g.uav == u].sort_values("t")
                ax.plot(p.x, p.y, color=UAV_COLORS[u], lw=1.2, marker="o", ms=2.5, zorder=2)
                ax.plot(p.x.iloc[0], p.y.iloc[0], marker="s", ms=6, color=UAV_COLORS[u], mec="white", zorder=3)
                ax.plot(p.x.iloc[-1], p.y.iloc[-1], marker="o", ms=8, color=UAV_COLORS[u], mec="white", mew=1.2,
                        zorder=3)
            cov = g[g.t >= 0].groupby("t").coverage.first().mean()
            ax.set_title(f"{'MOMA-TD3 (paper)' if m == 'ma3' else 'Đề xuất'} · góc "
                         f"{'QoS' if corner == 'qos' else 'năng lượng'} · độ phủ {cov:.2f}",
                         fontsize=8.5, color=INK, loc="left")
            ax.set_aspect("equal")
            ax.set_xlim(*fx)
            ax.set_ylim(*fy)
    for ax in axes[1]:
        ax.set_xlabel("x (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)")
    fig.suptitle("Quỹ đạo 3 UAV nhìn từ trên (■ xuất phát, ● kết thúc, ★ đỉnh tam giác r = 600 m); "
                 f"episode test {200000 + ep}, seed {seed}", fontsize=8.5, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, out, "fig_trajectories")


def formation_distance(late, vertices):
    """Mean distance to the triangle with the vertices given to the UAVs by the permutation that is best over
    the whole episode (each UAV keeps one vertex), per (method, seed, corner, ep). dist_vertex uses UAV id order,
    which is the prior's assignment but overstates the distance of methods without the prior, whose UAVs may
    hold the triangle with the vertices swapped"""
    perms = list(itertools.permutations(range(len(vertices))))
    res = {}
    for key, g in late.groupby(["method", "seed", "corner", "ep"]):
        g = g.sort_values(["uav", "t"])
        pos = g[["x", "y", "h"]].to_numpy().reshape(len(vertices), -1, 3)     # uav, t, xyz
        d = np.linalg.norm(pos[:, None] - vertices[None, :, None], axis=3).mean(axis=2)   # uav, vertex
        res[key] = min(np.mean([d[u, p[u]] for u in range(len(p))]) for p in perms)
    return pd.Series(res).rename_axis(["method", "seed", "corner", "ep"])


def behaviour(traj, out, env_config, r=600.0):
    with open(env_config, encoding="utf-8") as f:
        system = yaml.safe_load(f)["system"]
    fx, fy, fh = system["uav_fly_range_x"], system["uav_fly_range_y"], system["uav_fly_range_h"]
    a = np.pi / 2 + 2 * np.pi * np.arange(3) / 3   # prior.formation_vertices
    vertices = np.stack([sum(fx) / 2 + r * np.cos(a), sum(fy) / 2 + r * np.sin(a), np.full(3, float(fh[1]))], 1)
    t = traj.sort_values(["method", "seed", "corner", "ep", "uav", "t"]).copy()
    key = ["method", "seed", "corner", "ep", "uav"]
    t["move_m"] = np.hypot(t.groupby(key).x.diff(), t.groupby(key).y.diff())
    s = t[t.t >= 0]
    # Skip the flight out: the first 3 steps
    late = s[s.t >= 3]
    rows = late.groupby(["method", "corner"]).agg(
        speed_mps=("move_m", lambda v: v.mean() / 60.0), dist_vertex_m=("dist_vertex", "mean"),
        power=("power", "mean"), bw_embb=("bw_embb", "mean"), bw_urllc=("bw_urllc", "mean"),
        bw_mmtc=("bw_mmtc", "mean"))
    rows.insert(2, "dist_formation_m", formation_distance(late, vertices).groupby(["method", "corner"]).mean())
    rows["coverage"] = s.groupby(["method", "corner"]).coverage.mean()
    rows["qos"] = s.groupby(["method", "corner"]).qos.mean()
    rows = rows.reindex([(m, c) for m in TRAINED for c in ("qos", "energy", "fair")])
    rows.to_csv(os.path.join(out, "behaviour.csv"), float_format="%.4g")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(rows)


def main():
    parser = argparse.ArgumentParser(description="Figures of the 3-UAV results")
    parser.add_argument("--results", default="results-v1.1")
    parser.add_argument("--trajectories", default=None, help="Default <results>/eval_test/trajectories.csv")
    parser.add_argument("--env_config", default="config/environment/ma3.yaml", help="For the fly range")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    learning_curves(args.results, args.out)
    hv_dots(args.results, args.out)
    ablation(args.results, args.out)
    fronts(args.results, args.out)
    path = args.trajectories or os.path.join(args.results, "eval_test", "trajectories.csv")
    if os.path.exists(path):
        traj = pd.read_csv(path)
        trajectories(traj, args.out, args.env_config)
        behaviour(traj, args.out, args.env_config)
    else:
        print(f"{path} not found: skipping the trajectory figure and behaviour.csv")


if __name__ == "__main__":
    main()

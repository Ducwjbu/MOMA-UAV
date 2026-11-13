"""How close is the r=600 triangle to the best 3-point coverage placement of the 2 km area? (CLAUDE.md, formation
prior; report/01_bai_toan_va_phuong_phap.md)

Compares the RMS distance from a uniform point of the 2000 x 2000 m area to its nearest UAV for the prior's
triangle (radius 600 m around the centre, first vertex at 90 degrees) and for the best of 20 Lloyd (k-means) runs
with 3 centres, on 400k uniform points. Also the RMS for other radii.

Run from the repo root: py report/scripts/triangle_kmeans.py
"""
import numpy as np


def main():
    rng = np.random.default_rng(0)
    P = rng.uniform(0, 2000, (400000, 2))

    def rms(C):
        d = np.min(np.linalg.norm(P[:, None] - C[None], axis=2), 1)
        return np.sqrt((d ** 2).mean())

    a = np.pi / 2 + 2 * np.pi * np.arange(3) / 3
    best = None
    for _ in range(20):
        C = P[rng.choice(len(P), 3, replace=False)]
        for _ in range(100):
            lab = np.argmin(np.linalg.norm(P[:, None] - C[None], axis=2), 1)
            C = np.array([P[lab == k].mean(0) for k in range(3)])
        r = rms(C)
        if best is None or r < best[0]:
            best = (r, C)
    tri = rms(np.c_[1000 + 600 * np.cos(a), 1000 + 600 * np.sin(a)])
    print(f"triangle r=600 RMS {tri:.1f} m, best 3-means RMS {best[0]:.1f} m at {best[1].round(0).tolist()}: "
          f"{100 * (tri / best[0] - 1):.2f}% above")
    for r in (500, 550, 600, 650, 700):
        print(f"  r={r}: RMS {rms(np.c_[1000 + r * np.cos(a), 1000 + r * np.sin(a)]):.1f} m")


if __name__ == "__main__":
    main()

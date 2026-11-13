"""
Explicit city of ITU-R P.1410-6 (Sec. 2.1.4-2.1.5): buildings of equal footprint on a regular grid,
alpha = built-up area ratio, beta = buildings per km^2, Rayleigh heights with mode gamma (Eq. 17).

On a regular grid with beta buildings per km^2 the pitch is 1000 / sqrt(beta) m, and a footprint covering a
share alpha of each cell is a square of side W = 1000 sqrt(alpha / beta) m, leaving streets S = pitch - W.
A ray then crosses sqrt(alpha * beta) buildings per km on average, which is P.1410's Eq. 18.
"""
import numpy as np
from numba import njit


class City:
    def __init__(self, area, built_up_ratio, buildings_per_km2, height_scale, rng=np.random, shift=None):
        """shift (v2): the grid is moved by -shift (m, 0 <= shift < pitch per axis), so that the street grid sits
        differently in the area each episode; None keeps the grid at the origin, as in v1.1. Inside, positions
        are taken in grid coordinates, xy + shift"""
        self.area = np.asarray(area, dtype=np.float64)  # (width, height) in m
        self.pitch = 1000.0 / np.sqrt(buildings_per_km2)
        self.width = 1000.0 * np.sqrt(built_up_ratio / buildings_per_km2)
        self.street = self.pitch - self.width
        self.offset = self.street / 2.0  # each footprint is centred in its grid cell
        self.shift = None if shift is None else np.asarray(shift, dtype=np.float64)
        span = self.area if self.shift is None else self.area + self.shift
        self.n_cells = np.ceil(span / self.pitch).astype(int)  # (nx, ny), the last cells may overhang
        self.heights = rng.rayleigh(height_scale, size=(self.n_cells[0], self.n_cells[1]))
        # Footprint edges along each axis: every cell contributes its two edges
        self.edges = [np.sort(np.concatenate([np.arange(n) * self.pitch + self.offset,
                                              np.arange(n) * self.pitch + self.offset + self.width]))
                      for n in self.n_cells]

    def _grid(self, xy: np.ndarray) -> np.ndarray:
        """Area coordinates (..., 2) to grid coordinates"""
        return xy if self.shift is None else xy + self.shift

    def _grid3(self, p: np.ndarray) -> np.ndarray:
        """Segment ends (N, 3) to grid coordinates (heights unchanged)"""
        if self.shift is None:
            return p
        q = p.copy()
        q[:, :2] += self.shift
        return q

    def nearest_crossing(self, xy: np.ndarray) -> np.ndarray:
        """The street crossing nearest to each point (crossings are at the grid lines), in area coordinates"""
        if self.shift is None:
            return np.round(xy / self.pitch) * self.pitch
        return np.round((xy + self.shift) / self.pitch) * self.pitch - self.shift

    def in_street(self, xy: np.ndarray) -> np.ndarray:
        """Per axis, True where the coordinate lies in a street (x: a north-south street, y: an east-west one)"""
        g = self._grid(xy)
        local = g - np.floor(g / self.pitch) * self.pitch
        return (local < self.offset) | (local >= self.offset + self.width)

    def building_at(self, xy: np.ndarray):
        """Cell indices (i, j) of the footprint containing each point, -1 where the point is on a street"""
        return self._building_at_grid(self._grid(np.atleast_2d(xy)))

    def _building_at_grid(self, xy: np.ndarray):
        """building_at for points already in grid coordinates"""
        xy = np.atleast_2d(xy)
        cell = np.floor(xy / self.pitch).astype(int)
        local = xy - cell * self.pitch
        inside = np.all((local >= self.offset) & (local < self.offset + self.width), axis=1)
        inside &= np.all((cell >= 0) & (cell < self.n_cells), axis=1)
        cell[~inside] = -1
        return cell

    def is_indoor(self, xy: np.ndarray) -> np.ndarray:
        return self.building_at(xy)[:, 0] >= 0

    def roof_height(self, xy: np.ndarray) -> np.ndarray:
        """Height of the building under each point, 0 over streets"""
        cell = self.building_at(xy)
        roof = np.zeros(len(cell))
        on = cell[:, 0] >= 0
        roof[on] = self.heights[cell[on, 0], cell[on, 1]]
        return roof

    def line_of_sight(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Same result as line_of_sight_numpy, computed segment by segment in numba: each segment only visits
        the cuts it crosses, instead of sorting all footprint edges for every segment (about 10x faster)."""
        start = np.ascontiguousarray(self._grid3(np.atleast_2d(start).astype(np.float64)), dtype=np.float64)
        end = np.ascontiguousarray(self._grid3(np.atleast_2d(end).astype(np.float64)), dtype=np.float64)
        return _los_segments(start, end, self.edges[0], self.edges[1], self.pitch, self.offset, self.width,
                             self.n_cells[0], self.n_cells[1], self.heights)

    def line_of_sight_numpy(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """
        True where the straight segment start -> end clears every building. Shapes (N, 3) -> (N,).
        The segment is cut at every footprint edge it crosses, so each piece lies inside one footprint or
        on a street; a piece inside a footprint blocks when the segment's lower end over it is below the roof.
        The building containing `start` is skipped: an indoor UE's own walls are the O2I loss.
        """
        start = self._grid3(np.atleast_2d(start).astype(np.float64))
        end = self._grid3(np.atleast_2d(end).astype(np.float64))
        delta = end - start
        cuts = [np.zeros((len(start), 1)), np.ones((len(start), 1))]
        for axis in range(2):
            d = delta[:, axis:axis + 1]
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (self.edges[axis][np.newaxis, :] - start[:, axis:axis + 1]) / d
            t[~np.isfinite(t) | (t <= 0.0) | (t >= 1.0)] = 1.0  # unused cuts collapse onto the end
            cuts.append(t)
        t = np.sort(np.concatenate(cuts, axis=1), axis=1)
        t0, t1 = t[:, :-1], t[:, 1:]
        mid = (t0 + t1) / 2.0
        xy = start[:, np.newaxis, :2] + mid[..., np.newaxis] * delta[:, np.newaxis, :2]
        cell = np.floor(xy / self.pitch).astype(int)
        local = xy - cell * self.pitch
        inside = np.all((local >= self.offset) & (local < self.offset + self.width), axis=2)
        inside &= np.all((cell >= 0) & (cell < self.n_cells), axis=2)
        inside &= t1 > t0
        own = self._building_at_grid(start[:, :2])
        inside &= ~((cell[..., 0] == own[:, 0:1]) & (cell[..., 1] == own[:, 1:2]))
        roof = np.zeros(inside.shape)
        roof[inside] = self.heights[cell[inside][:, 0], cell[inside][:, 1]]
        lowest = start[:, 2:3] + np.minimum(t0 * delta[:, 2:3], t1 * delta[:, 2:3])
        return ~np.any(inside & (lowest < roof), axis=1)


@njit(cache=True)
def _los_segments(start, end, edges_x, edges_y, pitch, offset, width, nx, ny, heights):
    """City.line_of_sight_numpy, one segment at a time, with the same arithmetic in the same order"""
    n = start.shape[0]
    out = np.ones(n, dtype=np.bool_)
    cuts = np.empty(2 + edges_x.shape[0] + edges_y.shape[0])
    for k in range(n):
        sx, sy, sz = start[k, 0], start[k, 1], start[k, 2]
        dx, dy, dz = end[k, 0] - sx, end[k, 1] - sy, end[k, 2] - sz
        m = 0
        cuts[m] = 0.0
        m += 1
        cuts[m] = 1.0
        m += 1
        if dx != 0.0:
            for e in range(edges_x.shape[0]):
                t = (edges_x[e] - sx) / dx
                if t > 0.0 and t < 1.0:
                    cuts[m] = t
                    m += 1
        if dy != 0.0:
            for e in range(edges_y.shape[0]):
                t = (edges_y[e] - sy) / dy
                if t > 0.0 and t < 1.0:
                    cuts[m] = t
                    m += 1
        ts = np.sort(cuts[:m])
        # The building that contains the start (skipped: an indoor UE's own walls are the O2I loss)
        oi = int(np.floor(sx / pitch))
        oj = int(np.floor(sy / pitch))
        lx, ly = sx - oi * pitch, sy - oj * pitch
        if not (lx >= offset and lx < offset + width and ly >= offset and ly < offset + width
                and oi >= 0 and oi < nx and oj >= 0 and oj < ny):
            oi, oj = -1, -1
        for q in range(m - 1):
            t0, t1 = ts[q], ts[q + 1]
            if not t1 > t0:
                continue
            mid = (t0 + t1) / 2.0
            x = sx + mid * dx
            y = sy + mid * dy
            ci = int(np.floor(x / pitch))
            cj = int(np.floor(y / pitch))
            ux, uy = x - ci * pitch, y - cj * pitch
            if not (ux >= offset and ux < offset + width and uy >= offset and uy < offset + width):
                continue
            if not (ci >= 0 and ci < nx and cj >= 0 and cj < ny):
                continue
            if ci == oi and cj == oj:
                continue
            lowest = sz + min(t0 * dz, t1 * dz)
            if lowest < heights[ci, cj]:
                out[k] = False
                break
    return out

"""
Explicit city of ITU-R P.1410-6 (Sec. 2.1.4-2.1.5): buildings of equal footprint on a regular grid,
alpha = built-up area ratio, beta = buildings per km^2, Rayleigh heights with mode gamma (Eq. 17).

On a regular grid with beta buildings per km^2 the pitch is 1000 / sqrt(beta) m, and a footprint covering a
share alpha of each cell is a square of side W = 1000 sqrt(alpha / beta) m, leaving streets S = pitch - W.
A ray then crosses sqrt(alpha * beta) buildings per km on average, which is P.1410's Eq. 18.
"""
import numpy as np


class City:
    def __init__(self, area, built_up_ratio, buildings_per_km2, height_scale, rng=np.random):
        self.area = np.asarray(area, dtype=np.float64)  # (width, height) in m
        self.pitch = 1000.0 / np.sqrt(buildings_per_km2)
        self.width = 1000.0 * np.sqrt(built_up_ratio / buildings_per_km2)
        self.street = self.pitch - self.width
        self.offset = self.street / 2.0  # each footprint is centred in its grid cell
        self.n_cells = np.ceil(self.area / self.pitch).astype(int)  # (nx, ny), the last cells may overhang
        self.heights = rng.rayleigh(height_scale, size=(self.n_cells[0], self.n_cells[1]))
        # Footprint edges along each axis: every cell contributes its two edges
        self.edges = [np.sort(np.concatenate([np.arange(n) * self.pitch + self.offset,
                                              np.arange(n) * self.pitch + self.offset + self.width]))
                      for n in self.n_cells]

    def building_at(self, xy: np.ndarray):
        """Cell indices (i, j) of the footprint containing each point, -1 where the point is on a street"""
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
        """
        True where the straight segment start -> end clears every building. Shapes (N, 3) -> (N,).
        The segment is cut at every footprint edge it crosses, so each piece lies inside one footprint or
        on a street; a piece inside a footprint blocks when the segment's lower end over it is below the roof.
        The building containing `start` is skipped: an indoor UE's own walls are the O2I loss.
        """
        start = np.atleast_2d(start).astype(np.float64)
        end = np.atleast_2d(end).astype(np.float64)
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
        own = self.building_at(start[:, :2])
        inside &= ~((cell[..., 0] == own[:, 0:1]) & (cell[..., 1] == own[:, 1:2]))
        roof = np.zeros(inside.shape)
        roof[inside] = self.heights[cell[inside][:, 0], cell[inside][:, 1]]
        lowest = start[:, 2:3] + np.minimum(t0 * delta[:, 2:3], t1 * delta[:, 2:3])
        return ~np.any(inside & (lowest < roof), axis=1)

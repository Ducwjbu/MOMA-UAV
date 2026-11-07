"""
traffic_map.py
==============
Spatial traffic density of Lee, Zhou, Zhong, Niu, Zhou, Zhang, "Spatial Modeling of the Traffic Density in Cellular
Networks", IEEE Wireless Communications 21(1):80-88, 2014 (DOI 10.1109/MWC.2014.6757900).

The area is a grid of square pixels. A Gaussian random field is a sum of L sinusoids (Eq. 1),

    rho_G[m, n] = 2 / sqrt(L) * sum_l cos(i_l x[m, n] + phi_l) cos(j_l y[m, n] + psi_l),

with angular frequencies i_l, j_l ~ U(0, omega_max) and phases phi_l, psi_l ~ U(0, 2 pi); L = 10 is large enough
(p. 84). The traffic density is log-normal, rho = exp(sigma rho_G + mu) (Eq. 2). omega_max sets the coherence
distance, where the 2D autocorrelation of rho falls to half its peak (p. 83). A map whose coherence distance is
off by more than a tolerance (30 m for urban, p. 85) is redrawn, because one draw may produce a much different one.

Parameters come from Table 1 (urban DL: mu 18.93, sigma 2.3991, coherence 71.4 m, omega_max 0.011592 rad/m) with
10 m pixels (3/300 km, p. 82). mu only scales the density, so it does not change where UEs are placed; the env keeps
its own load calibration.
"""

import numpy as np


def density_map(area, pixel: float, sigma: float, omega_max: float, n_sinusoids: int = 10, rng=np.random):
    """Log-normal traffic density (Eq. 1-2 with mu = 0) on the pixel grid, shape (nx, ny); pixel centres at
    (k + 0.5) * pixel"""
    nx, ny = (np.ceil(np.asarray(area, dtype=np.float64) / pixel)).astype(int)
    x = (np.arange(nx) + 0.5) * pixel
    y = (np.arange(ny) + 0.5) * pixel
    i = rng.uniform(0.0, omega_max, n_sinusoids)
    j = rng.uniform(0.0, omega_max, n_sinusoids)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_sinusoids)
    psi = rng.uniform(0.0, 2.0 * np.pi, n_sinusoids)
    cx = np.cos(np.outer(x, i) + phi)  # (nx, L)
    cy = np.cos(np.outer(y, j) + psi)  # (ny, L)
    field = 2.0 / np.sqrt(n_sinusoids) * cx @ cy.T  # (nx, ny), Eq. 1
    return np.exp(sigma * field)  # Eq. 2


def coherence_distance(density: np.ndarray, pixel: float, eps: float = 0.01) -> float:
    """Mean distance from the origin of the points where the normalized 2D sample autocorrelation of the density
    is 0.5 +- eps (Lee et al., p. 83). The autocorrelation is of the mean-removed map, without wrap-around"""
    d = density - density.mean()
    nx, ny = d.shape
    f = np.fft.rfft2(d, s=(2 * nx, 2 * ny))
    acf = np.fft.fftshift(np.fft.irfft2(f * np.conj(f), s=(2 * nx, 2 * ny)))
    acf /= acf.max()
    cx, cy = np.unravel_index(np.argmax(acf), acf.shape)
    # Only lags up to half the map, where the estimate rests on enough pixel pairs
    gx, gy = np.meshgrid(np.arange(2 * nx) - cx, np.arange(2 * ny) - cy, indexing='ij')
    r = np.hypot(gx, gy) * pixel
    near = (np.abs(acf - 0.5) <= eps) & (np.abs(gx) <= nx // 2) & (np.abs(gy) <= ny // 2)
    return float(r[near].mean()) if near.any() else float('nan')


class TrafficMap:
    """One episode's density map, redrawn until its coherence distance is within tolerance of the target"""

    def __init__(self, area, params, rng=np.random):
        self.area = np.asarray(area, dtype=np.float64)
        self.pixel = float(params['pixel'])
        target, tol = float(params['coherence_distance']), float(params['coherence_tolerance'])
        for attempt in range(int(params.get('max_redraws', 50))):
            density = density_map(self.area, self.pixel, float(params['sigma']), float(params['omega_max']),
                                  int(params.get('n_sinusoids', 10)), rng)
            self.coherence = coherence_distance(density, self.pixel)
            if abs(self.coherence - target) <= tol:
                break
        self.redraws = attempt
        self.density = density
        self._cdf = np.cumsum(density.ravel())
        self._cdf /= self._cdf[-1]

    def sample(self, rng=np.random) -> np.ndarray:
        """A position (x, y) drawn with probability proportional to the density: a pixel, then uniform inside it"""
        k = min(int(np.searchsorted(self._cdf, rng.random(), side='right')), self._cdf.size - 1)
        ix, iy = np.unravel_index(k, self.density.shape)
        xy = (np.array([ix, iy]) + rng.random(2)) * self.pixel
        return np.minimum(xy, self.area)

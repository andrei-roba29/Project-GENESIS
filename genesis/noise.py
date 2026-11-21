"""Seeded 3D value noise + fractal Brownian motion on the unit sphere.

Sampling noise in 3D (x, y, z on the sphere) avoids seams at the
antimeridian and pinching at the poles that plague 2D lat/lon noise.
"""
from __future__ import annotations

import numpy as np


class Noise3D:
    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.perm = rng.permutation(512).astype(np.int32)
        self.seed = seed

    def _hash(self, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
        p = self.perm
        return p[(p[(p[ix & 255] + iy) & 255] + iz) & 255].astype(np.float64) / 255.0

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * (3.0 - 2.0 * t)

    def value(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Trilinear value noise in [0, 1]."""
        ix = np.floor(x).astype(np.int32)
        iy = np.floor(y).astype(np.int32)
        iz = np.floor(z).astype(np.int32)
        fx = x - ix
        fy = y - iy
        fz = z - iz
        ux, uy, uz = self._fade(fx), self._fade(fy), self._fade(fz)

        c000 = self._hash(ix, iy, iz)
        c100 = self._hash(ix + 1, iy, iz)
        c010 = self._hash(ix, iy + 1, iz)
        c110 = self._hash(ix + 1, iy + 1, iz)
        c001 = self._hash(ix, iy, iz + 1)
        c101 = self._hash(ix + 1, iy, iz + 1)
        c011 = self._hash(ix, iy + 1, iz + 1)
        c111 = self._hash(ix + 1, iy + 1, iz + 1)

        x00 = c000 + (c100 - c000) * ux
        x10 = c010 + (c110 - c010) * ux
        x01 = c001 + (c101 - c001) * ux
        x11 = c011 + (c111 - c011) * ux
        y0 = x00 + (x10 - x00) * uy
        y1 = x01 + (x11 - x01) * uy
        return y0 + (y1 - y0) * uz

    def fbm(self, x, y, z, octaves: int = 5, lacunarity: float = 2.0,
            gain: float = 0.5) -> np.ndarray:
        """Fractal Brownian motion, roughly in [-1, 1]."""
        total = np.zeros_like(x, dtype=np.float64)
        amp, freq, norm = 1.0, 1.0, 0.0
        for _ in range(octaves):
            total += amp * (self.value(x * freq, y * freq, z * freq) * 2.0 - 1.0)
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / max(norm, 1e-9)

    def ridged(self, x, y, z, octaves: int = 5, lacunarity: float = 2.0,
               gain: float = 0.5) -> np.ndarray:
        """Ridged multifractal in [0, 1] — sharp crests for mountain ranges."""
        total = np.zeros_like(x, dtype=np.float64)
        amp, freq, norm = 1.0, 1.0, 0.0
        for _ in range(octaves):
            n = 1.0 - np.abs(self.value(x * freq, y * freq, z * freq) * 2.0 - 1.0)
            total += amp * n * n
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / max(norm, 1e-9)


def sphere_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray):
    """Convert lat/lon degrees to unit-sphere coordinates."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return (np.cos(lat) * np.cos(lon), np.sin(lat), np.cos(lat) * np.sin(lon))


def haversine_km(lat1, lon1, lat2, lon2, radius_km: float) -> np.ndarray:
    """Great-circle distance in km (vectorised, spherical Earth)."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = np.deg2rad(np.asarray(lat2) - np.asarray(lat1))
    dl = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

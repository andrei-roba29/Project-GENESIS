"""Procedural geography: tectonic plates, continents, mountains, rifts.

Approach (all seeded, fully deterministic):
1. Scatter N plate seeds on the sphere; each cell belongs to its nearest seed.
2. Give every plate a random horizontal velocity (tangent vector).
3. Near plate boundaries, relative motion creates uplift (convergent ->
   mountain ranges), subsidence (divergent -> rift valleys) or shear.
4. Add multi-scale fractal continent noise + ridged noise for detail.
5. Pick the sea level so the ocean covers the configured fraction.
"""
from __future__ import annotations

import numpy as np

from .config import PlanetConfig
from .noise import Noise3D, sphere_xyz


def _fibonacci_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    """Evenly spread unit vectors + jitter."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i
    pts = np.stack([np.cos(phi) * np.cos(theta), np.sin(phi), np.cos(phi) * np.sin(theta)], axis=1)
    jitter = rng.normal(0.0, 0.08, size=pts.shape)
    pts = pts + jitter
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def generate_terrain(planet, cfg: PlanetConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    noise = Noise3D(cfg.seed + 1)
    h, w = planet.height, planet.width
    x, y, z = planet.x, planet.y, planet.z

    # --- tectonic plates -------------------------------------------------
    seeds = _fibonacci_sphere(cfg.plate_count, rng)          # (P, 3)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)  # (N, 3)
    dots = pts @ seeds.T                                     # (N, P)
    plate = np.argmax(dots, axis=1).reshape(h, w).astype(np.int16)
    planet.plate_id = plate

    # Random tangent velocity per plate.
    vel = rng.normal(0.0, 1.0, size=(cfg.plate_count, 3))
    vel -= np.sum(vel * seeds, axis=1, keepdims=True) * seeds   # make tangent
    norms = np.linalg.norm(vel, axis=1, keepdims=True)
    vel = vel / np.maximum(norms, 1e-9)

    # --- boundary stress ---------------------------------------------------
    # A cell is a boundary cell if any 4-neighbour belongs to another plate.
    # Convergence is measured along the direction to that neighbour.
    stress = np.zeros((h, w), np.float64)   # + convergent, - divergent
    boundary = np.zeros((h, w), bool)
    cell_vel = vel[plate]                    # (h, w, 3)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted_plate = np.roll(plate, shift=(-dr, -dc), axis=(0, 1))
        diff = shifted_plate != plate
        if dr == -1:
            diff[0, :] = False
        if dr == 1:
            diff[-1, :] = False
        if not diff.any():
            continue
        # Direction toward the neighbour (tangent, unit).
        nx = np.roll(x, shift=(-dr, -dc), axis=(0, 1))
        ny = np.roll(y, shift=(-dr, -dc), axis=(0, 1))
        nz = np.roll(z, shift=(-dr, -dc), axis=(0, 1))
        dx, dy, dz = nx - x, ny - y, nz - z
        dn = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
        dx, dy, dz = dx / dn, dy / dn, dz / dn
        # Relative velocity of neighbour plate w.r.t. this plate.
        rvx = vel[shifted_plate, 0] - cell_vel[..., 0]
        rvy = vel[shifted_plate, 1] - cell_vel[..., 1]
        rvz = vel[shifted_plate, 2] - cell_vel[..., 2]
        closing = -(rvx * dx + rvy * dy + rvz * dz)  # >0 when plates approach
        stress = np.where(diff, np.maximum(stress, closing), stress)
        boundary |= diff
    # Distance-to-boundary falloff so ranges have width.
    stress = _blur(stress, boundary, iterations=2)

    # --- base elevation ------------------------------------------------------
    base = noise.fbm(x * 2.2, y * 2.2, z * 2.2, octaves=cfg.noise_octaves)
    detail = noise.fbm(x * 6.0, y * 6.0, z * 6.0, octaves=4)
    ridged = noise.ridged(x * 3.0, y * 3.0, z * 3.0, octaves=cfg.noise_octaves)
    island = noise.fbm(x * 9.0, y * 9.0, z * 9.0, octaves=3)

    elev = base * cfg.continent_height_m
    elev += detail * 500.0
    # Convergent boundaries -> mountain belts; divergent -> rifts.
    elev += np.clip(stress, 0.0, None) * cfg.mountain_height_m * (0.35 + 0.65 * ridged)
    elev -= np.clip(-stress, 0.0, None) * cfg.rift_depth_m
    # Small island arcs in oceans.
    elev += np.where(base < -0.25, np.clip(island, 0.0, None) * 900.0, 0.0)

    # --- sea level to hit the target ocean fraction -------------------------
    sea = np.percentile(elev, cfg.ocean_fraction * 100.0) + cfg.sea_level_offset_m
    elev = elev - sea
    planet.elevation = elev.astype(np.float32)
    planet.land = planet.elevation > 0

    # --- slope (max gradient to 4-neighbours, normalised) -------------------
    gx = np.gradient(planet.elevation, axis=1)
    gy = np.gradient(planet.elevation, axis=0)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    planet.slope = np.clip(grad / 800.0, 0.0, 1.0).astype(np.float32)

    return {
        "plate_count": int(cfg.plate_count),
        "boundary_cells": int(boundary.sum()),
        "sea_level_m": round(float(sea), 1),
        "ocean_fraction": round(float((~planet.land).mean()), 3),
    }


def _blur(values: np.ndarray, mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Spread boundary stress to nearby cells (simple 3x3 max filter)."""
    out = values.copy()
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="edge")
        neigh = np.stack([
            padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
            padded[1:-1, :-2], padded[1:-1, 2:],
            padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:],
        ])
        out = np.maximum(out, neigh.max(axis=0) * 0.75)
    return out

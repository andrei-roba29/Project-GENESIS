"""Hydrology: depression filling, drainage, rivers, lakes, wetlands.

Rivers are derived from terrain drainage (D8 flow accumulation over the
filled elevation surface), never drawn arbitrarily. Lakes sit in filled
depressions; wetlands fringe lakes, rivers and low-lying ground.
"""
from __future__ import annotations

import heapq
import numpy as np

# 8-neighbour offsets (dr, dc, is_diagonal).
_OFFSETS = [(-1, -1, True), (-1, 0, False), (-1, 1, True), (0, -1, False),
            (0, 1, False), (1, -1, True), (1, 0, False), (1, 1, True)]


def priority_flood(elevation: np.ndarray, land: np.ndarray) -> np.ndarray:
    """Barnes priority-flood: raise depressions to their spill elevation."""
    h, w = elevation.shape
    filled = elevation.astype(np.float64).copy()
    visited = np.zeros((h, w), bool)
    heap: list[tuple[float, int, int]] = []
    # Seed with all ocean cells (outlets).
    ocean = np.argwhere(~land)
    for r, c in ocean:
        visited[r, c] = True
        heapq.heappush(heap, (filled[r, c], int(r), int(c)))
    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc, _ in _OFFSETS:
            nr, nc = r + dr, (c + dc) % w
            if nr < 0 or nr >= h or visited[nr, nc]:
                continue
            visited[nr, nc] = True
            filled[nr, nc] = max(elevation[nr, nc], z)
            heapq.heappush(heap, (filled[nr, nc], nr, nc))
    return filled.astype(np.float32)


def flow_accumulation(filled: np.ndarray, planet, threshold: int) -> dict:
    """D8 flow accumulation + Strahler order over the filled surface.

    Ocean cells are sinks: rivers terminate at the coast, so accumulation is
    only propagated through land cells.
    """
    h, w = filled.shape
    land_flat = planet.land.ravel()
    accum = np.ones(h * w, np.float64)
    order = np.ones(h * w, np.uint8)
    # Process from high to low so every contributor is handled first.
    flat_order = np.argsort(filled.ravel())[::-1]
    # Precompute per-cell lowest neighbour (downhill target).
    target = np.full(h * w, -1, np.int64)
    flat = filled.ravel()
    for idx in flat_order:
        if not land_flat[idx]:
            continue  # ocean cells are outlets, not conduits
        r, c = divmod(int(idx), w)
        z = flat[idx]
        best, best_slope = -1, 0.0
        for dr, dc, diag in _OFFSETS:
            nr, nc = r + dr, (c + dc) % w
            if nr < 0 or nr >= h:
                continue
            dz = z - flat[nr * w + nc]
            if dz <= 0:
                continue
            dist = 1.4142 if diag else 1.0
            slope = dz / dist
            if slope > best_slope:
                best_slope, best = slope, nr * w + nc
        target[idx] = best
    for idx in flat_order:
        t = target[idx]
        if t < 0:
            continue
        accum[t] += accum[idx]
        o_here, o_up = int(order[idx]), int(order[t])
        if o_here > o_up:
            order[t] = o_here
        elif o_here == o_up:
            order[t] = min(9, o_up + 1)
    river = accum > threshold
    return {"accum": accum.reshape(h, w).astype(np.float32),
            "order": order.reshape(h, w), "river": river.reshape(h, w)}


def distance_to_ocean(land: np.ndarray, planet) -> np.ndarray:
    """Dijkstra distance (km) from every land cell to the nearest ocean."""
    h, w = land.shape
    dist = np.full((h, w), np.inf)
    heap: list[tuple[float, int, int]] = []
    for r, c in np.argwhere(~land):
        dist[r, c] = 0.0
        heapq.heappush(heap, (0.0, int(r), int(c)))
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for dr, dc, diag in _OFFSETS:
            nr, nc = r + dr, (c + dc) % w
            if nr < 0 or nr >= h:
                continue
            step = planet.cell_km[r, c] * (1.4142 if diag else 1.0)
            nd = d + step
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(heap, (nd, nr, nc))
    return np.where(np.isfinite(dist), dist, 0.0).astype(np.float32)


def _smooth(a: np.ndarray, iterations: int = 2) -> np.ndarray:
    """3x3 mean smoothing (reduces micro-relief so drainage basins connect)."""
    out = a.astype(np.float64)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="edge")
        out = (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
               + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
               + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]) / 9.0
    return out.astype(np.float32)


def generate_hydrology(planet, cfg) -> dict:
    planet.filled_elevation = priority_flood(planet.elevation, planet.land)
    depth = planet.filled_elevation - planet.elevation

    # Lakes: land cells sitting in a filled depression.
    planet.lake = planet.land & (depth > 0.5)

    # Route flow over a lightly smoothed filled surface so basins connect
    # across filled flats and micro-relief.
    routing = _smooth(planet.filled_elevation, iterations=2)
    flow = flow_accumulation(routing, planet, cfg.river_threshold_cells)
    planet.flow_accum = flow["accum"]
    planet.river_order = flow["order"]
    planet.river = flow["river"] & planet.land

    # Wetlands: shallow depressions + cells adjacent to lakes/rivers on flat ground.
    near_water = planet.lake | planet.river
    near_water = _dilate(near_water, planet)
    planet.wetland = (planet.land & ~planet.lake & ~planet.river
                      & (((depth > 0.0) & (depth <= 2.0)) | (near_water & (planet.slope < 0.15))))

    planet.dist_ocean_km = distance_to_ocean(planet.land, planet)

    return {
        "river_cells": int(planet.river.sum()),
        "lake_cells": int(planet.lake.sum()),
        "wetland_cells": int(planet.wetland.sum()),
        "max_flow_accum": round(float(planet.flow_accum.max()), 1),
        "max_river_order": int(planet.river_order.max()),
    }


def _dilate(mask: np.ndarray, planet) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for dr, dc, _ in _OFFSETS:
        shifted = np.roll(mask, shift=(-dr, -dc), axis=(0, 1))
        if dr == -1:
            shifted[0, :] = False
        if dr == 1:
            shifted[-1, :] = False
        out |= shifted
    return out

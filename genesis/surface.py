"""Surface detail: continuous fractal micro-relief for close-up viewing.

The simulation grid (e.g. 180x90) is deliberately coarse — that is the right
resolution for climate and ecology, but up close it turns into visible blocks.
Rather than simulated at every scale (which would cost time and memory), the
fine structure is a *deterministic function of position*: multi-octave value
noise evaluated at the requested resolution, seeded from the world seed.

Important: this layer is detail only. It is added as a small elevation
perturbation for display and never feeds back into hydrology, climate or
ecology, so the simulation stays sharp while the surface looks continuous.
"""
from __future__ import annotations

import numpy as np

from .noise import Noise3D, sphere_xyz

_MAX_SIZE = 2048          # texture edge cap (preview-friendly, CPU-safe)


def _octave_limit(cfg, planet=None) -> float:
    """Coarsest scale, in degrees, of the fractal detail.

    Detail must never be coarser than the simulation cell size, otherwise the
    synthetic relief would fight the real terrain instead of enriching it.
    """
    cell_deg = 360.0 / float(getattr(cfg, "width", 180))
    if planet is not None:
        cell_deg = 360.0 / float(planet.width)
    return max(cell_deg * 4.0, 1.0)


# Canonical octave band shared by the whole-planet and close-up textures.
# Because both start from the same turn counts, the low-frequency structure is
# identical in either view, so zooming in reveals finer octaves instead of
# substituting a different landscape.
TURNS_MIN = 4
TURNS_MAX_WHOLE = 64
TURNS_MAX_PATCH = 1024


def _fbm(x, y, z, noise, turns_min: int, turns_max: int):
    """fBm cascade over power-of-two turn counts, plus its ridged component."""
    total = np.zeros(x.shape, np.float64)
    ridged = np.zeros(x.shape, np.float64)
    amp, amp_sum, rid_sum = 1.0, 0.0, 0.0
    turns = turns_min
    while turns <= turns_max:
        total += noise.value(x * turns, y * turns, z * turns) * amp
        amp_sum += amp
        if TURNS_MIN * 2 <= turns <= max(TURNS_MIN * 2, turns_max // 2):
            ridged += noise.ridged(x * turns, y * turns, z * turns, octaves=2) * amp
            rid_sum += amp
        amp *= 0.55
        turns *= 2
    field = total / max(amp_sum, 1e-9)
    if rid_sum > 0:
        field = field * 0.78 + (ridged / rid_sum - 0.45) * 0.9
    return field


def _normalise(field: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(field, [0.5, 99.5])
    return np.clip((field - lo) / max(hi - lo, 1e-9) * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)


def _grid(lat0: float, lon0: float, span_lat: float, span_lon: float, width: int, height: int):
    """Equirectangular lat/lon grid centred on (lat0, lon0)."""
    lats = lat0 + span_lat * (0.5 - (np.arange(height) + 0.5) / height)
    lons = lon0 + span_lon * ((np.arange(width) + 0.5) / width - 0.5)
    return np.meshgrid(lons, np.clip(lats, -90.0, 90.0))


def detail_field(cfg, width: int, height: int, seed_offset: int = 7, planet=None) -> np.ndarray:
    """Whole-planet fractal relief (equirectangular, seamless in longitude)."""
    width = int(np.clip(width, 16, _MAX_SIZE))
    height = int(np.clip(height, 8, _MAX_SIZE // 2))
    lon_g, lat_g = _grid(0.0, 0.0, 180.0, 360.0, width, height)
    x, y, z = sphere_xyz(lat_g, lon_g)
    noise = Noise3D(cfg.seed + seed_offset)
    return _normalise(_fbm(x, y, z, noise, TURNS_MIN, TURNS_MAX_WHOLE))


def detail_patch(cfg, lat0: float, lon0: float, span_lat: float, span_lon: float,
                 width: int, height: int, seed_offset: int = 7) -> np.ndarray:
    """Fractal relief for one close-up region, with finer octaves than the globe."""
    width = int(np.clip(width, 32, _MAX_SIZE))
    height = int(np.clip(height, 32, _MAX_SIZE // 2))
    lon_g, lat_g = _grid(lat0, lon0, span_lat, span_lon, width, height)
    x, y, z = sphere_xyz(lat_g, lon_g)
    noise = Noise3D(cfg.seed + seed_offset)
    # Stop the cascade where features would fall below a couple of pixels.
    deg_per_px = max(span_lat / height, span_lon / width)
    turns_max = TURNS_MAX_PATCH
    while turns_max > TURNS_MIN and 360.0 / turns_max < deg_per_px * 2.5:
        turns_max //= 2
    return _normalise(_fbm(x, y, z, noise, TURNS_MIN, max(turns_max, TURNS_MIN)))


def relief_shade(field: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Hillshade from a height field, scaled to the patch's own relief.

    Adaptive scaling matters: a whole-planet texture and a 2-degree close-up
    have wildly different gradients for the same terrain, so a fixed constant
    would leave one of them flat and the other blown out.
    """
    gy, gx = np.gradient(field)
    scale = 1.0 / max(float(np.abs(gx).mean() + np.abs(gy).mean()), 1e-6)
    slope_x = gx * scale * 0.55 * strength
    slope_y = gy * scale * 0.55 * strength
    nrm = np.sqrt(slope_x ** 2 + slope_y ** 2 + 1.0)
    # Light from the upper-left.
    shade = (-slope_x * 0.72 - slope_y * 0.72 + 0.85) / nrm
    return np.clip(0.28 + 1.15 * shade, 0.0, 1.45).astype(np.float32)


def render_detail(cfg, planet, layer: str, lat0: float, lon0: float,
                  span_lat: float, span_lon: float, width: int, height: int) -> np.ndarray:
    """RGB 0..1 close-up texture of the surface for one lat/lon region."""
    lon_g, lat_g = _grid(lat0, lon0, span_lat, span_lon, width, height)
    d = detail_patch(cfg, lat0, lon0, span_lat, span_lon, width, height)
    shade = relief_shade(d)

    elev = veg = temp = None
    if planet is not None:
        row = (90.0 - lat_g) / 180.0 * planet.height - 0.5
        col = ((lon_g + 180.0) % 360.0) / 360.0 * planet.width - 0.5
        elev = _sample_bilinear(planet.elevation, row, col)
        veg = _sample_bilinear(planet.veg_cover, row, col)
        temp = _sample_bilinear(planet.temp_current, row, col)

    rgb = np.zeros(d.shape + (3,), np.float32)

    if elev is None:
        # No simulation data (should not happen): plain shaded rock.
        rgb[..., 0] = 0.42 * shade
        rgb[..., 1] = 0.38 * shade
        rgb[..., 2] = 0.30 * shade
        return np.clip(rgb, 0.0, 1.0)

    # Coastlines are fractal: nudge the interpolated elevation with the detail
    # field so the shore gets natural crenellation instead of following the
    # simulation grid's square cell edges.
    elev = elev + d * 170.0
    land = elev > 0.0
    # Soft coast: 0 = deep water, 1 = high land.
    t_land = np.clip((elev + 400.0) / 800.0, 0.0, 1.0)
    # Beach band just above sea level.
    beach = np.exp(-((elev - 60.0) / 170.0) ** 2) * land

    # Slope from the real (interpolated) elevation field: steep ground shows
    # rock, flat ground holds soil and vegetation.
    gy, gx = np.gradient(elev)
    steep = np.clip(np.hypot(gx, gy) / 55.0, 0.0, 1.0)

    if layer == "temperature" and temp is not None:
        t = np.clip((temp + 20.0) / 70.0, 0.0, 1.0)
        rgb[..., 0] = 0.15 + t * 1.5
        rgb[..., 1] = 0.75 - np.abs(t - 0.45) * 1.1
        rgb[..., 2] = 0.95 - t * 1.25
        rgb *= shade[..., None]
    elif layer in ("vegetation", "soil", "moisture", "biome") and veg is not None:
        v = np.clip(veg, 0.0, 1.0)
        rgb[..., 0] = 0.46 - 0.36 * v
        rgb[..., 1] = 0.36 + 0.44 * v
        rgb[..., 2] = 0.22 + 0.03 * v
        rgb *= shade[..., None]
        rgb[~land] = (0.09, 0.30, 0.50)
    elif layer == "water":
        rgb[..., 0] = 0.12 + 0.06 * d
        rgb[..., 1] = 0.44 + 0.08 * d
        rgb[..., 2] = 0.62 + 0.12 * d
        rgb *= shade[..., None]
        rgb[land] = (0.42, 0.40, 0.26)
    else:
        # Terrain: rock and soil from real elevation, relief from the fractal
        # hillshade, vegetation tinted in only where the ground is gentle.
        e = np.clip((elev + 1200.0) / 5200.0, 0.0, 1.0)
        rock = np.stack([0.46 + 0.40 * e, 0.41 + 0.34 * e, 0.33 + 0.34 * e], axis=-1)
        # Vegetation is strong on flats, weak on cliffs (rock shows through).
        g = np.clip(veg, 0.0, 1.0) * 0.92 * (1.0 - 0.75 * steep)
        # Lowland foliage is deeper green, high ground more olive; the detail
        # field varies it so a forest is not one flat colour.
        tint = d * 0.5 + 0.5
        green = np.stack([
            0.10 + 0.26 * e + 0.12 * tint,
            0.30 + 0.20 * e + 0.14 * tint,
            0.09 + 0.10 * e + 0.06 * tint,
        ], axis=-1)
        land_rgb = rock * (1.0 - g[..., None]) + green * g[..., None]
        # Sand at the waterline, snow high up.
        land_rgb = land_rgb * (1.0 - beach[..., None] * 0.75) + \
                   np.array([0.80, 0.74, 0.55]) * beach[..., None] * 0.75
        # Snow line sits above most of the range; only the true summits whiten
        # and they still show rock through the cover.
        snow = np.clip((elev - 5200.0) / 1500.0, 0.0, 1.0) * (0.55 + 0.45 * steep)
        land_rgb = land_rgb * (1.0 - snow[..., None] * 0.85) + \
                   np.array([0.90, 0.93, 0.98]) * snow[..., None] * 0.85
        land_rgb = land_rgb * shade[..., None]

        # Ocean: shallows brighten toward the coast, deeps go dark; the detail
        # field adds gentle swell so the water is not a flat colour.
        depth = np.clip(-elev / 3800.0, 0.0, 1.0)
        sea_rgb = np.stack([
            0.06 + 0.10 * (1.0 - depth),
            0.26 + 0.24 * (1.0 - depth),
            0.46 + 0.26 * (1.0 - depth),
        ], axis=-1)
        sea_rgb *= (0.92 + 0.16 * (d * 0.5 + 0.5))[..., None]
        # Surf line right at the shore.
        surf = np.exp(-((elev + 30.0) / 55.0) ** 2) * (~land)
        sea_rgb = sea_rgb * (1.0 - surf[..., None] * 0.55) + \
                  np.array([0.75, 0.86, 0.92]) * surf[..., None] * 0.55

        rgb = sea_rgb * (1.0 - t_land[..., None]) + land_rgb * t_land[..., None]

    return np.clip(rgb, 0.0, 1.0)


def _sample_bilinear(field: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """True bilinear sample of a simulation grid at fractional row/col."""
    h, w = field.shape
    f = field.astype(np.float32)
    row = np.clip(row, 0.0, h - 1.0)
    col = np.mod(col, w)
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c1 = (c0 + 1) % w
    fr = (row - r0).astype(np.float32)
    fc = (col - c0).astype(np.float32)
    top = f[r0, c0] * (1 - fc) + f[r0, c1] * fc
    bot = f[r1, c0] * (1 - fc) + f[r1, c1] * fc
    return top * (1 - fr) + bot * fr

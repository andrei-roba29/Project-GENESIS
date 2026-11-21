"""Soil, natural resources and biome classification.

Soil state evolves with climate, vegetation and erosion; biomes are derived
from actual environmental conditions (temperature, precipitation, elevation,
moisture) — never painted on arbitrarily.
"""
from __future__ import annotations

import numpy as np

from .planet import BIOME_NAMES

# Biome ids (mirror planet.BIOME_NAMES).
(OCEAN, COASTAL_OCEAN, POLAR_OCEAN, ICE_CAP, TUNDRA, BOREAL, TEMPERATE,
 MEDITERRANEAN, GRASSLAND, SAVANNA, RAINFOREST, HOT_DESERT, COLD_DESERT,
 ALPINE, WETLAND, MANGROVE, FRESHWATER) = range(17)


def init_soil(planet, cfg, rng: np.random.Generator) -> dict:
    land = planet.land
    elev_km = np.maximum(planet.elevation, 0.0) / 1000.0
    noise = rng.random(planet.elevation.shape).astype(np.float32)

    # Parent material: rocky on mountains, deeper in lowlands.
    planet.soil_depth = np.where(land, np.clip(120.0 - elev_km * 18.0 + noise * 30.0, 5.0, 160.0), 0).astype(np.float32)
    planet.soil_retention = np.clip(0.35 + noise * 0.4 + np.where(planet.wetland, 0.25, 0.0), 0.0, 1.0).astype(np.float32)
    planet.soil_acidity = np.clip(6.2 + noise * 1.2 - np.where(planet.precip_annual > 1800, 0.5, 0.0), 4.0, 8.5).astype(np.float32)
    planet.soil_organic = np.where(land, np.clip(0.25 + noise * 0.3, 0, 1), 0).astype(np.float32)
    planet.soil_nutrients = np.where(land, np.clip(0.3 + noise * 0.4, 0, 1), 0).astype(np.float32)
    planet.soil_moisture = np.clip(planet.precip_annual / 2500.0, 0.0, 1.0).astype(np.float32)
    planet.soil_moisture = np.where(land, planet.soil_moisture, 1.0)
    planet.soil_erosion = np.zeros_like(planet.soil_erosion)
    planet.soil_fertility = _fertility(planet)

    compute_resources(planet, cfg)
    classify_biomes(planet, cfg)
    return {"mean_fertility": round(float(planet.soil_fertility[land].mean()), 3) if land.any() else 0}


def _fertility(planet) -> np.ndarray:
    f = (0.35 * planet.soil_organic + 0.30 * planet.soil_nutrients
         + 0.20 * np.clip(planet.soil_moisture, 0, 1)
         + 0.15 * np.clip(1.0 - np.abs(planet.soil_acidity - 6.5) / 2.5, 0, 1))
    return np.clip(f, 0.0, 1.0).astype(np.float32)


def step_soil(planet, cfg, days: int = 1) -> None:
    """Slow soil evolution: erosion, leaching, organic-matter turnover."""
    land = planet.land
    # Erosion: steep + wet + bare ground loses soil.
    erosion_rate = planet.slope * np.clip(planet.precip_current / 40.0, 0, 1) * (1.0 - planet.veg_cover) * 0.002
    planet.soil_erosion = np.clip(planet.soil_erosion + erosion_rate * days, 0, 1).astype(np.float32)
    planet.soil_depth = np.where(land, np.clip(planet.soil_depth - erosion_rate * 400.0 * days, 3.0, 200.0),
                                 planet.soil_depth).astype(np.float32)
    # Leaching acidifies wet soils slowly.
    leach = np.clip(planet.precip_current / 60.0, 0, 1) * 0.0004
    planet.soil_acidity = np.clip(planet.soil_acidity - leach * days, 3.8, 8.8).astype(np.float32)
    planet.soil_fertility = _fertility(planet)


def compute_resources(planet, cfg) -> None:
    """Per-cell natural-resource abundance (0..1)."""
    land = planet.land
    elev_km = np.maximum(planet.elevation, 0.0) / 1000.0
    depth_km = np.maximum(-planet.elevation, 0.0) / 1000.0
    arid = np.clip(1.0 - planet.precip_annual / 900.0, 0.0, 1.0)

    planet.resources["freshwater"] = np.clip(
        (planet.river * 0.8 + planet.lake * 0.9 + planet.glacier * 0.7
         + planet.soil_moisture * 0.4 + planet.wetland * 0.5), 0, 1).astype(np.float32)
    planet.resources["stone"] = np.clip(elev_km * 0.35 + planet.slope * 0.5 + (planet.elevation > 1500) * 0.3, 0, 1).astype(np.float32)
    planet.resources["sand"] = np.clip(arid * 0.7 + np.where(land & (planet.elevation < 60), 0.35, 0.0), 0, 1).astype(np.float32)
    planet.resources["clay"] = np.clip(planet.wetland * 0.7 + planet.lake * 0.4 + planet.soil_moisture * 0.3, 0, 1).astype(np.float32)
    planet.resources["salt"] = np.clip(arid * 0.6 + depth_km * 0.15 + np.where(land & (planet.elevation < 30) & (planet.precip_annual < 400), 0.4, 0.0), 0, 1).astype(np.float32)
    # Minerals/ores concentrate near tectonic boundaries and mountain belts.
    boundary = (planet.plate_id != np.roll(planet.plate_id, 1, axis=1)) | (planet.plate_id != np.roll(planet.plate_id, -1, axis=1))
    planet.resources["minerals"] = np.clip(boundary * 0.5 + elev_km * 0.25, 0, 1).astype(np.float32)
    planet.resources["ores"] = np.clip(boundary * 0.4 + np.clip(elev_km - 1.0, 0, 1) * 0.4, 0, 1).astype(np.float32)
    planet.resources["timber"] = np.clip(planet.veg_biomass / 250.0, 0, 1).astype(np.float32)


def classify_biomes(planet, cfg) -> None:
    """Rule-based biome map from climate + terrain (Whittaker-style)."""
    land = planet.land
    t = planet.temp_mean
    p = planet.precip_annual
    elev = planet.elevation
    lat = np.abs(planet.lat_grid)
    b = np.zeros(t.shape, np.uint8)

    # Oceans.
    b[~land] = OCEAN
    b[(~land) & (planet.elevation > -200)] = COASTAL_OCEAN
    b[(~land) & (lat > 60)] = POLAR_OCEAN

    # Ice.
    b[land & planet.glacier] = ICE_CAP
    b[land & (t < -12)] = ICE_CAP

    rest = land & (b == 0)
    b[rest & (t < 0) & (p < 600)] = TUNDRA
    b[rest & (t >= 0) & (t < 8) & (p >= 400)] = BOREAL
    b[rest & (t >= 8) & (t < 18) & (p >= 600) & (p < 1200)] = TEMPERATE
    b[rest & (t >= 12) & (t < 24) & (p >= 350) & (p < 800) & (lat > 28) & (lat < 45)] = MEDITERRANEAN
    b[rest & (t >= 8) & (t < 24) & (p >= 300) & (p < 700) & ~((lat > 28) & (lat < 45))] = GRASSLAND
    b[rest & (t >= 20) & (p >= 500) & (p < 1300)] = SAVANNA
    b[rest & (t >= 22) & (p >= 1300)] = RAINFOREST
    b[rest & (p < 250) & (t >= 18)] = HOT_DESERT
    b[rest & (p < 350) & (t < 18) & (t >= -12)] = COLD_DESERT
    # Alpine overrides where high and cold but not ice.
    b[rest & (elev > 2500) & (t < 8)] = ALPINE
    # Wetlands and mangroves.
    b[rest & planet.wetland] = WETLAND
    b[rest & (t >= 20) & (p >= 800) & (planet.elevation < 40) & (planet.dist_ocean_km < 30)] = MANGROVE
    # Freshwater bodies.
    b[planet.lake] = FRESHWATER
    # Fallback for anything unclassified.
    b[rest & (b == 0)] = GRASSLAND
    planet.biome = b

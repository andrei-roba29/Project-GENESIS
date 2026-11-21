"""Flora engine: aggregated vegetation populations (never per-leaf).

Five functional groups (trees, shrubs, grasses, ferns/mosses, aquatic) are
simulated per grid cell with germination, logistic growth, photosynthesis
proxy, water/nutrient consumption, seed dispersal, competition, mortality and
decomposition. A species catalog describes the plants each group contains.
"""
from __future__ import annotations

import numpy as np

from .planet import VEG_GROUPS

# Species catalog: traits used for suitability + UI display.
PLANT_SPECIES = [
    dict(name="Oak", group="trees", t_min=-5, t_max=30, p_min=500, p_max=2000,
         growth=0.06, lifespan=300, dispersal=0.02, light=0.8, water=0.5),
    dict(name="Pine", group="trees", t_min=-20, t_max=25, p_min=400, p_max=1500,
         growth=0.05, lifespan=400, dispersal=0.03, light=0.9, water=0.35),
    dict(name="Palm", group="trees", t_min=15, t_max=35, p_min=800, p_max=3000,
         growth=0.08, lifespan=120, dispersal=0.04, light=1.0, water=0.7),
    dict(name="Birch", group="trees", t_min=-10, t_max=22, p_min=500, p_max=1800,
         growth=0.07, lifespan=150, dispersal=0.05, light=0.7, water=0.6),
    dict(name="Shrub", group="shrubs", t_min=-10, t_max=32, p_min=250, p_max=1800,
         growth=0.10, lifespan=60, dispersal=0.06, light=0.7, water=0.4),
    dict(name="Grass", group="grasses", t_min=-5, t_max=35, p_min=200, p_max=2000,
         growth=0.25, lifespan=5, dispersal=0.10, light=1.0, water=0.45),
    dict(name="Fern", group="ferns", t_min=-5, t_max=28, p_min=700, p_max=3000,
         growth=0.12, lifespan=20, dispersal=0.08, light=0.4, water=0.8),
    dict(name="Moss", group="ferns", t_min=-15, t_max=25, p_min=500, p_max=3000,
         growth=0.10, lifespan=10, dispersal=0.12, light=0.3, water=0.9),
    dict(name="Aquatic Plant", group="aquatic", t_min=0, t_max=32, p_min=0, p_max=9999,
         growth=0.20, lifespan=3, dispersal=0.15, light=0.8, water=1.0),
    dict(name="Cactus", group="shrubs", t_min=10, t_max=40, p_min=0, p_max=350,
         growth=0.03, lifespan=150, dispersal=0.02, light=1.0, water=0.1),
]

# Carrying-capacity parameters per group: (temp optimum, precip optimum, moisture need).
_GROUP_K = {
    "trees": dict(t_opt=15.0, p_opt=1100.0, moisture=0.35, max_cover=0.95),
    "shrubs": dict(t_opt=18.0, p_opt=700.0, moisture=0.25, max_cover=0.70),
    "grasses": dict(t_opt=18.0, p_opt=600.0, moisture=0.20, max_cover=0.90),
    "ferns": dict(t_opt=12.0, p_opt=1600.0, moisture=0.55, max_cover=0.60),
    "aquatic": dict(t_opt=20.0, p_opt=1200.0, moisture=0.90, max_cover=0.80),
}
_GROUP_R = {"trees": 0.06, "shrubs": 0.10, "grasses": 0.25, "ferns": 0.12, "aquatic": 0.20}
_GROUP_BIOMASS = {"trees": 260.0, "shrubs": 60.0, "grasses": 18.0, "ferns": 12.0, "aquatic": 25.0}


def _suitability(planet, group: str) -> np.ndarray:
    """Environmental suitability 0..1 for a functional group."""
    k = _GROUP_K[group]
    t = planet.temp_current
    p = planet.precip_annual
    temp_s = np.exp(-(((t - k["t_opt"]) / 16.0) ** 2))
    precip_s = np.exp(-(((np.log1p(p) - np.log1p(k["p_opt"])) / 1.1) ** 2))
    moist_s = np.clip(planet.soil_moisture / max(k["moisture"], 0.05), 0.0, 1.0)
    fert_s = np.clip(planet.soil_fertility * 1.4, 0.0, 1.0)
    light_s = np.clip(1.15 - planet.cloud * 0.7, 0.1, 1.0)
    s = temp_s * (0.35 + 0.65 * precip_s) * (0.25 + 0.75 * moist_s) * (0.4 + 0.6 * fert_s) * light_s
    if group == "aquatic":
        s = np.where(planet.lake | planet.wetland | planet.river, s, 0.0)
    else:
        s = np.where(planet.land & ~planet.lake & ~planet.glacier, s, 0.0)
    return np.clip(s, 0.0, 1.0).astype(np.float32)


def init_flora(planet, cfg, rng: np.random.Generator) -> dict:
    """Seed initial vegetation from suitability (one-time colonisation)."""
    for g in VEG_GROUPS:
        s = _suitability(planet, g)
        cover = s * _GROUP_K[g]["max_cover"] * (0.6 + 0.4 * rng.random(s.shape))
        planet.veg[g] = cover.astype(np.float32)
    _sync_totals(planet)
    return {"mean_cover": round(float(planet.veg_cover[planet.land].mean()), 3) if planet.land.any() else 0}


def _sync_totals(planet) -> None:
    total = np.zeros_like(planet.veg_cover)
    biomass = np.zeros_like(planet.veg_biomass)
    for g in VEG_GROUPS:
        total += planet.veg[g]
        biomass += planet.veg[g] * _GROUP_BIOMASS[g]
    # Competition: total cover cannot exceed 1.
    over = total > 1.0
    if over.any():
        scale = np.where(over, 1.0 / np.maximum(total, 1e-9), 1.0)
        for g in VEG_GROUPS:
            planet.veg[g] = (planet.veg[g] * scale).astype(np.float32)
        total = np.minimum(total, 1.0)
        biomass = biomass * scale
    planet.veg_cover = total.astype(np.float32)
    planet.veg_biomass = biomass.astype(np.float32)


def step_flora(planet, cfg, rng: np.random.Generator, pollination: np.ndarray | None = None,
               seed_boost: np.ndarray | None = None) -> dict:
    """Advance vegetation by one week. pollination/seed_boost are per-cell
    0..1 factors supplied by the fauna engine (bees, birds, mammals)."""
    if pollination is None:
        pollination = np.zeros_like(planet.veg_cover)
    if seed_boost is None:
        seed_boost = np.zeros_like(planet.veg_cover)

    mortality_total = np.zeros_like(planet.veg_cover)
    for g in VEG_GROUPS:
        cover = planet.veg[g]
        suit = _suitability(planet, g)
        k_cap = suit * _GROUP_K[g]["max_cover"]
        r = _GROUP_R[g] * (1.0 + 0.25 * pollination)

        # Logistic growth toward carrying capacity.
        growth = r * cover * np.clip(1.0 - cover / np.maximum(k_cap, 1e-6), 0.0, 1.0)
        # Germination on suitable bare ground.
        germ = np.where((cover < 0.02) & (suit > 0.35), suit * 0.02 * (1.0 + pollination), 0.0)
        # Seed dispersal from neighbours (4-neighbour mean).
        neigh = _neighbour_mean(cover)
        dispersal = neigh * 0.05 * suit * (1.0 + 0.5 * seed_boost)
        new_cover = cover + growth + germ + dispersal

        # Mortality: background + drought + flood + heat.
        drought = np.clip(1.0 - planet.soil_moisture / 0.25, 0.0, 1.0) * 0.02
        heat = np.clip((planet.temp_current - 38.0) / 10.0, 0.0, 1.0) * 0.05
        mortality = cover * (0.004 + drought + heat)
        new_cover = np.clip(new_cover - mortality, 0.0, 1.0)
        mortality_total += mortality
        planet.veg[g] = new_cover.astype(np.float32)

        # Water + nutrient consumption (photosynthesis proxy).
        planet.soil_moisture = np.clip(planet.soil_moisture - growth * 0.10, 0.0, 1.0).astype(np.float32)
        planet.soil_nutrients = np.clip(planet.soil_nutrients - growth * 0.06, 0.0, 1.0).astype(np.float32)

    # Decomposition: dead biomass returns organic matter + nutrients.
    planet.soil_organic = np.clip(planet.soil_organic + mortality_total * 0.35, 0, 1).astype(np.float32)
    planet.soil_nutrients = np.clip(planet.soil_nutrients + mortality_total * 0.25, 0, 1).astype(np.float32)
    _sync_totals(planet)
    return {"mean_cover": round(float(planet.veg_cover[planet.land].mean()), 4) if planet.land.any() else 0}


def _neighbour_mean(a: np.ndarray) -> np.ndarray:
    padded = np.pad(a, 1, mode="edge")
    return (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0

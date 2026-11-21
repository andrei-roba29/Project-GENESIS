"""Ecological interaction engine.

Couples the systems together and computes ecosystem health. Everything here
is emergent from the other engines — no decorative subsystems:
rain -> soil moisture -> vegetation -> herbivores -> predators ->
decomposition -> nutrients -> vegetation, plus disturbance (fire, flood,
drought) and recovery.
"""
from __future__ import annotations

import numpy as np

from .fauna import SPECIES


def ecosystem_health(planet, fauna) -> dict:
    land = planet.land
    if not land.any():
        return {"health": 0.0}
    veg = float(planet.veg_cover[land].mean())
    fert = float(planet.soil_fertility[land].mean())
    moist = float(planet.soil_moisture[land].mean())
    # Biodiversity: species present relative to catalog size.
    present = len(fauna.species_in_radius()) if fauna else 0
    biodiv = present / max(len(SPECIES), 1)
    # Disturbance load: recent fire/drought/flood pressure.
    disturb = float(np.clip((planet.fire_days[land] > 0).mean() * 0.5
                            + (planet.drought_days[land] >= 30).mean() * 0.5, 0, 1))
    health = float(np.clip(0.35 * veg + 0.20 * fert + 0.15 * moist + 0.30 * biodiv - 0.25 * disturb, 0, 1))
    return {
        "health": round(health, 4),
        "vegetation": round(veg, 4),
        "fertility": round(fert, 4),
        "moisture": round(moist, 4),
        "biodiversity": round(biodiv, 4),
        "species_present": present,
        "disturbance": round(disturb, 4),
        "mean_biomass_tha": round(float(planet.veg_biomass[land].mean()), 2),
        "total_biomass_mt": round(float(planet.veg_biomass[land].sum() * planet.cell_area_km2[land].mean() / 1e6), 3),
    }


def nutrient_cycle_stats(planet) -> dict:
    land = planet.land
    return {
        "mean_organic_matter": round(float(planet.soil_organic[land].mean()), 4) if land.any() else 0,
        "mean_nutrients": round(float(planet.soil_nutrients[land].mean()), 4) if land.any() else 0,
        "mean_acidity_ph": round(float(planet.soil_acidity[land].mean()), 2) if land.any() else 0,
        "mean_soil_depth_cm": round(float(planet.soil_depth[land].mean()), 1) if land.any() else 0,
        "mean_erosion": round(float(planet.soil_erosion[land].mean()), 4) if land.any() else 0,
        "mean_retention": round(float(planet.soil_retention[land].mean()), 4) if land.any() else 0,
    }

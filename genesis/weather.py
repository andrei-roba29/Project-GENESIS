"""Daily weather: stochastic phenomena with real environmental consequences.

Each cell gets a weather state (clear, rain, heavy rain, snow, fog,
thunderstorm, lightning, hail, heatwave, cold wave, drought, storm, flood,
wildfire). Consequences are applied immediately: soil moisture, vegetation
stress, flooding, fire, drought — the ecosystem engine reacts to all of it.
"""
from __future__ import annotations

import numpy as np

# Weather ids.
(CLEAR, RAIN, HEAVY_RAIN, SNOW, FOG, THUNDERSTORM, LIGHTNING, HAIL,
 HEATWAVE, COLD_WAVE, DROUGHT, STORM, FLOOD, WILDFIRE) = range(14)

WEATHER_NAMES = ["Clear", "Rain", "Heavy Rain", "Snow", "Fog", "Thunderstorm",
                 "Lightning", "Hail", "Heatwave", "Cold Wave", "Drought",
                 "Storm", "Flood", "Wildfire"]


def step_weather(planet, cfg, rng: np.random.Generator) -> dict:
    """Advance weather by one simulated day. Returns per-phenomenon counts."""
    land = planet.land
    t = planet.temp_current
    seasonal = getattr(planet, "_precip_seasonal", np.ones_like(t))
    mean_daily = np.maximum(planet.precip_annual / 365.25 * seasonal, 0.0)

    # --- precipitation event ------------------------------------------------
    # Rain needs both cloud cover and humid air (deserts stay dry).
    humid_factor = np.clip((planet.humidity - 25.0) / 60.0, 0.12, 1.0)
    rain_prob = np.clip(planet.cloud * 0.95 * humid_factor, 0.0, 0.97)
    event = rng.random(t.shape) < rain_prob
    scale = np.maximum(mean_daily * 2.2, 0.05)
    precip = np.where(event, rng.exponential(scale), 0.0).astype(np.float32)
    # Oceans always evaporate into the atmosphere (no surface accumulation).
    planet.precip_current = np.where(land, precip, precip * 0.0)

    w = np.full(t.shape, CLEAR, np.uint8)

    # --- classify -------------------------------------------------------------
    is_rain = land & (precip >= 0.5)
    is_snow = land & (precip >= 0.5) & (t < 1.0)
    is_heavy = land & (precip >= 20.0) & (t >= 1.0)
    is_rain_only = is_rain & ~is_snow & ~is_heavy

    w[is_rain_only] = RAIN
    w[is_heavy] = HEAVY_RAIN
    w[is_snow] = SNOW

    # Fog: humid, cool, calm mornings.
    fog = land & (planet.humidity > 78) & (t < 16) & (planet.wind_speed < 6) & (precip < 0.5)
    w[fog] = FOG

    # Thunderstorms: warm + humid + unstable.
    storm_cell = land & (t > 19) & (planet.humidity > 65) & (precip >= 2.0) & (rng.random(t.shape) < 0.35)
    w[storm_cell] = THUNDERSTORM
    lightning = storm_cell & (rng.random(t.shape) < 0.55)
    w[lightning] = LIGHTNING
    hail = storm_cell & (t < 24) & (rng.random(t.shape) < 0.18)
    w[hail] = HAIL

    # Storms: strong wind + rain.
    big_storm = land & (planet.wind_speed > 20) & (precip >= 5.0)
    w[big_storm] = STORM

    # --- persistent anomalies ---------------------------------------------------
    hot = land & (t > planet.temp_mean + 9.0) & (t > 28.0)
    planet.heat_days = np.where(hot, planet.heat_days + 1, 0).astype(np.int16)
    w[planet.heat_days >= 3] = HEATWAVE

    cold = land & (t < planet.temp_mean - 9.0) & (t < -5.0)
    planet.cold_days = np.where(cold, planet.cold_days + 1, 0).astype(np.int16)
    w[planet.cold_days >= 3] = COLD_WAVE

    dry = land & (precip < 0.5) & (planet.soil_moisture < 0.15)
    planet.drought_days = np.where(dry, planet.drought_days + 1, 0).astype(np.int16)
    w[planet.drought_days >= 30] = DROUGHT

    # --- consequences -------------------------------------------------------------
    # Soil moisture: infiltration - evaporation - drainage/percolation.
    evap = np.clip(0.003 + 0.0012 * np.maximum(t, 0.0) * (0.5 + planet.wind_speed / 20.0), 0.0, 0.05)
    runoff = np.clip(precip / 100.0 - 0.25, 0.0, None) * (0.3 + planet.slope)
    drainage = np.clip(planet.soil_moisture - 0.45, 0.0, None) * 0.05
    planet.soil_moisture = np.clip(
        planet.soil_moisture + precip / 100.0 * planet.soil_retention - evap - runoff * 0.4 - drainage,
        0.0, 1.0).astype(np.float32)

    # Flood: heavy rain on flat, saturated ground.
    flood = land & (precip >= 25.0) & (planet.slope < 0.2) & (planet.soil_moisture > 0.55)
    w[flood] = FLOOD
    if flood.any():
        planet.veg_cover = np.where(flood, planet.veg_cover * 0.7, planet.veg_cover).astype(np.float32)
        planet.veg_biomass = np.where(flood, planet.veg_biomass * 0.7, planet.veg_biomass).astype(np.float32)
        # Sediment deposition boosts fertility.
        planet.soil_fertility = np.where(flood, np.clip(planet.soil_fertility + 0.05, 0, 1),
                                         planet.soil_fertility).astype(np.float32)
        planet.soil_nutrients = np.where(flood, np.clip(planet.soil_nutrients + 0.04, 0, 1),
                                         planet.soil_nutrients).astype(np.float32)

    # Wildfire: lightning on dry, vegetated ground.
    fire = land & lightning & (planet.soil_moisture < 0.18) & (planet.veg_cover > 0.25)
    w[fire] = WILDFIRE
    if fire.any():
        planet.veg_cover = np.where(fire, planet.veg_cover * 0.25, planet.veg_cover).astype(np.float32)
        planet.veg_biomass = np.where(fire, planet.veg_biomass * 0.2, planet.veg_biomass).astype(np.float32)
        for g in planet.veg:
            planet.veg[g] = np.where(fire, planet.veg[g] * 0.25, planet.veg[g]).astype(np.float32)
        # Ash returns nutrients; fire scars the soil.
        planet.soil_nutrients = np.where(fire, np.clip(planet.soil_nutrients + 0.12, 0, 1),
                                         planet.soil_nutrients).astype(np.float32)
        planet.soil_organic = np.where(fire, np.clip(planet.soil_organic - 0.08, 0, 1),
                                       planet.soil_organic).astype(np.float32)
        planet.fire_days = np.where(fire, planet.fire_days + 1, planet.fire_days).astype(np.int16)

    # Drought stress: vegetation slowly dies back.
    drought = planet.drought_days >= 30
    if drought.any():
        planet.veg_cover = np.where(drought, planet.veg_cover * 0.995, planet.veg_cover).astype(np.float32)
        planet.veg_biomass = np.where(drought, planet.veg_biomass * 0.995, planet.veg_biomass).astype(np.float32)

    planet.weather = w
    counts = {WEATHER_NAMES[i]: int((w == i).sum()) for i in range(len(WEATHER_NAMES))}
    return counts

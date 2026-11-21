"""Engine: orchestrates generation and the scheduled simulation tick.

Generation order (each stage tested before the next):
  terrain -> hydrology -> climate -> soil/resources/biomes -> flora -> fauna

Tick order (different update frequencies, never a full recompute per frame):
  daily: weather, soil moisture/erosion
  weekly: flora growth
  monthly: climate refresh, fauna, resources, stats snapshot
"""
from __future__ import annotations

import numpy as np

from .config import PlanetConfig, apply_preset
from .planet import Planet
from .clock import SimClock
from . import geography, hydrology, climate, weather, soil, flora, fauna, ecology


class Engine:
    def __init__(self, cfg: PlanetConfig | None = None):
        self.cfg = cfg or PlanetConfig()
        self.planet: Planet | None = None
        self.fauna = None
        self.clock = SimClock()
        self.generation_report: dict = {}

    # ------------------------------------------------------------------
    def generate(self, cfg: PlanetConfig | None = None) -> dict:
        if cfg is not None:
            self.cfg = cfg
        cfg = self.cfg
        if cfg.preset != "custom":
            apply_preset(cfg, cfg.preset)
        planet = Planet(cfg)
        rng = np.random.default_rng(cfg.seed)
        report = {"seed": cfg.seed, "resolution": f"{cfg.width}x{cfg.height}"}
        report["geography"] = geography.generate_terrain(planet, cfg)
        report["hydrology"] = hydrology.generate_hydrology(planet, cfg)
        report["climate"] = climate.compute_climate(planet, cfg)
        report["soil"] = soil.init_soil(planet, cfg, rng)
        report["flora"] = flora.init_flora(planet, cfg, rng)
        self.planet = planet
        self.fauna = fauna.FaunaEngine(planet, cfg)
        self.clock = SimClock()
        planet.sim_day = 0.0
        planet.version += 1
        self._log("world", f"World generated (seed {cfg.seed}, {cfg.width}x{cfg.height})")
        report["fauna"] = {"species_in_radius": len(self.fauna.species_in_radius()),
                           "radius_km": round(self.fauna.active_radius_km, 1),
                           "anchor_region": self.fauna.anchor_region}
        report["summary"] = planet.summary()
        self.generation_report = report
        return report

    # ------------------------------------------------------------------
    def tick_days(self, days: float) -> dict:
        """Advance the simulation by `days` simulated days."""
        if self.planet is None:
            raise RuntimeError("No world generated yet")
        planet, cfg = self.planet, self.cfg
        rng = planet.rng
        remaining = float(days)
        weather_counts: dict = {}
        while remaining > 1e-9:
            step = min(1.0, remaining)
            self.clock.advance(step)
            planet.sim_day = self.clock.sim_day
            # Daily systems.
            weather_counts = weather.step_weather(planet, cfg, rng)
            soil.step_soil(planet, cfg, days=int(step))
            # Weekly flora (with fauna feedbacks).
            if self.clock.due("flora"):
                flora.step_flora(planet, cfg, rng,
                                 pollination=self.fauna.pollination_map(),
                                 seed_boost=self.fauna.seed_dispersal_map())
                self.clock.mark("flora")
            # Monthly systems.
            if self.clock.due("climate"):
                climate.update_seasonal(planet, cfg)
                self.clock.mark("climate")
            if self.clock.due("fauna"):
                self.fauna.step(planet, cfg)
                self.clock.mark("fauna")
            if self.clock.due("resources"):
                soil.compute_resources(planet, cfg)
                soil.classify_biomes(planet, cfg)
                self.clock.mark("resources")
            if self.clock.due("stats"):
                self._snapshot_stats(weather_counts)
                self._log_events(weather_counts)
                self.clock.mark("stats")
            remaining -= step
        planet.version += 1
        return {"sim_day": round(planet.sim_day, 2), "date": planet.date_string(),
                "weather": weather_counts}

    # ------------------------------------------------------------------
    def _snapshot_stats(self, weather_counts: dict) -> None:
        planet = self.planet
        health = ecology.ecosystem_health(planet, self.fauna)
        planet.stats_history.append({
            "day": round(planet.sim_day, 1), "date": planet.date_string(),
            "mean_temp": round(float(planet.temp_current[planet.land].mean()), 2) if planet.land.any() else 0,
            "veg_cover": health["vegetation"], "health": health["health"],
            "species": health["species_present"],
            "total_animals": round(float(self.fauna.populations.sum()), 1),
            "weather": {k: v for k, v in weather_counts.items() if v > 0},
        })
        if len(planet.stats_history) > 500:
            planet.stats_history = planet.stats_history[-500:]

    def _log_events(self, weather_counts: dict) -> None:
        planet = self.planet
        total_cells = planet.land.size
        for name in ("Flood", "Wildfire", "Drought", "Storm", "Heatwave", "Cold Wave"):
            n = weather_counts.get(name, 0)
            if n > max(10, total_cells * 0.002):
                self._log(name.lower(), f"{name} affecting {n} cells")

    def _log(self, kind: str, text: str) -> None:
        self.planet.history.append({"day": round(self.planet.sim_day, 1),
                                    "date": self.planet.date_string(),
                                    "kind": kind, "text": text})
        if len(self.planet.history) > 500:
            self.planet.history = self.planet.history[-500:]

    # ------------------------------------------------------------------
    def cell_report(self, lat: float, lon: float) -> dict:
        from .planet import BIOME_NAMES
        from .weather import WEATHER_NAMES
        p = self.planet
        row, col = p.cell_at(lat, lon)
        region = int(p.region_id[row, col])
        fauna_here = []
        if self.fauna is not None:
            for si, pop in enumerate(self.fauna.populations[region]):
                if pop > 1.0:
                    fauna_here.append({"species": fauna.SPECIES_NAMES[si],
                                       "population": round(float(pop), 1)})
        fauna_here.sort(key=lambda d: -d["population"])
        return {
            "lat": round(float(p.lat[row]), 3), "lon": round(float(p.lon[col]), 3),
            "elevation_m": round(float(p.elevation[row, col]), 1),
            "land": bool(p.land[row, col]),
            "biome": BIOME_NAMES[int(p.biome[row, col])],
            "local_season": p.season_at(1 if lat >= 0 else -1),
            "temp_c": round(float(p.temp_current[row, col]), 2),
            "temp_mean_c": round(float(p.temp_mean[row, col]), 2),
            "precip_mm_year": round(float(p.precip_annual[row, col]), 1),
            "precip_today_mm": round(float(p.precip_current[row, col]), 2),
            "humidity_pct": round(float(p.humidity[row, col]), 1),
            "pressure_hpa": round(float(p.pressure[row, col]), 1),
            "wind_ms": round(float(p.wind_speed[row, col]), 1),
            "wind_dir_deg": round(float(np.degrees(p.wind_dir[row, col]) % 360), 0),
            "cloud_pct": round(float(p.cloud[row, col]) * 100, 1),
            "weather": WEATHER_NAMES[int(p.weather[row, col])],
            "soil": {
                "fertility": round(float(p.soil_fertility[row, col]), 3),
                "moisture": round(float(p.soil_moisture[row, col]), 3),
                "nutrients": round(float(p.soil_nutrients[row, col]), 3),
                "organic": round(float(p.soil_organic[row, col]), 3),
                "acidity_ph": round(float(p.soil_acidity[row, col]), 2),
                "depth_cm": round(float(p.soil_depth[row, col]), 1),
                "erosion": round(float(p.soil_erosion[row, col]), 3),
                "retention": round(float(p.soil_retention[row, col]), 3),
            },
            "vegetation": {
                "cover": round(float(p.veg_cover[row, col]), 3),
                "biomass_tha": round(float(p.veg_biomass[row, col]), 1),
                "groups": {g: round(float(p.veg[g][row, col]), 3) for g in p.veg},
            },
            "water": {
                "river": bool(p.river[row, col]),
                "river_order": int(p.river_order[row, col]),
                "flow_accum": round(float(p.flow_accum[row, col]), 1),
                "lake": bool(p.lake[row, col]),
                "wetland": bool(p.wetland[row, col]),
                "glacier": bool(p.glacier[row, col]),
                "dist_ocean_km": round(float(p.dist_ocean_km[row, col]), 1),
            },
            "resources": {r: round(float(p.resources[r][row, col]), 3) for r in p.resources},
            "fauna": fauna_here[:12],
            "region": region,
            "drought_days": int(p.drought_days[row, col]),
            "fire_days": int(p.fire_days[row, col]),
        }

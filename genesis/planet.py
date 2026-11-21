"""Planet grid data model.

The world is a regular latitude/longitude grid. Every simulation layer is a
2D NumPy array of shape (height, width). Row 0 is the north pole, the last
row the south pole; column 0 is longitude -180.
"""
from __future__ import annotations

import numpy as np

from .config import PlanetConfig
from .noise import sphere_xyz


# Biome ids (kept in sync with web/js/biomes.js and soil.BIOMES).
BIOME_NAMES = [
    "Ocean", "Coastal Ocean", "Polar Ocean", "Ice Cap", "Tundra",
    "Boreal Forest", "Temperate Forest", "Mediterranean", "Grassland",
    "Savanna", "Tropical Rainforest", "Hot Desert", "Cold Desert",
    "Alpine", "Wetland", "Mangrove", "Freshwater",
]

# Vegetation functional groups.
VEG_GROUPS = ["trees", "shrubs", "grasses", "ferns", "aquatic"]

# Resource names.
RESOURCES = ["freshwater", "stone", "sand", "clay", "salt", "minerals", "ores", "timber"]


class Planet:
    def __init__(self, cfg: PlanetConfig):
        self.cfg = cfg
        w, h = cfg.width, cfg.height
        self.width, self.height = w, h
        # Cell-centre coordinates.
        self.lon = -180.0 + (np.arange(w) + 0.5) * (360.0 / w)
        self.lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
        self.lon_grid, self.lat_grid = np.meshgrid(self.lon, self.lat)
        self.x, self.y, self.z = sphere_xyz(self.lat_grid, self.lon_grid)
        # Cell size in km (varies with latitude).
        self.cell_km = 2.0 * np.pi * cfg.radius_km * np.cos(np.deg2rad(self.lat_grid)) / w
        self.cell_km = np.maximum(self.cell_km, 0.05)
        self.cell_area_km2 = self.cell_km * (np.pi * cfg.radius_km / h)

        # --- terrain / hydrology ---
        self.elevation = np.zeros((h, w), np.float32)        # m, 0 = sea level
        self.land = np.zeros((h, w), bool)
        self.plate_id = np.zeros((h, w), np.int16)
        self.slope = np.zeros((h, w), np.float32)            # 0..1
        self.filled_elevation = np.zeros((h, w), np.float32)
        self.flow_accum = np.zeros((h, w), np.float32)
        self.river_order = np.zeros((h, w), np.uint8)
        self.river = np.zeros((h, w), bool)
        self.lake = np.zeros((h, w), bool)
        self.wetland = np.zeros((h, w), bool)
        self.glacier = np.zeros((h, w), bool)
        self.dist_ocean_km = np.zeros((h, w), np.float32)

        # --- climate (annual means + current seasonal/weather state) ---
        self.temp_mean = np.zeros((h, w), np.float32)        # deg C
        self.temp_current = np.zeros((h, w), np.float32)
        self.temp_amplitude = np.zeros((h, w), np.float32)   # seasonal amplitude
        self.precip_annual = np.zeros((h, w), np.float32)    # mm/year
        self.precip_current = np.zeros((h, w), np.float32)   # mm/day (weather)
        self.humidity = np.zeros((h, w), np.float32)         # %
        self.pressure = np.zeros((h, w), np.float32)         # hPa
        self.wind_speed = np.zeros((h, w), np.float32)       # m/s
        self.wind_dir = np.zeros((h, w), np.float32)         # rad, direction wind blows TO
        self.cloud = np.zeros((h, w), np.float32)            # 0..1

        # --- soil ---
        self.soil_fertility = np.zeros((h, w), np.float32)
        self.soil_moisture = np.zeros((h, w), np.float32)
        self.soil_nutrients = np.zeros((h, w), np.float32)
        self.soil_organic = np.zeros((h, w), np.float32)
        self.soil_acidity = np.zeros((h, w), np.float32)     # pH
        self.soil_depth = np.zeros((h, w), np.float32)       # cm
        self.soil_erosion = np.zeros((h, w), np.float32)     # cumulative 0..1
        self.soil_retention = np.zeros((h, w), np.float32)   # water retention 0..1

        # --- biomes / vegetation / resources ---
        self.biome = np.zeros((h, w), np.uint8)
        self.veg_cover = np.zeros((h, w), np.float32)        # total 0..1
        self.veg_biomass = np.zeros((h, w), np.float32)      # t/ha
        self.veg = {g: np.zeros((h, w), np.float32) for g in VEG_GROUPS}
        self.resources = {r: np.zeros((h, w), np.float32) for r in RESOURCES}

        # --- weather state ---
        self.weather = np.zeros((h, w), np.uint8)            # WEATHER_* id
        self.drought_days = np.zeros((h, w), np.int16)
        self.heat_days = np.zeros((h, w), np.int16)
        self.cold_days = np.zeros((h, w), np.int16)
        self.fire_days = np.zeros((h, w), np.int16)

        # --- fauna regions (coarse blocks) ---
        self.region_deg = 10.0
        self.region_rows = int(np.ceil(180.0 / self.region_deg))
        self.region_cols = int(np.ceil(360.0 / self.region_deg))
        self.region_id = self._compute_region_id()
        self.n_regions = self.region_rows * self.region_cols
        # Region centroids (lat, lon).
        self.region_lat = np.zeros(self.n_regions, np.float32)
        self.region_lon = np.zeros(self.n_regions, np.float32)
        self._compute_region_centroids()

        # --- simulation state ---
        self.sim_day = 0.0            # days since Year 1, Day 1
        self.version = 0              # bumped whenever layers change
        self.history: list[dict] = []
        self.stats_history: list[dict] = []
        self.rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------
    def _compute_region_id(self) -> np.ndarray:
        r = np.clip(((90.0 - self.lat_grid) / self.region_deg).astype(int), 0, self.region_rows - 1)
        c = np.clip(((self.lon_grid + 180.0) / self.region_deg).astype(int), 0, self.region_cols - 1)
        return (r * self.region_cols + c).astype(np.int32)

    def _compute_region_centroids(self):
        for rid in range(self.n_regions):
            mask = self.region_id == rid
            if mask.any():
                self.region_lat[rid] = float(self.lat_grid[mask].mean())
                self.region_lon[rid] = float(self.lon_grid[mask].mean())

    # ------------------------------------------------------------------
    @property
    def day_of_year(self) -> int:
        return int(self.sim_day % 365.25) + 1

    @property
    def year(self) -> int:
        return int(self.sim_day // 365.25) + 1

    def date_string(self) -> str:
        return f"Year {self.year}, Day {self.day_of_year}"

    def season(self) -> str:
        """Northern-hemisphere season name for the current day."""
        d = self.day_of_year
        if 80 <= d < 172:
            return "Spring"
        if 172 <= d < 266:
            return "Summer"
        if 266 <= d < 355:
            return "Autumn"
        return "Winter"

    def cell_at(self, lat: float, lon: float) -> tuple[int, int]:
        """Grid indices for a geographic coordinate."""
        row = int(np.clip((90.0 - lat) / 180.0 * self.height, 0, self.height - 1))
        col = int(np.clip((lon + 180.0) / 360.0 * self.width, 0, self.width - 1))
        return row, col

    def region_at(self, lat: float, lon: float) -> int:
        row, col = self.cell_at(lat, lon)
        return int(self.region_id[row, col])

    def neighbors4(self, row: int, col: int):
        h, w = self.height, self.width
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, (col + dc) % w
            if 0 <= r < h:
                yield r, c

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        land = self.land
        n = land.size
        return {
            "name": self.cfg.world_name,
            "seed": self.cfg.seed,
            "preset": self.cfg.preset,
            "width": self.width, "height": self.height,
            "radius_km": self.cfg.radius_km,
            "circumference_km": round(self.cfg.circumference_km, 1),
            "traversal_minutes": round(self.cfg.traversal_minutes, 1),
            "movement_speed_kmh": self.cfg.movement_speed_kmh,
            "ocean_fraction": round(float((~land).mean()), 3),
            "land_fraction": round(float(land.mean()), 3),
            "mean_elevation_m": round(float(self.elevation[land].mean()), 1) if land.any() else 0,
            "max_elevation_m": round(float(self.elevation.max()), 1),
            "min_elevation_m": round(float(self.elevation.min()), 1),
            "mean_temp_c": round(float(self.temp_mean[land].mean()), 2) if land.any() else 0,
            "mean_precip_mm": round(float(self.precip_annual[land].mean()), 1) if land.any() else 0,
            "mean_veg_cover": round(float(self.veg_cover[land].mean()), 3) if land.any() else 0,
            "river_cells": int(self.river.sum()),
            "lake_cells": int(self.lake.sum()),
            "wetland_cells": int(self.wetland.sum()),
            "glacier_cells": int(self.glacier.sum()),
            "sim_day": round(self.sim_day, 2),
            "date": self.date_string(),
            "season": self.season(),
            "year": self.year,
            "day_of_year": self.day_of_year,
            "paused": self.cfg.paused,
            "speed_multiplier": self.cfg.speed_multiplier,
            "days_per_real_second": round(self.cfg.days_per_real_second, 3),
            "human_population": self.cfg.human_population,
            "fauna_radius_km": round(self.cfg.fauna_radius_km(), 1),
            "version": self.version,
            "n_cells": int(n),
        }

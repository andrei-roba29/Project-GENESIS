"""Local persistence: save/load the entire world as JSON.

Grid arrays are stored as base64-encoded float32/int bytes (compact and
exact). No cloud, no database server — plain files in ./saves.
"""
from __future__ import annotations

import base64
import json
import os

import numpy as np

_FLOAT_ARRAYS = [
    "elevation", "slope", "filled_elevation", "flow_accum", "dist_ocean_km",
    "temp_mean", "temp_current", "temp_amplitude", "precip_annual",
    "precip_current", "humidity", "pressure", "wind_speed", "wind_dir", "cloud",
    "soil_fertility", "soil_moisture", "soil_nutrients", "soil_organic",
    "soil_acidity", "soil_depth", "soil_erosion", "soil_retention",
    "veg_cover", "veg_biomass",
]
_BOOL_ARRAYS = ["land", "river", "lake", "wetland", "glacier"]
_INT_ARRAYS = {"plate_id": np.int16, "river_order": np.uint8, "biome": np.uint8,
               "weather": np.uint8, "region_id": np.int32,
               "drought_days": np.int16, "heat_days": np.int16,
               "cold_days": np.int16, "fire_days": np.int16}


def _encode(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _decode(s: str, dtype, shape) -> np.ndarray:
    raw = base64.b64decode(s.encode("ascii"))
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def world_to_dict(planet, fauna, clock) -> dict:
    data = {
        "format": "project-genesis-save-v1",
        "config": planet.cfg.to_dict(),
        "sim_day": planet.sim_day,
        "clock": clock.to_dict(),
        "history": planet.history[-500:],
        "stats_history": planet.stats_history[-500:],
        "arrays": {},
        "fauna": {
            "populations": _encode(fauna.populations),
            "suitability": _encode(fauna.suitability),
            "anchor_region": fauna.anchor_region,
        },
    }
    for name in _FLOAT_ARRAYS:
        data["arrays"][name] = _encode(getattr(planet, name))
    for name in _BOOL_ARRAYS:
        data["arrays"][name] = _encode(getattr(planet, name))
    for name, dt in _INT_ARRAYS.items():
        data["arrays"][name] = _encode(getattr(planet, name).astype(dt))
    for g, arr in planet.veg.items():
        data["arrays"][f"veg_{g}"] = _encode(arr)
    for r, arr in planet.resources.items():
        data["arrays"][f"res_{r}"] = _encode(arr)
    return data


def dict_to_world(data: dict):
    """Rebuild (planet, fauna, clock) from a save dict."""
    from .config import PlanetConfig
    from .planet import Planet
    from .clock import SimClock
    from .fauna import FaunaEngine

    cfg = PlanetConfig.from_dict(data["config"])
    planet = Planet(cfg)
    planet.sim_day = float(data.get("sim_day", 0.0))
    shape = (cfg.height, cfg.width)
    arrays = data["arrays"]
    for name in _FLOAT_ARRAYS:
        setattr(planet, name, _decode(arrays[name], np.float32, shape))
    for name in _BOOL_ARRAYS:
        setattr(planet, name, _decode(arrays[name], bool, shape))
    for name, dt in _INT_ARRAYS.items():
        setattr(planet, name, _decode(arrays[name], dt, shape))
    for g in planet.veg:
        planet.veg[g] = _decode(arrays[f"veg_{g}"], np.float32, shape)
    for r in planet.resources:
        planet.resources[r] = _decode(arrays[f"res_{r}"], np.float32, shape)
    planet.history = data.get("history", [])
    planet.stats_history = data.get("stats_history", [])
    planet.version = 1

    clock = SimClock.from_dict(data.get("clock", {"sim_day": planet.sim_day}))
    from .fauna import SPECIES
    fauna = FaunaEngine.__new__(FaunaEngine)
    fauna.planet = planet
    fauna.cfg = cfg
    fauna.n_species = len(SPECIES)
    fauna.populations = _decode(data["fauna"]["populations"], np.float32,
                                (planet.n_regions, fauna.n_species))
    fauna.suitability = _decode(data["fauna"]["suitability"], np.float32,
                                (planet.n_regions, fauna.n_species))
    fauna.anchor_region = int(data["fauna"]["anchor_region"])
    fauna.active_regions = np.zeros(planet.n_regions, bool)
    fauna.refresh_active_area()
    return planet, fauna, clock


def save_world(planet, fauna, clock, directory: str, name: str) -> str:
    os.makedirs(directory, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip() or "world"
    path = os.path.join(directory, f"{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(world_to_dict(planet, fauna, clock), f)
    return path


def load_world(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return dict_to_world(json.load(f))


def list_saves(directory: str) -> list[dict]:
    out = []
    if not os.path.isdir(directory):
        return out
    for fn in sorted(os.listdir(directory)):
        if fn.endswith(".json"):
            path = os.path.join(directory, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    head = json.load(f)
                out.append({"name": fn[:-5], "path": path,
                            "date": head.get("config", {}).get("world_name", ""),
                            "sim_day": head.get("sim_day", 0),
                            "seed": head.get("config", {}).get("seed")})
            except Exception:
                continue
    return out

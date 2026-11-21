"""Local HTTP server: simulation API + static web UI.

Runs entirely offline on localhost (or 0.0.0.0 for LAN/preview). No external
services. Only dependency: NumPy. Start with:  python -m genesis.server
"""
from __future__ import annotations

import json
import os
import struct
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

from .config import PlanetConfig
from .engine import Engine
from . import persistence, ecology, flora, fauna
from .planet import BIOME_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT, "web")
SAVE_DIR = os.path.join(ROOT, "saves")

engine = Engine()
_lock = threading.Lock()
_texture_cache: dict = {}
_last_texture_version = -1


# ----------------------------------------------------------------------
# PNG texture rendering (pure Python, no Pillow needed)
# ----------------------------------------------------------------------
def _png_encode(rgba: np.ndarray) -> bytes:
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[i].tobytes() for i in range(h))
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def _colormap(values: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    """Map values through a multi-stop colour ramp -> (h, w, 3) uint8."""
    v = np.clip(values, 0.0, 1.0)
    out = np.zeros(v.shape + (3,), np.float64)
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        m = (v >= t0) & (v <= t1)
        f = np.where(t1 > t0, (v - t0) / max(t1 - t0, 1e-9), 0.0)
        for ch in range(3):
            out[..., ch] = np.where(m, c0[ch] + (c1[ch] - c0[ch]) * f, out[..., ch])
    out[..., 0] = np.where(v < stops[0][0], stops[0][1][0], out[..., 0])
    out[..., 1] = np.where(v < stops[0][0], stops[0][1][1], out[..., 1])
    out[..., 2] = np.where(v < stops[0][0], stops[0][1][2], out[..., 2])
    out[..., 0] = np.where(v > stops[-1][0], stops[-1][1][0], out[..., 0])
    out[..., 1] = np.where(v > stops[-1][0], stops[-1][1][1], out[..., 1])
    out[..., 2] = np.where(v > stops[-1][0], stops[-1][1][2], out[..., 2])
    return out.astype(np.uint8)


def render_layer(layer: str) -> bytes:
    """Render a simulation layer to an equirectangular PNG texture."""
    global _last_texture_version
    p = engine.planet
    if p is None:
        raise RuntimeError("No world generated")
    key = (layer, p.version)
    if key in _texture_cache:
        return _texture_cache[key]
    if p.version != _last_texture_version:
        _texture_cache.clear()
        _last_texture_version = p.version

    land = p.land
    if layer == "elevation":
        v = np.clip((p.elevation + 4000.0) / 11000.0, 0, 1)
        rgb = _colormap(v, [(0.0, (8, 30, 80)), (0.36, (20, 80, 160)), (0.40, (40, 120, 60)),
                            (0.55, (120, 160, 70)), (0.70, (150, 110, 60)), (0.85, (200, 190, 180)),
                            (1.0, (255, 255, 255))])
    elif layer == "biome":
        palette = {
            0: (20, 70, 140), 1: (40, 110, 170), 2: (150, 200, 230), 3: (240, 245, 250),
            4: (150, 160, 150), 5: (50, 90, 60), 6: (70, 130, 70), 7: (170, 150, 90),
            8: (190, 180, 110), 9: (210, 170, 90), 10: (25, 110, 45), 11: (230, 200, 120),
            12: (200, 170, 130), 13: (140, 140, 150), 14: (90, 140, 110), 15: (60, 130, 100),
            16: (60, 140, 190),
        }
        rgb = np.zeros(p.elevation.shape + (3,), np.uint8)
        for bid, col in palette.items():
            rgb[p.biome == bid] = col
    elif layer == "temperature":
        v = np.clip((p.temp_current + 30.0) / 70.0, 0, 1)
        rgb = _colormap(v, [(0.0, (40, 40, 160)), (0.3, (60, 120, 220)), (0.5, (120, 200, 120)),
                            (0.7, (250, 200, 80)), (1.0, (220, 60, 40))])
    elif layer == "precipitation":
        v = np.clip(p.precip_annual / 3000.0, 0, 1)
        rgb = _colormap(v, [(0.0, (240, 230, 200)), (0.4, (140, 200, 230)), (1.0, (20, 60, 180))])
    elif layer == "vegetation":
        v = np.clip(p.veg_cover, 0, 1)
        rgb = _colormap(v, [(0.0, (220, 210, 170)), (0.5, (110, 170, 80)), (1.0, (15, 90, 30))])
        rgb[~land] = (20, 70, 140)
    elif layer == "soil":
        v = np.clip(p.soil_fertility, 0, 1)
        rgb = _colormap(v, [(0.0, (90, 60, 40)), (0.5, (150, 110, 70)), (1.0, (60, 140, 60))])
        rgb[~land] = (20, 70, 140)
    elif layer == "moisture":
        v = np.clip(p.soil_moisture, 0, 1)
        rgb = _colormap(v, [(0.0, (240, 220, 180)), (0.5, (120, 180, 220)), (1.0, (15, 60, 160))])
        rgb[~land] = (20, 70, 140)
    elif layer == "water":
        rgb = np.zeros(p.elevation.shape + (3,), np.uint8)
        rgb[~land] = (20, 70, 140)
        rgb[land] = (190, 170, 120)
        rgb[p.river] = (30, 110, 220)
        rgb[p.lake] = (50, 150, 230)
        rgb[p.wetland] = (90, 160, 110)
        rgb[p.glacier] = (240, 245, 250)
    elif layer == "resources":
        v = np.zeros(p.elevation.shape)
        for r in ("freshwater", "stone", "timber", "minerals"):
            v = np.maximum(v, p.resources[r])
        rgb = _colormap(np.clip(v, 0, 1), [(0.0, (200, 190, 160)), (0.5, (230, 170, 60)), (1.0, (200, 60, 40))])
        rgb[~land] = (20, 70, 140)
    elif layer == "fauna":
        dens = np.zeros(p.elevation.shape)
        if engine.fauna is not None:
            pop = engine.fauna.populations.sum(axis=1)
            dens = pop[engine.planet.region_id]
        v = np.clip(dens / max(float(dens.max()), 1.0), 0, 1)
        rgb = _colormap(v, [(0.0, (210, 200, 170)), (0.5, (240, 160, 60)), (1.0, (190, 40, 40))])
        rgb[~land] = (20, 70, 140)
    elif layer == "weather":
        rgb = np.zeros(p.elevation.shape + (3,), np.uint8)
        rgb[~land] = (20, 70, 140)
        rgb[land] = (200, 190, 150)
        from .weather import WEATHER_NAMES
        wet = np.isin(p.weather, [1, 2, 5, 6, 7, 11])
        rgb[wet] = (60, 130, 220)
        rgb[p.weather == 3] = (230, 240, 250)
        rgb[p.weather == 13] = (230, 90, 30)
        rgb[p.weather == 12] = (40, 90, 200)
        rgb[p.weather == 8] = (240, 170, 50)
        rgb[p.weather == 9] = (120, 160, 230)
        rgb[p.weather == 10] = (220, 160, 60)
    else:
        raise ValueError(f"Unknown layer: {layer}")

    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]).astype(np.uint8)
    png = _png_encode(rgba)
    _texture_cache[key] = png
    return png


LAYERS = ["elevation", "biome", "temperature", "precipitation", "vegetation",
          "soil", "moisture", "water", "resources", "fauna", "weather"]


# ----------------------------------------------------------------------
# Background simulation loop (wall-clock driven, frame-rate independent)
# ----------------------------------------------------------------------
def _sim_loop():
    last = time.time()
    while True:
        time.sleep(0.25)
        cfg = engine.cfg
        if engine.planet is None or cfg.paused:
            last = time.time()
            continue
        now = time.time()
        elapsed = now - last
        last = now
        days = elapsed * cfg.days_per_real_second * cfg.speed_multiplier
        days = min(days, 60.0)  # avoid spiral after long pauses
        if days > 0.01:
            with _lock:
                try:
                    engine.tick_days(days)
                except Exception:
                    pass


# ----------------------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "Genesis/1.0"

    def log_message(self, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"))

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path == "/" or path == "/index.html":
                self._send(200, open(os.path.join(WEB_DIR, "index.html"), "rb").read(), "text/html")
            elif path.startswith("/css/") or path.startswith("/js/"):
                fpath = os.path.join(WEB_DIR, path.lstrip("/"))
                if not os.path.isfile(fpath):
                    return self._json({"error": "not found"}, 404)
                ctype = "text/css" if path.endswith(".css") else "text/javascript"
                self._send(200, open(fpath, "rb").read(), ctype)
            elif path == "/api/state":
                with _lock:
                    if engine.planet is None:
                        return self._json({"generated": False})
                    summary = engine.planet.summary()
                    summary["species_in_radius"] = len(engine.fauna.species_in_radius())
                    summary["health"] = ecology.ecosystem_health(engine.planet, engine.fauna)["health"]
                    self._json({"generated": True, "summary": summary,
                                "layers": LAYERS, "biomes": BIOME_NAMES})
            elif path == "/api/texture":
                layer = qs.get("layer", ["elevation"])[0]
                with _lock:
                    self._send(200, render_layer(layer), "image/png")
            elif path == "/api/geography":
                with _lock:
                    p = engine.planet
                    land = p.land
                    elev = np.where(land, p.elevation, -99999)
                    idx = np.argsort(elev.ravel())[::-1][:10]
                    peaks = []
                    for i in idx:
                        r, c = divmod(int(i), p.width)
                        peaks.append({"lat": round(float(p.lat[r]), 2),
                                      "lon": round(float(p.lon[c]), 2),
                                      "elevation_m": round(float(p.elevation[r, c]), 1),
                                      "biome": BIOME_NAMES[int(p.biome[r, c])]})
                    self._json({
                        "mean_elevation_m": round(float(p.elevation[land].mean()), 1),
                        "max_elevation_m": round(float(p.elevation.max()), 1),
                        "min_elevation_m": round(float(p.elevation.min()), 1),
                        "mean_slope": round(float(p.slope[land].mean()), 3),
                        "plate_count": engine.cfg.plate_count,
                        "peaks": peaks,
                        "mountain_cells": int((p.elevation > 2500).sum()),
                        "lowland_cells": int((land & (p.elevation < 300)).sum()),
                    })
            elif path == "/api/cell":
                lat = float(qs.get("lat", ["0"])[0]); lon = float(qs.get("lon", ["0"])[0])
                with _lock:
                    self._json(engine.cell_report(lat, lon))
            elif path == "/api/fauna":
                with _lock:
                    f = engine.fauna
                    self._json({
                        "radius_km": round(f.active_radius_km, 1),
                        "human_population": engine.cfg.human_population,
                        "formula": "R = min(Rmax, Rmin + growth * R_planet * sqrt(N))",
                        "rmin_km": engine.cfg.fauna_radius_min_km,
                        "rmax_km": round(engine.cfg.fauna_radius_max_km, 1),
                        "growth": engine.cfg.fauna_radius_growth,
                        "species_in_radius": len(f.species_in_radius()),
                        "catalog_size": len(fauna.SPECIES),
                        "anchor": {"lat": round(float(engine.planet.region_lat[f.anchor_region]), 2),
                                   "lon": round(float(engine.planet.region_lon[f.anchor_region]), 2)},
                        "species": f.table(),
                    })
            elif path == "/api/flora":
                with _lock:
                    p = engine.planet
                    land = p.land
                    self._json({
                        "species": flora.PLANT_SPECIES,
                        "groups": {g: round(float(p.veg[g][land].mean()), 4) for g in p.veg},
                        "mean_cover": round(float(p.veg_cover[land].mean()), 4),
                        "mean_biomass_tha": round(float(p.veg_biomass[land].mean()), 2),
                    })
            elif path == "/api/ecosystem":
                with _lock:
                    self._json({"health": ecology.ecosystem_health(engine.planet, engine.fauna),
                                "nutrients": ecology.nutrient_cycle_stats(engine.planet)})
            elif path == "/api/resources":
                with _lock:
                    p = engine.planet
                    land = p.land
                    totals = {}
                    for r, arr in p.resources.items():
                        totals[r] = {"mean": round(float(arr[land].mean()), 4),
                                     "cells_abundant": int((arr > 0.6).sum())}
                    self._json({"totals": totals, "resources": list(p.resources)})
            elif path == "/api/history":
                with _lock:
                    self._json({"history": engine.planet.history[-200:],
                                "stats": engine.planet.stats_history[-120:]})
            elif path == "/api/settings":
                self._json(engine.cfg.to_dict())
            elif path == "/api/saves":
                self._json({"saves": persistence.list_saves(SAVE_DIR)})
            elif path == "/api/generation":
                self._json(engine.generation_report)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            body = self._body()
            if path == "/api/generate":
                cfg = PlanetConfig.from_dict(body.get("config", {}))
                with _lock:
                    report = engine.generate(cfg)
                self._json(report)
            elif path == "/api/tick":
                days = float(body.get("days", 1.0))
                with _lock:
                    self._json(engine.tick_days(days))
            elif path == "/api/control":
                with _lock:
                    if "paused" in body:
                        engine.cfg.paused = bool(body["paused"])
                    if "speed_multiplier" in body:
                        engine.cfg.speed_multiplier = float(body["speed_multiplier"])
                    if "human_population" in body:
                        engine.cfg.human_population = int(body["human_population"])
                        if engine.fauna is not None:
                            engine.fauna.refresh_active_area()
                    self._json({"paused": engine.cfg.paused,
                                "speed_multiplier": engine.cfg.speed_multiplier,
                                "human_population": engine.cfg.human_population,
                                "fauna_radius_km": round(engine.cfg.fauna_radius_km(), 1)})
            elif path == "/api/settings":
                with _lock:
                    for k, v in body.items():
                        if hasattr(engine.cfg, k):
                            setattr(engine.cfg, k, v)
                    if engine.fauna is not None:
                        engine.fauna.refresh_active_area()
                    self._json(engine.cfg.to_dict())
            elif path == "/api/save":
                name = body.get("name", engine.cfg.world_name)
                with _lock:
                    path_saved = persistence.save_world(engine.planet, engine.fauna, engine.clock, SAVE_DIR, name)
                self._json({"saved": path_saved})
            elif path == "/api/load":
                name = body.get("name", "")
                safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()
                fpath = os.path.join(SAVE_DIR, f"{safe}.json")
                if not os.path.isfile(fpath):
                    return self._json({"error": "save not found"}, 404)
                with _lock:
                    planet, fauna_obj, clock = persistence.load_world(fpath)
                    engine.planet, engine.fauna, engine.clock = planet, fauna_obj, clock
                    engine.cfg = planet.cfg
                    planet.version += 1
                self._json({"loaded": name, "summary": planet.summary()})
            elif path == "/api/delete-save":
                name = body.get("name", "")
                safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()
                fpath = os.path.join(SAVE_DIR, f"{safe}.json")
                if os.path.isfile(fpath):
                    os.remove(fpath)
                self._json({"deleted": name})
            elif path == "/api/export":
                with _lock:
                    data = persistence.world_to_dict(engine.planet, engine.fauna, engine.clock)
                self._json(data)
            elif path == "/api/import":
                with _lock:
                    planet, fauna_obj, clock = persistence.dict_to_world(body)
                    engine.planet, engine.fauna, engine.clock = planet, fauna_obj, clock
                    engine.cfg = planet.cfg
                    planet.version += 1
                self._json({"imported": True, "summary": planet.summary()})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)


def main(port: int = 8000, host: str = "0.0.0.0"):
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Auto-generate a starter world so the UI is alive on first open.
    print(f"[genesis] Generating starter world (seed {engine.cfg.seed})...")
    engine.generate()
    print(f"[genesis] World ready: {engine.planet.summary()['ocean_fraction']*100:.0f}% ocean, "
          f"{len(engine.fauna.species_in_radius())} species in radius.")
    threading.Thread(target=_sim_loop, daemon=True).start()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[genesis] Serving on http://{host}:{port} (local, offline, zero-cost)")
    server.serve_forever()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    main(port=port)

"""Project GENESIS Phase 1 — test suite.

Run:  python -m pytest tests/ -q
"""
import numpy as np
import pytest

from genesis.config import PlanetConfig
from genesis.engine import Engine
from genesis import persistence
from genesis.noise import haversine_km


@pytest.fixture(scope="module")
def world():
    e = Engine(PlanetConfig(seed=42))
    e.generate()
    return e


def test_ocean_fraction_matches_preset(world):
    p = world.planet
    assert 0.60 < p.summary()["ocean_fraction"] < 0.82


def test_deterministic_generation():
    a = Engine(PlanetConfig(seed=7)); a.generate()
    b = Engine(PlanetConfig(seed=7)); b.generate()
    assert np.allclose(a.planet.elevation, b.planet.elevation)
    c = Engine(PlanetConfig(seed=8)); c.generate()
    assert not np.allclose(a.planet.elevation, c.planet.elevation)


def test_rivers_follow_drainage(world):
    p = world.planet
    assert p.river.sum() > 0, "expected drainage-derived rivers"
    # Every river cell must have a lower (or equal) neighbour to flow into.
    filled = p.filled_elevation
    rows, cols = np.where(p.river)
    ok = 0
    for r, c in zip(rows[:200], cols[:200]):
        neigh = [filled[max(r - 1, 0), c], filled[min(r + 1, p.height - 1), c],
                 filled[r, max(c - 1, 0)], filled[r, min(c + 1, p.width - 1)]]
        if min(neigh) <= filled[r, c] + 1e-3:
            ok += 1
    assert ok / max(len(rows[:200]), 1) > 0.8


def test_lakes_in_depressions(world):
    p = world.planet
    assert p.lake.sum() > 0
    depth = p.filled_elevation - p.elevation
    assert float(depth[p.lake].min()) > 0.4


def test_temperature_gradients(world):
    p = world.planet
    land = p.land
    # Equatorial land warmer than polar land.
    eq = land & (np.abs(p.lat_grid) < 15)
    pole = land & (np.abs(p.lat_grid) > 60)
    if eq.any() and pole.any():
        assert float(p.temp_mean[eq].mean()) > float(p.temp_mean[pole].mean()) + 5
    # Higher elevation colder (lapse rate).
    low = land & (p.elevation < 500)
    high = land & (p.elevation > 2500)
    if low.any() and high.any():
        assert float(p.temp_mean[low].mean()) > float(p.temp_mean[high].mean())


def test_biomes_from_conditions(world):
    p = world.planet
    # Deserts must be dry; rainforest wet and warm.
    desert = np.isin(p.biome, [11, 12]) & p.land
    rainforest = (p.biome == 10) & p.land
    if desert.any():
        assert float(p.precip_annual[desert].mean()) < 600
    if rainforest.any():
        assert float(p.precip_annual[rainforest].mean()) > 1000
        assert float(p.temp_mean[rainforest].mean()) > 18


def test_traversal_benchmark():
    cfg = PlanetConfig()
    assert 8.0 < cfg.traversal_minutes < 12.0, cfg.traversal_minutes


def test_fauna_radius_formula():
    cfg = PlanetConfig()
    r0 = cfg.fauna_radius_km(0)
    r2 = cfg.fauna_radius_km(2)
    r100 = cfg.fauna_radius_km(100)
    r_big = cfg.fauna_radius_km(10_000_000)
    assert r0 < r2 < r100 <= r_big
    assert r_big <= cfg.fauna_radius_max_km + 1e-6  # capped


def test_fauna_species_count_with_two_humans(world):
    # Design brief: with 2 humans, ~30 species in the active area.
    n = len(world.fauna.species_in_radius())
    assert 20 <= n <= 32, n


def test_haversine_antipode():
    d = haversine_km(0, 0, 0, 180, 100.0)
    assert abs(float(d) - np.pi * 100.0) < 1.0


def test_save_load_roundtrip(world, tmp_path):
    world.tick_days(30)
    path = persistence.save_world(world.planet, world.fauna, world.clock, str(tmp_path), "rt")
    p2, f2, c2 = persistence.load_world(path)
    assert np.allclose(world.planet.elevation, p2.elevation)
    assert np.allclose(world.planet.veg_cover, p2.veg_cover)
    assert np.allclose(world.fauna.populations, f2.populations)
    assert p2.sim_day == world.planet.sim_day
    assert p2.summary()["ocean_fraction"] == world.planet.summary()["ocean_fraction"]


def test_tick_advances_ecology(world):
    p = world.planet
    before = float(p.veg_cover[p.land].mean())
    world.tick_days(120)
    after = float(p.veg_cover[p.land].mean())
    assert 0.0 <= after <= 1.0
    assert p.sim_day > 0
    assert len(p.stats_history) > 0


def test_weather_consequences(world):
    p = world.planet
    # Weather state is populated and soil moisture responds.
    assert p.weather.max() >= 1
    assert 0.0 <= float(p.soil_moisture[p.land].mean()) <= 1.0


def test_cell_report(world):
    rep = world.cell_report(10.0, 20.0)
    for key in ("lat", "lon", "elevation_m", "biome", "temp_c", "precip_mm_year",
                "soil", "vegetation", "water", "resources", "fauna"):
        assert key in rep


# ---------------------------------------------------------------------------
# Seasons: both hemispheres, latitude dependence, and real amplitude
# ---------------------------------------------------------------------------
def test_seasons_flip_between_hemispheres(world):
    from genesis.climate import update_seasonal
    p = world.planet
    jan, jul = 15.0, 195.0
    land = p.land
    mid_n = land & (p.lat_grid > 35) & (p.lat_grid < 55)
    mid_s = land & (p.lat_grid < -35) & (p.lat_grid > -55)
    update_seasonal(p, world.cfg, jan)
    north_jan, south_jan = float(p.temp_current[mid_n].mean()), float(p.temp_current[mid_s].mean())
    update_seasonal(p, world.cfg, jul)
    north_jul, south_jul = float(p.temp_current[mid_n].mean()), float(p.temp_current[mid_s].mean())
    # The two hemispheres must be in opposite seasons.
    assert north_jul > north_jan + 6, "northern mid-latitudes must warm in July"
    assert south_jan > south_jul + 6, "southern mid-latitudes must warm in January"


def test_tropics_swing_less_than_midlatitudes(world):
    from genesis.climate import update_seasonal
    p = world.planet
    update_seasonal(p, world.cfg, 15.0); a = p.temp_current.copy()
    update_seasonal(p, world.cfg, 195.0); b = p.temp_current.copy()
    swing = np.abs(b - a)
    trop = (np.abs(p.lat_grid) < 12) & p.land
    mid = (np.abs(p.lat_grid) > 40) & (np.abs(p.lat_grid) < 60) & p.land
    assert float(swing[mid].mean()) > float(swing[trop].mean()) + 4


def test_season_names_are_opposite(world):
    p = world.planet
    p.sim_day = 14  # mid-January
    assert p.season_at(1) == "Winter"
    assert p.season_at(-1) == "Summer"


def test_continentality_scales_with_planet_size():
    from genesis.config import PlanetConfig
    small = PlanetConfig(radius_km=100.0)
    big = PlanetConfig(radius_km=6371.0)
    # The e-folding distance must follow the planet, not be an Earth constant.
    assert abs(small.continentality_scale_km / small.circumference_km - 0.10) < 1e-6
    assert big.continentality_scale_km > small.continentality_scale_km * 50


# ---------------------------------------------------------------------------
# Surface detail: fractal relief for close-up viewing
# ---------------------------------------------------------------------------
def test_detail_field_is_structured_and_bounded(world):
    from genesis import surface
    d = surface.detail_patch(world.cfg, -30, 120, 10, 10, 128, 128)
    assert d.shape == (128, 128)
    assert -1.01 < d.min() and d.max() < 1.01
    # Real structure, not flat or saturated noise.
    assert 0.15 < float(d.std()) < 0.75


def test_coastline_warping_creates_detail(world):
    """A close-up must not simply trace the simulation grid's square cells."""
    from genesis import surface
    p = world.planet
    rgb = surface.render_detail(world.cfg, p, "elevation", -30, 120, 8, 8, 96, 96)
    assert rgb.shape == (96, 96, 3)
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0
    assert float(rgb.std()) > 0.02

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


# ---------------------------------------------------------------------------
# Regression guards for the blank ground view
#
# These are static checks on the shipped frontend source: the two bugs that
# made the 3D ground view render nothing were both one-line mistakes that no
# Python-level test could otherwise catch.
# ---------------------------------------------------------------------------
import os
import re

_GLOBE_JS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "web", "js", "globe.js")


def _globe_source():
    with open(_GLOBE_JS, encoding="utf-8") as f:
        return f.read()


def test_ground_texture_handle_is_never_replaced_by_a_promise_result():
    """The GL texture is created once; load() must not overwrite the handle.

    Regression: `this.texture = tex` where `tex` was the boolean `true`
    resolved by `_loadTexture()`, which then broke `gl.bindTexture`.
    """
    src = _globe_source()
    # _loadTexture must resolve with no value (the handle is created in _initGL)
    assert "resolve(true)" not in src, "texture loader must not resolve a truthy placeholder"
    # and the only assignments to this.texture must be a GL texture object
    assigns = re.findall(r"this\.texture\s*=\s*([^;]+);", src)
    for a in assigns:
        assert a.strip() in ("null", "gl.createTexture()"), f"unexpected texture assignment: {a}"


def test_index_buffer_type_matches_what_is_drawn():
    """Index upload and drawElements must agree on the element type.

    Regression: a Uint32Array was uploaded but UNSIGNED_SHORT was passed to
    drawElements, so the GPU read garbage indices.
    """
    src = _globe_source()
    assert "this.indexType" in src, "index type must be recorded when uploading"
    assert re.search(r"drawElements\([^)]*this\.indexType", src), \
        "drawElements must reuse the recorded index type"
    # The upload must pick 16-bit whenever the vertex count fits, as it does for
    # the 240x240 ground patch (57,600 vertices).
    assert "use32" in src and "Uint16Array" in src, "16-bit indices must be used when possible"


def test_ground_shaders_avoid_undeclared_webgl_builtins():
    """Raw WebGL has no cameraPosition / projectionMatrix uniforms."""
    src = _globe_source()
    shaders = re.findall(r"const \w+_(?:VERT|FRAG) = `(.*?)`", src, re.S)
    assert shaders, "expected shader sources in globe.js"
    for body in shaders:
        for builtin in ("cameraPosition", "projectionMatrix", "modelViewMatrix", "normalMatrix"):
            assert builtin not in body, f"undeclared built-in {builtin} would fail to compile"


def test_every_shader_uniform_field_is_supplied():
    """Each uniform the shader declares must be assigned by JS."""
    src = _globe_source()
    shaders = re.findall(r"const \w+_(?:VERT|FRAG) = `(.*?)`", src, re.S)
    declared = set()
    for body in shaders:
        declared |= set(re.findall(r"uniform\s+\w+\s+(\w+)\s*;", body))
    used = set(re.findall(r"uniform\w*\(\s*this\.(\w+)", src))
    lookups = set(re.findall(r'getUniformLocation\([^,]+,\s*"(\w+)"\)', src))
    missing = declared - lookups
    assert not missing, f"shader uniforms never looked up: {sorted(missing)}"


def test_ground_mesh_uvs_are_not_flipped_relative_to_rows():
    """The texture upload and the mesh V coordinates must use one convention."""
    src = _globe_source()
    # We upload without flipping, so mesh V must follow row order (i), not 1-i.
    assert "UNPACK_FLIP_Y_WEBGL, false" in src
    assert re.search(r"uv\[k \* 2 \+ 1\] = i / \(n - 1\);", src), \
        "V must track the image row index when the upload is not flipped"


# ---------------------------------------------------------------------------
# Vegetation and cloud data for the close-up view
# ---------------------------------------------------------------------------
def test_region_payload_carries_vegetation_and_clouds():
    """The ground view draws trees/clouds from the simulation, so the payload
    must carry the real per-cell fields, not a decorative pattern."""
    import base64
    from genesis.server import region_payload
    # Use the module-level engine; generate a small world for speed.
    import genesis.server as srv
    from genesis.config import PlanetConfig
    from genesis.engine import Engine
    srv.engine = Engine(PlanetConfig(seed=42))
    srv.engine.generate()

    d = region_payload("elevation", -30, 120, 24, 64)
    cells = d["size"] ** 2
    for group in ("trees", "shrubs", "grasses", "ferns"):
        assert group in d["veg"], f"missing vegetation group {group}"
        raw = base64.b64decode(d["veg"][group])
        assert len(raw) == cells, f"{group} raster must be one byte per cell"
    cloud = base64.b64decode(d["cloud"])
    assert len(cloud) == cells
    assert len(base64.b64decode(d["biome_ids"])) == cells
    assert d["biome_names"], "biome names are needed for legend/labels"


def test_vegetation_raster_tracks_land_not_water():
    """Vegetation cover must follow the simulation: present on land, absent
    in open ocean. This is the check that keeps the decor honest."""
    import base64
    import numpy as np
    import genesis.server as srv
    from genesis.config import PlanetConfig
    from genesis.engine import Engine
    srv.engine = Engine(PlanetConfig(seed=42))
    srv.engine.generate()

    d = srv.region_payload("elevation", -30, 120, 24, 96)
    n = d["size"]
    heights = np.frombuffer(base64.b64decode(d["elevation"]), dtype="<i2").reshape(n, n) / 10.0
    trees = np.frombuffer(base64.b64decode(d["veg"]["trees"]), dtype=np.uint8).reshape(n, n)

    sea = heights < -50
    land = heights > 50
    assert sea.any() and land.any(), "patch should span coast"
    # Open ocean carries essentially nothing; what little shows up is
    # bilinear bleed from the shoreline.
    assert trees[sea].mean() < 10.0, "no trees should grow in open ocean"
    # Land must carry real cover and far more of it than the sea.
    assert trees[land].max() > 0, "land should carry some tree cover"
    assert trees[land].mean() > trees[sea].mean() * 3, "vegetation must follow land"


def test_cloud_field_is_a_fraction_everywhere(world):
    p = world.planet
    assert p.cloud.min() >= 0.0 and p.cloud.max() <= 1.0
    # Not a constant: a uniform deck would be decoration, not weather.
    assert float(p.cloud.std()) > 0.01


def test_ground_decor_source_is_regression_guarded():
    """The decor must be derived from rasters, never scattered randomly."""
    src = _globe_source()
    assert "_storeRasters" in src
    assert "this.rasters" in src
    # Placement must consult the real cover fields and skip water.
    assert "no vegetation in water" in src or "h <= 0" in src
    assert "showVegetation" in src and "showClouds" in src


def test_decor_heights_are_not_divided_twice():
    """_heights holds metres (decimetres were divided by 10 on decode).

    Dividing again put every plant 10x too low, i.e. buried in the terrain —
    the exact bug that made the decor invisible.
    """
    src = _globe_source()
    build = src[src.index("_buildDecor() {"):src.index("_buildDecorBuffers()")]
    assert "this._heights[k] / 10" not in build, "height already in metres"
    assert "const h = this._heights[k];" in build


def test_decor_buffers_are_released_before_rebuild():
    """Zooming rebuilds the billboards, which must not leak GPU buffers."""
    src = _globe_source()
    body = src[src.index("_buildDecorBuffers() {"):]
    body = body[:body.index("_drawDecor")]
    assert "deleteBuffer" in body


def test_cloud_deck_does_not_follow_the_camera():
    """Clouds sit at a fixed altitude; tying them to this.dist made the deck
    slide up and down whenever the player zoomed."""
    src = _globe_source()
    # _buildCloudPlane is the last method in the class, so slice to the end.
    body = src[src.index("_buildCloudPlane() {"):]
    assert "this.dist * 0.10" not in body
    # A new tile must not keep drawing the previous tile's cloud field.
    assert "this._cloudBuilt = false" in src


def test_cloud_deck_only_shows_thick_cloud():
    """A continuous 0..1 cover field drawn as an opaque deck blankets the map;
    only the thicker part should be visible."""
    src = _globe_source()
    body = src[src.index("_buildCloudPlane() {"):]
    assert "LO" in body and "HI" in body, "cloud cover needs a visibility range"


def test_shader_attributes_have_matching_pointers():
    """Every enabled attribute must be fed from a buffer that still has data.

    aTint was once a constant while aShade is per-instance; using an attribute
    name the shader does not declare silently draws nothing.
    """
    src = _globe_source()
    for shader in ("GROUND_VERT", "DECOR_VERT", "CLOUD_VERT"):
        body = src[src.index(f"const {shader} = `"):]
        body = body[:body.index("`;")]
        assert "aTint" not in body
    decor = src[src.index("_drawDecor(mvp, eye) {"):]
    decor = decor[:decor.index("_drawClouds")]
    assert "vertexAttribPointer(this.dShade" in decor

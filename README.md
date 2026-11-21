# Project GENESIS — Phase 1: Living Miniature Planet

A persistent, interactive planetary simulation: a small Earth-like world with
procedural geography, hydrology, climate, weather, soil, biomes, dynamic
vegetation and population-based fauna — all running **locally, offline, with
zero recurring cost**. No paid APIs, no cloud, no subscriptions. The only
dependency is [NumPy](https://numpy.org).

> **Phase 1 scope:** build the living natural world first. There are **no
> human entities yet** — humanity arrives in a later phase. The fauna
> simulation radius is driven by a configurable "human population" parameter
> (default 2, per the design brief) so the population-dependent system can be
> exercised and tuned now.

## Quick start

```bash
pip install numpy          # the only dependency
python -m genesis.server   # serves the game UI at http://localhost:8000
```

Open http://localhost:8000 — a starter world (seed 42) is generated
automatically. Drag to rotate, scroll to zoom, click to inspect any location.

Headless usage:

```bash
python -m genesis.cli generate --seed 42 --preset earthlike
python -m genesis.cli generate --seed 7 --width 360 --height 180 --tick 365 --save my-world
python -m genesis.cli load my-world
python -m pytest tests/ -q
```

## The planet

- **Finite & traversable:** default radius 100 km (circumference 628 km).
  At the default expedition speed (3770 km/h) one full crossing takes
  **≈ 10 real minutes**. Movement speed and simulation clock are independent
  settings — the benchmark can be retuned without touching the clock.
- **Simulation clock:** 1 real hour = 100 simulated years
  (1 sim year ≈ 36 real seconds, 1 sim day ≈ 0.099 real seconds), with
  pause/resume and 0.1×–100× speed control. Time is frame-rate independent;
  systems update on schedules (weather/soil daily, flora weekly, fauna and
  climate monthly) — the planet is never recomputed per frame.

## Systems (all real, all coupled)

| System | Implementation |
|---|---|
| **Geography** | Tectonic plates (Voronoi seeds + velocities): convergent boundaries → mountain belts, divergent → rifts. Multi-scale fractal + ridged noise on the sphere. Sea level tuned to the target ocean fraction. |
| **Hydrology** | Barnes priority-flood depression filling, D8 flow accumulation + Strahler order. Rivers follow drainage (never drawn arbitrarily); lakes sit in filled depressions; wetlands fringe water; glaciers high & cold. |
| **Climate** | Insolation from latitude/tilt/season; temperature = insolation − 6.5 °C/km lapse rate; continentality from distance-to-ocean; precipitation from ITCZ / subtropical highs / westerlies + orographic lift; three-cell pressure & wind bands. |
| **Seasons** | Real axial tilt physics: the two hemispheres are always in opposite seasons, the tropics barely swing while mid-latitudes swing hardest, and land far from the sea swings harder than coasts. Continentality is normalised to the planet's own circumference, so it does not depend on Earth-sized constants. Precipitation belts migrate with the sun. |
| **Weather** | Daily stochastic phenomena — rain, heavy rain, snow, fog, thunderstorms, lightning, hail, heatwaves, cold waves, droughts, storms, floods, wildfires — each with immediate consequences (soil moisture, vegetation stress, sediment, fire scars). |
| **Soil & resources** | Fertility, moisture, nutrients, organic matter, acidity, depth, erosion, retention evolve with climate, vegetation and slope. Freshwater, stone, sand, clay, salt, minerals, ores, timber derived from terrain & climate. |
| **Biomes** | Rule-based from actual conditions (Whittaker-style): rainforest, savanna, temperate/boreal forest, grassland, Mediterranean, tundra, hot/cold desert, alpine, wetland, mangrove, ice cap, marine zones. |
| **Flora** | Aggregated populations per cell (5 functional groups, 10-species catalog): germination, logistic growth, photosynthesis proxy, water/nutrient consumption, seed dispersal, competition, mortality, decomposition. |
| **Fauna** | 30-species catalog (mammals, birds, fish, insects, reptiles, amphibians). Region-based populations (one number per species per region). Active radius **R = min(Rmax, Rmin + growth·R_planet·√N)** using real spherical distances. With N = 2, ~29 species live in the active area. Populations follow habitat suitability, food, water, predation and competition — never spawned arbitrarily. |
| **Ecology** | Rain → soil → vegetation → herbivores → predators → decomposition → nutrients → vegetation, plus fire/flood/drought disturbance and recovery. Pollinators boost plant reproduction; birds/mammals disperse seeds. |
| **Persistence** | Local JSON saves (`saves/` folder) with exact float arrays — close the app, resume the same world later. Export/import JSON from the browser too. |

## Interface

- **3D globe** (WebGL, zero dependencies) + flat-map mode, 11 data layers
  (elevation, biomes, temperature, precipitation, vegetation, soil, moisture,
  water, resources, fauna, weather).
- **True zoom-in — 3D ground view.** Scroll toward the surface and the view
  hands over from the globe to a rendered landscape built from the real
  simulation heightfield: drag to orbit, dolly down into valleys, adjust the
  region size (0.5°–40°) and vertical exaggeration. Coastlines, river valleys
  and mountain belts keep their real positions; the fine relief on top is
  deterministic fractal detail that never feeds back into the simulation, so
  ecology and hydrology stay exact while the surface looks continuous.
- **11 panels:** World, Geography, Climate, Weather, Hydrology, Flora, Fauna,
  Ecosystem, Resources, Settings, History — every control performs a real
  function.
- **Location inspector:** coordinates, elevation, biome, temperature,
  precipitation, water, soil, vegetation, resources and animal populations
  for any clicked point.
- **Settings:** seed, preset (earthlike / archipelago / desert / frozen),
  resolution, radius, tilt, ocean fraction, plates, movement speed, fauna
  population driver, save/load/export/import.

## Architecture

```
genesis/
  config.py      PlanetConfig, presets, traversal + fauna-radius formulas
  noise.py       Seeded 3D value noise / fBm on the sphere
  planet.py      Grid data model (all simulation layers)
  geography.py   Tectonics + terrain
  hydrology.py   Priority-flood, D8 flow accumulation, rivers/lakes
  climate.py     Insolation, temperature, precipitation, pressure, wind
  weather.py     Daily phenomena + consequences
  surface.py     Fractal close-up relief + terrain texture rendering
  soil.py        Soil evolution, resources, biome classification
  flora.py       Vegetation functional groups + species catalog
  fauna.py       30-species catalog, population engine, radius
  ecology.py     Health index, nutrient cycle
  clock.py       Simulation clock + update scheduler
  persistence.py JSON save/load (base64 float arrays)
  engine.py      Generation pipeline + scheduled tick
  server.py      Local HTTP API + PNG texture rendering + static UI
  cli.py         Headless commands
web/             Vanilla-JS frontend (WebGL globe, panels, charts)
tests/           pytest suite (determinism, drainage, climate, radius, saves)
```

## Seasons at a glance

The clock shows both hemispheres (they are always opposite), plus the solar
declination. Switch to the **Temperature** layer and let time run: the warm
bands visibly migrate north and south. The `Climate` panel shows the model and
the measured mean seasonal swing.

## Performance

- 180×90 world: generation ≈ 0.4 s, 90-day tick ≈ 0.2 s.
- 360×180 world: generation ≈ 1 s, 680 drainage-derived rivers.
- Simulation data is fully separated from rendering; textures are cached per
  layer and only re-rendered when the world version changes.

## Costs

**Zero.** No paid APIs, no cloud, no map/weather/terrain services, no
subscriptions, no build step. Runs offline on any machine with Python 3.10+.

"""Fauna engine: population-based animal simulation.

Animals are NOT individual agents. Each species has one population number
per region (a coarse geographic block); a single "deer population" may
represent hundreds of deer. Populations are generated and adjusted from
habitat suitability, food, water, climate and vegetation — never spawned
arbitrarily next to humans.

The active simulation area is a spherical radius around the habitat anchor,
driven by the living human population:
    R = min(Rmax, Rmin + growth * R_planet * sqrt(N))
With N = 2 the catalog supports ~30 species in the active area.
"""
from __future__ import annotations

import numpy as np

from .noise import haversine_km

# diet: herbivore | carnivore | omnivore | insectivore | piscivore | scavenger
SPECIES = [
    # name, class, diet, trophic, biomes, t_min, t_max, p_min, capacity, growth, prey
    dict(name="Red Deer", cls="Mammal", diet="herbivore", trophic=1, biomes=[5, 6, 8], t_min=-10, t_max=28, p_min=400, cap=900, growth=0.10, prey=[]),
    dict(name="Rabbit", cls="Mammal", diet="herbivore", trophic=1, biomes=[6, 8, 9, 11], t_min=-8, t_max=32, p_min=250, cap=1400, growth=0.22, prey=[]),
    dict(name="Wild Boar", cls="Mammal", diet="omnivore", trophic=1, biomes=[5, 6, 7, 10], t_min=-8, t_max=30, p_min=400, cap=700, growth=0.12, prey=[]),
    dict(name="Beaver", cls="Mammal", diet="herbivore", trophic=1, biomes=[5, 6, 14], t_min=-10, t_max=25, p_min=500, cap=350, growth=0.10, prey=[]),
    dict(name="Red Fox", cls="Mammal", diet="omnivore", trophic=2, biomes=[4, 5, 6, 8, 9], t_min=-20, t_max=32, p_min=200, cap=500, growth=0.14, prey=["Rabbit", "Rodent"]),
    dict(name="Wolf", cls="Mammal", diet="carnivore", trophic=3, biomes=[4, 5, 6], t_min=-25, t_max=28, p_min=300, cap=160, growth=0.08, prey=["Red Deer", "Wild Boar"]),
    dict(name="Brown Bear", cls="Mammal", diet="omnivore", trophic=3, biomes=[5, 6], t_min=-15, t_max=28, p_min=400, cap=120, growth=0.06, prey=["Red Deer", "Wild Boar"]),
    dict(name="Lynx", cls="Mammal", diet="carnivore", trophic=3, biomes=[5], t_min=-25, t_max=25, p_min=300, cap=90, growth=0.07, prey=["Rabbit", "Rodent"]),
    dict(name="Rodent", cls="Mammal", diet="herbivore", trophic=1, biomes=[4, 5, 6, 7, 8, 9, 10, 11, 12], t_min=-15, t_max=38, p_min=150, cap=2200, growth=0.30, prey=[]),
    dict(name="Bat", cls="Mammal", diet="insectivore", trophic=2, biomes=[5, 6, 7, 10], t_min=-5, t_max=32, p_min=300, cap=800, growth=0.12, prey=["Insect"]),
    dict(name="Golden Eagle", cls="Bird", diet="carnivore", trophic=3, biomes=[4, 5, 6, 8, 13], t_min=-15, t_max=30, p_min=200, cap=110, growth=0.06, prey=["Rabbit", "Rodent"]),
    dict(name="Owl", cls="Bird", diet="carnivore", trophic=3, biomes=[5, 6, 7, 10], t_min=-15, t_max=30, p_min=250, cap=200, growth=0.08, prey=["Rodent", "Frog"]),
    dict(name="Songbird", cls="Bird", diet="insectivore", trophic=2, biomes=[5, 6, 7, 9, 10], t_min=-10, t_max=32, p_min=300, cap=1500, growth=0.18, prey=["Insect"]),
    dict(name="Duck", cls="Bird", diet="omnivore", trophic=1, biomes=[6, 14, 16], t_min=-10, t_max=28, p_min=400, cap=600, growth=0.14, prey=[]),
    dict(name="Vulture", cls="Bird", diet="scavenger", trophic=3, biomes=[8, 9, 11, 12], t_min=-5, t_max=38, p_min=150, cap=140, growth=0.07, prey=[]),
    dict(name="Honey Bee", cls="Insect", diet="herbivore", trophic=1, biomes=[6, 7, 8, 9, 10], t_min=5, t_max=35, p_min=300, cap=3000, growth=0.25, prey=[]),
    dict(name="Ant", cls="Insect", diet="omnivore", trophic=1, biomes=[5, 6, 7, 8, 9, 10, 11, 12], t_min=-5, t_max=40, p_min=150, cap=4000, growth=0.30, prey=[]),
    dict(name="Butterfly", cls="Insect", diet="herbivore", trophic=1, biomes=[6, 7, 8, 9, 10], t_min=8, t_max=35, p_min=300, cap=1800, growth=0.22, prey=[]),
    dict(name="Beetle", cls="Insect", diet="scavenger", trophic=1, biomes=[5, 6, 7, 8, 9, 10, 11], t_min=-5, t_max=38, p_min=150, cap=3500, growth=0.28, prey=[]),
    dict(name="Insect", cls="Insect", diet="herbivore", trophic=1, biomes=[5, 6, 7, 8, 9, 10, 11, 12], t_min=-5, t_max=40, p_min=150, cap=5000, growth=0.35, prey=[]),
    dict(name="Trout", cls="Fish", diet="insectivore", trophic=2, biomes=[16, 5, 6], t_min=-2, t_max=22, p_min=500, cap=900, growth=0.16, prey=["Insect"]),
    dict(name="Freshwater Fish", cls="Fish", diet="omnivore", trophic=2, biomes=[16, 6, 9, 10, 14], t_min=0, t_max=32, p_min=300, cap=1200, growth=0.18, prey=["Insect"]),
    dict(name="Salmon", cls="Fish", diet="piscivore", trophic=3, biomes=[16, 5], t_min=-2, t_max=20, p_min=600, cap=500, growth=0.10, prey=["Freshwater Fish", "Trout"]),
    dict(name="Frog", cls="Amphibian", diet="insectivore", trophic=2, biomes=[6, 10, 14, 16], t_min=0, t_max=30, p_min=500, cap=1000, growth=0.20, prey=["Insect"]),
    dict(name="Snake", cls="Reptile", diet="carnivore", trophic=2, biomes=[6, 7, 8, 9, 11], t_min=2, t_max=38, p_min=200, cap=600, growth=0.12, prey=["Rodent", "Frog"]),
    dict(name="Lizard", cls="Reptile", diet="insectivore", trophic=2, biomes=[7, 8, 9, 11, 12], t_min=5, t_max=42, p_min=100, cap=900, growth=0.16, prey=["Insect"]),
    dict(name="Turtle", cls="Reptile", diet="omnivore", trophic=1, biomes=[9, 10, 14, 16], t_min=5, t_max=35, p_min=300, cap=400, growth=0.08, prey=[]),
    dict(name="Crocodile", cls="Reptile", diet="carnivore", trophic=3, biomes=[10, 14], t_min=15, t_max=38, p_min=700, cap=80, growth=0.05, prey=["Duck", "Freshwater Fish"]),
    dict(name="Shark", cls="Fish", diet="carnivore", trophic=3, biomes=[0, 1], t_min=5, t_max=30, p_min=0, cap=150, growth=0.05, prey=["Freshwater Fish"]),
    dict(name="Whale", cls="Mammal", diet="carnivore", trophic=3, biomes=[0, 2], t_min=-5, t_max=25, p_min=0, cap=60, growth=0.03, prey=["Shark"]),
]

SPECIES_NAMES = [s["name"] for s in SPECIES]
SPECIES_INDEX = {n: i for i, n in enumerate(SPECIES_NAMES)}
CLASSES = sorted({s["cls"] for s in SPECIES})


class FaunaEngine:
    """Population matrix: populations[region, species]."""

    def __init__(self, planet, cfg):
        self.planet = planet
        self.cfg = cfg
        self.n_species = len(SPECIES)
        self.populations = np.zeros((planet.n_regions, self.n_species), np.float32)
        self.suitability = np.zeros((planet.n_regions, self.n_species), np.float32)
        self.anchor_region = 0
        self.active_regions = np.zeros(planet.n_regions, bool)
        self._compute_suitability()
        self._place_anchor()
        self.refresh_active_area()
        self._seed_populations()

    # ------------------------------------------------------------------
    def _region_stats(self):
        """Per-region mean climate/vegetation (computed once per call)."""
        p = self.planet
        n = p.n_regions
        temp = np.zeros(n); precip = np.zeros(n); veg = np.zeros(n)
        moisture = np.zeros(n); counts = np.zeros(n)
        flat_region = p.region_id.ravel()
        for arr, out in ((p.temp_mean, temp), (p.precip_annual, precip),
                         (p.veg_biomass, veg), (p.soil_moisture, moisture)):
            np.add.at(out, flat_region, arr.ravel())
        np.add.at(counts, flat_region, 1.0)
        counts = np.maximum(counts, 1.0)
        return temp / counts, precip / counts, veg / counts, moisture / counts

    def _compute_suitability(self):
        p = self.planet
        temp, precip, veg, moisture = self._region_stats()
        # Dominant biome per region.
        biome_counts = np.zeros((p.n_regions, 17), np.float32)
        np.add.at(biome_counts, (p.region_id.ravel(), p.biome.ravel()), 1.0)
        dom_biome = np.argmax(biome_counts, axis=1)
        for si, sp in enumerate(SPECIES):
            biome_ok = np.isin(dom_biome, sp["biomes"]).astype(np.float32)
            t_ok = np.exp(-(((temp - (sp["t_min"] + sp["t_max"]) / 2.0) / ((sp["t_max"] - sp["t_min"]) / 1.6)) ** 2))
            p_ok = np.clip(precip / max(sp["p_min"], 1.0), 0.0, 1.0) if sp["p_min"] > 0 else np.ones_like(precip)
            veg_ok = np.clip(veg / 60.0, 0.05, 1.0) if sp["diet"] in ("herbivore", "omnivore") else np.ones_like(veg)
            water_ok = np.clip(moisture * 1.5, 0.1, 1.0)
            self.suitability[:, si] = biome_ok * t_ok * (0.3 + 0.7 * p_ok) * veg_ok * water_ok

    def _place_anchor(self):
        """Anchor = region with the highest total habitat suitability.

        Phase 1 has no human entities; the anchor stands in for the future
        population centre. When entities exist, anchors become their positions.
        """
        total = self.suitability.sum(axis=1)
        self.anchor_region = int(np.argmax(total))

    def refresh_active_area(self):
        """Recompute which regions fall inside the population-driven radius."""
        p = self.planet
        radius = self.cfg.fauna_radius_km()
        a_lat, a_lon = p.region_lat[self.anchor_region], p.region_lon[self.anchor_region]
        dist = haversine_km(a_lat, a_lon, p.region_lat, p.region_lon, self.cfg.radius_km)
        self.active_regions = dist <= radius
        self.active_radius_km = float(radius)
        self.anchor_distance_km = dist

    def _seed_populations(self):
        rng = np.random.default_rng(self.cfg.seed + 7)
        for si, sp in enumerate(SPECIES):
            k = sp["cap"] * self.suitability[:, si]
            seeded = np.where(self.active_regions, k * (0.3 + 0.4 * rng.random(len(k))), 0.0)
            self.populations[:, si] = seeded

    # ------------------------------------------------------------------
    def species_in_radius(self) -> list[int]:
        """Species with a viable population inside the active radius."""
        totals = self.populations[self.active_regions].sum(axis=0)
        return [i for i, t in enumerate(totals) if t > 5.0]

    def step(self, planet, cfg) -> dict:
        """Advance fauna by one month."""
        self.refresh_active_area()
        pops = self.populations
        # Food factors per region.
        veg = self._region_biomass()
        water = self._region_moisture()
        totals = pops.sum(axis=0)

        new_pops = pops.copy()
        for si, sp in enumerate(SPECIES):
            k = sp["cap"] * self.suitability[:, si]
            diet = sp["diet"]
            if diet == "herbivore":
                food = np.clip(veg / 80.0, 0.0, 1.2)
            elif diet == "insectivore":
                food = np.clip(pops[:, SPECIES_INDEX["Insect"]] / 2500.0, 0.0, 1.2)
            elif diet == "piscivore":
                food = np.clip(pops[:, SPECIES_INDEX["Freshwater Fish"]] / 600.0, 0.0, 1.2)
            elif diet == "scavenger":
                food = np.clip(totals.sum() / 20000.0, 0.1, 1.2) * np.ones_like(veg)
            elif diet == "carnivore":
                prey_total = np.zeros_like(veg)
                for prey in sp["prey"]:
                    prey_total += pops[:, SPECIES_INDEX[prey]]
                food = np.clip(prey_total / 400.0, 0.0, 1.2)
            else:  # omnivore
                food = np.clip(0.5 * veg / 80.0 + 0.5 * pops[:, SPECIES_INDEX["Insect"]] / 2500.0, 0.0, 1.2)
            water_f = np.clip(water * 1.6, 0.15, 1.0)
            k_eff = k * np.clip(food, 0.0, 1.0) * water_f
            growth = sp["growth"] * pops[:, si] * np.clip(1.0 - pops[:, si] / np.maximum(k_eff, 1.0), -1.0, 1.0)
            # Predation pressure on prey from their predators.
            predation = np.zeros_like(veg)
            for pj, other in enumerate(SPECIES):
                if sp["name"] in other["prey"]:
                    predation += pops[:, pj] * 0.0008
            growth -= pops[:, si] * np.clip(predation, 0.0, 0.2)
            # Environmental mortality (drought, flood, fire handled via water/veg).
            growth -= pops[:, si] * np.clip(0.02 - water_f * 0.015, 0.0, 0.05)
            new_pops[:, si] = np.clip(pops[:, si] + growth, 0.0, None)
        # Only the active radius is simulated in detail; outside decays to zero.
        new_pops[~self.active_regions] = 0.0
        self.populations = new_pops.astype(np.float32)
        return {"species_in_radius": len(self.species_in_radius()),
                "total_animals": round(float(self.populations.sum()), 1)}

    # ------------------------------------------------------------------
    def _region_biomass(self) -> np.ndarray:
        out = np.zeros(self.planet.n_regions)
        np.add.at(out, self.planet.region_id.ravel(), self.planet.veg_biomass.ravel())
        counts = np.bincount(self.planet.region_id.ravel(), minlength=self.planet.n_regions)
        return out / np.maximum(counts, 1)

    def _region_moisture(self) -> np.ndarray:
        out = np.zeros(self.planet.n_regions)
        np.add.at(out, self.planet.region_id.ravel(), self.planet.soil_moisture.ravel())
        counts = np.bincount(self.planet.region_id.ravel(), minlength=self.planet.n_regions)
        return out / np.maximum(counts, 1)

    def pollination_map(self) -> np.ndarray:
        """Per-cell pollination factor from bee/butterfly populations."""
        p = self.planet
        idx = p.region_id
        bees = self.populations[:, SPECIES_INDEX["Honey Bee"]] + self.populations[:, SPECIES_INDEX["Butterfly"]]
        return np.clip(bees[idx] / 1500.0, 0.0, 1.0).astype(np.float32)

    def seed_dispersal_map(self) -> np.ndarray:
        """Per-cell seed-dispersal factor from birds + mammals."""
        p = self.planet
        idx = p.region_id
        birds = (self.populations[:, SPECIES_INDEX["Songbird"]]
                 + self.populations[:, SPECIES_INDEX["Duck"]]
                 + self.populations[:, SPECIES_INDEX["Red Deer"]])
        return np.clip(birds[idx] / 1200.0, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    def table(self) -> list[dict]:
        rows = []
        totals = self.populations.sum(axis=0)
        active_totals = self.populations[self.active_regions].sum(axis=0)
        for si, sp in enumerate(SPECIES):
            best_region = int(np.argmax(self.populations[:, si])) if totals[si] > 0 else -1
            rows.append({
                "name": sp["name"], "class": sp["cls"], "diet": sp["diet"],
                "population": round(float(totals[si]), 1),
                "in_radius": round(float(active_totals[si]), 1),
                "suitability": round(float(self.suitability[best_region, si]), 3) if best_region >= 0 else 0.0,
                "habitat": ", ".join(_biome_name(b) for b in sp["biomes"]),
                "region_lat": round(float(self.planet.region_lat[best_region]), 1) if best_region >= 0 else None,
                "region_lon": round(float(self.planet.region_lon[best_region]), 1) if best_region >= 0 else None,
            })
        return rows


def _biome_name(bid: int) -> str:
    from .planet import BIOME_NAMES
    return BIOME_NAMES[bid] if 0 <= bid < len(BIOME_NAMES) else str(bid)

"""Planet configuration, presets and derived physical quantities.

All values are plain data so they can be serialised to JSON for save/load.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math


@dataclass
class PlanetConfig:
    """Full user-facing configuration for one generated world."""

    # Identity / generation
    seed: int = 42
    preset: str = "earthlike"          # earthlike | archipelago | desert | frozen | custom
    width: int = 180                   # grid columns (longitude)
    height: int = 90                   # grid rows (latitude)

    # Planet physics
    radius_km: float = 100.0           # miniature Earth (1/64 scale)
    axial_tilt_deg: float = 23.5
    ocean_fraction: float = 0.71       # target share of ocean cells
    sea_level_offset_m: float = 0.0

    # Terrain generation
    plate_count: int = 9
    noise_octaves: int = 5
    mountain_height_m: float = 6500.0
    continent_height_m: float = 3200.0
    rift_depth_m: float = 1800.0

    # Hydrology
    river_threshold_cells: int = 15    # flow-accumulation threshold for a river cell

    # Climate
    lapse_rate_c_per_km: float = 6.5
    greenhouse_offset_c: float = 0.0

    # Fauna radius driver. Phase 1 has NO human entities; this parameter
    # stands in for the future living population so the population-dependent
    # simulation radius can be exercised and tuned. Default 2 matches the
    # design brief ("with 2 human entities, ~30 species in the active area").
    human_population: int = 2
    fauna_radius_min_km: float = 40.0
    fauna_radius_growth: float = 0.55  # multiplied by planet radius * sqrt(N)
    fauna_radius_max_fraction: float = 0.45  # of pi * R (just under half circumference)

    # Movement-time calibration (independent of the simulation clock).
    # Default: 2*pi*100 km / 3770 km/h = 10.0 minutes to cross the planet.
    movement_speed_kmh: float = 3770.0  # expedition speed

    # Simulation clock: 1 real hour = 100 simulated years by default.
    sim_years_per_real_hour: float = 100.0
    speed_multiplier: float = 1.0
    paused: bool = False

    # Save/load
    world_name: str = "Genesis World"

    # -- derived quantities -------------------------------------------------
    @property
    def circumference_km(self) -> float:
        return 2.0 * math.pi * self.radius_km

    @property
    def traversal_minutes(self) -> float:
        """Time to walk/drive the full circumference at movement speed."""
        if self.movement_speed_kmh <= 0:
            return float("inf")
        return self.circumference_km / self.movement_speed_kmh * 60.0

    @property
    def days_per_real_second(self) -> float:
        """Default clock rate: sim_years_per_real_hour -> days per second."""
        return self.sim_years_per_real_hour * 365.25 / 3600.0

    @property
    def fauna_radius_max_km(self) -> float:
        return self.fauna_radius_max_fraction * math.pi * self.radius_km

    def fauna_radius_km(self, human_population: int | None = None) -> float:
        """Population-dependent animal simulation radius (spherical, km).

        R = min(Rmax, Rmin + growth * R_planet * sqrt(N))
        More humans -> larger active area; capped below half the circumference.
        """
        n = self.human_population if human_population is None else human_population
        r = self.fauna_radius_min_km + self.fauna_radius_growth * self.radius_km * math.sqrt(max(0, n))
        return min(self.fauna_radius_max_km, r)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlanetConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


PRESETS: dict[str, dict] = {
    "earthlike": dict(ocean_fraction=0.71, axial_tilt_deg=23.5, plate_count=9,
                      mountain_height_m=6500.0, continent_height_m=3200.0),
    "archipelago": dict(ocean_fraction=0.82, axial_tilt_deg=20.0, plate_count=12,
                        mountain_height_m=3500.0, continent_height_m=1800.0),
    "desert": dict(ocean_fraction=0.55, axial_tilt_deg=25.0, plate_count=7,
                   mountain_height_m=4500.0, continent_height_m=2600.0),
    "frozen": dict(ocean_fraction=0.65, axial_tilt_deg=28.0, plate_count=8,
                   mountain_height_m=5500.0, continent_height_m=2800.0,
                   greenhouse_offset_c=-12.0),
}


def apply_preset(cfg: PlanetConfig, preset: str) -> PlanetConfig:
    if preset in PRESETS:
        for k, v in PRESETS[preset].items():
            setattr(cfg, k, v)
        cfg.preset = preset
    return cfg

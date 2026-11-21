"""Climate engine: temperature, precipitation, pressure, wind, seasons.

Scientific approximations (not full atmospheric physics):
- Insolation from latitude, axial tilt and day of year (standard formula).
- Mean temperature from insolation minus an elevation lapse rate, with
  continentality (distance from ocean) controlling seasonal amplitude.
- Precipitation from latitude bands (ITCZ, subtropical highs, westerlies)
  modulated by ocean distance, orographic lift and temperature.
- Pressure/wind from the classic three-cell circulation bands.
"""
from __future__ import annotations

import numpy as np

from .config import PlanetConfig


def _insolation(lat_deg: np.ndarray, decl_deg: np.ndarray) -> np.ndarray:
    """Daily-mean insolation factor in [0, 1] (normalised to equatorial equinox)."""
    phi = np.deg2rad(lat_deg)
    d = np.deg2rad(decl_deg)
    tan_phi, tan_d = np.tan(phi), np.tan(d)
    cos_h0 = np.clip(-tan_phi * tan_d, -1.0, 1.0)
    h0 = np.arccos(cos_h0)
    s = (h0 * np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.sin(h0)) / np.pi
    return np.clip(s, 0.0, None)


def declination_deg(day_of_year: float, tilt_deg: float) -> float:
    """Solar declination: 0 at the equinoxes, +/- tilt at the solstices."""
    return tilt_deg * np.sin(2.0 * np.pi * (day_of_year - 80.0) / 365.25)


def compute_climate(planet, cfg: PlanetConfig, day_of_year: float | None = None) -> dict:
    """(Re)compute annual climate + current seasonal state."""
    lat = planet.lat_grid
    land = planet.land
    elev_km = np.maximum(planet.elevation, 0.0) / 1000.0
    doy = planet.day_of_year if day_of_year is None else day_of_year
    decl = declination_deg(doy, cfg.axial_tilt_deg)

    # --- annual mean insolation (12 monthly samples) ---
    s_annual = np.zeros_like(lat)
    for m in range(12):
        s_annual += _insolation(lat, declination_deg(m * 30.4 + 15, cfg.axial_tilt_deg))
    s_annual /= 12.0
    s_norm = s_annual / max(float(s_annual.max()), 1e-9)

    # --- temperature ---
    t_mean = 40.0 * s_norm - 15.0 + cfg.greenhouse_offset_c
    t_mean = t_mean - cfg.lapse_rate_c_per_km * elev_km
    # Oceans moderate temperature (heat capacity).
    t_mean = np.where(land, t_mean, t_mean * 0.92 + 2.0)
    planet.temp_mean = t_mean.astype(np.float32)

    # Seasonal amplitude: small over oceans, large inland. Continentality is
    # normalised by the planet's own size, so the same physics works for a
    # 100 km miniature world and a full-size planet (see PlanetConfig).
    cont = 1.0 - np.exp(-planet.dist_ocean_km / cfg.continentality_scale_km)
    amp_land = cfg.seasonal_amp_min_c + cfg.seasonal_amp_range_c * cont
    amp = np.where(land, amp_land, cfg.ocean_seasonal_amp_c)
    # Upper-troposphere amplification: high ground has thinner air and swings
    # harder between seasons (scaled down by the miniature plane's gravity).
    amp = amp * (1.0 + cfg.seasonal_elevation_gain * elev_km)
    planet.temp_amplitude = amp.astype(np.float32)
    # Seasonal range of the insolation itself, used to scale the swing by
    # latitude so the tropics stay equable while mid-latitudes get real seasons.
    planet.temp_cos_lat = np.cos(np.radians(lat)).astype(np.float32)

    # --- precipitation (annual, mm) ---
    # ITCZ band follows the declination; subtropical dry belts at ~30 deg;
    # westerlies wet band at ~55 deg; polar desert.
    itcz = 2400.0 * np.exp(-(((lat - decl) / 13.0) ** 2))
    westerlies = 1100.0 * np.exp(-(((np.abs(lat) - 55.0) / 16.0) ** 2))
    base = itcz + westerlies + 150.0
    # Moisture supply decays inland; orographic lift adds precipitation.
    moisture = np.where(land, 0.25 + 0.75 * np.exp(-planet.dist_ocean_km / 600.0), 1.0)
    orographic = 1.0 + 0.55 * elev_km
    warm = np.clip(0.40 + (t_mean + 10.0) / 60.0, 0.25, 1.25)
    precip = base * moisture * orographic * warm
    precip = np.where(land, precip, np.maximum(precip, 600.0))
    planet.precip_annual = np.clip(precip, 0.0, 4500.0).astype(np.float32)

    # --- pressure (hPa): equatorial low, subtropical highs, subpolar lows ---
    shift = decl * 0.4
    p = (1013.0
         - 5.0 * np.exp(-(((lat - shift) / 14.0) ** 2))
         + 7.0 * np.exp(-(((np.abs(lat - shift) - 30.0) / 11.0) ** 2))
         - 8.0 * np.exp(-(((np.abs(lat - shift) - 60.0) / 12.0) ** 2))
         + 9.0 * np.exp(-(((np.abs(lat - shift) - 90.0) / 16.0) ** 2)))
    planet.pressure = p.astype(np.float32)

    # --- wind: trades / westerlies / polars + speed from pressure gradient ---
    alat = np.abs(lat - shift)
    # Direction wind blows TO (radians, 0 = east, pi/2 = north).
    # Trades blow westward, westerlies eastward, polars westward.
    to_east = np.where((alat > 30) & (alat < 60), 1.0, -1.0)
    # Coriolis deflection: NE in N hemisphere westerlies, SE in S, etc.
    hemi = np.where(lat >= 0, 1.0, -1.0)
    wind_dir = np.arctan2(0.45 * hemi * to_east, to_east)
    dp_dlat = np.abs(np.gradient(p, axis=0))
    speed = 2.5 + 14.0 * np.clip(dp_dlat / 0.6, 0.0, 1.0)
    planet.wind_dir = wind_dir.astype(np.float32)
    planet.wind_speed = speed.astype(np.float32)

    # --- humidity + cloud ---
    rh = 25.0 + 75.0 * planet.precip_annual / (planet.precip_annual + 450.0)
    rh -= np.clip(t_mean - 28.0, 0.0, 20.0) * 0.6
    planet.humidity = np.clip(rh, 5.0, 100.0).astype(np.float32)
    planet.cloud = np.clip(planet.humidity / 100.0 * 0.45 + planet.precip_annual / 6000.0,
                           0.0, 1.0).astype(np.float32)

    # --- glaciers: high + cold, or permanent polar ice ---
    planet.glacier = land & (((planet.elevation > 2200.0) & (t_mean < 0.0))
                             | ((np.abs(lat) > 78.0) & (t_mean < -8.0)))

    update_seasonal(planet, cfg, doy)
    return {
        "mean_temp_c": round(float(t_mean[land].mean()), 2) if land.any() else 0,
        "mean_precip_mm": round(float(planet.precip_annual[land].mean()), 1) if land.any() else 0,
        "glacier_cells": int(planet.glacier.sum()),
        "declination_deg": round(float(decl), 2),
    }


def update_seasonal(planet, cfg: PlanetConfig, day_of_year: float | None = None) -> None:
    """Refresh current temperature / precipitation for the season.

    Both hemispheres are handled explicitly: northern summer is southern
    winter, the tropics barely swing while mid-latitudes get the full range,
    and land far from the sea swings harder than maritime coasts.
    """
    doy = planet.day_of_year if day_of_year is None else day_of_year
    decl = declination_deg(doy, cfg.axial_tilt_deg)
    # +1 at the June solstice, -1 at the December solstice.
    s = np.sin(2.0 * np.pi * (doy - 80.0) / 365.25)

    # Polar-equator insolation contrast grows with tilt, so the seasonal swing
    # is strongest in high latitudes and nearly flat at the equator.
    coslat = planet.temp_cos_lat
    lat_factor = 1.0 - coslat ** 2 * 0.85
    hemi = np.where(planet.lat_grid >= 0, 1.0, -1.0)

    planet.temp_current = (planet.temp_mean
                           + planet.temp_amplitude * s * lat_factor * hemi).astype(np.float32)

    # Precipitation follows the weather belts, which migrate with the sun:
    # the ITCZ and the westerlies shift toward the summer hemisphere.
    shift = decl * 0.6
    itcz = np.exp(-(((planet.lat_grid - shift) / 13.0) ** 2))
    west = np.exp(-(((np.abs(planet.lat_grid - shift) - 55.0) / 16.0) ** 2))
    season_norm = cfg.seasonal_precip_amplitude
    planet._precip_seasonal = (1.0 + season_norm * (itcz * s * 1.4
                                                    + west * s * hemi * 0.55)).astype(np.float32)
    # Current seasonal temperature/precip factors are exposed for the UI.
    planet._declination_deg = float(decl)
    planet._sun_season = float(s)

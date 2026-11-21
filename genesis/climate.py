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

    # Seasonal amplitude: small over oceans, large inland.
    amp_land = 4.0 + 26.0 * (1.0 - np.exp(-planet.dist_ocean_km / 450.0))
    amp = np.where(land, amp_land, 3.0)
    planet.temp_amplitude = amp.astype(np.float32)

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
    """Refresh current temperature / precipitation for the season."""
    doy = planet.day_of_year if day_of_year is None else day_of_year
    phase = np.cos(2.0 * np.pi * (doy - 200.0) / 365.25)   # peak ~mid-July (N)
    hemi = np.where(planet.lat_grid >= 0, 1.0, -1.0)
    planet.temp_current = (planet.temp_mean + planet.temp_amplitude * phase * hemi).astype(np.float32)
    # Precipitation peaks with the local warm season near the ITCZ.
    seasonal = 1.0 + 0.35 * phase * hemi * np.exp(-((planet.lat_grid / 25.0) ** 2))
    planet._precip_seasonal = seasonal.astype(np.float32)

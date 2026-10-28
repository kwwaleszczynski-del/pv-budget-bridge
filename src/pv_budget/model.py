"""The plant model shared by the budget and the expected-energy chain (pvlib building blocks).

POA irradiance -> cell temperature (PVsyst-type heat balance, or back-of-module sensor + conduction offset)
-> DC power (PVWatts: linear in irradiance, linear temperature coefficient, lumped DC losses)
-> AC power per inverter (PVWatts inverter efficiency curve) -> AC losses.

The inverter curve is written out here so that the same call returns AC power with and without the AC
limit: the difference is the clipping loss. Measured POA has no beam/diffuse split, so incidence-angle and
soiling effects are not modelled per interval; they sit in ``dc_loss`` and in the budget's soiling assumption.

Usage:
    from pv_budget import model
    sp = model.solar_position(index, dt_h, plant)
    tc = model.cell_temperature(plant, poa, t_amb, wind)
    pac_unc, pac = model.ac_power(plant, poa, tc)       # kW per inverter, without and with the AC limit
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from pv_budget.config import Plant

ETA_REF = 0.9637   # reference efficiency of the PVWatts inverter curve


def midpoints(index: pd.DatetimeIndex, dt_h: float) -> pd.DatetimeIndex:
    return index + pd.Timedelta(hours=dt_h / 2)


def solar_position(index: pd.DatetimeIndex, dt_h: float, plant: Plant) -> pd.DataFrame:
    sp = pvlib.solarposition.get_solarposition(midpoints(index, dt_h), plant.latitude, plant.longitude,
                                               altitude=plant.altitude)
    sp.index = index
    return sp[["apparent_zenith", "zenith", "apparent_elevation", "azimuth"]]


def poa_from_components(plant: Plant, sp: pd.DataFrame, ghi, dni, dhi, tilt=None, azimuth=None,
                        model: str = "perez") -> pd.Series:
    tilt = plant.tilt if tilt is None else tilt
    azimuth = plant.azimuth if azimuth is None else azimuth
    dni_extra = pvlib.irradiance.get_extra_radiation(sp.index)
    am = pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"])
    out = pvlib.irradiance.get_total_irradiance(tilt, azimuth, sp["apparent_zenith"], sp["azimuth"],
                                                dni, ghi, dhi, dni_extra=dni_extra, airmass=am,
                                                albedo=plant.albedo, model=model)
    return out["poa_global"].fillna(0.0).clip(lower=0)


def clearsky_poa(plant: Plant, sp: pd.DataFrame, tilt=None, azimuth=None) -> pd.Series:
    loc = pvlib.location.Location(plant.latitude, plant.longitude, altitude=plant.altitude)
    times = sp.index
    cs = loc.get_clearsky(times, model="ineichen", solar_position=sp.assign(
        apparent_zenith=sp["apparent_zenith"], zenith=sp["zenith"]))
    return poa_from_components(plant, sp, cs["ghi"], cs["dni"], cs["dhi"], tilt, azimuth)


def cell_temperature(plant: Plant, poa, t_amb=None, wind=None, t_mod=None) -> np.ndarray:
    poa = np.asarray(poa, float)
    if plant.temperature_source == "module" and t_mod is not None:
        return np.asarray(t_mod, float) + poa / 1000.0 * plant.module_delta_t
    wind = 1.0 if wind is None else np.nan_to_num(np.asarray(wind, float), nan=1.0)
    return np.asarray(pvlib.temperature.pvsyst_cell(poa, np.asarray(t_amb, float), wind_speed=wind,
                                                    u_c=plant.u_c, u_v=plant.u_v), float)


def temperature_factor(plant: Plant, t_cell) -> np.ndarray:
    return 1.0 + plant.gamma_pdc * (np.asarray(t_cell, float) - 25.0)


def inverter(pdc, pac0: float, eta_nom: float) -> tuple[np.ndarray, np.ndarray]:
    """PVWatts inverter: AC power without and with the AC limit ``pac0`` (kW)."""
    pdc = np.asarray(pdc, float)
    pdc0 = pac0 / eta_nom
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = pdc / pdc0
        eta = eta_nom / ETA_REF * (-0.0162 * zeta - 0.0059 / zeta + 0.9858)
        pac = np.where(pdc > 0, eta * pdc, 0.0)
    pac = np.where(np.isfinite(pac), np.maximum(pac, 0.0), np.nan)
    return pac, np.minimum(pac, pac0)


def degradation(plant: Plant, times: pd.DatetimeIndex) -> np.ndarray:
    if plant.cod is None or plant.degradation_per_year == 0:
        return np.ones(len(times))
    cod = pd.Timestamp(plant.cod, tz="UTC") if pd.Timestamp(plant.cod).tz is None else pd.Timestamp(plant.cod)
    years = np.maximum((times - cod) / pd.Timedelta(days=365.25), 0.0)
    return (1.0 - plant.degradation_per_year) ** np.asarray(years, float)


def ac_power(plant: Plant, poa, t_cell, derate=1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AC power per inverter after AC losses (kW): without the inverter limit, and with it.

    ``derate`` multiplies the irradiance reaching the cells (degradation, soiling or snow, one value per
    interval or a scalar), so that it acts before the inverter limit as it does in the field.
    """
    poa = np.asarray(poa, float)
    f = temperature_factor(plant, t_cell)
    g = poa / 1000.0 * np.asarray(derate, float)
    unc, lim = {}, {}
    for inv in plant.inverters:
        pdc = inv.dc_kwp * g * f * (1.0 - plant.dc_loss)
        a, b = inverter(pdc, inv.ac_kw, plant.inverter_eta)
        unc[inv.id] = a * (1.0 - plant.ac_loss)
        lim[inv.id] = b * (1.0 - plant.ac_loss)
    return pd.DataFrame(unc), pd.DataFrame(lim)


def ac_limits(plant: Plant) -> pd.Series:
    """Inverter AC limits as seen at the meter side of the AC losses, kW."""
    return pd.Series({i.id: i.ac_kw * (1.0 - plant.ac_loss) for i in plant.inverters})

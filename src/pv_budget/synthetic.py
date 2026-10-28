"""A synthetic 1.32 MWp plant in central Poland with a known answer for every bridge component.

Weather: ERA5 at 51.9 N 19.5 E (Open-Meteo archive, hourly 2021-2024), spread to 15 minutes with a constant
clear-sky index and diffuse fraction within each hour (hourly means are kept). The "true" plant differs from
the configured model on purpose (temperature coefficient, heat loss factors, DC losses, incidence-angle loss,
degradation rate), so the bridge has a real unexplained residual.

True losses, each logged or not as an O&M team would:

- soiling builds up on dry days and is washed off by rain (>= 4 mm/day); not logged;
- snow cover from ERA5 snowfall, sliding off above 0 degC and with sun; not logged;
- inverter faults (random, some multi-day), logged as ``outage`` when longer than 4 h;
- one planned maintenance day per inverter and year, a grid outage per year, one force-majeure event;
- grid-operator set-points (redispatch on sunny spring and summer middays), logged and in the set-point signal;
- communication gaps (power and met data missing).

Output, in ``out_dir``: ``long.parquet`` (the tool's long format, naive Europe/Warsaw time, so the autumn DST
hour appears twice), ``events.csv`` (local time), ``truth.parquet`` (per-interval true losses, UTC).

Usage:
    python -m pv_budget synthetic --era5 PATH --ref data/pl_demo/sarah3.parquet --out data/pl_demo
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from pv_budget import model
from pv_budget.config import Config

log = logging.getLogger(__name__)

TZ = "Europe/Warsaw"
TRUE = {"gamma_pdc": -0.0038, "u_c": 26.0, "u_v": 1.2, "dc_loss": 0.075, "degradation_per_year": 0.006,
        "iam_b0": 0.05, "calibration": 1.01}


def weather_15min(era5: pd.DataFrame, plant) -> pd.DataFrame:
    """Hourly ERA5 (values for the hour ending at the stamp) -> 15-minute interval starts, UTC."""
    era5 = era5.copy()
    era5.index = era5.index - pd.Timedelta("1h")           # stamp = start of the averaged hour
    idx = pd.date_range(era5.index[0], era5.index[-1] + pd.Timedelta("45min"), freq="15min")
    sp = model.solar_position(idx, 0.25, plant)
    loc = pvlib.location.Location(plant.latitude, plant.longitude, altitude=plant.altitude)
    cs = loc.get_clearsky(idx + pd.Timedelta("7.5min"), model="ineichen")
    cs.index = idx
    hour = idx.floor("1h")
    cs_h = cs["ghi"].groupby(hour).transform("mean").to_numpy()
    ghi_h = era5["shortwave_radiation"].reindex(hour).to_numpy()
    dhi_h = era5["diffuse_radiation"].reindex(hour).to_numpy()
    kt = np.where(cs_h > 1, ghi_h / np.maximum(cs_h, 1), 0.0)
    ghi = np.where(cs["ghi"].to_numpy() > 0, kt * cs["ghi"].to_numpy(), 0.0)
    # hours where the sun is up for only part of the hour: put the hourly energy where the sun is
    fd = np.clip(np.where(ghi_h > 0, dhi_h / np.maximum(ghi_h, 1e-6), 1.0), 0, 1)
    dhi = fd * ghi
    cosz = np.cos(np.radians(sp["zenith"].to_numpy()))
    dni = np.where(cosz > 0.02, (ghi - dhi) / np.maximum(cosz, 0.02), 0.0)
    dni = np.clip(dni, 0, 1100)
    hourly = era5.reindex(hour)
    temp = era5["temperature_2m"].copy()
    temp.index = temp.index + pd.Timedelta("1h")              # instantaneous at the original stamp
    t15 = temp.reindex(temp.index.union(idx)).interpolate("time").reindex(idx).bfill().ffill()
    wind = era5["wind_speed_10m"].copy()
    wind.index = wind.index + pd.Timedelta("1h")
    w15 = wind.reindex(wind.index.union(idx)).interpolate("time").reindex(idx).bfill().ffill()
    return pd.DataFrame({"ghi": ghi, "dhi": dhi, "dni": dni, "t_amb": t15.to_numpy(), "wind": w15.to_numpy(),
                         "precip": hourly["precipitation"].to_numpy() / 4,
                         "snowfall": hourly["snowfall"].to_numpy() / 4}, index=idx)


def sensor_poa(plant, sp, w: pd.DataFrame, tilt=None, azimuth=None, shading=None) -> np.ndarray:
    """POA seen by a pyranometer, optionally misoriented or with its beam blocked at low morning sun."""
    tilt = plant.tilt if tilt is None else tilt
    azimuth = plant.azimuth if azimuth is None else azimuth
    dni_extra = pvlib.irradiance.get_extra_radiation(sp.index)
    am = pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"])
    comp = pvlib.irradiance.get_total_irradiance(tilt, azimuth, sp["apparent_zenith"], sp["azimuth"],
                                                 w["dni"], w["ghi"], w["dhi"], dni_extra=dni_extra, airmass=am,
                                                 albedo=plant.albedo, model="perez")
    beam = comp["poa_direct"].fillna(0).to_numpy()
    diffuse = (comp["poa_diffuse"]).fillna(0).to_numpy()
    if shading is not None:
        az0, az1, el = shading
        blocked = (sp["azimuth"].to_numpy() >= az0) & (sp["azimuth"].to_numpy() <= az1) & \
                  (sp["apparent_elevation"].to_numpy() <= el)
        beam = np.where(blocked, 0.0, beam)
    return np.clip(beam + diffuse, 0, None)


def _soiling(precip_daily: pd.Series, rate: float = 0.0012, rain_mm: float = 4.0, cap: float = 0.08) -> pd.Series:
    s, out = 0.0, []
    for p in precip_daily.to_numpy():
        s = 0.0 if p >= rain_mm else min(cap, s + rate)
        out.append(s)
    return pd.Series(out, index=precip_daily.index)


def _snow(w: pd.DataFrame, poa: np.ndarray) -> np.ndarray:
    c, out = 0.0, np.zeros(len(w))
    snowfall, temp = w["snowfall"].to_numpy(), w["t_amb"].to_numpy()
    for k in range(len(w)):
        if temp[k] < 1.0:
            c = min(1.0, c + snowfall[k] / 1.5)
        melt = 0.04 * max(temp[k], 0.0) + 0.05 * poa[k] / 1000.0 * (temp[k] > -3)
        c = max(0.0, c - melt)
        out[k] = c
    return out


def generate(cfg: Config, era5: pd.DataFrame, poa_ref: pd.Series | None, out_dir: Path, seed: int = 7,
             sensor: dict | None = None) -> dict:
    """Write long.parquet, events.csv and truth.parquet; returns paths and a summary."""
    rng = np.random.default_rng(seed)
    p = cfg.plant
    w = weather_15min(era5, p)
    idx = w.index
    sp = model.solar_position(idx, 0.25, p)
    local = idx.tz_convert(TZ)
    true = p.__class__(**{**p.__dict__, **{k: v for k, v in TRUE.items() if k in p.__dict__}})
    poa_true = sensor_poa(p, sp, w)
    aoi = pvlib.irradiance.aoi(p.tilt, p.azimuth, sp["apparent_zenith"], sp["azimuth"]).to_numpy()
    comp = pvlib.irradiance.get_total_irradiance(p.tilt, p.azimuth, sp["apparent_zenith"], sp["azimuth"],
                                                 w["dni"], w["ghi"], w["dhi"], model="perez",
                                                 dni_extra=pvlib.irradiance.get_extra_radiation(idx),
                                                 airmass=pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"]))
    iam = pvlib.iam.ashrae(aoi, b=TRUE["iam_b0"])
    g_eff = np.nan_to_num(comp["poa_direct"].to_numpy() * iam) + np.nan_to_num(comp["poa_diffuse"].to_numpy())
    tc = model.cell_temperature(true, poa_true, w["t_amb"].to_numpy(), w["wind"].to_numpy())
    deg = model.degradation(true, idx)
    precip_day = pd.Series(w["precip"].to_numpy(), index=local).groupby(local.date).sum()
    soil_day = _soiling(precip_day)
    soil = soil_day.reindex(local.date).to_numpy()
    snow = _snow(w, poa_true)
    x0, _ = model.ac_power(true, g_eff, tc, deg)
    x1, _ = model.ac_power(true, g_eff, tc, deg * (1 - soil))
    x2, _ = model.ac_power(true, g_eff, tc, deg * (1 - soil) * (1 - snow))
    pac0 = model.ac_limits(p)
    inv = [i.id for i in p.inverters]
    lim = pd.DataFrame({i: np.minimum(x2[i], pac0[i]) for i in inv})
    T = len(idx)
    down = pd.DataFrame(False, index=range(T), columns=inv)
    cap = pd.DataFrame(1.0, index=range(T), columns=inv)
    gap = np.zeros(T, bool)
    events = []
    day_mask = poa_true > 50

    def add(cat, k0, k1, devs, limit=np.nan, log_it=True):
        if log_it:
            events.append({"start": local[k0], "end": local[min(k1, T - 1)], "device": devs if isinstance(devs, str)
                           else "*", "category": cat, "limit": limit})

    years = sorted(set(local.year))

    def at(ts):
        return int(idx.searchsorted(ts.tz_convert("UTC").floor("15min")))
    # inverter faults
    for y in years:
        for i in inv:
            for _ in range(rng.poisson(1.5)):
                start = pd.Timestamp(f"{y}-01-01", tz=TZ) + pd.Timedelta(days=float(rng.uniform(0, 364)))
                dur_h = float(np.clip(rng.lognormal(np.log(8), 1.0), 1, 96))
                k0 = at(start)
                k1 = min(T - 1, k0 + int(dur_h * 4))
                down.loc[k0:k1, i] = True
                if dur_h > 4:
                    add("outage", k0, k1, i)
        # planned maintenance: one working day per inverter in April-May
        for n, i in enumerate(inv):
            d = pd.Timestamp(f"{y}-04-10 08:00", tz=TZ) + pd.Timedelta(days=7 * n + int(rng.integers(0, 20)))
            k0 = at(d)
            k1 = k0 + 8 * 4
            down.loc[k0:k1, i] = True
            add("planned_maintenance", k0, k1, i)
        # grid outage (distribution line), a few hours
        d = pd.Timestamp(f"{y}-01-01 06:00", tz=TZ) + pd.Timedelta(days=float(rng.uniform(60, 300)))
        k0 = at(d)
        k1 = k0 + int(rng.integers(8, 24))
        down.loc[k0:k1, :] = True
        add("grid_outage", k0, k1, "*")
        # grid-operator set-points on sunny middays, April-August
        cand = [d for d in pd.date_range(f"{y}-04-01", f"{y}-08-31", freq="D")
                if precip_day.get(d.date(), 1.0) == 0.0]
        for d in rng.choice(cand, size=min(10, len(cand)), replace=False):
            d = pd.Timestamp(d).tz_localize(TZ) + pd.Timedelta(hours=int(rng.integers(10, 12)))
            k0 = at(d)
            k1 = k0 + int(rng.integers(8, 20))
            lim_frac = float(rng.choice([0.0, 0.3, 0.5, 0.6]))
            cap.loc[k0:k1, :] = lim_frac
            add("grid_command", k0, k1, "*", lim_frac)
        # communication gaps
        for _ in range(3):
            d = pd.Timestamp(f"{y}-01-01", tz=TZ) + pd.Timedelta(days=float(rng.uniform(0, 364)))
            k0 = at(d)
            gap[k0:k0 + int(rng.integers(8, 48))] = True
    # one force-majeure event (lightning strike), 36 h
    d = pd.Timestamp(f"{years[1]}-07-14 15:00", tz=TZ)
    k0 = at(d)
    down.loc[k0:k0 + 36 * 4, :] = True
    add("force_majeure", k0, k0 + 36 * 4, "*")

    up = ~down.to_numpy()
    capped = np.minimum(lim.to_numpy(), cap.to_numpy() * pac0[inv].to_numpy())
    noise = 1.0 + rng.normal(0, 0.003, size=capped.shape)
    power = np.where(up, capped, 0.0) * noise
    power[gap, :] = np.nan
    # sensor
    s = dict(sensor or {})
    poa_sensor = sensor_poa(p, sp, w, s.get("tilt"), s.get("azimuth"), s.get("shading"))
    poa_meas = poa_sensor * TRUE["calibration"] * (1 + rng.normal(0, 0.005, T))
    t_mod = tc - poa_true / 1000 * 3.0 + rng.normal(0, 0.3, T)
    met = pd.DataFrame({"poa": poa_meas, "t_amb": w["t_amb"].to_numpy() + rng.normal(0, 0.2, T),
                        "t_mod": t_mod, "wind": w["wind"].to_numpy(), "precip": w["precip"].to_numpy()}, index=idx)
    met.loc[gap] = np.nan
    met.loc[gap, "precip"] = w["precip"].to_numpy()[gap]

    frames = [pd.DataFrame({"timestamp": local.tz_localize(None), "device_id": "WS01", "signal": c,
                            "value": met[c].to_numpy()}) for c in met.columns]
    frames += [pd.DataFrame({"timestamp": local.tz_localize(None), "device_id": i, "signal": "p_ac",
                             "value": power[:, n]}) for n, i in enumerate(inv)]
    frames.append(pd.DataFrame({"timestamp": local.tz_localize(None), "device_id": "PPC", "signal": "setpoint",
                                "value": cap[inv[0]].to_numpy()}))
    if poa_ref is not None:
        r = poa_ref.copy()
        r = r[(r.index >= idx[0]) & (r.index <= idx[-1])]
        frames.append(pd.DataFrame({"timestamp": r.index.tz_convert(TZ).tz_localize(None), "device_id": "SAT",
                                    "signal": "poa_ref", "value": r.to_numpy()}))
    long = pd.concat(frames, ignore_index=True).dropna(subset=["value"]).reset_index(drop=True)
    ev = pd.DataFrame(events)
    ev["start"] = pd.to_datetime(ev["start"]).dt.tz_localize(None)
    ev["end"] = pd.to_datetime(ev["end"]).dt.tz_localize(None)
    # truth per interval, plant totals (kW), in the bridge's sequential definition
    x0s, x1s, x2s = x0.sum(axis=1).to_numpy(), x1.sum(axis=1).to_numpy(), x2.sum(axis=1).to_numpy()
    lims = lim.to_numpy()
    truth = pd.DataFrame({
        "soiling": x0s - x1s, "snow": x1s - x2s, "clipping": x2s - lims.sum(axis=1),
        # as observable: downtime hidden in a data gap counts as gap, like in the estimator
        # (a fault during a zero set-point is invisible: it stays curtailment)
        "availability": np.where(~up & ~gap[:, None] & (cap.to_numpy() > 0.01), lims, 0.0).sum(axis=1),
        "curtailment": np.where(up | gap[:, None] | (cap.to_numpy() <= 0.01), lims - capped, 0.0).sum(axis=1),
        "data_gaps": np.where(gap[:, None], capped, 0.0).sum(axis=1),
        "potential": x0s, "actual": np.nan_to_num(power).sum(axis=1),
        "poa_true": poa_true, "soiling_ratio": 1 - soil, "snow_cover": snow}, index=idx)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out_dir / "long.parquet")
    ev.to_csv(out_dir / "events.csv", index=False, date_format="%Y-%m-%d %H:%M")
    truth.to_parquet(out_dir / "truth.parquet")
    log.info("synthetic plant: %d intervals, %d events, actual %.0f MWh", T, len(ev),
             truth["actual"].sum() * 0.25 / 1000)
    return {"long": out_dir / "long.parquet", "events": out_dir / "events.csv", "truth": out_dir / "truth.parquet"}

"""NREL PVDAQ (public, OEDI) rooftop systems of one school campus in Las Vegas, as long-format input files.

Source: Deline, C., Perry, K., Deceglie, M., Muller, M., Sekulic, W., & Jordan, D. (2021). Photovoltaic Data
Acquisition (PVDAQ) Public Datasets. Open Energy Data Initiative (OEDI), NREL. https://doi.org/10.25984/1846021,
CC BY 4.0. One CSV per system and day; 15-minute values in local standard time (UTC-8, no daylight saving).

Systems used (system metadata JSON in the same bucket): 34 (building A, 146.6 kWp, one 135 kW inverter, tilt
11.2 deg), 1276 (building B, 68.5 kWp, one 50 kW inverter, tilt 5 deg), 1278 (building D, 171.4 kWp, a 100 kW
and a 50 kW inverter, tilt not given). All three log the same plane-of-array sensor of the site's met station.

Quirks handled here: power channels are in hectowatts until 4 August 2018 and in watts afterwards (the switch
is found from the power-to-irradiance ratio); temperatures in degF and wind in mph; timestamps jitter by a
second around the quarter hours.

Usage:
    python -m pv_budget pvdaq --raw data/pvdaq/raw --out data/pvdaq [--copy-from DIR]
"""
from __future__ import annotations

import logging
import shutil
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BASE_URL = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/pvdata"
YEARS = (2016, 2017, 2018)
TZ = "Etc/GMT+8"
SYSTEMS = {
    34: {"INV1": "ac_power_hw"},
    1276: {"INV1": "ac_power_hw"},
    1278: {"INV1": "inv1_ac_power_hw", "INV2": "inv2_ac_power_hw"},
}
MET = ["poa_irradiance", "ambient_temp_f", "module_temp_f", "wind_speed_mph"]   # mapped in configs/pvdaq_*.yaml


def day_url(sid: int, day: pd.Timestamp) -> str:
    return (f"{BASE_URL}/system_id={sid}/year={day.year}/month={day.month}/day={day.day}/"
            f"system_{sid}__date_{day:%Y_%m_%d}.csv")


def fetch(raw_dir: Path, systems=tuple(SYSTEMS), years=YEARS, copy_from: Path | None = None,
          workers: int = 4) -> int:
    """Daily files into raw_dir/system_<id>/; copied from a local mirror when given, else downloaded."""
    jobs = []
    for sid in systems:
        out = Path(raw_dir) / f"system_{sid}"
        out.mkdir(parents=True, exist_ok=True)
        for day in pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D"):
            name = f"system_{sid}__date_{day:%Y_%m_%d}.csv"
            if not (out / name).exists():
                jobs.append((sid, day, out / name))

    def get(job):
        sid, day, path = job
        if copy_from is not None:
            src = Path(copy_from) / f"system_{sid}" / path.name
            if src.exists():
                shutil.copyfile(src, path)
                return 1
            return 0
        try:
            with urllib.request.urlopen(day_url(sid, day), timeout=60) as r:
                path.write_bytes(r.read())
            return 1
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return 0
            raise

    with ThreadPoolExecutor(workers) as pool:
        n = sum(pool.map(get, jobs))
    log.info("PVDAQ: %d of %d missing daily files %s", n, len(jobs), "copied" if copy_from else "downloaded")
    return n


def read_system(raw_dir: Path, sid: int, years=YEARS) -> pd.DataFrame:
    files = sorted((Path(raw_dir) / f"system_{sid}").glob(f"system_{sid}__date_*.csv"))
    files = [p for p in files if int(p.stem.split("_date_")[1][:4]) in years and p.stat().st_size > 0]
    frames = [f for f in (pd.read_csv(p) for p in files) if "measured_on" in f.columns]
    raw = pd.concat(frames, ignore_index=True)
    raw.index = pd.to_datetime(raw.pop("measured_on")).dt.round("15min")
    raw = raw[~raw.index.duplicated()].sort_index()
    raw.columns = [c.rsplit("__", 1)[0] for c in raw.columns]
    return raw


def kw_factor(raw: pd.Series, poa: pd.Series, jump: float = 20.0) -> np.ndarray:
    """kW per raw unit: 0.1 while the channel reports hectowatts, 0.001 after it switches to watts."""
    ratio = (raw.where(raw > 0) / poa.where(poa > 200)).to_numpy()
    day = np.flatnonzero(np.isfinite(ratio))
    factor = np.full(len(raw), 0.1)
    if not len(day):
        return factor
    ref = np.nanmedian(ratio[day[: max(100, len(day) // 4)]])
    watt = ratio[day] > jump * ref
    if watt.any():
        tail = np.cumsum(watt[::-1])[::-1] / np.arange(len(watt), 0, -1)
        k = np.flatnonzero(watt & (tail >= 0.95))
        if len(k):
            factor[day[k[0]]:] = 0.001
            log.info("power channel switches from hW to W at %s", raw.index[day[k[0]]])
    return factor


def to_long(raw_dir: Path, sid: int, years=YEARS) -> pd.DataFrame:
    """One system in the tool's long format: naive local standard time (data.timezone = Etc/GMT+8), met signals
    under their PVDAQ names and units (renamed and converted by the YAML mapping), AC power in kW."""
    raw = read_system(raw_dir, sid, years)
    grid = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31 23:45", freq="15min")
    raw = raw.reindex(grid)
    frames = []
    for col in MET:
        if col in raw.columns:
            v = pd.to_numeric(raw[col], errors="coerce")
            frames.append(pd.DataFrame({"timestamp": grid, "device_id": "MET", "signal": col,
                                        "value": v.to_numpy()}))
    poa = pd.to_numeric(raw["poa_irradiance"], errors="coerce")
    for inv, col in SYSTEMS[sid].items():
        v = pd.to_numeric(raw[col], errors="coerce")
        kw = (v * kw_factor(v, poa)).clip(lower=0)
        frames.append(pd.DataFrame({"timestamp": grid, "device_id": inv, "signal": "p_ac",
                                    "value": kw.to_numpy()}))
    out = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    return out.reset_index(drop=True)


def fetch_era5_weather(cache: Path, lat: float = 36.1952, lon: float = -115.1582, years=YEARS) -> pd.DataFrame:
    """Hourly ERA5 precipitation (mm, hour ending at the stamp) and 2 m temperature from the Open-Meteo
    archive (one request, cached). Rain days anchor the soiling estimate; the site has no rain gauge."""
    import json
    cache = Path(cache)
    if cache.exists():
        return pd.read_parquet(cache)
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           f"&start_date={min(years)}-01-01&end_date={max(years)}-12-31"
           f"&hourly=precipitation,temperature_2m&timezone=GMT&models=era5")
    with urllib.request.urlopen(url, timeout=120) as r:
        js = json.loads(r.read())
    h = js["hourly"]
    out = pd.DataFrame({"precip": h["precipitation"], "t2m": h["temperature_2m"]},
                       index=pd.DatetimeIndex(pd.to_datetime(h["time"]), name="time").tz_localize("UTC"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def add_precip(long: pd.DataFrame, weather: pd.DataFrame, label: str = "end") -> pd.DataFrame:
    """Append hourly precipitation (signal precip, stamps shifted to the data's label) and 2 m temperature
    (signal t_amb_ref)."""
    shift = pd.Timedelta(hours={"start": -1.0, "end": 0.0, "center": -0.5}[label])
    t = (weather.index + shift).tz_convert(TZ).tz_localize(None)
    pr = pd.DataFrame({"timestamp": t, "device_id": "MET", "signal": "precip", "value": weather["precip"].to_numpy()})
    # 2 m temperature is instantaneous at the stamp; it only fills gaps of the site's ambient sensor
    ta = pd.DataFrame({"timestamp": weather.index.tz_convert(TZ).tz_localize(None), "device_id": "MET",
                       "signal": "t_amb_ref", "value": weather["t2m"].to_numpy()})
    return pd.concat([long, pr, ta], ignore_index=True)


def add_reference(long: pd.DataFrame, poa_ref: pd.Series, label: str = "start") -> pd.DataFrame:
    """Append an hourly reference POA (index = UTC hour starts) as signal poa_ref, labelled like the data."""
    shift = pd.Timedelta(hours={"start": 0.0, "end": 1.0, "center": 0.5}[label])
    t = (poa_ref.index + shift).tz_convert(TZ).tz_localize(None)
    ref = pd.DataFrame({"timestamp": t, "device_id": "MET", "signal": "poa_ref", "value": poa_ref.to_numpy()})
    return pd.concat([long, ref], ignore_index=True)

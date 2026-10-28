"""Generic plant input: a long table (timestamp, device_id, signal, value) or a wide table with a column mapping,
put on one regular UTC grid whose timestamps mark the start of each interval.

Standard signals (other names are mapped with ``data.signals`` / ``data.columns`` in the YAML):

- plant level: ``poa`` (W/m2), ``poa_ref`` (satellite or model POA, W/m2), ``ghi`` (W/m2), ``t_amb`` and
  ``t_mod`` (degC), ``t_amb_ref`` (e.g. reanalysis, fills gaps of ``t_amb``), ``wind`` (m/s), ``precip`` (mm per interval), ``snow_depth`` (cm), ``setpoint`` (fraction of
  the AC rating, or kW with unit kW), ``e_export`` (meter energy, kWh per interval);
- inverter level (device = inverter id): ``p_ac`` (kW), ``setpoint``.

Timestamps:

- strings with a UTC offset (``2024-10-27T02:30:00+02:00``) are exact and ``data.timezone`` is ignored;
- naive timestamps are read in ``data.timezone``. In a zone with daylight saving time the autumn hour occurs
  twice; within each device and signal the first occurrence is taken as summer time and the second as winter
  time (file order). A repeated hour seen only once, and the skipped spring hour, cannot be placed and are
  dropped with a warning (both are at night);
- ``data.label`` says whether a timestamp marks the start, end or centre of its averaging interval.

Signals with different intervals (5, 10, 15 min, 1 h) are averaged up or spread down to ``data.interval``
(default: the coarsest interval found). Energy-type signals (``e_export``, ``precip``) are summed, not averaged.

Usage:
    from pv_budget import io
    cfg = load_config("configs/pl_demo.yaml")
    data = io.load(cfg, "data/pl_demo/long.parquet", events="data/pl_demo/events.csv")
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pv_budget.config import Config

log = logging.getLogger(__name__)

MET_SIGNALS = ["poa", "poa_ref", "ghi", "t_amb", "t_amb_ref", "t_mod", "wind", "precip", "snow_depth"]
SUM_SIGNALS = {"e_export", "precip"}
EVENT_CATEGORIES = {"outage", "planned_maintenance", "force_majeure", "grid_command", "grid_outage",
                    "cleaning", "other"}

UNITS = {  # unit -> (scale, offset) to the standard unit
    "w": (0.001, 0.0), "kw": (1.0, 0.0), "mw": (1000.0, 0.0), "hw": (0.1, 0.0),
    "wh": (0.001, 0.0), "kwh": (1.0, 0.0), "mwh": (1000.0, 0.0),
    "w/m2": (1.0, 0.0), "kw/m2": (1000.0, 0.0),
    "degc": (1.0, 0.0), "c": (1.0, 0.0), "degf": (5 / 9, -32 * 5 / 9), "f": (5 / 9, -32 * 5 / 9),
    "k": (1.0, -273.15),
    "m/s": (1.0, 0.0), "mph": (0.44704, 0.0), "km/h": (1 / 3.6, 0.0),
    "%": (0.01, 0.0), "fraction": (1.0, 0.0), "mm": (1.0, 0.0), "cm": (1.0, 0.0), "m": (100.0, 0.0),
}


@dataclass
class PlantData:
    index: pd.DatetimeIndex            # UTC, start of each interval
    dt_h: float                        # interval length in hours
    met: pd.DataFrame                  # plant-level signals
    power: pd.DataFrame                # AC power per inverter, kW
    setpoint: pd.DataFrame | None = None   # allowed share of each inverter's AC rating (1 = no limit)
    meter: pd.Series | None = None     # export energy per interval, kWh
    events: pd.DataFrame = field(default_factory=lambda: empty_events())

    def copy(self) -> "PlantData":
        return PlantData(self.index, self.dt_h, self.met.copy(), self.power.copy(),
                         None if self.setpoint is None else self.setpoint.copy(),
                         None if self.meter is None else self.meter.copy(), self.events.copy())


def empty_events() -> pd.DataFrame:
    return pd.DataFrame({"start": pd.Series(dtype="datetime64[ns, UTC]"),
                         "end": pd.Series(dtype="datetime64[ns, UTC]"),
                         "device": pd.Series(dtype=object), "category": pd.Series(dtype=object),
                         "limit": pd.Series(dtype=float)})


STUCK_CHECK = {"t_amb", "t_mod", "wind"}


def drop_stuck(s: pd.Series, n: int, name: str = "") -> pd.Series:
    """Readings that repeat the exact same value for ``n`` or more intervals are a stuck sensor: set to NaN."""
    v = s.to_numpy()
    change = np.r_[True, v[1:] != v[:-1]]
    run_id = np.cumsum(change)
    run_len = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
    stuck = (run_len >= n) & np.isfinite(v)
    if stuck.any():
        log.warning("%s: %d readings in stuck runs (>= %d equal values) set to missing", name, int(stuck.sum()), n)
    return s.mask(stuck)


def convert(values: pd.Series, unit: str | None) -> pd.Series:
    if unit is None:
        return values
    key = unit.strip().lower().replace("°", "deg").replace("²", "2").replace(" ", "")
    if key not in UNITS:
        raise ValueError(f"unknown unit {unit!r}")
    scale, offset = UNITS[key]
    return values * scale + offset


_OFFSET = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


def parse_times(ts: pd.Series, tz: str, groups: pd.Series | None = None) -> pd.Series:
    """Timestamps to UTC. Naive ones are local time in ``tz``; repeated autumn hours are resolved by order."""
    if pd.api.types.is_datetime64_any_dtype(ts) and getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert("UTC")
    if not pd.api.types.is_datetime64_any_dtype(ts):
        s = ts.astype(str).str.strip()
        if s.str.contains(_OFFSET).all():
            return pd.to_datetime(s, utc=True, format="ISO8601")
        ts = pd.to_datetime(s, format="mixed")
    if str(tz).upper() == "UTC":
        return ts.dt.tz_localize("UTC")
    local = ts.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    bad = local.isna() & ts.notna()
    if bad.any():
        sub = ts[bad]
        keys = (groups[bad] if groups is not None else pd.Series(0, index=sub.index)).astype(str)
        g = pd.DataFrame({"k": keys, "t": sub})
        rank = g.groupby(["k", "t"]).cumcount().to_numpy()
        size = g.groupby(["k", "t"])["t"].transform("size").to_numpy()
        try_dst = sub.dt.tz_localize(tz, ambiguous=np.ones(len(sub), bool), nonexistent="NaT")
        try_std = sub.dt.tz_localize(tz, ambiguous=np.zeros(len(sub), bool), nonexistent="NaT")
        ambiguous = try_dst.notna().to_numpy()
        resolved = pd.Series(pd.NaT, index=sub.index, dtype=try_dst.dtype)
        first = ambiguous & (size >= 2) & (rank == 0)
        second = ambiguous & (size >= 2) & (rank == 1)
        resolved[first] = try_dst[first]
        resolved[second] = try_std[second]
        local = local.astype(try_dst.dtype)
        local[bad] = resolved
        n_single = int((ambiguous & (size < 2)).sum())
        n_skip = int((~ambiguous).sum())
        n_extra = int((ambiguous & (rank >= 2)).sum())
        if n_single or n_skip or n_extra:
            log.warning("timestamps dropped around DST changes: %d ambiguous seen once, %d in the skipped "
                        "spring hour, %d repeated more than twice", n_single, n_skip, n_extra)
    return local.dt.tz_convert("UTC")


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def wide_to_long(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    tcol = cfg.data.timestamp_column
    frames = []
    for col, spec in cfg.data.columns.items():
        if col not in df.columns:
            log.warning("mapped column %r not in the file", col)
            continue
        frames.append(pd.DataFrame({"timestamp": df[tcol], "device_id": spec.get("device", cfg.data.plant_device),
                                    "signal": spec["signal"],
                                    "value": convert(pd.to_numeric(df[col], errors="coerce"), spec.get("unit"))}))
    if not frames:
        raise ValueError("no mapped columns found in the wide file")
    return pd.concat(frames, ignore_index=True)


def map_long(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.rename(columns={cfg.data.timestamp_column: "timestamp"})
    missing = {"timestamp", "device_id", "signal", "value"} - set(df.columns)
    if missing:
        raise ValueError(f"long format needs columns timestamp, device_id, signal, value; missing {missing}")
    df = df[["timestamp", "device_id", "signal", "value"]].copy()
    df["device_id"] = df["device_id"].astype(str).map(lambda d: cfg.data.devices.get(d, d))
    df["signal"] = df["signal"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for name, spec in cfg.data.signals.items():
        m = df["signal"] == name
        if m.any():
            df.loc[m, "value"] = convert(df.loc[m, "value"], spec.get("unit"))
            df.loc[m, "signal"] = spec.get("signal", name)
    return df


def _interval(t: pd.Series) -> pd.Timedelta:
    d = t.sort_values().diff().dropna()
    d = d[d > pd.Timedelta(0)]
    return d.median().round("1min") if len(d) else pd.Timedelta("1h")


def to_grid(long: pd.DataFrame, interval: str | None, label: str) -> tuple[pd.DataFrame, pd.Timedelta]:
    """Long rows with UTC timestamps -> wide frame on a regular grid, columns (device, signal)."""
    long = long.dropna(subset=["timestamp"])
    steps = {k: _interval(g) for k, g in long.groupby(["device_id", "signal"])["timestamp"]}
    target = pd.Timedelta(interval) if interval else max(steps.values())
    series = {}
    for (dev, sig), g in long.groupby(["device_id", "signal"]):
        step = steps[(dev, sig)]
        t = g["timestamp"] - {"start": 0, "end": 1, "center": 0.5}[label] * step
        s = pd.Series(g["value"].to_numpy(), index=pd.DatetimeIndex(t).round(step))
        s = s.groupby(level=0).sum() if sig in SUM_SIGNALS else s.groupby(level=0).mean()
        if step < target:
            r = s.resample(target, label="left", closed="left")
            s = r.sum(min_count=1) if sig in SUM_SIGNALS else r.mean()
        elif step > target:
            n = int(round(step / target))
            fine = pd.date_range(s.index.min(), s.index.max() + step - target, freq=target)
            s = s.reindex(fine).ffill(limit=n - 1)
            if sig in SUM_SIGNALS:
                s = s / n
        series[(dev, sig)] = s
    start = min(s.index.min() for s in series.values()).floor(target)
    end = max(s.index.max() for s in series.values())
    grid = pd.date_range(start, end, freq=target)
    wide = pd.DataFrame({k: s.reindex(grid) for k, s in series.items()})
    wide.columns = pd.MultiIndex.from_tuples(wide.columns, names=["device", "signal"])
    return wide, target


def read_events(path, tz: str = "UTC") -> pd.DataFrame:
    """Event log CSV: start, end, device (inverter id, or * / PLANT for the plant), category, optional limit."""
    if path is None:
        return empty_events()
    ev = pd.read_csv(path)
    for c in ("start", "end", "device", "category"):
        if c not in ev.columns:
            raise ValueError(f"event log needs column {c}")
    out = pd.DataFrame({"start": parse_times(ev["start"], tz), "end": parse_times(ev["end"], tz),
                        "device": ev["device"].astype(str).replace({"PLANT": "*"}),
                        "category": ev["category"].astype(str).str.strip().str.lower(),
                        "limit": pd.to_numeric(ev.get("limit", np.nan), errors="coerce")})
    unknown = set(out["category"]) - EVENT_CATEGORIES
    if unknown:
        log.warning("event categories not used by the tool: %s", sorted(unknown))
    return out.sort_values("start", ignore_index=True)


def load(cfg: Config, path=None, events=None, frame: pd.DataFrame | None = None) -> PlantData:
    raw = frame if frame is not None else read_table(path)
    long = wide_to_long(raw, cfg) if cfg.data.format == "wide" else map_long(raw, cfg)
    long["timestamp"] = parse_times(long["timestamp"], cfg.data.timezone,
                                    long["device_id"] + "|" + long["signal"])
    wide, step = to_grid(long, cfg.data.interval, cfg.data.label)
    idx = wide.index
    inv_ids = [i.id for i in cfg.plant.inverters]
    devices = set(wide.columns.get_level_values("device"))
    extra = devices - set(inv_ids) - {cfg.data.plant_device}
    met = pd.DataFrame(index=idx)
    for sig in MET_SIGNALS:
        cols = [c for c in wide.columns if c[1] == sig and c[0] not in inv_ids]
        if cols:
            if len(cols) > 1:
                log.warning("%d devices report %s; their mean is used", len(cols), sig)
            met[sig] = wide[cols].mean(axis=1)
            if sig in STUCK_CHECK:
                met[sig] = drop_stuck(met[sig], int(round(pd.Timedelta("6h") / step)), sig)
    if "t_amb_ref" in met:   # e.g. reanalysis 2 m temperature: fills gaps of the site's ambient sensor
        n = int(met["t_amb"].isna().sum()) if "t_amb" in met else len(met)
        met["t_amb"] = met["t_amb"].fillna(met["t_amb_ref"]) if "t_amb" in met else met["t_amb_ref"]
        log.info("t_amb: %d missing values filled from t_amb_ref", n)
    power = pd.DataFrame(index=idx)
    for i in inv_ids:
        if (i, "p_ac") in wide.columns:
            power[i] = wide[(i, "p_ac")]
        else:
            log.warning("no p_ac for inverter %s", i)
            power[i] = np.nan
    unused = [d for d in extra if (d, "p_ac") in wide.columns]
    if unused:
        log.warning("p_ac of devices not in the plant config ignored: %s", sorted(unused))
    setpoint = None
    sp_cols = [c for c in wide.columns if c[1] == "setpoint"]
    if sp_cols:
        ac = {i.id: i.ac_kw for i in cfg.plant.inverters}
        unit = next((s.get("unit") for s in list(cfg.data.signals.values()) + list(cfg.data.columns.values())
                     if s.get("signal") == "setpoint"), None)
        setpoint = pd.DataFrame(1.0, index=idx, columns=inv_ids)
        plant_sp = [c for c in sp_cols if c[0] not in inv_ids]
        if plant_sp:
            v = wide[plant_sp[0]]
            if unit and unit.lower() in ("kw", "mw", "w"):
                v = v / cfg.plant.ac_kw
            for i in inv_ids:
                setpoint[i] = v
        for i in inv_ids:
            if (i, "setpoint") in wide.columns:
                v = wide[(i, "setpoint")]
                setpoint[i] = v / ac[i] if unit and unit.lower() in ("kw", "mw", "w") else v
        setpoint = setpoint.ffill().fillna(1.0).clip(0, 1)
    meter = None
    m_cols = [c for c in wide.columns if c[1] == "e_export"]
    if m_cols:
        meter = wide[m_cols].sum(axis=1, min_count=1)
    ev = read_events(events, cfg.data.timezone) if events is not None else empty_events()
    log.info("loaded %d intervals of %s, %s to %s, %d inverters, met signals %s", len(idx), step,
             idx[0], idx[-1], len(inv_ids), list(met.columns))
    return PlantData(idx, step / pd.Timedelta("1h"), met, power, setpoint, meter, ev)


def to_long(data: PlantData, tz: str | None = None) -> pd.DataFrame:
    """Back to the long format; ``tz`` writes naive local timestamps (the DST hour then appears twice)."""
    t = data.index if tz is None else data.index.tz_convert(tz).tz_localize(None)
    frames = [pd.DataFrame({"timestamp": t, "device_id": "PLANT", "signal": c, "value": data.met[c].to_numpy()})
              for c in data.met.columns]
    frames += [pd.DataFrame({"timestamp": t, "device_id": c, "signal": "p_ac", "value": data.power[c].to_numpy()})
               for c in data.power.columns]
    if data.setpoint is not None:
        frames += [pd.DataFrame({"timestamp": t, "device_id": c, "signal": "setpoint",
                                 "value": data.setpoint[c].to_numpy()}) for c in data.setpoint.columns]
    if data.meter is not None:
        frames.append(pd.DataFrame({"timestamp": t, "device_id": "METER", "signal": "e_export",
                                    "value": data.meter.to_numpy()}))
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["value"]).reset_index(drop=True)

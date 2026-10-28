"""Losses and sensor faults of known size, injected into a plant's data to test the bridge and the QC.

Each function returns a modified copy of the data and the truth of what was injected. Energy truths are in
kWh of AC energy taken out of the measured series, so they do not depend on any model; only the soiling
injection needs one (for intervals where the inverter was clipping, the unclipped power is not measured: it
is estimated from the model scaled to the same day's unclipped intervals).

Usage:
    from pv_budget import inject
    data2, truth = inject.outage(data, "INV1", "2017-04-10", days=3)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv_budget import model
from pv_budget.config import Config
from pv_budget.io import PlantData


def _ts(day, tz) -> pd.Timestamp:
    return pd.Timestamp(day).tz_localize(tz).tz_convert("UTC")


def _event(data: PlantData, start, end, device, category, limit=np.nan) -> None:
    ev = pd.DataFrame({"start": [start], "end": [end], "device": [device], "category": [category],
                       "limit": [limit]})
    data.events = pd.concat([data.events, ev], ignore_index=True).sort_values("start", ignore_index=True)


def outage(data: PlantData, inverter: str, start: str, days: float, tz: str, log: bool = True):
    d = data.copy()
    t0 = _ts(start, tz)
    t1 = t0 + pd.Timedelta(days=days)
    m = (d.index >= t0) & (d.index < t1)
    removed = float(d.power.loc[m, inverter].fillna(0).sum() * d.dt_h)
    d.power.loc[m, inverter] = 0.0
    if log:
        _event(d, t0, t1, inverter, "outage")
    return d, {"component": "availability", "kwh": removed, "start": str(t0), "end": str(t1)}


def curtail(data: PlantData, cfg: Config, days: list[str], share: float, tz: str, hours=(9, 17)):
    """Cap every inverter at ``share`` of its AC rating on the given days, with set-point signal and event."""
    d = data.copy()
    limits = model.ac_limits(cfg.plant)
    if d.setpoint is None:
        d.setpoint = pd.DataFrame(1.0, index=d.index, columns=d.power.columns)
    removed = 0.0
    for day in days:
        t0 = _ts(pd.Timestamp(day) + pd.Timedelta(hours=hours[0]), tz)
        t1 = _ts(pd.Timestamp(day) + pd.Timedelta(hours=hours[1]), tz)
        m = (d.index >= t0) & (d.index < t1)
        for c in d.power.columns:
            cap = share * limits[c]
            before = d.power.loc[m, c]
            d.power.loc[m, c] = before.clip(upper=cap)
            removed += float((before - before.clip(upper=cap)).fillna(0).sum() * d.dt_h)
            d.setpoint.loc[m, c] = share
        _event(d, t0, t1, "*", "grid_command", share)
    return d, {"component": "curtailment", "kwh": removed, "days": list(days)}


def soiling(data: PlantData, cfg: Config, start: str, end: str, peak: float, tz: str, poa: pd.Series,
            t_cell: np.ndarray):
    """Soiling rising linearly from 0 to ``peak`` between start and end, then a logged cleaning.

    Applied before the inverter limit: an interval that was clipping loses only what falls below the limit.
    Truth: AC energy removed (net), and the DC-side loss (unclipped power x soiling) that the soiling bar
    itself should show; the difference reappears in the clipping bar.
    """
    d = data.copy()
    t0, t1 = _ts(start, tz), _ts(end, tz)
    m = (d.index >= t0) & (d.index < t1)
    frac = np.zeros(len(d.index))
    frac[m] = peak * (d.index[m] - t0) / (t1 - t0)
    unc, _ = model.ac_power(cfg.plant, poa.to_numpy(), t_cell, model.degradation(cfg.plant, d.index))
    limits = model.ac_limits(cfg.plant)
    day = d.index.tz_convert(tz).normalize()
    net = dc = 0.0
    for c in d.power.columns:
        P = d.power[c].to_numpy()
        lim = np.nanmax(P[m]) if np.isfinite(P[m]).any() else limits[c]
        lim = min(lim, limits[c])
        clipped = P >= 0.97 * lim
        u = unc[c].to_numpy()
        ok = ~clipped & (P > 0.05 * lim) & (u > 0)
        k = pd.Series(np.where(ok, P / np.where(u > 0, u, 1), np.nan), index=d.index).groupby(day).transform("median")
        k = k.fillna(np.nanmedian(k)).to_numpy()
        pot = np.where(clipped, u * k, P)              # unclipped power: measured, or model x day factor
        new = np.where(clipped, np.minimum(lim, pot * (1 - frac)), P * (1 - frac))
        new = np.where(np.isfinite(P), new, np.nan)
        net += float(np.nansum(P[m] - new[m]) * d.dt_h)
        dc += float(np.nansum(np.where(np.isfinite(P), pot, 0)[m] * frac[m]) * d.dt_h)
        d.power[c] = new
    _event(d, t1, t1 + pd.Timedelta(hours=1), "*", "cleaning")
    return d, {"component": "soiling", "kwh": net, "kwh_dc": dc, "start": str(t0), "end": str(t1),
               "mean_loss": float(frac[m].mean())}


def pyranometer_drift(data: PlantData, start: str, end: str, total: float, tz: str):
    """Sensor reading falls linearly from 0 to ``total`` (e.g. -0.03) between start and end, and stays there."""
    d = data.copy()
    t0, t1 = _ts(start, tz), _ts(end, tz)
    f = np.clip((d.index - t0) / (t1 - t0), 0, 1)
    factor = 1 + total * np.asarray(f, float)
    d.met["poa"] = d.met["poa"] * factor
    return d, {"component": "sensor_drift", "total": total, "start": str(t0), "end": str(t1),
               "factor": pd.Series(factor, index=d.index)}


def clock_shift(data: PlantData, start: str, end: str, minutes: int, tz: str):
    """The POA logger's clock runs ``minutes`` late between start and end (values stamped later)."""
    d = data.copy()
    t0, t1 = _ts(start, tz), _ts(end, tz)
    steps = int(round(minutes / 60 / d.dt_h))
    m = (d.index >= t0) & (d.index < t1)
    shifted = d.met["poa"].shift(steps)
    d.met.loc[m, "poa"] = shifted[m]
    return d, {"component": "clock", "minutes": minutes, "start": str(t0), "end": str(t1)}


def sensor_soiling(data: PlantData, episodes: list[tuple[str, str]], peak: float, tz: str):
    """Sensor soiling: linear build-up to ``peak`` over each episode, then a cleaning (recovery)."""
    d = data.copy()
    factor = np.ones(len(d.index))
    for a, b in episodes:
        t0, t1 = _ts(a, tz), _ts(b, tz)
        m = (d.index >= t0) & (d.index < t1)
        factor[m] = 1 - peak * (d.index[m] - t0) / (t1 - t0)
    d.met["poa"] = d.met["poa"] * factor
    return d, {"component": "sensor_soiling", "peak": peak, "episodes": episodes,
               "factor": pd.Series(factor, index=d.index)}

"""Guarantee tests in the style of EPC / O&M contracts, configured in YAML (see configs/contract_example.yaml).

The clauses in the example file are illustrative: thresholds, exclusions and damages formulas differ from one
contract to the next and must be checked against the actual agreement. Nothing here is legal advice.

Availability guarantee
    A = sum of capacity-weighted daylight intervals with the inverter running / all counted intervals.
    Counted: daylight (POA above ``daylight_wm2``), data recorded, not covered by an excluded event category
    (e.g. force majeure, grid-operator instruction, grid outage). Planned maintenance is excluded up to an
    allowance of capacity-weighted daylight hours per period; beyond it, it counts as downtime.
    ``basis: energy`` uses 1 - lost energy / expected energy over the same intervals instead.
    Shortfall energy = (A_guaranteed - A) x expected energy of the counted intervals; damages = shortfall
    energy x price, capped. Optional bonus above a threshold.

Performance-ratio guarantee
    Measured PR (plain, temperature-corrected to 25 degC, or weather-corrected) over intervals with POA above
    ``min_irradiance_wm2``, all inverters recorded and no excluded event. The guaranteed value falls each
    contract year by ``annual_degradation`` (relative), and a measurement tolerance is subtracted.
    Shortfall energy = (PR_guaranteed_effective - PR) x reference yield of the same intervals.

Usage:
    python -m pv_budget run ... --contract configs/contract_example.yaml
    table = contract.evaluate(spec, analysis)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pv_budget import kpi
from pv_budget.losses import event_mask

log = logging.getLogger(__name__)


def load(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _periods(local: pd.DatetimeIndex, period: str, start: pd.Timestamp | None) -> pd.Index:
    if period == "month":
        return pd.Index([f"{y}-{m:02d}" for y, m in zip(local.year, local.month)])
    if period == "quarter":
        return pd.Index([f"{y}-Q{(m - 1) // 3 + 1}" for y, m in zip(local.year, local.month)])
    if start is None:
        return pd.Index([str(y) for y in local.year])
    # contract years counted from the contract start date
    s = start.tz_localize(local.tz) if start.tzinfo is None else start.tz_convert(local.tz)
    n = np.floor((local - s) / pd.Timedelta(days=365.25)).astype(int) + 1
    return pd.Index([f"CY{k}" for k in n])


def availability_test(spec: dict, state: pd.DataFrame, weights: pd.Series, frame: pd.DataFrame, events,
                      dt_h: float, periods: pd.Index, excluded: list[str], allowance_h: float,
                      lim: np.ndarray) -> pd.DataFrame:
    """``lim``: expected AC power per interval and inverter (after soiling, snow and clipping), kW."""
    inv = list(state.columns)
    w = weights[inv].to_numpy()
    daylight = frame["poa"].to_numpy() > spec.get("daylight_wm2", 50)
    st = state.to_numpy()
    known = (st != "gap") & daylight[:, None]
    excl = event_mask(events, state.index, inv, [c for c in excluded if c != "planned_maintenance"]).to_numpy()
    pm = event_mask(events, state.index, inv, ["planned_maintenance"]).to_numpy() & known \
        if "planned_maintenance" in excluded else np.zeros_like(known)
    # planned maintenance: excluded until the period's allowance (capacity-weighted daylight hours) is used
    pm_h = (pm.astype(float) @ w) * dt_h
    pm_ok = np.zeros_like(pm)
    for per in pd.unique(periods):
        pos = np.flatnonzero(periods == per)
        cum = np.cumsum(pm_h[pos])
        within = cum <= allowance_h + 1e-9
        pm_ok[pos] = pm[pos] & within[:, None]
    counted = known & ~excl & ~pm_ok
    up = (st == "up") & counted
    down = (st == "down") & counted
    rows = []
    for per in pd.unique(periods):
        pos = np.flatnonzero(periods == per)
        num = float((up[pos].astype(float) @ w).sum())
        den = float((counted[pos].astype(float) @ w).sum())
        a_time = num / den if den > 0 else np.nan
        e_lost = float(np.where(down[pos], lim[pos], 0).sum() * dt_h)
        e_exp = float(np.where(counted[pos], lim[pos], 0).sum() * dt_h)
        a_energy = 1 - e_lost / e_exp if e_exp and e_exp > 0 else np.nan
        a = a_energy if spec.get("basis", "time") == "energy" else a_time
        g = float(spec["guaranteed"])
        short = max(0.0, g - a) if np.isfinite(a) else np.nan
        e_short = short * e_exp if np.isfinite(e_exp) else np.nan
        ld = min(e_short / 1000 * spec.get("ld_price_per_mwh", 0.0), spec.get("ld_cap", np.inf)) \
            if np.isfinite(e_short) else np.nan
        bonus = 0.0
        if spec.get("bonus_threshold") is not None and np.isfinite(a) and a > spec["bonus_threshold"]:
            bonus = (a - spec["bonus_threshold"]) * e_exp / 1000 * spec.get("ld_price_per_mwh", 0.0) * \
                spec.get("bonus_share", 0.0)
        rows.append({"period": per, "test": "availability", "basis": spec.get("basis", "time"),
                     "measured": a, "availability_time": a_time, "availability_energy": a_energy,
                     "guaranteed": g, "pass": bool(a >= g) if np.isfinite(a) else None,
                     "shortfall_kwh": e_short, "damages": ld, "bonus": bonus,
                     "excluded_share": 1 - den / max(float((known[pos].astype(float) @ w).sum()), 1e-9),
                     "planned_maintenance_h": float(pm_h[pos].sum())})
    return pd.DataFrame(rows)


def pr_test(spec: dict, frame: pd.DataFrame, state: pd.DataFrame, events, plant, t_typ: float, dt_h: float,
            periods: pd.Index, excluded: list[str], start: pd.Timestamp | None) -> pd.DataFrame:
    inv = list(state.columns)
    st = state.to_numpy()
    poa = frame["poa"].to_numpy()
    complete = ~(st == "gap").any(axis=1)
    excl = event_mask(events, state.index, inv, excluded).to_numpy().any(axis=1)
    m = (poa >= spec.get("min_irradiance_wm2", 50)) & complete & ~excl
    if spec.get("exclude_curtailment", True):
        m &= ~frame["curtailed"].to_numpy()
    daylight = poa >= spec.get("min_irradiance_wm2", 50)
    metric = spec.get("metric", "pr_wc")
    ref_key = {"pr": "ref", "pr_25": "ref_25", "pr_wc": "ref_wc"}[metric]
    rows = []
    for k, per in enumerate(pd.unique(periods)):
        pos = periods == per
        terms = kpi.pr_terms(frame, plant, t_typ, dt_h, mask=m & pos)
        val = kpi.pr_values(terms)[metric]
        year_n = k + 1
        if start is not None:
            first = frame.index[np.flatnonzero(pos)[0]]
            s = start.tz_localize("UTC") if start.tzinfo is None else start
            year_n = int(np.floor((first - s) / pd.Timedelta(days=365.25))) + 1
        g = float(spec["guaranteed"]) * (1 - spec.get("annual_degradation", 0.0)) ** max(year_n - 1, 0)
        g_eff = g - spec.get("tolerance", 0.0)
        completeness = (m & pos).sum() / max((daylight & pos).sum(), 1)
        enough = completeness >= spec.get("min_data_share", 0.0)
        short = max(0.0, g_eff - val) if np.isfinite(val) else np.nan
        e_short = short * terms[ref_key] if np.isfinite(short) else np.nan
        ld = min(e_short / 1000 * spec.get("ld_price_per_mwh", 0.0), spec.get("ld_cap", np.inf)) \
            if np.isfinite(e_short) else np.nan
        rows.append({"period": per, "test": "performance_ratio", "basis": metric, "measured": val,
                     "guaranteed": g, "guaranteed_effective": g_eff,
                     "pass": (bool(val >= g_eff) if np.isfinite(val) else None) if enough else None,
                     "shortfall_kwh": e_short, "damages": ld if enough else np.nan, "bonus": 0.0,
                     "data_share": completeness, "contract_year": year_n})
    return pd.DataFrame(rows)


def evaluate(spec: dict, analysis) -> pd.DataFrame:
    cfg, loss, data = analysis.cfg, analysis.loss, analysis.data
    frame = loss.frame
    tz = cfg.plant.timezone
    local = frame.index.tz_convert(tz)
    start = pd.Timestamp(spec["contract_start"]) if spec.get("contract_start") else None
    periods = _periods(local, spec.get("period", "year"), start)
    excluded = list(spec.get("excluded_events", []))
    t_typ = float(analysis.budget["T_typ"].iloc[0])
    out = []
    if "availability" in spec:
        out.append(availability_test(spec["availability"], loss.state, loss.weights, frame, data.events,
                                     data.dt_h, periods, excluded, spec.get("planned_maintenance_allowance_h", 0.0),
                                     expected_per_inverter(frame, cfg, list(loss.state.columns))))
    if "performance_ratio" in spec:
        out.append(pr_test(spec["performance_ratio"], frame, loss.state, data.events, cfg.plant, t_typ, data.dt_h,
                           periods, excluded, start))
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    log.info("contract tests: %d rows, %d failed", len(res), int((res["pass"] == False).sum()) if len(res) else 0)  # noqa: E712
    return res


def expected_per_inverter(frame: pd.DataFrame, cfg, inverters: list[str]) -> np.ndarray:
    """Expected AC power per interval and inverter after soiling, snow and clipping (kW)."""
    from pv_budget import model
    p = cfg.plant
    derate = model.degradation(p, frame.index) * frame["soiling_ratio"].to_numpy() *         (1 - frame["snow_cover"].to_numpy())
    _, lim = model.ac_power(p, frame["poa"].to_numpy(), frame["t_cell"].to_numpy(), derate)
    return lim[inverters].to_numpy()

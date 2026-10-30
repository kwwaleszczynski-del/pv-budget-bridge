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

The availability, the PR, the guarantee step-down and the damages are computed by the res-asset-kpi library
(``res_asset_kpi.pv``, ``res_asset_kpi.contracts``).

Usage:
    python -m pv_budget run ... --contract configs/contract_example.yaml
    table = contract.evaluate(spec, analysis)
"""
from __future__ import annotations

import logging
from pathlib import Path

import dataclasses

import numpy as np
import pandas as pd
import yaml
from res_asset_kpi import contracts, pv

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
    st = state.to_numpy()
    # running: up (or asleep), stopped: down, no data: gap
    running = pd.DataFrame(np.where(st == "gap", np.nan, (st != "down").astype(float)), index=state.index,
                           columns=inv)
    excl = event_mask(events, state.index, inv, [c for c in excluded if c != "planned_maintenance"]).to_numpy()
    pm = event_mask(events, state.index, inv, ["planned_maintenance"]).to_numpy() \
        if "planned_maintenance" in excluded else None
    av = pv.availability(running, dt_h, capacity=weights[inv].to_numpy(), poa_wm2=frame["poa"].to_numpy(),
                         daylight_wm2=spec.get("daylight_wm2", 50), excluded=excl, planned=pm,
                         planned_allowance_h=allowance_h, expected_kw=lim, periods=np.asarray(periods))
    basis = spec.get("basis", "time")
    measured = av["availability_energy" if basis == "energy" else "availability_time"]
    price = spec.get("ld_price_per_mwh", 0.0)
    bonus = None if spec.get("bonus_threshold") is None else \
        contracts.Bonus(threshold=spec["bonus_threshold"], share=spec.get("bonus_share", 0.0), price_per_mwh=price)
    g = contracts.Guarantee(guaranteed=float(spec["guaranteed"]),
                            damages=contracts.Damages(method="energy", price_per_mwh=price, cap=spec.get("ld_cap")),
                            bonus=bonus)
    r = contracts.assess(g, measured, basis_energy_mwh=av["expected_kwh"].to_numpy() / 1000)
    return pd.DataFrame({"period": av.index, "test": "availability", "basis": basis,
                         "measured": measured.to_numpy(), "availability_time": av["availability_time"].to_numpy(),
                         "availability_energy": av["availability_energy"].to_numpy(), "guaranteed": g.guaranteed,
                         "pass": r["passed"].to_numpy(),
                         "shortfall_kwh": (r["shortfall"] * av["expected_kwh"]).to_numpy(),
                         "damages": r["damages"].to_numpy(), "bonus": r["bonus"].to_numpy(),
                         "excluded_share": av["excluded_share"].to_numpy(),
                         "planned_maintenance_h": av["planned_h"].to_numpy()})


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
    ref_attr = {"pr": "reference_kwh", "pr_25": "reference_25_kwh", "pr_wc": "reference_wc_kwh"}[metric]
    pers = pd.unique(periods)
    terms, years, share = [], [], []
    for k, per in enumerate(pers):
        pos = periods == per
        terms.append(kpi.pr_terms(frame, plant, t_typ, dt_h, mask=m & pos))
        year_n = k + 1
        if start is not None:
            first = frame.index[np.flatnonzero(pos)[0]]
            s = start.tz_localize("UTC") if start.tzinfo is None else start
            year_n = int(np.floor((first - s) / pd.Timedelta(days=365.25))) + 1
        years.append(year_n)
        share.append((m & pos).sum() / max((daylight & pos).sum(), 1))
    measured = pd.Series([kpi.pr_values(t)[metric] for t in terms], index=pers, dtype=float)
    ref = np.array([getattr(t, ref_attr) for t in terms], dtype=float)
    g = contracts.Guarantee(guaranteed=float(spec["guaranteed"]),
                            annual_degradation=spec.get("annual_degradation", 0.0),
                            tolerance=spec.get("tolerance", 0.0), min_data_share=spec.get("min_data_share", 0.0),
                            damages=contracts.Damages(method="energy", cap=spec.get("ld_cap"),
                                                      price_per_mwh=spec.get("ld_price_per_mwh", 0.0)))
    r = contracts.assess(g, measured, contract_year=years, basis_energy_mwh=ref / 1000, data_share=share)
    stepped = contracts.effective_guarantee(dataclasses.replace(g, tolerance=0.0), years)
    return pd.DataFrame({"period": pers, "test": "performance_ratio", "basis": metric,
                         "measured": measured.to_numpy(),
                         "guaranteed": stepped, "guaranteed_effective": r["guaranteed_effective"].to_numpy(),
                         "pass": r["passed"].to_numpy(), "shortfall_kwh": r["shortfall"].to_numpy() * ref,
                         "damages": r["damages"].to_numpy(), "bonus": 0.0, "data_share": share,
                         "contract_year": years})


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

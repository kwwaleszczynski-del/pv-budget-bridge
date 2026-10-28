"""Performance ratio, temperature-corrected PR, availability and a year-on-year degradation estimate.

Definitions (written from public descriptions; this is not a certified implementation of IEC 61724-1):

- PR = final yield / reference yield = (E_AC / P0) / (H_POA / G_STC), with P0 the DC rating at STC and
  G_STC = 1 kW/m2;
- PR_25 (temperature-corrected to 25 degC): each interval's reference yield is multiplied by
  1 + gamma (T_cell - 25), so a hot month no longer looks like poor performance;
- PR_wc (weather-corrected, Dierauf et al. 2013, NREL/TP-5200-57991): the same correction but towards
  T_typ, the irradiance-weighted average cell temperature of a typical year (here: the budget's TMY), so the
  annual PR_wc is close to the plain annual PR and the monthly values no longer swing with the seasons;
- availability, time-based: capacity-weighted share of daylight intervals (POA above a threshold) in which
  the inverters ran, intervals without data left out; energy-based: 1 - lost energy / (lost + expected).

Usage:
    from pv_budget import kpi
    terms = kpi.pr_terms(loss.frame, cfg.plant, t_typ, dt_h)
    kpi.pr_values(terms)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv_budget.config import Plant


def pr_terms(frame: pd.DataFrame, plant: Plant, t_typ: float, dt_h: float, mask=None) -> dict:
    """Sums for the PR family over the intervals in ``mask``: energy (kWh) and reference yields (kWh)."""
    m = np.ones(len(frame), bool) if mask is None else np.asarray(mask, bool)
    g = frame["poa"].to_numpy()[m] / 1000.0
    tc = frame["t_cell"].to_numpy()[m]
    e = float(frame["actual"].to_numpy()[m].sum() * dt_h)
    p0 = plant.dc_kwp
    return {"E": e, "H": float(g.sum() * dt_h),
            "ref": float(p0 * g.sum() * dt_h),
            "ref_25": float(p0 * np.sum(g * (1 + plant.gamma_pdc * (tc - 25.0))) * dt_h),
            "ref_wc": float(p0 * np.sum(g * (1 + plant.gamma_pdc * (tc - t_typ))) * dt_h) if np.isfinite(t_typ)
            else np.nan}


def pr_values(t: dict) -> dict:
    def div(a, b):
        return a / b if b and np.isfinite(b) and b > 0 else np.nan
    return {"pr": div(t["E"], t["ref"]), "pr_25": div(t["E"], t["ref_25"]), "pr_wc": div(t["E"], t["ref_wc"])}


def availability(frame: pd.DataFrame) -> dict:
    up, down = frame["up_w"].sum(), frame["down_w"].sum()
    lost, exp = frame["availability"].sum(), frame["expected"].sum()
    return {"availability_time": up / (up + down) if up + down > 0 else np.nan,
            "availability_energy": 1 - lost / (lost + exp) if lost + exp > 0 else np.nan}


def yoy_degradation(pi: pd.Series, min_pairs: int = 30, n_boot: int = 500, seed: int = 0) -> dict:
    """Year-on-year rate of the daily performance index (median of PI(d) / PI(d - 365) - 1), %/year.

    A diagnostic, in the spirit of the YoY method used for PV degradation studies: it sees soiling and
    sensor drift too, which is why the bridge's degradation bar uses the assumed rate instead.
    """
    s = pi.dropna()
    s = s[s > 0]
    prev = s.copy()
    prev.index = prev.index + pd.Timedelta(days=365)
    pairs = (s / prev.reindex(s.index)).dropna()
    if len(pairs) < min_pairs:
        return {"rate": np.nan, "low": np.nan, "high": np.nan, "pairs": int(len(pairs))}
    r = pairs.to_numpy() - 1
    rng = np.random.default_rng(seed)
    boot = [np.median(rng.choice(r, len(r))) for _ in range(n_boot)]
    return {"rate": float(np.median(r)), "low": float(np.percentile(boot, 2.5)),
            "high": float(np.percentile(boot, 97.5)), "pairs": int(len(pairs))}

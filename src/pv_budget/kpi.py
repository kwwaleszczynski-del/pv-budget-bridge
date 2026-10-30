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

The PR family is computed by the res-asset-kpi library (``res_asset_kpi.pv.pr_terms``).

Usage:
    from pv_budget import kpi
    terms = kpi.pr_terms(loss.frame, cfg.plant, t_typ, dt_h)
    kpi.pr_values(terms)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from res_asset_kpi import pv

from pv_budget.config import Plant


def pr_terms(frame: pd.DataFrame, plant: Plant, t_typ: float, dt_h: float, mask=None) -> pv.PRTerms:
    """Sums for the PR family over the intervals in ``mask``: energy and reference energies (kWh)."""
    return pv.pr_terms(frame["actual"], frame["poa"], plant.dc_kwp, dt_h, t_cell_c=frame["t_cell"],
                       gamma_per_k=plant.gamma_pdc, t_typ_c=t_typ, mask=mask)


def pr_values(t: pv.PRTerms) -> dict:
    return {"pr": t.pr, "pr_25": t.pr_temperature_corrected, "pr_wc": t.pr_weather_corrected}


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

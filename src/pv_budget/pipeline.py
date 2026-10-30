"""One plant from input to monthly results: sensor QC -> loss accounting -> bridge -> KPIs.

Usage:
    from pv_budget import pipeline
    res = pipeline.analyze(cfg, data, budget_table)
    res.monthly          # bridge bars + PR + availability per month
    res.annual           # the same summed per year
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pv_budget import bridge, kpi, losses, model, qc
from pv_budget.config import Config
from pv_budget.io import PlantData

log = logging.getLogger(__name__)


@dataclass
class Analysis:
    cfg: Config
    data: PlantData
    qc: qc.QCResult
    loss: losses.LossResult
    budget: pd.DataFrame
    monthly: pd.DataFrame
    annual: pd.DataFrame
    yoy: dict


def analyze(cfg: Config, data: PlantData, budget_table: pd.DataFrame, use_qc: bool = True) -> Analysis:
    sp = model.solar_position(data.index, data.dt_h, cfg.plant)
    q = qc.run(cfg, data, sp)
    if use_qc:
        poa = q.poa
    else:
        raw = data.met["poa"].clip(lower=0)
        poa = raw.fillna(q.poa)
    loss = losses.run(cfg, data, poa)
    br = bridge.monthly(cfg, loss, budget_table, data.dt_h)
    t_typ = float(budget_table["T_typ"].iloc[0])
    fr = loss.frame
    local = fr.index.tz_convert(cfg.plant.timezone)
    key = pd.Index([f"{y}-{m:02d}" for y, m in zip(local.year, local.month)])
    extra = []
    for period, pos in pd.Series(np.arange(len(fr)), index=key).groupby(level=0):
        sub = fr.iloc[pos.to_numpy()]
        terms = kpi.pr_terms(sub, cfg.plant, t_typ, data.dt_h)
        extra.append({"period": period, **kpi.pr_values(terms), **kpi.availability(sub),
                      "ref_yield": terms.reference_kwh, "ref_yield_wc": terms.reference_wc_kwh})
    br = br.join(pd.DataFrame(extra).set_index("period"))
    br["pr_budget"] = br["month"].map(budget_table["PR"]) if "PR" in budget_table else np.nan
    annual = annual_table(br, cfg, t_typ, fr, data.dt_h)
    yoy = kpi.yoy_degradation(loss.daily["pi_no_snow"])
    log.info("%s: %d months, actual %.1f MWh vs budget %.1f MWh, residual %.1f MWh", cfg.plant.name, len(br),
             br["E_actual"].sum() / 1000, br["E_budget"].sum() / 1000, br["residual"].sum() / 1000)
    return Analysis(cfg, data, q, loss, budget_table, br, annual, yoy)


def annual_table(br: pd.DataFrame, cfg: Config, t_typ: float, fr: pd.DataFrame, dt_h: float) -> pd.DataFrame:
    sum_cols = ["E_budget", "E_actual", "H_actual", "H_budget"] + bridge.BARS + \
               [f"shapley_{c}" for c in bridge.BARS] + [f"loss_{c}" for c in bridge.COMPONENTS] + \
               [f"budget_{c}" for c in bridge.COMPONENTS]
    a = br.groupby("year")[[c for c in sum_cols if c in br]].sum()
    a["months"] = br.groupby("year").size()
    a["coverage"] = br.groupby("year")["coverage"].sum() / 12
    local = fr.index.tz_convert(cfg.plant.timezone)
    for y in a.index:
        sub = fr[local.year == y]
        a.loc[y, ["pr", "pr_25", "pr_wc"]] = list(kpi.pr_values(kpi.pr_terms(sub, cfg.plant, t_typ, dt_h)).values())
        av = kpi.availability(sub)
        a.loc[y, "availability_time"] = av["availability_time"]
        a.loc[y, "availability_energy"] = av["availability_energy"]
    return a

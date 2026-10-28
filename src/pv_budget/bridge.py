"""Monthly energy bridge (waterfall) from the P50 budget to the measured production.

Default method, "sequential (loss accounting)", in this fixed order:

1. irradiance   E_b (r_irr - 1), r_irr = M25(actual POA) / M25(budget POA): model AC energy at 25 degC,
                 so it is the POA ratio plus the inverter's part-load curve;
2. temperature  E_b r_irr (r_T - 1), r_T = [M/M25](actual) / [M/M25](budget): the cell-temperature factor of
                 the actual month against the budget's;
3. degradation  E_b r_irr r_T (r_D - 1), r_D = degradation factor of the month / factor the budget assumed.
   This gives E_w, the budget re-stated for the weather that occurred and the plant's age;
4. soiling, snow, clipping, availability, curtailment, data gaps: each bar = budget-assumed loss - measured
   loss, in kWh. The measured loss fractions come from the per-interval chain in ``losses`` (scale-free) and
   are turned into kWh on the actual plant's potential, E_act / prod(1 - l_k), so they are the energy that
   was really lost, not a model's estimate of it. The budget-assumed losses are spread over E_w in the same
   order, B_k = X0_b prod_{j<k}(1 - b_j) b_k with X0_b = E_w / prod(1 - b_k);
5. residual     whatever is left = potential of the actual plant - potential the budget implies for this
                 weather (model error, unmodelled losses, sensor level, open soiling).

Bars add up to E_act - E_b exactly in every month. The weather steps carry the order effects (temperature
on top of irradiance, etc.); the loss bars do not depend on the order of the weather steps.

Alternative ("shapley"): the same month written as a product E_act = E_b prod f_k (weather ratios, loss
ratios (1 - l_k)/(1 - b_k), residual), with each factor's bar the Shapley value of the product game. It is
order-free, but no bar equals a measured quantity. Both are computed; ``bridge.method`` picks the headline.

Usage:
    from pv_budget import bridge
    table = bridge.monthly(cfg, loss, budget_table, dt_h)
"""
from __future__ import annotations

import logging
from math import factorial

import numpy as np
import pandas as pd

from pv_budget import model
from pv_budget.config import Config
from pv_budget.losses import COMPONENTS, LossResult

log = logging.getLogger(__name__)

WEATHER = ["irradiance", "temperature", "degradation"]
MIN_COVERAGE = 0.2   # months with less data than this share (dataset edges) are left out
BARS = WEATHER + COMPONENTS + ["residual"]
BUDGET_LOSS = {"soiling": "soiling", "snow": "snow", "clipping": "clip", "availability": None,
               "curtailment": "curtailment", "data_gaps": None}


def shapley_product(base: float, factors: np.ndarray) -> np.ndarray:
    """Shapley values of v(S) = base * prod_{k in S} f_k; they sum to base * (prod f - 1)."""
    f = np.asarray(factors, float)
    n = len(f)
    w = np.array([factorial(s) * factorial(n - 1 - s) / factorial(n) for s in range(n)])
    out = np.zeros(n)
    for k in range(n):
        others = np.delete(f, k)
        e = np.zeros(n)          # elementary symmetric polynomials of the other factors
        e[0] = 1.0
        for x in others:
            e[1:] = e[1:] + x * e[:-1]
        out[k] = base * (f[k] - 1.0) * float(np.dot(w, e))
    return out


def monthly(cfg: Config, loss: LossResult, budget: pd.DataFrame, dt_h: float) -> pd.DataFrame:
    p = cfg.plant
    fr = loss.frame
    idx = fr.index
    local = idx.tz_convert(p.timezone)
    key = local.year * 100 + local.month
    deg = model.degradation(p, idx)
    poa, tc = fr["poa"].to_numpy(), fr["t_cell"].to_numpy()
    u25, _ = model.ac_power(p, poa, np.full(len(poa), 25.0))
    u, _ = model.ac_power(p, poa, tc)
    g = poa / 1000
    ex = pd.DataFrame({"M25": u25.sum(axis=1).to_numpy(), "M": u.sum(axis=1).to_numpy(),
                       "x0nd": fr["potential"].to_numpy() / deg, "gf": g * (1 + p.gamma_pdc * (tc - 25)),
                       "g": g, "gt": g * tc}, index=idx)
    cols = ["potential"] + COMPONENTS + ["expected", "actual"]
    sums = (fr[cols].groupby(key).sum() * dt_h).join((ex.groupby(key).sum() * dt_h))
    sums["H"] = fr["poa"].groupby(key).sum() * dt_h / 1000
    sums["n"] = fr.groupby(key).size()
    sums["t_amb"] = fr["t_amb"].groupby(key).mean() if "t_amb" in fr else np.nan
    rows = []
    for k, s in sums.iterrows():
        y, m = divmod(int(k), 100)
        b = budget.loc[m]
        hours = pd.Timestamp(year=y, month=m, day=1).days_in_month * 24
        frac = min(1.0, s["n"] * dt_h / hours)
        if frac < MIN_COVERAGE:
            log.info("%d-%02d skipped: data cover %.0f%% of the month", y, m, 100 * frac)
            continue
        Eb = b["E"] * frac
        # weather
        if np.isfinite(b["M25"]):
            r_irr = s["M25"] / (b["M25"] * frac)
            r_T = (s["M"] / s["M25"]) / (b["M"] / b["M25"])
        else:  # monthly budget table: POA ratio; cell temperatures shifted by the ambient difference
            r_irr = s["H"] / (b["H"] * frac)
            r_T = 1.0
            if np.isfinite(b.get("t_amb", np.nan)) and np.isfinite(s["t_amb"]) and s["g"] > 0:
                d_t = s["t_amb"] - b["t_amb"]
                phi_a = s["gf"] / s["g"]
                phi_b = (s["g"] + p.gamma_pdc * (s["gt"] - (d_t + 25) * s["g"])) / s["g"]
                r_T = phi_a / phi_b
        D_a = s["potential"] / s["x0nd"] if s["x0nd"] > 0 else 1.0
        r_D = D_a / b["D"]
        w_irr = Eb * (r_irr - 1)
        w_T = Eb * r_irr * (r_T - 1)
        w_D = Eb * r_irr * r_T * (r_D - 1)
        Ew = Eb * r_irr * r_T * r_D
        # measured loss fractions (sequential, model scale)
        rest, ell = s["potential"], {}
        for c in COMPONENTS:
            ell[c] = s[c] / rest if rest > 0 else 0.0
            rest -= s[c]
        keep = np.prod([1 - ell[c] for c in COMPONENTS])
        if keep > 0.05 and s["actual"] > 0:
            pot = s["actual"] / keep
        else:
            pot = s["potential"]
            log.warning("%d-%02d: almost no expected production left; losses kept on the model scale", y, m)
        # budget-assumed fractions
        bfr = {"soiling": b["soiling"], "snow": b["snow"], "clipping": b["clip"],
               "availability": 1 - b["availability"], "curtailment": b["curtailment"], "data_gaps": 0.0}
        x0b = Ew / np.prod([1 - bfr[c] for c in COMPONENTS])
        row = {"year": y, "month": m, "period": f"{y}-{m:02d}", "coverage": frac, "E_budget": Eb,
               "E_actual": s["actual"], "H_actual": s["H"], "H_budget": b["H"] * frac, "r_irr": r_irr,
               "r_temp": r_T, "r_deg": r_D, "E_weather": Ew, "potential_actual": pot,
               "potential_budget": x0b, "irradiance": w_irr, "temperature": w_T, "degradation": w_D}
        acc_a, acc_b = pot, x0b
        for c in COMPONENTS:
            La = acc_a * ell[c]
            Bb = acc_b * bfr[c]
            row[f"loss_{c}"] = La
            row[f"budget_{c}"] = Bb
            row[f"frac_{c}"] = ell[c]
            row[c] = Bb - La
            acc_a -= La
            acc_b -= Bb
        row["residual"] = s["actual"] - Eb - sum(row[c] for c in WEATHER + COMPONENTS)
        # Shapley allocation of the same month, as a product of factors
        f = [r_irr, r_T, r_D] + [(1 - ell[c]) / (1 - bfr[c]) for c in COMPONENTS] + \
            [pot / x0b if x0b > 0 else 1.0]
        sh = shapley_product(Eb, np.array(f))
        for c, v in zip(BARS, sh):
            row[f"shapley_{c}"] = v
        row["closure"] = s["actual"] - Eb - sum(row[c] for c in BARS)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("period")
    if cfg.bridge.method == "shapley":
        for c in BARS:
            out[f"sequential_{c}"] = out[c]
            out[c] = out[f"shapley_{c}"]
    return out

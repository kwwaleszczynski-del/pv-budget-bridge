import numpy as np
import pandas as pd
import pytest

from pv_budget import bridge, inject, losses, pipeline
from pv_budget.losses import soiling_ratio
from conftest import make_cfg, synthetic_plant


def flat_budget(cfg, data):
    """A budget equal to the plant model on the same weather (so weather bars are ~0)."""
    from pv_budget import model
    loss = losses.run(cfg, data, data.met["poa"])
    fr = loss.frame
    local = fr.index.tz_convert(cfg.plant.timezone)
    rows = []
    poa, tc = fr["poa"].to_numpy(), fr["t_cell"].to_numpy()
    u25, _ = model.ac_power(cfg.plant, poa, np.full(len(poa), 25.0))
    u, _ = model.ac_power(cfg.plant, poa, tc)
    for m in range(1, 13):
        sel = local.month == m
        n = max(sel.sum(), 1)
        days = pd.Timestamp(year=2023, month=m, day=1).days_in_month * 24 / data.dt_h
        s = days / n
        rows.append({"month": m, "E": fr["actual"][sel].sum() * data.dt_h * s, "H": poa[sel].sum() * data.dt_h / 1000 * s,
                     "M25": u25.sum(axis=1)[sel].sum() * data.dt_h * s, "M": u.sum(axis=1)[sel].sum() * data.dt_h * s,
                     "clip": 0.0, "soiling": 0.0, "snow": 0.0, "availability": 1.0, "curtailment": 0.0, "D": 1.0,
                     "T_typ": 40.0, "PR": 0.8})
    return pd.DataFrame(rows).set_index("month").fillna(0.0)


def test_shapley_product_properties():
    f = np.array([1.1, 0.9, 0.95])
    sh = bridge.shapley_product(100.0, f)
    assert sh.sum() == pytest.approx(100.0 * (f.prod() - 1))
    # two factors: phi_1 = base [(f1 - 1) + (f1 - 1)(f2 - 1) / 2]
    sh2 = bridge.shapley_product(100.0, np.array([1.2, 0.5]))
    assert sh2[0] == pytest.approx(100 * (0.2 + 0.2 * -0.5 / 2))
    assert sh2[1] == pytest.approx(100 * (-0.5 + 0.2 * -0.5 / 2))
    # symmetric factors get equal shares
    s3 = bridge.shapley_product(1.0, np.array([0.9, 0.9, 1.0]))
    assert s3[0] == pytest.approx(s3[1]) and s3[2] == pytest.approx(0.0)


def test_bridge_closes_and_finds_an_outage():
    cfg, data = synthetic_plant(days=40)
    bt = flat_budget(cfg, data)
    base = pipeline.analyze(cfg, data, bt)
    assert np.allclose(base.monthly["closure"], 0.0, atol=1e-6)
    d2, truth = inject.outage(data, "A", "2023-05-20", days=3, tz="UTC")
    res = pipeline.analyze(cfg, d2, bt)
    assert np.allclose(res.monthly["closure"], 0.0, atol=1e-6)
    got = -(res.monthly["availability"] - base.monthly["availability"]).sum()
    assert got == pytest.approx(truth["kwh"], rel=0.03)
    # nothing leaks into the unexplained bar
    assert abs((res.monthly["residual"] - base.monthly["residual"]).sum()) < 0.03 * truth["kwh"]


def test_curtailment_set_point_is_curtailment_not_availability():
    cfg, data = synthetic_plant(days=20)
    bt = flat_budget(cfg, data)
    base = pipeline.analyze(cfg, data, bt)
    days = ["2023-05-03", "2023-05-04", "2023-05-05"]
    d2, truth = inject.curtail(data, cfg, days, 0.5, "UTC")
    res = pipeline.analyze(cfg, d2, bt)
    got = -(res.monthly["curtailment"] - base.monthly["curtailment"]).sum()
    assert got == pytest.approx(truth["kwh"], rel=0.03)
    assert abs((res.monthly["availability"] - base.monthly["availability"]).sum()) < 1.0


def test_sequential_bars_by_hand():
    """Weather steps of the sequential bridge: E_b (r-1), E_b r_irr (r_T - 1), E_b r_irr r_T (r_D - 1)."""
    cfg, data = synthetic_plant(days=31, start="2023-05-01")
    bt = flat_budget(cfg, data)
    bt.loc[5, "E"] *= 0.9            # budget 10 % lower than the model on the same weather
    res = pipeline.analyze(cfg, data, bt)
    row = res.monthly.loc["2023-05"]
    eb = row["E_budget"]
    assert row["irradiance"] == pytest.approx(eb * (row["r_irr"] - 1))
    assert row["temperature"] == pytest.approx(eb * row["r_irr"] * (row["r_temp"] - 1))
    assert row["degradation"] == pytest.approx(eb * row["r_irr"] * row["r_temp"] * (row["r_deg"] - 1))
    # the model matches the plant, so a 10 % lower budget shows as +1/0.9 - 1 in the unexplained bar
    assert row["residual"] / eb == pytest.approx(1 / 0.9 - 1, abs=0.01)


def test_soiling_ratio_sawtooth():
    cfg = make_cfg()
    days = pd.date_range("2023-04-01", periods=120, freq="D")
    rng = np.random.default_rng(3)
    loss = np.zeros(120)
    loss[40:70] = np.linspace(0, 0.06, 30)        # 30 days of build-up, cleaned on day 70
    pi = pd.Series(0.85 * (1 - loss) * (1 + rng.normal(0, 0.002, 120)), index=days)
    # day 39 rained without a step (nothing to wash off): the build-up is counted from there
    sr, cleanings = soiling_ratio(pi, {days[39], days[70]}, cfg)
    assert len(cleanings) == 1 and cleanings[0]["date"] == days[70]
    est = 1 - sr.to_numpy()
    assert est[:35].max() < 0.005
    assert est[40:70].mean() == pytest.approx(loss[40:70].mean(), abs=0.004)

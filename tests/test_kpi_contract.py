"""PR and guarantee formulas against examples worked out by hand."""
import numpy as np
import pandas as pd
import pytest

from pv_budget import contract, kpi
from pv_budget.io import empty_events
from conftest import make_cfg


def frame(poa, tc, actual):
    idx = pd.date_range("2024-06-01 10:00", periods=len(poa), freq="1h", tz="UTC")
    return pd.DataFrame({"poa": poa, "t_cell": tc, "actual": actual}, index=idx)


def test_pr_family_by_hand():
    # P0 = 240 kWp (two 120 kWp inverters), gamma = -0.4 %/K, two one-hour intervals
    cfg = make_cfg()
    f = frame([500.0, 1000.0], [35.0, 45.0], [96.0, 192.0])
    t = kpi.pr_terms(f, cfg.plant, t_typ=40.0, dt_h=1.0)
    # reference yield: 240 x (0.5 + 1.0) = 360 kWh -> PR = 288 / 360 = 0.8
    assert t.energy_kwh == pytest.approx(288.0)
    assert t.reference_kwh == pytest.approx(360.0)
    # to 25 degC: 240 x (0.5 x (1 - 0.04) + 1.0 x (1 - 0.08)) = 240 x 1.40 = 336
    assert t.reference_25_kwh == pytest.approx(336.0)
    # weather-corrected to T_typ = 40 degC: 240 x (0.5 x 1.02 + 1.0 x 0.98) = 240 x 1.49 = 357.6
    assert t.reference_wc_kwh == pytest.approx(357.6)
    v = kpi.pr_values(t)
    assert v["pr"] == pytest.approx(0.8)
    assert v["pr_25"] == pytest.approx(288 / 336)
    assert v["pr_wc"] == pytest.approx(288 / 357.6)


def test_yoy_recovers_a_known_trend():
    days = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    rng = np.random.default_rng(1)
    pi = pd.Series(0.85 * (1 - 0.01) ** ((days - days[0]).days / 365.25) * (1 + rng.normal(0, 0.01, len(days))),
                   index=days)
    r = kpi.yoy_degradation(pi)
    assert r["rate"] == pytest.approx(-0.01, abs=0.002)
    assert r["low"] < r["rate"] < r["high"]


def availability_inputs():
    """One day, 10 daylight hours, two inverters of equal size (weights 0.5)."""
    idx = pd.date_range("2024-06-01 07:00", periods=10, freq="1h", tz="UTC")
    st = pd.DataFrame("up", index=idx, columns=["A", "B"])
    st.iloc[0:2, 0] = "down"     # A down 2 h: excluded by a force-majeure event
    st.iloc[4:7, 1] = "down"     # B down 3 h: 1 h planned maintenance (allowance) + 2 h fault
    st.iloc[9, 0] = "gap"        # no data for A in the last hour
    events = pd.DataFrame({"start": [idx[0], idx[4]], "end": [idx[2], idx[5]], "device": ["A", "B"],
                           "category": ["force_majeure", "planned_maintenance"], "limit": [np.nan, np.nan]})
    fr = pd.DataFrame({"poa": 600.0}, index=idx)
    lim = np.full((10, 2), 80.0)   # expected 80 kW per inverter every hour
    return st, events, fr, lim


def test_availability_guarantee_by_hand():
    st, events, fr, lim = availability_inputs()
    spec = {"basis": "time", "guaranteed": 0.95, "ld_price_per_mwh": 100.0, "ld_cap": 1e9}
    w = pd.Series({"A": 0.5, "B": 0.5})
    per = pd.Index(["2024"] * 10)
    out = contract.availability_test(spec, st, w, fr, events, 1.0, per,
                                     ["force_majeure", "planned_maintenance"], allowance_h=0.5, lim=lim)
    r = out.iloc[0]
    # counted: 20 inverter-hours - 1 gap - 2 force majeure - 1 planned (0.5 weighted h = allowance) = 16
    # down and counted: B's two fault hours -> A = 14 / 16 = 0.875
    assert r["availability_time"] == pytest.approx(14 / 16)
    # energy basis: lost 2 x 80 kWh of 16 x 80 kWh expected
    assert r["availability_energy"] == pytest.approx(1 - 160 / 1280)
    # shortfall (0.95 - 0.875) x 1280 kWh = 96 kWh -> 9.6 at 100 per MWh
    assert r["shortfall_kwh"] == pytest.approx(96.0)
    assert r["damages"] == pytest.approx(9.6)
    assert not r["pass"]


def test_planned_maintenance_beyond_allowance_counts():
    st, events, fr, lim = availability_inputs()
    w = pd.Series({"A": 0.5, "B": 0.5})
    out = contract.availability_test({"guaranteed": 0.9}, st, w, fr, events, 1.0, pd.Index(["x"] * 10),
                                     ["force_majeure", "planned_maintenance"], allowance_h=0.0, lim=lim)
    # no allowance: the planned hour is counted as downtime -> 14 / 17
    assert out.iloc[0]["availability_time"] == pytest.approx(14 / 17)


def test_pr_guarantee_with_degradation_and_tolerance():
    cfg = make_cfg()
    idx = pd.date_range("2024-06-01 10:00", periods=2, freq="1h", tz="UTC")
    fr = pd.DataFrame({"poa": [500.0, 1000.0], "t_cell": [25.0, 25.0], "actual": [90.0, 180.0],
                       "curtailed": False}, index=idx)
    st = pd.DataFrame("up", index=idx, columns=["A", "B"])
    spec = {"metric": "pr", "guaranteed": 0.80, "annual_degradation": 0.01, "tolerance": 0.02,
            "ld_price_per_mwh": 50.0, "ld_cap": 1e9}
    # contract start 2021-07-01 -> June 2024 is contract year 3: 0.80 x 0.99^2 = 0.78408, minus 0.02 tolerance
    out = contract.pr_test(spec, fr, st, empty_events(), cfg.plant, 25.0, 1.0, pd.Index(["CY3"] * 2), [],
                           start=pd.Timestamp("2021-07-01"))
    r = out.iloc[0]
    assert r["contract_year"] == 3
    assert r["guaranteed"] == pytest.approx(0.78408)
    assert r["guaranteed_effective"] == pytest.approx(0.76408)
    # measured PR = 270 / (240 x 1.5) = 0.75 -> shortfall (0.76408 - 0.75) x 360 kWh = 5.0688 kWh
    assert r["measured"] == pytest.approx(0.75)
    assert r["shortfall_kwh"] == pytest.approx(5.0688)
    assert r["damages"] == pytest.approx(5.0688 / 1000 * 50.0)
    assert not r["pass"]

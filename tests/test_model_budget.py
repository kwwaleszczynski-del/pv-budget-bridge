import numpy as np
import pandas as pd
import pvlib
import pytest

from pv_budget import budget, model
from conftest import make_cfg


def test_inverter_matches_pvlib_pvwatts():
    pdc = np.array([0.0, 5.0, 40.0, 90.0, 120.0])
    unc, lim = model.inverter(pdc, 100.0, 0.96)
    ref = pvlib.inverter.pvwatts(pdc, 100.0 / 0.96, eta_inv_nom=0.96)
    assert np.allclose(lim, np.asarray(ref), atol=1e-9)
    assert unc[-1] > 100.0 and lim[-1] == pytest.approx(100.0)


def test_ac_power_derate_acts_before_the_limit():
    cfg = make_cfg()
    poa, tc = np.array([1100.0]), np.array([25.0])
    unc, lim = model.ac_power(cfg.plant, poa, tc)
    unc2, lim2 = model.ac_power(cfg.plant, poa, tc, derate=0.97)
    # 3 % less light on a clipping inverter: unlimited power drops, limited power stays at the limit
    assert unc2["A"].iloc[0] < unc["A"].iloc[0]
    assert lim2["A"].iloc[0] == pytest.approx(lim["A"].iloc[0])


def test_degradation_and_module_temperature():
    cfg = make_cfg()
    t = pd.DatetimeIndex(["2020-01-01", "2022-01-01"], tz="UTC")
    d = model.degradation(cfg.plant.__class__(**{**cfg.plant.__dict__, "degradation_per_year": 0.01}), t)
    assert d[0] == pytest.approx(1.0) and d[1] == pytest.approx(0.99 ** (731 / 365.25))
    p = cfg.plant.__class__(**{**cfg.plant.__dict__, "temperature_source": "module", "module_delta_t": 3.0})
    assert model.cell_temperature(p, [1000.0], t_mod=[40.0])[0] == pytest.approx(43.0)


def fake_tmy():
    cfg = make_cfg()
    idx = pd.date_range("1990-01-01", periods=8760, freq="1h", tz="UTC")
    loc = pvlib.location.Location(cfg.plant.latitude, cfg.plant.longitude)
    cs = loc.get_clearsky(idx + pd.Timedelta("30min"))
    cs.index = idx
    return cfg, pd.DataFrame({"ghi": cs["ghi"] * 0.7, "dni": cs["dni"] * 0.6, "dhi": cs["dhi"] * 1.2,
                              "temp_air": 10.0, "wind_speed": 3.0}, index=idx)


def test_budget_file_round_trip(tmp_path):
    cfg, tmy = fake_tmy()
    hourly = budget.simulate(cfg, tmy, 0.5)
    path = tmp_path / "budget.csv"
    budget.write_hourly(hourly, path, cfg, "test")
    back = budget.read_hourly(path, cfg)
    assert len(back) == 8760
    assert back["energy"].sum() == pytest.approx(hourly["E_Grid"].sum(), rel=1e-6)
    table = budget.monthly_table(cfg, hourly=back)
    assert table["E"].sum() == pytest.approx(hourly["E_Grid"].sum(), rel=1e-6)
    assert set(table.index) == set(range(1, 13))
    # budget PR is below 1 and the assumed losses are carried
    assert (table["PR"] < 0.95).all() and (table["availability"] == cfg.budget.availability).all()
    assert table["T_typ"].nunique() == 1


def test_monthly_budget_table(tmp_path):
    cfg = make_cfg()
    path = tmp_path / "monthly.csv"
    pd.DataFrame({"month": range(1, 13), "energy": 1000.0, "poa": 50.0, "t_amb": 10.0}).to_csv(path, index=False)
    t = budget.monthly_table(cfg, table=budget.read_monthly(path, cfg))
    assert t.loc[3, "E"] == 1000.0 and np.isnan(t.loc[3, "M25"])

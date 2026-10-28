import numpy as np
import pandas as pd
import pytest

from pv_budget import inject, model, qc
from conftest import make_cfg, synthetic_plant


def test_clock_shift_found_and_corrected():
    cfg, data = synthetic_plant(days=60, cloud=0.15)
    d2, truth = inject.clock_shift(data, "2023-05-20", "2023-06-10", 60, "UTC")
    res = qc.run(cfg, d2)
    hits = [f for f in res.findings if f["check"] == "clock" and f["status"] == "fail"]
    assert len(hits) == 1 and hits[0]["shift_min"] == 60
    assert abs((pd.Timestamp(hits[0]["start"]) - pd.Timestamp("2023-05-20")).days) <= 3
    # corrected POA close to the original inside the shifted period
    m = (data.index >= "2023-05-25") & (data.index < "2023-06-05")
    err = (res.poa[m] - data.met["poa"][m]).abs().sum() / data.met["poa"][m].sum()
    assert err < 0.02


def test_labelling_offset_reported():
    cfg, data = synthetic_plant(days=30, cloud=0.1)
    d2 = data.copy()
    d2.met["poa"] = data.met["poa"].shift(1)    # values stamped one 15-min interval late
    res = qc.run(cfg, d2)
    f = next(f for f in res.findings if f["check"] == "clock")
    assert f["status"] == "warn" and f["lag_min"] == pytest.approx(15, abs=4)


def test_orientation_fit_recovers_a_turned_sensor():
    cfg, data = synthetic_plant(days=60, cloud=0.1)
    sp = model.solar_position(data.index, data.dt_h, cfg.plant)
    # rebuild the sensor reading for a plane turned 20 degrees west, from the same clear-sky days
    import pvlib
    loc = pvlib.location.Location(cfg.plant.latitude, cfg.plant.longitude, altitude=cfg.plant.altitude)
    mid = model.midpoints(data.index, data.dt_h)
    cs = loc.get_clearsky(mid, solar_position=sp.set_index(mid))
    cs.index = data.index
    k = data.met["poa"] / model.poa_from_components(cfg.plant, sp, cs["ghi"], cs["dni"], cs["dhi"]).replace(0, np.nan)
    turned = model.poa_from_components(cfg.plant, sp, cs["ghi"], cs["dni"], cs["dhi"], azimuth=200.0)
    d2 = data.copy()
    d2.met["poa"] = (turned * k.fillna(0)).fillna(0)
    res = qc.run(cfg, d2)
    assert res.orientation["azimuth"] == pytest.approx(200.0, abs=4.0)
    assert res.orientation["tilt"] == pytest.approx(30.0, abs=4.0)
    assert any(f["check"] == "orientation" and f["status"] == "warn" for f in res.findings)


def test_morning_shading_found():
    cfg, data = synthetic_plant(days=60, cloud=0.1)
    sp = model.solar_position(data.index, data.dt_h, cfg.plant)
    d2 = data.copy()
    blocked = (sp["azimuth"] < 100) & (sp["apparent_elevation"] < 15)
    d2.met["poa"] = data.met["poa"].where(~blocked, data.met["poa"] * 0.3)
    res = qc.run(cfg, d2)
    sh = [f for f in res.findings if f["check"] == "shading" and f["status"] == "warn"]
    assert sh and sh[0]["az_to"] <= 105 and sh[0]["max_elevation"] <= 20
    clean = qc.run(cfg, data)
    assert not [f for f in clean.findings if f["check"] == "shading" and f["status"] == "warn"]


def test_gaps_filled_from_reference():
    cfg, data = synthetic_plant(days=20)
    d2 = data.copy()
    d2.met["poa_ref"] = data.met["poa"] * 0.95
    m = (data.index >= "2023-05-10 08:00") & (data.index < "2023-05-10 12:00")
    d2.met.loc[m, "poa"] = np.nan
    res = qc.run(cfg, d2)
    assert res.flags.loc[m, "filled"].all()
    assert res.poa[m].sum() == pytest.approx(data.met["poa"][m].sum(), rel=0.05)

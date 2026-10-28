import numpy as np
import pandas as pd
import pvlib
import pytest

from pv_budget import model
from pv_budget.config import from_dict
from pv_budget.io import PlantData, empty_events

PLANT = {"name": "TEST", "latitude": 52.0, "longitude": 20.0, "altitude": 100, "timezone": "Europe/Warsaw",
         "tilt": 30, "azimuth": 180, "inverters": [{"id": "A", "dc_kwp": 120, "ac_kw": 100},
                                                   {"id": "B", "dc_kwp": 120, "ac_kw": 100}],
         "gamma_pdc": -0.004, "cod": "2020-01-01", "degradation_per_year": 0.0}


def make_cfg(**sections):
    d = {"plant": dict(PLANT), "data": {"timezone": "UTC"}}
    for k, v in sections.items():
        d[k] = {**d.get(k, {}), **v}
    return from_dict(d)


def synthetic_plant(days=40, start="2023-05-01", dt="15min", seed=0, cloud=0.25, cfg=None):
    """Clear-sky weather with random cloudy days and a plant that follows the configured model exactly."""
    cfg = cfg or make_cfg()
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=int(days * pd.Timedelta("1D") / pd.Timedelta(dt)), freq=dt, tz="UTC")
    dt_h = pd.Timedelta(dt) / pd.Timedelta("1h")
    sp = model.solar_position(idx, dt_h, cfg.plant)
    loc = pvlib.location.Location(cfg.plant.latitude, cfg.plant.longitude, altitude=cfg.plant.altitude)
    cs = loc.get_clearsky(model.midpoints(idx, dt_h), solar_position=sp.set_index(model.midpoints(idx, dt_h)))
    cs.index = idx
    day = idx.normalize()
    kday = pd.Series(np.where(rng.random(days + 1) < cloud, rng.uniform(0.3, 0.7, days + 1), 1.0),
                     index=pd.date_range(idx[0].normalize(), periods=days + 1, freq="D"))
    k = kday.reindex(day).to_numpy()
    ghi, dni, dhi = cs["ghi"] * k, cs["dni"] * k ** 2, cs["dhi"] * (2 - k)
    poa = model.poa_from_components(cfg.plant, sp, ghi, dni, dhi)
    t_amb = 15 + 8 * np.sin(2 * np.pi * (idx.hour + idx.minute / 60 - 9) / 24)
    tc = model.cell_temperature(cfg.plant, poa.to_numpy(), t_amb, np.full(len(idx), 2.0))
    _, lim = model.ac_power(cfg.plant, poa.to_numpy(), tc, model.degradation(cfg.plant, idx))
    lim.index = idx
    met = pd.DataFrame({"poa": poa.to_numpy(), "t_amb": t_amb, "wind": 2.0}, index=idx)
    data = PlantData(idx, dt_h, met, lim.copy(), None, None, empty_events())
    return cfg, data


@pytest.fixture
def plant():
    return synthetic_plant()

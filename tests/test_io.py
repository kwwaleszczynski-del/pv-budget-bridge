import numpy as np
import pandas as pd
import pytest

from pv_budget import io
from pv_budget.config import from_dict
from conftest import PLANT


def cfg_with(**data):
    return from_dict({"plant": dict(PLANT), "data": data})


def test_autumn_dst_hour_resolved_by_order():
    # local Warsaw time, 15 min, across the night of 27 Oct 2024: 02:00-02:45 appears twice
    utc = pd.date_range("2024-10-26 22:00", "2024-10-27 03:00", freq="15min", tz="UTC")
    local = utc.tz_convert("Europe/Warsaw").tz_localize(None)
    assert local.duplicated().sum() == 4
    out = io.parse_times(pd.Series(local), "Europe/Warsaw", pd.Series(["A|p"] * len(local)))
    assert out.notna().all()
    assert (pd.DatetimeIndex(out) == utc).all()


def test_spring_gap_and_lonely_repeated_hour_dropped():
    s = pd.Series(pd.to_datetime(["2024-03-31 01:45", "2024-03-31 02:30", "2024-03-31 03:00",
                                  "2024-10-27 02:15"]))
    out = io.parse_times(s, "Europe/Warsaw")
    assert out.isna().tolist() == [False, True, False, True]
    assert out.iloc[0] == pd.Timestamp("2024-03-31 00:45", tz="UTC")
    assert out.iloc[2] == pd.Timestamp("2024-03-31 01:00", tz="UTC")


def test_offsets_are_exact():
    s = pd.Series(["2024-10-27T02:30:00+02:00", "2024-10-27T02:30:00+01:00"])
    out = io.parse_times(s, "Europe/Warsaw")
    assert list(out) == [pd.Timestamp("2024-10-27 00:30", tz="UTC"), pd.Timestamp("2024-10-27 01:30", tz="UTC")]


def test_long_format_mixed_intervals_and_label_end():
    t5 = pd.date_range("2024-06-01 10:05", "2024-06-01 11:00", freq="5min")      # labelled at interval end
    th = pd.DatetimeIndex(["2024-06-01 11:00"])
    df = pd.concat([
        pd.DataFrame({"timestamp": t5, "device_id": "A", "signal": "p_ac", "value": np.arange(12, dtype=float)}),
        pd.DataFrame({"timestamp": t5, "device_id": "B", "signal": "p_ac", "value": 1.0}),
        pd.DataFrame({"timestamp": th, "device_id": "WS", "signal": "poa", "value": [800.0]}),
    ])
    cfg = cfg_with(timezone="UTC", label="end", interval="15min")
    d = io.load(cfg, frame=df)
    assert d.dt_h == 0.25
    a = d.power["A"].dropna()
    assert a.index[0] == pd.Timestamp("2024-06-01 10:00", tz="UTC")
    assert a.tolist() == [1.0, 4.0, 7.0, 10.0]       # mean of three 5-min values per quarter hour
    assert d.met["poa"].dropna().tolist() == [800.0] * 4   # hourly value spread over its quarter hours


def test_wide_format_with_units():
    df = pd.DataFrame({"time": ["2024-06-01 10:00", "2024-06-01 10:15"], "Inv A [W]": [50000, 60000],
                       "Inv B [W]": [1000, 2000], "Tamb [F]": [32.0, 212.0], "Irr": [500, 600]})
    cfg = cfg_with(format="wide", timestamp_column="time", timezone="UTC",
                   columns={"Inv A [W]": {"device": "A", "signal": "p_ac", "unit": "W"},
                            "Inv B [W]": {"device": "B", "signal": "p_ac", "unit": "W"},
                            "Tamb [F]": {"signal": "t_amb", "unit": "degF"},
                            "Irr": {"signal": "poa", "unit": "W/m2"}})
    d = io.load(cfg, frame=df)
    assert d.power["A"].tolist() == [50.0, 60.0]
    assert d.met["t_amb"].round(6).tolist() == [0.0, 100.0]


def test_stuck_values_removed():
    s = pd.Series([1.0, 2.0] + [5.0] * 30 + [3.0])
    out = io.drop_stuck(s, 24)
    assert out.isna().sum() == 30 and out.iloc[0] == 1.0


def test_unknown_unit_raises():
    with pytest.raises(ValueError):
        io.convert(pd.Series([1.0]), "furlongs")

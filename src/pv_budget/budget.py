"""P50 budget: an hourly table in a layout close to a PVsyst hourly CSV export, or a monthly table.

In this repository the budget is computed with the plant model (pvlib) from the PVGIS typical meteorological
year, with the losses the budget assumes (soiling, snow, availability, curtailment) stated in the config.
A budget made elsewhere is read from a file instead (``budget.path``):

- hourly: ``;``-separated, optional comment lines, one header row with variable names, an optional units row,
  then one row per hour. Column names follow PVsyst variable names by default (GlobInc, T_Amb, WindVel,
  E_Grid, IL_Pmax) and can be remapped in ``budget.columns``. Only files written by this tool were tested;
  check the reader against your own export before relying on it;
- monthly: one row per calendar month with energy (kWh) and POA irradiation (kWh/m2), optional ambient
  temperature (degC) and clipping (fraction).

Usage:
    python -m pv_budget budget --config configs/pl_demo.yaml --out data/pl_demo/budget_hourly.csv
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pv_budget import model
from pv_budget.config import Config, monthly

log = logging.getLogger(__name__)


def fetch_tmy(cfg: Config, cache: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """PVGIS TMY for the plant location (hourly, UTC), cached as parquet + json."""
    import pvlib

    cache = Path(cache) if cache else None
    if cache and cache.with_suffix(".parquet").exists():
        meta = json.loads(cache.with_suffix(".json").read_text())
        return pd.read_parquet(cache.with_suffix(".parquet")), meta
    out = pvlib.iotools.get_pvgis_tmy(cfg.plant.latitude, cfg.plant.longitude, map_variables=True, timeout=120)
    data = out[0]
    meta = next(o for o in out[1:] if isinstance(o, dict) and "inputs" in o)
    info = {"irradiance_time_offset": meta["inputs"]["location"].get("irradiance_time_offset", 0.5),
            "radiation_db": meta["inputs"]["meteo_data"]["radiation_db"],
            "years": [meta["inputs"]["meteo_data"]["year_min"], meta["inputs"]["meteo_data"]["year_max"]],
            "months_selected": meta.get("months_selected")}
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(cache.with_suffix(".parquet"))
        cache.with_suffix(".json").write_text(json.dumps(info, indent=1))
    log.info("PVGIS TMY: %s %s", info["radiation_db"], info["years"])
    return data, info


def fetch_pvgis_hourly(cfg: Config, start: int, end: int, cache: Path, database: str | None = None) -> pd.Series:
    """PVGIS hourly POA on the nominal plane (satellite SARAH3 or reanalysis ERA5), W/m2, index = hour start."""
    import pvlib

    cache = Path(cache)
    if cache.exists():
        return pd.read_parquet(cache)["poa_ref"]
    kw = {"raddatabase": database} if database else {}
    d, *_ = pvlib.iotools.get_pvgis_hourly(cfg.plant.latitude, cfg.plant.longitude, start=start, end=end,
                                           surface_tilt=cfg.plant.tilt, surface_azimuth=cfg.plant.azimuth,
                                           components=True, map_variables=True, timeout=180, **kw)
    poa = d["poa_direct"] + d["poa_sky_diffuse"] + d["poa_ground_diffuse"]
    # PVGIS stamps each hour at the time its irradiance refers to (e.g. hh:30 or hh:10)
    poa.index = poa.index.floor("1h")
    out = poa.rename("poa_ref").to_frame()
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out["poa_ref"]


def simulate(cfg: Config, tmy: pd.DataFrame, offset_h: float) -> pd.DataFrame:
    """Hourly budget from a TMY with the plant model and the assumed losses (kWh per hour = kW)."""
    p, b = cfg.plant, cfg.budget
    start = tmy.index + pd.Timedelta(hours=offset_h - 0.5)
    sp = model.solar_position(start, 1.0, p)
    poa = model.poa_from_components(p, sp, tmy["ghi"].to_numpy(), tmy["dni"].to_numpy(), tmy["dhi"].to_numpy())
    tc = model.cell_temperature(p, poa.to_numpy(), tmy["temp_air"].to_numpy(), tmy["wind_speed"].to_numpy())
    months = (tmy.index.tz_convert("UTC") + pd.Timestamp("1990-01-15", tz=p.timezone).utcoffset()).month
    d_ref = budget_degradation(cfg)
    derate = d_ref * np.array([(1 - monthly(b.soiling, m)) * (1 - monthly(b.snow, m)) for m in months])
    unc, lim = model.ac_power(p, poa.to_numpy(), tc, derate)
    curt = np.array([monthly(b.curtailment, m) for m in months])
    e_grid = lim.sum(axis=1).to_numpy() * b.availability * (1 - curt)
    clip = (unc.sum(axis=1) - lim.sum(axis=1)).to_numpy()
    # rows stamped at the TMY hour in local standard time (no DST), as budget exports usually are
    std = pd.Timestamp("1990-01-15", tz=p.timezone).utcoffset()
    local = (tmy.index.tz_convert("UTC") + std).tz_localize(None)
    return pd.DataFrame({"date": local, "GlobHor": tmy["ghi"].to_numpy(),
                         "DiffHor": tmy["dhi"].to_numpy(), "T_Amb": tmy["temp_air"].to_numpy(),
                         "WindVel": tmy["wind_speed"].to_numpy(), "GlobInc": poa.to_numpy(),
                         "IL_Pmax": clip, "E_Grid": e_grid})


def budget_degradation(cfg: Config) -> float:
    ref = cfg.budget.reference_date or cfg.plant.cod
    if ref is None:
        return 1.0
    t = pd.DatetimeIndex([pd.Timestamp(ref)]).tz_localize("UTC")
    return float(model.degradation(cfg.plant, t)[0])


def write_hourly(df: pd.DataFrame, path: Path, cfg: Config, source: str) -> None:
    """Write in a PVsyst-like layout: comment lines, names row, units row, dd/mm/yy HH:mm dates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = [f"# P50 hourly budget, plant {cfg.plant.name}, {cfg.plant.dc_kwp:.1f} kWp DC / {cfg.plant.ac_kw:.1f} kW AC",
            f"# weather: {source}; model: pvlib (Perez, PVsyst-type cell temperature, PVWatts)",
            f"# assumed: soiling {cfg.budget.soiling}, snow {cfg.budget.snow}, availability "
            f"{cfg.budget.availability}, curtailment {cfg.budget.curtailment}",
            "# dates: start of hour, local standard time (no DST); energies in kWh per hour",
            "date;GlobHor;DiffHor;T_Amb;WindVel;GlobInc;IL_Pmax;E_Grid",
            ";W/m2;W/m2;degC;m/s;W/m2;kW;kW"]
    body = df.assign(date=df["date"].dt.strftime("%d/%m/%y %H:%M")).to_csv(
        sep=";", index=False, header=False, float_format="%.3f", lineterminator="\n")
    path.write_text("\n".join(head) + "\n" + body, encoding="utf-8")


def read_hourly(path, cfg: Config) -> pd.DataFrame:
    """Hourly budget file -> columns month, poa, t_amb, wind, energy (kWh), clipping (kWh, optional)."""
    b = cfg.budget
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    names_at = next(i for i, ln in enumerate(lines) if ln.split(b.sep)[0].strip() == b.columns["date"])
    first = names_at + 1
    if first < len(lines) and not _is_date(lines[first].split(b.sep)[0], b.date_format):
        first += 1   # units row
    df = pd.read_csv(path, sep=b.sep, skiprows=list(range(names_at)) + list(range(names_at + 1, first)),
                     encoding="utf-8", encoding_errors="replace")
    df.columns = [c.strip() for c in df.columns]
    t = pd.to_datetime(df[b.columns["date"]].astype(str).str.strip(), format=b.date_format)
    if b.label == "end":
        t = t - pd.Timedelta("1h")
    scale = {"kwh": 1.0, "wh": 0.001, "mwh": 1000.0}[b.energy_unit.lower()]
    out = pd.DataFrame({"month": t.dt.month, "poa": pd.to_numeric(df[b.columns["poa"]], errors="coerce"),
                        "energy": pd.to_numeric(df[b.columns["energy"]], errors="coerce") * scale})
    for k in ("t_amb", "wind", "clipping"):
        c = b.columns.get(k)
        if c and c in df.columns:
            out[k] = pd.to_numeric(df[c], errors="coerce") * (scale if k == "clipping" else 1.0)
    return out


def _is_date(s: str, fmt: str) -> bool:
    try:
        return pd.notna(pd.to_datetime(s.strip(), format=fmt)) and bool(s.strip())
    except (ValueError, TypeError):
        return False


def read_monthly(path, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"month", "energy", "poa"}
    if need - set(df.columns):
        raise ValueError(f"monthly budget needs columns {sorted(need)}")
    return df.set_index("month").sort_index()


def monthly_table(cfg: Config, hourly: pd.DataFrame | None = None, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per calendar month: what the bridge needs from the budget.

    E (kWh), H (kWh/m2), M25 and M (model AC energy without limit and without assumed losses, at 25 degC and at
    the budget's cell temperature), clip (clipping share of unlimited AC energy), the assumed loss fractions,
    the degradation factor, and T_typ (irradiance-weighted cell temperature of the whole year, for the
    weather-corrected PR).
    """
    p, b = cfg.plant, cfg.budget
    rows = []
    if hourly is not None:
        poa = hourly["poa"].to_numpy()
        t_amb = hourly["t_amb"].to_numpy() if "t_amb" in hourly else np.full(len(hourly), 20.0)
        wind = hourly["wind"].to_numpy() if "wind" in hourly else None
        tc = model.cell_temperature(p, poa, t_amb, wind)
        u25, _ = model.ac_power(p, poa, np.full(len(poa), 25.0))
        u, _ = model.ac_power(p, poa, tc)
        months = hourly["month"].to_numpy()
        derate = np.array([(1 - monthly(b.soiling, m)) * (1 - monthly(b.snow, m)) for m in months])
        du, dl = model.ac_power(p, poa, tc, derate * budget_degradation(cfg))
        clip_model = du.sum(axis=1).to_numpy() - dl.sum(axis=1).to_numpy()
        frame = pd.DataFrame({"month": months, "E": hourly["energy"].to_numpy(), "H": poa / 1000.0,
                              "M25": u25.sum(axis=1).to_numpy(), "M": u.sum(axis=1).to_numpy(),
                              "unc": du.sum(axis=1).to_numpy(), "clipkwh": clip_model,
                              "gt": poa * tc, "g": poa})
        g = frame.groupby("month").sum()
        t_typ = float(frame["gt"].sum() / frame["g"].sum())
        for m, r in g.iterrows():
            rows.append({"month": m, "E": r.E, "H": r.H, "M25": r.M25, "M": r.M,
                         "clip": r.clipkwh / r.unc if r.unc > 0 else 0.0, "hours": int((months == m).sum())})
    else:
        t_typ = np.nan
        for m, r in table.iterrows():
            rows.append({"month": int(m), "E": float(r["energy"]), "H": float(r["poa"]), "M25": np.nan,
                         "M": np.nan, "clip": float(r.get("clipping", 0.0) or 0.0),
                         "t_amb": float(r.get("t_amb", np.nan)), "hours": np.nan})
    out = pd.DataFrame(rows).set_index("month")
    out["soiling"] = [monthly(b.soiling, m) for m in out.index]
    out["snow"] = [monthly(b.snow, m) for m in out.index]
    out["availability"] = b.availability
    out["curtailment"] = [monthly(b.curtailment, m) for m in out.index]
    out["D"] = budget_degradation(cfg)
    out["T_typ"] = t_typ
    if hourly is not None:
        out["PR"] = out["E"] / (p.dc_kwp * out["H"])
    return out


def load(cfg: Config, data_dir: Path | None = None) -> pd.DataFrame:
    """The monthly budget table for a config: from budget.path, or computed from the PVGIS TMY."""
    b = cfg.budget
    if b.path:
        if b.format == "monthly":
            return monthly_table(cfg, table=read_monthly(b.path, cfg))
        return monthly_table(cfg, hourly=read_hourly(b.path, cfg))
    cache = Path(b.tmy_cache) if b.tmy_cache else (Path(data_dir) / "tmy" if data_dir else None)
    tmy, info = fetch_tmy(cfg, cache)
    hourly = simulate(cfg, tmy, info["irradiance_time_offset"])
    h = pd.DataFrame({"month": hourly["date"].dt.month, "poa": hourly["GlobInc"], "t_amb": hourly["T_Amb"],
                      "wind": hourly["WindVel"], "energy": hourly["E_Grid"], "clipping": hourly["IL_Pmax"]})
    return monthly_table(cfg, hourly=h)

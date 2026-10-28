"""Verification with injected losses and sensor faults of known size.

For every example plant (the synthetic Polish plant and three PVDAQ systems) the base data are analysed, then
again with one fault injected at a time; the change of the relevant bridge bar (injected run minus base run,
so the plant's own unknown losses cancel) is compared with what was injected:

- soiling 2 % (mean over one month, a linear build-up to 4 % ending in a logged cleaning);
- a 3-day outage of one inverter (logged);
- a limit to 90 % of the inverters' AC rating on the 10 sunniest days of a month (set-point + event);
- pyranometer drift falling linearly to -3 % over one year (and staying);
- a 1-hour clock shift of the POA logger over three to seven months;
- synthetic plant only (the sensor sees the weather's beam and diffuse parts): sensor soiling episodes,
  a misoriented sensor (+20 deg azimuth) and morning shading of the sensor.

The synthetic plant also has its full truth (every loss of every interval), compared month by month.

Usage:
    python -m pv_budget verify --jobs 4
"""
from __future__ import annotations

import json
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PLANS = {
    "pl_demo": {"tz": "Europe/Warsaw", "inv": "INV01", "soiling": ("2023-06-01", "2023-07-01"),
                "outage": "2023-07-10", "curtail": "2023-06", "drift": ("2023-01-01", "2024-01-01"),
                "clock": ("2022-03-27", "2022-10-30")},
    "pvdaq_34": {"tz": "Etc/GMT+8", "inv": "INV1", "soiling": ("2017-05-01", "2017-06-01"),
                 "outage": "2017-04-10", "curtail": "2017-04", "drift": ("2017-01-01", "2018-01-01"),
                 "clock": ("2017-04-01", "2017-07-01")},
}
PLANS["pvdaq_1276"] = dict(PLANS["pvdaq_34"])
PLANS["pvdaq_1278"] = dict(PLANS["pvdaq_34"])
BRIDGE_JOBS = ["base", "base_noqc", "soiling", "outage", "curtail", "drift", "drift_noqc", "clock", "clock_noqc"]
# an obstacle to the south-east (where the morning beam reaches a south-facing sensor)
SENSOR_JOBS = {"orientation": {"azimuth": 200.0}, "shading": {"shading": (100.0, 160.0, 20.0)}}
SENSOR_SOILING = [("2022-04-01", "2022-05-11"), ("2022-07-01", "2022-08-10"), ("2023-08-01", "2023-09-10")]


def _months(a: str, b: str) -> list[str]:
    return [f"{p.year}-{p.month:02d}" for p in pd.period_range(pd.Timestamp(a), pd.Timestamp(b) - pd.Timedelta(days=1),
                                                                 freq="M")]


def _job(plant: str, name: str, data_dir: str) -> dict:
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)
    from pv_budget import budget, cli, inject, io, model, pipeline, qc, synthetic
    from pv_budget.config import load_config

    data_dir = Path(data_dir)
    ex = cli.EXAMPLES[plant]
    plan = PLANS[plant]
    cfg = load_config(cli.ROOT / ex["config"])
    cfg.budget.tmy_cache = str(data_dir / ex["tmy"])
    bt = budget.load(cfg)
    tz = plan["tz"]
    if name in SENSOR_JOBS:
        era5 = pd.read_parquet(data_dir / "era5" / "era5_51.90_19.50_2021_2024.parquet")
        ref = pd.read_parquet(data_dir / "pl_demo" / "sarah3.parquet")["poa_ref"]
        out = data_dir / "pl_demo" / f"sensor_{name}"
        synthetic.generate(cfg, era5, ref, out, sensor=SENSOR_JOBS[name])
        data = io.load(cfg, out / "long.parquet", events=out / "events.csv")
        truth = {"component": name, **{k: list(v) if isinstance(v, tuple) else v for k, v in SENSOR_JOBS[name].items()}}
        res = pipeline.analyze(cfg, data, bt)
        return _pack(res, truth)
    data = io.load(cfg, data_dir / ex["input"], events=data_dir / ex["events"] if ex.get("events") else None)
    use_qc = not name.endswith("_noqc")
    kind = name.replace("_noqc", "")
    truth = {"component": "none"}
    if kind == "soiling":
        q = qc.run(cfg, data)
        met = data.met
        tc = model.cell_temperature(cfg.plant, q.poa.to_numpy(), met["t_amb"].interpolate(limit=8).to_numpy()
                                    if "t_amb" in met else None, met["wind"].to_numpy() if "wind" in met else None,
                                    met["t_mod"].interpolate(limit=8).to_numpy() if "t_mod" in met else None)
        window = plan["soiling"]
        if "precip" in met:   # rain washes soiling off: put the build-up in the driest 30 days of that summer
            rain = met["precip"].groupby(data.index.tz_convert(tz).normalize().tz_localize(None)).sum()
            y = window[0][:4]
            wet = (rain[f"{y}-04-01":f"{y}-09-30"] >= cfg.losses.rain_cleaning_mm).astype(int).rolling(30).sum()
            end = wet.idxmin() + pd.Timedelta(days=1)
            window = (f"{end - pd.Timedelta(days=30):%Y-%m-%d}", f"{end:%Y-%m-%d}")
        data, truth = inject.soiling(data, cfg, *window, peak=0.04, tz=tz, poa=q.poa, t_cell=tc)
        # the estimate may spread the build-up back to the previous rain: include the month before
        first = pd.Timestamp(window[0]) - pd.DateOffset(months=1)
        truth["months"] = _months(f"{first:%Y-%m}-01", window[1])
        truth["window"] = list(window)
    elif kind == "outage":
        data, truth = inject.outage(data, plan["inv"], plan["outage"], days=3, tz=tz)
        truth["months"] = [plan["outage"][:7]]
    elif kind == "curtail":
        local = data.index.tz_convert(tz)
        daily = data.met["poa"].groupby(local.normalize().tz_localize(None)).sum()
        daily = daily[[f"{d:%Y-%m}" == plan["curtail"] for d in daily.index]]
        days = [f"{d:%Y-%m-%d}" for d in daily.sort_values(ascending=False).index[:10]]
        data, truth = inject.curtail(data, cfg, sorted(days), 0.9, tz)
        truth["months"] = [plan["curtail"]]
    elif kind == "drift":
        data, truth = inject.pyranometer_drift(data, *plan["drift"], total=-0.03, tz=tz)
        truth.pop("factor")
        truth["months"] = _months(plan["drift"][0], f"{int(plan['drift'][1][:4]) + 1}-01-01")
    elif kind == "clock":
        data, truth = inject.clock_shift(data, *plan["clock"], minutes=60, tz=tz)
        truth["months"] = _months(*plan["clock"])
    elif kind == "sensor_soiling":
        data, truth = inject.sensor_soiling(data, SENSOR_SOILING, 0.05, tz)
        truth.pop("factor")
        truth["months"] = sorted({m for a, b in SENSOR_SOILING for m in _months(a, b)})
    res = pipeline.analyze(cfg, data, bt, use_qc=use_qc)
    return _pack(res, truth)


def _pack(res, truth) -> dict:
    d = res.qc.daily
    keep = [c for c in ["drift", "factor", "sensor_soiling", "lag_raw_min", "level"] if c in d]
    return {"monthly": res.monthly, "findings": res.qc.findings, "orientation": res.qc.orientation,
            "shading": res.qc.shading, "daily": d[keep] if len(d) else pd.DataFrame(), "truth": truth,
            "yoy": res.yoy}


def compare_bridge(base: dict, inj: dict, comp: str, extra: str | None = None) -> dict:
    from pv_budget.bridge import BARS
    t = inj["truth"]
    months = [m for m in t["months"] if m in base["monthly"].index and m in inj["monthly"].index]
    b, i = base["monthly"].loc[months], inj["monthly"].loc[months]
    delta = (i[BARS] - b[BARS]).sum()
    recovered = -float(delta[comp])
    truth_kwh = float(t.get("kwh_dc", t["kwh"])) if comp == "soiling" else float(t["kwh"])
    others = float(delta.drop([comp] + ([extra] if extra else [])).abs().sum())
    out = {"truth_kwh": truth_kwh, "recovered_kwh": recovered, "error_kwh": recovered - truth_kwh,
           "error_pct": 100 * (recovered - truth_kwh) / truth_kwh if truth_kwh else np.nan,
           "residual_change_kwh": float(delta["residual"]), "other_bars_abs_kwh": others,
           "budget_kwh": float(b["E_budget"].sum())}
    if comp == "soiling":
        net = -float(delta["soiling"] + delta["clipping"])
        out.update({"truth_net_kwh": float(t["kwh"]), "recovered_net_kwh": net,
                    "error_net_pct": 100 * (net - t["kwh"]) / t["kwh"] if t["kwh"] else np.nan})
    return out


def compare_sensor(base: dict, inj: dict, noqc: dict | None, base_noqc: dict | None = None) -> dict:
    t, f = inj["truth"], inj["findings"]
    months = [m for m in t.get("months", []) if m in base["monthly"].index]
    out = {}
    if noqc is not None and months:
        e = base["monthly"].loc[months, "E_budget"].sum()
        for tag, r, b0 in (("qc", inj, base), ("noqc", noqc, base_noqc or base)):
            dlt = r["monthly"].loc[months] - b0["monthly"].loc[months]
            out[f"irradiance_change_pct_{tag}"] = 100 * float(dlt["irradiance"].sum()) / e
            out[f"residual_change_pct_{tag}"] = 100 * float(dlt["residual"].sum()) / e
            out[f"abs_bar_change_pct_{tag}"] = 100 * float(dlt[[c for c in dlt.columns if c in (
                "irradiance", "temperature", "clipping", "availability", "soiling", "snow", "residual")]]
                .abs().sum().sum()) / e
    comp = t["component"]
    if comp == "sensor_drift":
        hit = [x for x in f if x["check"] == "drift" and x["status"] == "fail"]
        out["detected"] = bool(hit)
        if hit:
            out["onset_found"] = hit[0]["start"]
            out["onset_days_error"] = (pd.Timestamp(hit[0]["start"]) - pd.Timestamp(t["start"][:10])).days
        d = inj["daily"]
        if "drift" in d:
            end = pd.Timestamp(t["end"][:10])
            after = d.loc[d.index >= end, "drift"]
            out["drift_after_end_est"] = float(after.median() - 1) if len(after) else np.nan
            mid = d.loc[(d.index >= end - pd.Timedelta(days=200)) & (d.index < end - pd.Timedelta(days=150)), "drift"]
            out["drift_mid_est"] = float(mid.median() - 1) if len(mid) else np.nan
        out["drift_true_end"] = t["total"]
    elif comp == "clock":
        hit = [x for x in f if x["check"] == "clock" and x["status"] == "fail"]
        out["detected"] = bool(hit)
        if hit:
            best = max(hit, key=lambda x: (pd.Timestamp(x["end"]) - pd.Timestamp(x["start"])).days)
            out.update({"shift_found_min": best["shift_min"], "start_found": best["start"], "end_found": best["end"],
                        "start_days_error": (pd.Timestamp(best["start"]) - pd.Timestamp(t["start"][:10])).days,
                        "end_days_error": (pd.Timestamp(best["end"]) - (pd.Timestamp(t["end"][:10]) -
                                                                         pd.Timedelta(days=1))).days,
                        "periods_found": len(hit)})
    elif comp == "sensor_soiling":
        hit = [x for x in f if x["check"] == "sensor_soiling" and x["status"] == "warn"]
        found = 0
        for a, b in t["episodes"]:
            if any(pd.Timestamp(x["start"]) <= pd.Timestamp(b) + pd.Timedelta(days=10) and
                   pd.Timestamp(x["end"]) >= pd.Timestamp(a) for x in hit):
                found += 1
        out.update({"episodes_true": len(t["episodes"]), "episodes_found": found, "warnings": len(hit),
                    "peaks_found": [round(x["loss"], 4) for x in hit], "peak_true": t["peak"]})
    elif comp == "orientation":
        o = inj["orientation"]
        out.update({"azimuth_true": t["azimuth"], "azimuth_found": o.get("azimuth"), "tilt_found": o.get("tilt"),
                    "flagged": any(x["check"] == "orientation" and x["status"] == "warn" for x in f)})
    elif comp == "shading":
        out.update({"sector_true": t["shading"], "sectors_found": inj["shading"],
                    "flagged": any(x["check"] == "shading" and x["status"] == "warn" for x in f)})
    return out


def truth_table(base: dict, truth_path: Path) -> pd.DataFrame:
    from pv_budget.losses import COMPONENTS
    truth = pd.read_parquet(truth_path)
    local = truth.index.tz_convert("Europe/Warsaw")
    key = pd.Index([f"{y}-{m:02d}" for y, m in zip(local.year, local.month)])
    tm = truth.groupby(key).sum() * 0.25
    br = base["monthly"]
    rows = []
    for c in COMPONENTS:
        est, tru = br[f"loss_{c}"], tm[c].reindex(br.index)
        err = est - tru
        rows.append({"component": c, "truth_mwh": tru.sum() / 1000, "estimate_mwh": est.sum() / 1000,
                     "total_error_pct": 100 * (est.sum() - tru.sum()) / tru.sum() if tru.sum() > 0 else np.nan,
                     "monthly_mae_kwh": float(err.abs().mean()), "monthly_max_kwh": float(err.abs().max()),
                     "monthly_mae_pct_of_actual": 100 * float(err.abs().mean() / br["E_actual"].mean())})
    return pd.DataFrame(rows)


def run_all(data_dir: Path, results_dir: Path, jobs: int = 4) -> dict:
    from pv_budget.bridge import BARS

    tasks = [(p, n) for p in PLANS for n in BRIDGE_JOBS] + [("pl_demo", "sensor_soiling")] + \
            [("pl_demo", n) for n in SENSOR_JOBS]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        outs = list(pool.map(_job, [t[0] for t in tasks], [t[1] for t in tasks], [str(data_dir)] * len(tasks)))
    R = {t: o for t, o in zip(tasks, outs)}
    summary = {"bridge": {}, "sensor": {}, "clean_false_alarms": {}, "method_difference": {}}
    for p in PLANS:
        base = R[(p, "base")]
        summary["bridge"][p] = {
            "soiling_2pct": compare_bridge(base, R[(p, "soiling")], "soiling", extra="clipping"),
            "outage_3d": compare_bridge(base, R[(p, "outage")], "availability"),
            "limit_90pct": compare_bridge(base, R[(p, "curtail")], "curtailment")}
        b0 = R[(p, "base_noqc")]
        summary["sensor"][p] = {"drift_-3pct": compare_sensor(base, R[(p, "drift")], R[(p, "drift_noqc")], b0),
                                "clock_+1h": compare_sensor(base, R[(p, "clock")], R[(p, "clock_noqc")], b0)}
        flags = [f for f in base["findings"] if f["status"] in ("warn", "fail") and
                 f["check"] in ("clock", "drift", "sensor_soiling", "shading", "orientation")]
        summary["clean_false_alarms"][p] = [f"{f['check']}: {f['message']}" for f in flags]
        br = base["monthly"]
        diff = pd.concat({c: (br[c] - br[f"shapley_{c}"]).abs() / br["E_budget"] for c in BARS}, axis=1)
        summary["method_difference"][p] = {"max_pct_of_budget": 100 * float(diff.max().max()),
                                           "mean_pct_of_budget": 100 * float(diff.mean().mean()),
                                           "worst_bar": str(diff.max().idxmax())}
    for n in ["sensor_soiling", *SENSOR_JOBS]:
        summary["sensor"].setdefault("pl_demo", {})[n] = compare_sensor(R[("pl_demo", "base")], R[("pl_demo", n)],
                                                                        None)
    tt = truth_table(R[("pl_demo", "base")], data_dir / "pl_demo" / "base" / "truth.parquet")
    summary["synthetic_truth"] = tt.to_dict("records")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "verification.json").write_text(json.dumps(_clean(summary), indent=1, default=str))
    tt.to_csv(results_dir / "synthetic_truth.csv", index=False, float_format="%.3f")
    log.info("verification written to %s", results_dir / "verification.json")
    return summary


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x

"""Command line.

Usage:
    python -m pv_budget run --config plant.yaml --input data.csv [--events events.csv] [--contract contract.yaml]
                            [--budget-file budget.csv] --out out/plant [--month 2024-06]
    python -m pv_budget budget --config plant.yaml --out out/plant/budget_hourly.csv   # P50 from PVGIS TMY
    python -m pv_budget prepare [--pvdaq-mirror DIR] [--era5 FILE]   # public data for the examples (data/)
    python -m pv_budget examples [--jobs 4]       # the four example plants -> results/ and reports
    python -m pv_budget verify [--jobs 4]         # injected losses and sensor faults -> results/verification.json
    python -m pv_budget site                      # docs/index.html data
    python -m pv_budget all [--jobs 4]

Paths come from arguments or the environment (PV_BUDGET_DATA, default ./data; PV_BUDGET_RESULTS, default
./results); nothing is hard-coded to a machine.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

log = logging.getLogger("pv_budget")

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
    "pl_demo": {"config": "configs/pl_demo.yaml", "input": "pl_demo/base/long.parquet",
                "events": "pl_demo/base/events.csv", "tmy": "pl_demo/tmy"},
    "pvdaq_34": {"config": "configs/pvdaq_34.yaml", "input": "pvdaq/long_34.parquet", "tmy": "pvdaq/tmy"},
    "pvdaq_1276": {"config": "configs/pvdaq_1276.yaml", "input": "pvdaq/long_1276.parquet", "tmy": "pvdaq/tmy"},
    "pvdaq_1278": {"config": "configs/pvdaq_1278.yaml", "input": "pvdaq/long_1278.parquet", "tmy": "pvdaq/tmy"},
}


def data_dir(a) -> Path:
    return Path(getattr(a, "data", None) or os.environ.get("PV_BUDGET_DATA", "data"))


def results_dir(a) -> Path:
    return Path(getattr(a, "results", None) or os.environ.get("PV_BUDGET_RESULTS", "results"))


def analyze_example(key: str, data: Path, overrides: dict | None = None):
    from pv_budget import budget, io, pipeline
    from pv_budget.config import load_config

    ex = EXAMPLES[key]
    cfg = load_config(ROOT / ex["config"])
    cfg.budget.tmy_cache = str(data / ex["tmy"])
    for k, v in (overrides or {}).items():
        section, name = k.split(".")
        setattr(getattr(cfg, section), name, v)
    d = io.load(cfg, data / ex["input"], events=data / ex["events"] if ex.get("events") else None)
    return pipeline.analyze(cfg, d, budget.load(cfg))


def cmd_run(a):
    from pv_budget import budget, contract, io, pipeline, report
    from pv_budget.config import load_config

    cfg = load_config(a.config)
    if a.budget_file:
        cfg.budget.path = a.budget_file
    if a.tmy_cache:
        cfg.budget.tmy_cache = a.tmy_cache
    data = io.load(cfg, a.input, events=a.events)
    res = pipeline.analyze(cfg, data, budget.load(cfg, Path(a.out)), use_qc=not a.no_qc)
    tests = contract.evaluate(contract.load(a.contract), res) if a.contract else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    months = [a.month] if a.month else None
    report.write_excel(res, tests, out / "report.xlsx", months)
    report.write_html(res, tests, out / "report.html", months)
    res.monthly.to_csv(out / "monthly.csv")
    log.info("reports in %s", out)


def cmd_budget(a):
    from pv_budget import budget
    from pv_budget.config import load_config

    cfg = load_config(a.config)
    tmy, info = budget.fetch_tmy(cfg, Path(a.tmy_cache) if a.tmy_cache else None)
    hourly = budget.simulate(cfg, tmy, info["irradiance_time_offset"])
    budget.write_hourly(hourly, Path(a.out), cfg, f"PVGIS TMY ({info['radiation_db']}, {info['years']})")
    log.info("hourly budget %s: %.1f MWh/year", a.out, hourly["E_Grid"].sum() / 1000)


def cmd_prepare(a):
    import pandas as pd

    from pv_budget import budget, pvdaq, synthetic
    from pv_budget.config import load_config

    data = data_dir(a)
    # PVDAQ: daily files, long format, ERA5 POA reference (PVGIS) and precipitation (Open-Meteo)
    raw = data / "pvdaq" / "raw"
    pvdaq.fetch(raw, copy_from=Path(a.pvdaq_mirror) if a.pvdaq_mirror else None)
    weather = pvdaq.fetch_era5_weather(data / "pvdaq" / "era5_lv.parquet")
    for sid in pvdaq.SYSTEMS:
        cfg = load_config(ROOT / f"configs/pvdaq_{sid}.yaml")
        long = pvdaq.to_long(raw, sid)
        ref = budget.fetch_pvgis_hourly(cfg, 2016, 2018, data / "pvdaq" / f"era5_poa_{cfg.plant.tilt:g}.parquet",
                                        "PVGIS-ERA5")
        long = pvdaq.add_reference(long, ref, cfg.data.label)
        long = pvdaq.add_precip(long, weather, cfg.data.label)
        long.to_parquet(data / "pvdaq" / f"long_{sid}.parquet")
        budget.fetch_tmy(cfg, data / EXAMPLES[f"pvdaq_{sid}"]["tmy"])
        log.info("PVDAQ %s: %d rows", sid, len(long))
    # synthetic Polish plant
    cfg = load_config(ROOT / "configs/pl_demo.yaml")
    budget.fetch_tmy(cfg, data / "pl_demo" / "tmy")
    ref = budget.fetch_pvgis_hourly(cfg, 2021, 2023, data / "pl_demo" / "sarah3.parquet", "PVGIS-SARAH3")
    era5 = pd.read_parquet(a.era5 or data / "era5" / "era5_51.90_19.50_2021_2024.parquet")
    synthetic.generate(cfg, era5, ref, data / "pl_demo" / "base")
    # a one-week sample of the long format, for the README
    long = pd.read_parquet(data / "pl_demo" / "base" / "long.parquet")
    t = pd.to_datetime(long["timestamp"])
    sample = long[(t >= "2024-10-26") & (t < "2024-10-28")]
    sample.to_csv(ROOT / "examples" / "long_format_sample.csv", index=False, float_format="%.3f")


def cmd_examples(a):
    from concurrent.futures import ProcessPoolExecutor

    from pv_budget import report

    data, res_dir = data_dir(a), results_dir(a)
    keys = list(EXAMPLES)
    with ProcessPoolExecutor(max_workers=min(a.jobs, len(keys))) as pool:
        outs = list(pool.map(_example_job, keys, [str(data)] * len(keys), [str(res_dir)] * len(keys)))
    summary = {k: v for k, v in zip(keys, outs)}
    (res_dir / "examples.json").write_text(json.dumps(summary, indent=1, default=str))
    report.markdown_summary(summary, res_dir / "examples.md")
    if "pvdaq_34" in summary:  # the README figure
        report.waterfall_svg(summary["pvdaq_34"], res_dir / "waterfall_pvdaq_34.svg")


def _example_job(key: str, data: str, res_dir: str) -> dict:
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)
    from pv_budget import contract, report

    res = analyze_example(key, Path(data))
    spec = contract.load(ROOT / "configs" / "contract_example.yaml")
    tests = contract.evaluate(spec, res)
    out = Path(res_dir) / key
    out.mkdir(parents=True, exist_ok=True)
    res.monthly.to_csv(out / "monthly.csv", float_format="%.4f")
    res.annual.to_csv(out / "annual.csv", float_format="%.4f")
    tests.to_csv(out / "contract.csv", index=False, float_format="%.4f")
    (out / "qc_findings.json").write_text(json.dumps(res.qc.findings, indent=1, default=str))
    report.write_excel(res, tests, out / "report.xlsx")
    report.write_html(res, tests, out / "report.html")
    return report.site_payload(res, tests)


def cmd_verify(a):
    from pv_budget import verify

    verify.run_all(data_dir(a), results_dir(a), jobs=a.jobs)


def cmd_site(a):
    from pv_budget import site

    site.build(results_dir(a), ROOT / "docs")


def main(argv=None):
    p = argparse.ArgumentParser(prog="pv_budget", description=__doc__.split("\n")[0])
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse one plant and write the monthly reports")
    r.add_argument("--config", required=True)
    r.add_argument("--input", required=True, help="long or wide CSV/Parquet")
    r.add_argument("--events", help="event log CSV (start, end, device, category, limit)")
    r.add_argument("--contract", help="guarantee clauses YAML")
    r.add_argument("--budget-file", help="hourly or monthly budget file (default: PVGIS TMY + plant model)")
    r.add_argument("--tmy-cache", help="where to keep the downloaded PVGIS TMY")
    r.add_argument("--month", help="report only this month, YYYY-MM")
    r.add_argument("--no-qc", action="store_true", help="use the raw POA sensor (gaps still filled)")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)
    b = sub.add_parser("budget", help="P50 hourly budget from PVGIS TMY, PVsyst-like CSV layout")
    b.add_argument("--config", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--tmy-cache")
    b.set_defaults(func=cmd_budget)
    for name, func, hlp in [("prepare", cmd_prepare, "download / copy the public example data"),
                            ("examples", cmd_examples, "run the four example plants"),
                            ("verify", cmd_verify, "injected-loss verification"),
                            ("site", cmd_site, "build docs/ data")]:
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--data", help="data folder (default $PV_BUDGET_DATA or ./data)")
        s.add_argument("--results", help="results folder (default $PV_BUDGET_RESULTS or ./results)")
        s.add_argument("--jobs", type=int, default=4, help="parallel processes (default 4)")
        if name == "prepare":
            s.add_argument("--pvdaq-mirror", help="local folder with PVDAQ daily files to copy instead of download")
            s.add_argument("--era5", help="ERA5 parquet for the synthetic plant (Open-Meteo archive, 51.9N 19.5E)")
        s.set_defaults(func=func)
    al = sub.add_parser("all", help="prepare, examples, verify, site")
    al.add_argument("--data")
    al.add_argument("--results")
    al.add_argument("--jobs", type=int, default=4)
    al.add_argument("--pvdaq-mirror")
    al.add_argument("--era5")
    al.set_defaults(func=lambda a: [f(a) for f in (cmd_prepare, cmd_examples, cmd_verify, cmd_site)])
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    a.func(a)


if __name__ == "__main__":
    main()

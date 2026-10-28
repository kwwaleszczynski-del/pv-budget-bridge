"""Monthly reports: an Excel workbook (openpyxl) and a self-contained HTML page with a waterfall chart.

Excel sheets: Summary, Bridge (one row per month, conditional formatting on deviation, PR, availability and
residual), Waterfall (per reported month, as a stacked bar chart with an invisible base, the usual way to draw
a waterfall in a workbook), Losses (measured vs budget-assumed), Contract, QC, Downtime.

Usage:
    report.write_excel(analysis, contract_table, "out/report.xlsx", months=["2024-06"])
    report.write_html(analysis, contract_table, "out/report.html")
"""
from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from pv_budget.bridge import BARS, COMPONENTS

LABELS = {"irradiance": "Irradiance", "temperature": "Temperature", "degradation": "Degradation",
          "soiling": "Soiling", "snow": "Snow", "clipping": "Clipping", "availability": "Availability",
          "curtailment": "Curtailment", "data_gaps": "Data gaps", "residual": "Unexplained"}
HEAD = PatternFill("solid", fgColor="1F3A5F")
HEAD_FONT = Font(bold=True, color="FFFFFF")
RED = PatternFill("solid", fgColor="F8D7DA")
GREEN = PatternFill("solid", fgColor="D4EDDA")
AMBER = PatternFill("solid", fgColor="FFF3CD")
THIN = Border(bottom=Side(style="thin", color="BBBBBB"))
PLOTLY = ('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" '
          'integrity="sha384-cCVCZkAjYNxaYKbM8lsArLznDF/SvMFr1jcZrvOpSTCa0W40ZAdLzHCEulnUa5i7" '
          'crossorigin="anonymous" charset="utf-8"></script>')


def _safe(v):
    """Text that a spreadsheet would read as a formula (=, +, -, @, tab, CR) is stored as plain text."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def _table(ws, df: pd.DataFrame, start_row: int = 1, formats: dict | None = None, widths: dict | None = None):
    formats = formats or {}
    for j, c in enumerate(df.columns, 1):
        cell = ws.cell(row=start_row, column=j, value=_safe(str(c)))
        cell.fill, cell.font = HEAD, HEAD_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, row in enumerate(df.itertuples(index=False), start_row + 1):
        for j, v in enumerate(row, 1):
            if isinstance(v, (float, np.floating)) and not math.isfinite(v):
                v = None
            elif isinstance(v, np.generic):
                v = v.item()
            elif isinstance(v, pd.Timestamp):
                v = v.tz_localize(None) if v.tzinfo else v
            cell = ws.cell(row=i, column=j, value=_safe(v))
            cell.border = THIN
            fmt = formats.get(df.columns[j - 1])
            if fmt:
                cell.number_format = fmt
    for j, c in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(j)].width = (widths or {}).get(c, max(11, min(28, len(str(c)) + 3)))
    ws.freeze_panes = ws.cell(row=start_row + 1, column=2)
    ws.row_dimensions[start_row].height = 32


def _col(df: pd.DataFrame, name: str) -> str:
    return get_column_letter(list(df.columns).index(name) + 1)


def bridge_frame(br: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"Month": br.index, "Coverage": br["coverage"].to_numpy(),
                        "Budget P50 [kWh]": br["E_budget"].to_numpy(), "Actual [kWh]": br["E_actual"].to_numpy()})
    out["Deviation [kWh]"] = out["Actual [kWh]"] - out["Budget P50 [kWh]"]
    out["Deviation [%]"] = out["Deviation [kWh]"] / out["Budget P50 [kWh]"]
    for c in BARS:
        out[f"{LABELS[c]} [kWh]"] = br[c].to_numpy()
    out["Check [kWh]"] = br["closure"].to_numpy()
    out["POA actual [kWh/m2]"] = br["H_actual"].to_numpy()
    out["POA budget [kWh/m2]"] = br["H_budget"].to_numpy()
    out["PR"] = br["pr"].to_numpy()
    out["PR weather-corr."] = br["pr_wc"].to_numpy()
    out["PR budget"] = br["pr_budget"].to_numpy()
    out["Availability time"] = br["availability_time"].to_numpy()
    out["Availability energy"] = br["availability_energy"].to_numpy()
    return out


def write_excel(res, tests: pd.DataFrame | None, path, months: list[str] | None = None) -> None:
    cfg, br = res.cfg, res.monthly
    if months:
        br = br.loc[[m for m in months if m in br.index]]
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    p = cfg.plant
    tot_b, tot_a = br["E_budget"].sum(), br["E_actual"].sum()
    info = [("Plant", p.name), ("DC / AC [kW]", f"{p.dc_kwp:.1f} / {p.ac_kw:.1f}"),
            ("Location", f"{p.latitude:.4f}, {p.longitude:.4f}"), ("Report time zone", p.timezone),
            ("Months", f"{br.index[0]} .. {br.index[-1]}" if len(br) else "-"),
            ("Budget P50 [kWh]", tot_b), ("Actual [kWh]", tot_a),
            ("Deviation [%]", tot_a / tot_b - 1 if tot_b else None),
            ("Bridge method", cfg.bridge.method),
            ("Unexplained residual [kWh]", br["residual"].sum()),
            ("YoY performance trend [%/yr]", res.yoy.get("rate")),
            ("QC findings (warn/fail)", sum(f["status"] in ("warn", "fail") for f in res.qc.findings)),
            ("Note", "Working material. PR definitions follow public descriptions of IEC 61724-1 and NREL "
                     "weather-corrected PR; not a certified implementation. Contract clauses are examples.")]
    for i, (k, v) in enumerate(info, 1):
        ws.cell(row=i, column=1, value=k).font = Font(bold=True)
        c = ws.cell(row=i, column=2, value=None if isinstance(v, float) and not math.isfinite(v) else _safe(v))
        if "[%" in k:
            c.number_format = "0.0%"
        elif "kWh" in k:
            c.number_format = "#,##0"
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 30, 60

    # Bridge
    ws = wb.create_sheet("Bridge")
    bf = bridge_frame(br)
    fmt = {c: "#,##0" for c in bf.columns if "[kWh]" in c}
    fmt.update({c: "0.0%" for c in ["Coverage", "Deviation [%]", "PR", "PR weather-corr.", "PR budget",
                                    "Availability time", "Availability energy"]})
    fmt.update({c: "0.0" for c in bf.columns if "kWh/m2" in c})
    _table(ws, bf, formats=fmt)
    n = len(bf) + 1
    if n > 1:
        dev = _col(bf, "Deviation [%]")
        ws.conditional_formatting.add(f"{dev}2:{dev}{n}", ColorScaleRule(
            start_type="num", start_value=-0.15, start_color="F8696B", mid_type="num", mid_value=0,
            mid_color="FFFFFF", end_type="num", end_value=0.15, end_color="63BE7B"))
        for c in ("Availability time", "Availability energy"):
            col = _col(bf, c)
            ws.conditional_formatting.add(f"{col}2:{col}{n}", CellIsRule(operator="lessThan", formula=["0.98"],
                                                                         fill=RED))
        pr, prb = _col(bf, "PR"), _col(bf, "PR budget")
        ws.conditional_formatting.add(f"{pr}2:{pr}{n}", FormulaRule(formula=[f"{pr}2<{prb}2-0.03"], fill=AMBER))
        rc, bc = _col(bf, "Unexplained [kWh]"), _col(bf, "Budget P50 [kWh]")
        ws.conditional_formatting.add(f"{rc}2:{rc}{n}", FormulaRule(formula=[f"ABS({rc}2)>0.05*{bc}2"], fill=RED))
        for c in [f"{LABELS[k]} [kWh]" for k in BARS]:
            col = _col(bf, c)
            ws.conditional_formatting.add(f"{col}2:{col}{n}", CellIsRule(operator="lessThan", formula=["0"],
                                                                         font=Font(color="9C0006")))

    # Waterfall sheets (at most 3 months; the HTML report has all)
    for period in list(br.index)[-3:] if months else []:
        _waterfall_sheet(wb, br.loc[period], period)

    # Losses
    ws = wb.create_sheet("Losses")
    lf = pd.DataFrame({"Month": br.index})
    for c in COMPONENTS:
        lf[f"{LABELS[c]} measured [kWh]"] = br[f"loss_{c}"].to_numpy()
        lf[f"{LABELS[c]} budget [kWh]"] = br[f"budget_{c}"].to_numpy()
    _table(ws, lf, formats={c: "#,##0" for c in lf.columns if "kWh" in c})

    # Contract
    if tests is not None and len(tests):
        ws = wb.create_sheet("Contract")
        tf = tests.copy()
        tf["pass"] = tf["pass"].map({True: "PASS", False: "FAIL"}).fillna("n/a")
        cols = [c for c in ["period", "test", "basis", "measured", "guaranteed", "guaranteed_effective", "pass",
                            "shortfall_kwh", "damages", "bonus", "excluded_share", "data_share"] if c in tf]
        tf = tf[cols]
        _table(ws, tf, formats={"measured": "0.00%", "guaranteed": "0.00%", "guaranteed_effective": "0.00%",
                                "shortfall_kwh": "#,##0", "damages": "#,##0", "bonus": "#,##0",
                                "excluded_share": "0.0%", "data_share": "0.0%"})
        col, n = _col(tf, "pass"), len(tf) + 1
        ws.conditional_formatting.add(f"{col}2:{col}{n}", CellIsRule(operator="equal", formula=['"FAIL"'], fill=RED))
        ws.conditional_formatting.add(f"{col}2:{col}{n}", CellIsRule(operator="equal", formula=['"PASS"'],
                                                                     fill=GREEN))
        ws.cell(row=n + 2, column=1, value="Example clauses only (configs/contract_example.yaml); check against "
                                           "the actual contract.").font = Font(italic=True)

    # QC
    ws = wb.create_sheet("QC")
    qf = pd.DataFrame([{"check": f["check"], "status": f["status"], "message": f["message"]}
                       for f in res.qc.findings])
    if len(qf):
        _table(ws, qf, widths={"message": 110, "check": 18, "status": 10})
        col, n = "B", len(qf) + 1
        for word, fill in (("fail", RED), ("warn", AMBER), ("ok", GREEN)):
            ws.conditional_formatting.add(f"{col}2:{col}{n}", CellIsRule(operator="equal", formula=[f'"{word}"'],
                                                                         fill=fill))

    # Downtime
    dt = res.loss.downtime
    if dt is not None and len(dt):
        ws = wb.create_sheet("Downtime")
        tz = cfg.plant.timezone
        df = dt.assign(start=dt["start"].dt.tz_convert(tz).dt.tz_localize(None),
                       end=dt["end"].dt.tz_convert(tz).dt.tz_localize(None))
        if months:
            key = df["start"].dt.strftime("%Y-%m")
            df = df[key.isin(months)]
        _table(ws, df, formats={"hours": "0.00", "energy_kwh": "#,##0.0", "start": "yyyy-mm-dd hh:mm",
                                "end": "yyyy-mm-dd hh:mm"}, widths={"start": 18, "end": 18})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def _waterfall_sheet(wb, row: pd.Series, period: str) -> None:
    ws = wb.create_sheet(f"Waterfall {period}")
    names = ["Budget P50"] + [LABELS[c] for c in BARS] + ["Actual"]
    vals = [row["E_budget"]] + [row[c] for c in BARS] + [row["E_actual"]]
    base, up, down, level = [], [], [], 0.0
    for k, v in enumerate(vals):
        if k == 0 or k == len(vals) - 1:
            base.append(0.0)
            up.append(v)
            down.append(0.0)
            level = v
        else:
            new = level + v
            base.append(min(level, new))
            up.append(v if v > 0 else 0.0)
            down.append(-v if v < 0 else 0.0)
            level = new
    df = pd.DataFrame({"Step": names, "Value [kWh]": vals, "base": base, "increase": up, "decrease": down})
    _table(ws, df, formats={c: "#,##0" for c in df.columns if c != "Step"}, widths={"Step": 16})
    chart = BarChart()
    chart.type, chart.grouping, chart.overlap = "col", "stacked", 100
    chart.title, chart.y_axis.title = f"Budget to actual, {period}", "kWh"
    data = Reference(ws, min_col=3, min_row=1, max_col=5, max_row=len(df) + 1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=2, max_row=len(df) + 1))
    chart.series[0].graphicalProperties.noFill = True
    chart.series[0].graphicalProperties.line.noFill = True
    chart.series[1].graphicalProperties.solidFill = "3C8D5A"
    chart.series[2].graphicalProperties.solidFill = "C0504D"
    chart.legend = None
    chart.width, chart.height = 24, 11
    ws.add_chart(chart, "H2")


# ---------------------------------------------------------------- JSON for HTML and the docs site
def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), 5)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (pd.Timestamp,)):
        return str(x)
    return x


def site_payload(res, tests: pd.DataFrame | None) -> dict:
    cfg, br, q = res.cfg, res.monthly, res.qc
    p = cfg.plant
    keep = ["E_budget", "E_actual", "H_actual", "H_budget", "coverage", "pr", "pr_25", "pr_wc", "pr_budget",
            "availability_time", "availability_energy", "r_irr", "r_temp", "r_deg"] + BARS + \
        [f"shapley_{c}" for c in BARS] + [f"loss_{c}" for c in COMPONENTS] + [f"budget_{c}" for c in COMPONENTS]
    months = [{"period": k, **{c: r[c] for c in keep if c in br}} for k, r in br.iterrows()]
    annual = [{"year": int(y), **{c: r[c] for c in r.index}} for y, r in res.annual.iterrows()]
    d = q.daily
    daily = {}
    if len(d):
        cols = [c for c in ["lag_raw_min", "lag_min", "clear", "level_cs", "level_ref", "level_plant", "level",
                            "factor"] if c in d]
        daily = {"date": [f"{t:%Y-%m-%d}" for t in d.index], **{c: d[c].astype(float).tolist() for c in cols}}
    shading = q.shading_stats.to_dict("list") if q.shading_stats is not None and len(q.shading_stats) else {}
    dt = res.loss.downtime
    down = dt.groupby("category")["energy_kwh"].agg(["sum", "count"]).reset_index().to_dict("records") \
        if dt is not None and len(dt) else []
    return _clean({"plant": {"name": p.name, "dc_kwp": p.dc_kwp, "ac_kw": p.ac_kw, "tilt": p.tilt,
                             "azimuth": p.azimuth, "timezone": p.timezone, "lat": p.latitude, "lon": p.longitude,
                             "method": cfg.bridge.method},
                   "months": months, "annual": annual, "findings": q.findings, "orientation": q.orientation,
                   "shading": q.shading, "shading_stats": shading, "qc_daily": daily, "downtime": down,
                   "yoy": res.yoy, "budget": res.budget.reset_index().to_dict("records"),
                   "contract": tests.to_dict("records") if tests is not None else [],
                   "cleanings": [{"date": f"{c['date']:%Y-%m-%d}", "step": c["step"]} for c in res.loss.cleanings]})


def script_json(obj) -> str:
    """JSON that cannot close the surrounding <script> element or break it with line separators."""
    return (json.dumps(obj, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def write_html(res, tests, path, months: list[str] | None = None) -> None:
    payload = site_payload(res, tests)
    if months:
        payload["months"] = [m for m in payload["months"] if m["period"] in months]
    html = HTML_TEMPLATE.replace("__TITLE__", escape(f"{res.cfg.plant.name} budget bridge")) \
        .replace("__PLOTLY__", PLOTLY).replace("__DATA__", script_json(payload)) \
        .replace("__LABELS__", json.dumps(LABELS)).replace("__BARS__", json.dumps(BARS))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")


def markdown_summary(summary: dict, path) -> None:
    lines = ["| plant | months | budget MWh | actual MWh | irradiance | temperature | degradation | soiling | snow | "
             "clipping | availability | curtailment | data gaps | unexplained | PR | PR_wc |",
             "|---" * 16 + "|"]
    for key, s in summary.items():
        m = s["months"]
        tot = {c: sum((r.get(c) or 0) for r in m) for c in ["E_budget", "E_actual"] + BARS}
        e_b = tot["E_budget"] or 1
        pr = np.nanmean([a["pr"] for a in s["annual"] if a.get("pr") is not None])
        prw = np.nanmean([a["pr_wc"] for a in s["annual"] if a.get("pr_wc") is not None])
        cells = [f"{100 * tot[c] / e_b:+.1f} %" for c in BARS]
        lines.append(f"| {s['plant']['name']} | {len(m)} | {tot['E_budget'] / 1000:,.1f} | {tot['E_actual'] / 1000:,.1f} | "
                     + " | ".join(cells) + f" | {pr:.3f} | {prw:.3f} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
__PLOTLY__
<style>
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --border:rgba(11,11,11,.10); --up:#2e8b57; --down:#c0504d; --total:#2a78d6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --page:#0d0d0d;
  --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --border:rgba(255,255,255,.10);
  --up:#3fae74; --down:#d9625e; --total:#3987e5; } }
body { margin:0; background:var(--page); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1100px; margin:0 auto; padding:20px 16px 40px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:24px 0 6px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:12px; margin-top:12px; }
table { border-collapse:collapse; width:100%; font-size:13px; } th, td { padding:4px 6px; border-bottom:1px solid var(--grid);
  text-align:right; } th:first-child, td:first-child { text-align:left; }
.wrap { overflow-x:auto; } select { font:inherit; } .ok { color:var(--up); } .fail { color:var(--down); }
.warn { color:#b8860b; } .note { color:var(--muted); font-size:12px; }
#wf { height:420px; }
</style>
</head>
<body><main>
<h1 id="title"></h1>
<p class="note">Working material. PR definitions follow public descriptions of IEC 61724-1 and the NREL
weather-corrected PR (not a certified implementation); contract clauses are examples to be checked against the
actual agreement.</p>
<div class="card"><label>Month <select id="month"></select></label><div id="wf"></div></div>
<h2>Monthly bridge [kWh]</h2><div class="card wrap"><table id="bridge"></table></div>
<h2>Contract tests (example clauses)</h2><div class="card wrap"><table id="contract"></table></div>
<h2>Sensor QC</h2><div class="card wrap"><table id="qc"></table></div>
</main>
<script>
const D = __DATA__, L = __LABELS__, B = __BARS__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (v, d=0) => v == null ? "" : Number(v).toLocaleString("en-US", {maximumFractionDigits:d, minimumFractionDigits:d});
document.getElementById("title").textContent = D.plant.name + " - budget to actual";
const sel = document.getElementById("month");
D.months.forEach((m, i) => { const o = document.createElement("option"); o.value = i; o.textContent = m.period; sel.appendChild(o); });
sel.value = D.months.length - 1;
function draw() {
  const m = D.months[sel.value];
  const x = ["Budget P50", ...B.map(b => L[b]), "Actual"];
  const y = [m.E_budget, ...B.map(b => m[b]), m.E_actual];
  const measure = ["absolute", ...B.map(() => "relative"), "total"];
  Plotly.react("wf", [{type:"waterfall", x, y, measure, text: y.map(v => fmt(v)), textposition:"outside",
    increasing:{marker:{color:css("--up")}}, decreasing:{marker:{color:css("--down")}}, totals:{marker:{color:css("--total")}},
    connector:{line:{color:css("--muted"), width:1}}, hovertemplate:"%{x}: %{y:,.0f} kWh<extra></extra>"}],
    {margin:{t:30,r:10,b:60,l:60}, paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
     font:{color:css("--ink")}, yaxis:{title:"kWh", gridcolor:css("--grid")}, showlegend:false,
     title:{text: m.period + "  (sum of steps = actual - budget)", font:{size:13}}}, {displayModeBar:false, responsive:true});
}
sel.onchange = draw; draw();
const bt = document.getElementById("bridge");
bt.innerHTML = "<tr><th>Month</th><th>Budget</th>" + B.map(b => "<th>" + L[b] + "</th>").join("") +
  "<th>Actual</th><th>PR</th><th>PR wc</th><th>Avail.</th></tr>" + D.months.map(m => "<tr><td>" + esc(m.period) + "</td><td>" +
  fmt(m.E_budget) + "</td>" + B.map(b => "<td>" + fmt(m[b]) + "</td>").join("") + "<td>" + fmt(m.E_actual) + "</td><td>" +
  fmt(m.pr, 3) + "</td><td>" + fmt(m.pr_wc, 3) + "</td><td>" + fmt(100 * (m.availability_time ?? NaN), 2) + "%</td></tr>").join("");
const ct = document.getElementById("contract");
ct.innerHTML = "<tr><th>Period</th><th>Test</th><th>Measured</th><th>Guaranteed (eff.)</th><th>Result</th><th>Damages</th></tr>" +
  D.contract.map(r => "<tr><td>" + esc(r.period) + "</td><td>" + esc(r.test) + " (" + esc(r.basis) + ")</td><td>" + fmt(100 * r.measured, 2) +
  "%</td><td>" + fmt(100 * (r.guaranteed_effective ?? r.guaranteed), 2) + "%</td><td class='" + (r.pass === true ? "ok" : r.pass === false ? "fail" : "") +
  "'>" + (r.pass === true ? "pass" : r.pass === false ? "fail" : "n/a") + "</td><td>" + fmt(r.damages) + "</td></tr>").join("");
document.getElementById("qc").innerHTML = "<tr><th>Check</th><th>Status</th><th style='text-align:left'>Finding</th></tr>" +
  D.findings.map(f => "<tr><td>" + esc(f.check) + "</td><td class='" + esc(f.status) + "'>" + esc(f.status) + "</td><td style='text-align:left'>" +
  esc(f.message) + "</td></tr>").join("");
</script>
</body></html>
"""

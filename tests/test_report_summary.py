"""README summary tables and the static waterfall from a hand-made two-month summary."""
from pv_budget import report
from pv_budget.bridge import BARS


def _summary():
    months = []
    for e_actual, irr in ((90.0, -6.0), (95.0, -2.0)):
        row = {c: 0.0 for c in BARS}
        row.update(E_budget=100.0, E_actual=e_actual, irradiance=irr, residual=e_actual - 100.0 - irr)
        months.append(row)
    return {"x": {"plant": {"name": "X", "dc_kwp": 10.0}, "months": months,
                  "annual": [{"pr": 0.8, "pr_wc": 0.81}]}}


def test_markdown_summary_two_tables(tmp_path):
    report.markdown_summary(_summary(), tmp_path / "s.md")
    energy, bars = (tmp_path / "s.md").read_text(encoding="utf-8").strip().split("\n\n")
    assert energy.splitlines()[2] == "| X | 10.0 | 2 | 0.2 | 0.2 | 92.5 % | 0.800 | 0.810 |"
    cells = bars.splitlines()[2].split(" | ")
    assert cells[1] == "-4.0 %" and cells[-1] == "-3.5 % |"


def test_waterfall_svg_ends_at_actual(tmp_path):
    report.waterfall_svg(_summary()["x"], tmp_path / "w.svg")
    svg = (tmp_path / "w.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert ">92.5</text>" in svg and ">-4.0</text>" in svg and ">-3.5</text>" in svg

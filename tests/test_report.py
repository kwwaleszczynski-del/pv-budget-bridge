import pandas as pd
from openpyxl import Workbook

from pv_budget import io
from pv_budget.report import _safe, _table, script_json


def test_script_json_cannot_close_the_script_element():
    out = script_json({"name": "</script><img src=x onerror=alert(1)>", "x": "a & b"})
    assert "<" not in out and ">" not in out and "&" not in out


def test_formula_like_text_is_stored_as_text():
    wb = Workbook()
    _table(wb.active, pd.DataFrame({"category": ["=HYPERLINK(\"http://x\")", "@cmd", "outage"], "kwh": [-5.0, 1.0, 2.0]}))
    vals = [c.value for c in wb.active["A"][1:]]
    assert vals == ["'=HYPERLINK(\"http://x\")", "'@cmd", "outage"]
    assert wb.active["B2"].value == -5.0 and _safe(-5.0) == -5.0


def test_unknown_event_category_read_as_other(tmp_path):
    p = tmp_path / "ev.csv"
    p.write_text("start,end,device,category\n2024-06-01 10:00,2024-06-01 12:00,*,=cmd()\n")
    ev = io.read_events(p, "UTC")
    assert ev["category"].tolist() == ["other"]

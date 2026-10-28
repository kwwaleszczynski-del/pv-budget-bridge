"""GitHub Pages site: docs/index.html (static) + docs/data.js (results of `examples` and `verify`).

Usage:
    python -m pv_budget site
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def build(results: Path, docs: Path) -> None:
    examples = json.loads((Path(results) / "examples.json").read_text())
    verification = json.loads((Path(results) / "verification.json").read_text())
    for p in examples.values():          # the page only needs a thinned QC history
        d = p.get("qc_daily") or {}
        for k in list(d):
            if k not in ("date", "lag_raw_min", "clear", "level_cs", "level_ref", "level_plant", "level"):
                d.pop(k)
        p.pop("budget", None)
    payload = {"examples": examples, "verification": verification}
    docs = Path(docs)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "data.js").write_text("window.DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
                                  encoding="utf-8")
    log.info("docs/data.js: %.0f kB", (docs / "data.js").stat().st_size / 1000)

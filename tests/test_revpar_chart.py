"""The reported-RevPAR chart renders headlessly at the promised size."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from scripts import make_revpar_chart as chart
from src import analysis


def _fake_comparison(seed: int) -> analysis.RevparComparison:
    rng = np.random.RandomState(seed)
    days = pd.date_range("2018-10-01", "2026-06-30", freq="D")
    daily = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0002, 0.004, len(days))), index=days)
    ends = pd.DatetimeIndex(pd.date_range("2019-03-31", "2026-06-30", freq="QE"))
    rows = [
        {
            "ticker": t,
            "fiscal_quarter": f"Q{e.quarter} {e.year}",
            "period_end": e,
            "revpar_yoy_pct": float(v),
            "source_url": "https://example.com",
        }
        for e, v in zip(ends, rng.normal(3.0, 5.0, len(ends)), strict=True)
        for t in ("MAR", "HLT", "H")
    ]
    return analysis.compare_index_to_reported(daily, pd.DataFrame(rows))


def test_two_panel_chart_is_1600_by_900(tmp_path, monkeypatch):
    out = tmp_path / "chart.png"
    monkeypatch.setattr(chart, "OUT_PNG", out)
    chart.make_chart(_fake_comparison(1), "tsa", _fake_comparison(2), "composite")
    with Image.open(out) as img:
        assert img.size == (1600, 900)


def test_chart_without_overlay_also_renders(tmp_path, monkeypatch):
    out = tmp_path / "chart.png"
    monkeypatch.setattr(chart, "OUT_PNG", out)
    chart.make_chart(_fake_comparison(3), "composite")
    assert out.exists()

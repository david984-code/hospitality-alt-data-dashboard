"""Tests for the regime-change notifier's state logic (no network, no email)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src import notify


def _fake_res(gate: int, alerts: list[str]) -> SimpleNamespace:
    monthly = pd.DataFrame(
        {"tsa_yoy": [3.0, 2.0], "tsa_accel": [0.5, -0.3], "gate": [1, gate]},
        index=pd.PeriodIndex(["2026-04", "2026-05"], freq="M"),
    )
    anomalies = pd.DataFrame(
        {
            "signal": ["tsa_yoy", "brand_MAR"],
            "alert": ["YES" if s in alerts else "" for s in ("tsa_yoy", "brand_MAR")],
        }
    )
    return SimpleNamespace(signals=SimpleNamespace(monthly=monthly), anomalies=anomalies)


def test_current_state():
    cur = notify.current_state(_fake_res(gate=0, alerts=["tsa_yoy"]))
    assert cur["gate"] == "OFF"
    assert cur["alerts"] == ["tsa_yoy"]
    assert cur["as_of"] == "2026-05"


def test_has_changed():
    cur = notify.current_state(_fake_res(gate=1, alerts=[]))
    assert notify.has_changed(None, cur) is True  # first run always notifies
    assert notify.has_changed(cur, cur) is False
    assert notify.has_changed({"gate": "OFF", "alerts": []}, cur) is True  # gate flip
    assert notify.has_changed({"gate": "ON", "alerts": ["x"]}, cur) is True  # alert change


def test_compose_mentions_flip():
    cur = notify.current_state(_fake_res(gate=0, alerts=[]))
    subject, body = notify.compose({"gate": "ON", "alerts": []}, cur)
    assert "ON -> OFF" in subject
    assert "Signal gate: OFF" in body

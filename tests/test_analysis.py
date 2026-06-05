"""Unit tests for the analysis math, on synthetic data with known answers.

These target the most-broken-if-wrong code: the YoY/zscore transforms, the
lead-lag nowcast correlation, the top-2 backtest accounting, and the anomaly
and upcoming-earnings filters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import analysis
from src.analysis import Signals


def test_yoy_simple_doubling():
    idx = pd.period_range("2022-01", periods=13, freq="M")
    s = pd.Series(range(1, 14), index=idx, dtype=float)
    out = analysis.yoy(s, 12)
    # 13th value (13) vs 1st (1) -> +1200%
    assert np.isnan(out.iloc[0])
    assert out.iloc[12] == 1200.0


def test_zscore_is_standardized():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = analysis.zscore(s)
    assert abs(z.mean()) < 1e-9
    # pandas std is sample (ddof=1); the standardized series has std 1
    assert abs(z.std() - 1.0) < 1e-9


def test_demand_nowcast_perfect_coincidence():
    # 36 monthly points; accommodation employment == TSA monthly level.
    months = pd.date_range("2021-01-01", periods=36, freq="MS")
    level = pd.Series(100.0 + np.arange(36), index=months)
    tsa_daily = level.copy()  # resample("ME").mean() collapses to the same level
    accom_emp = level.copy()

    nc = analysis.demand_nowcast(tsa_daily, accom_emp, max_lag=6)
    assert nc.r_coincident > 0.99
    assert nc.best_lag_months == 0
    assert nc.best_r > 0.99
    assert "r" in nc.table.columns


def _signals_with_one_gate() -> tuple[pd.DataFrame, Signals]:
    """Prices rising +10%/+5%/0 for MAR/HLT/H, one accelerating month (Jan)."""
    idx = pd.date_range("2022-01-31", periods=4, freq="ME")
    prices = pd.DataFrame(
        {
            "MAR": [100.0, 110.0, 121.0, 133.1],
            "HLT": [100.0, 105.0, 110.25, 115.76],
            "H": [100.0, 100.0, 100.0, 100.0],
        },
        index=idx,
    )
    months = pd.PeriodIndex(["2022-01", "2022-02", "2022-03"], freq="M")
    monthly = pd.DataFrame(
        {
            "tsa_yoy": [5.0, 4.0, 3.0],
            "tsa_accel": [1.0, -1.0, -1.0],  # only Jan accelerates
            "gate": [1, 0, 0],
            "brand_MAR": [9.0, 1.0, 1.0],
            "brand_HLT": [5.0, 1.0, 1.0],
            "brand_H": [1.0, 1.0, 1.0],
        },
        index=months,
    )
    return prices, Signals(monthly=monthly, weekly=pd.DataFrame())


def test_backtest_top2_picks_and_accounts():
    prices, signals = _signals_with_one_gate()
    bt = analysis.backtest_top2(prices, signals, hold_months=1)

    # One gated month -> top-2 by brand momentum = MAR, HLT.
    assert bt.n_rebalances == 1
    assert bt.n_trades == 2
    assert set(bt.trades["ticker"]) == {"MAR", "HLT"}
    # Both forward returns positive (+10%, +5%) -> 100% hit rate.
    assert bt.hit_rate == 1.0
    assert bt.mean_return == 7.5
    # H was never picked.
    assert "H" not in set(bt.trades["ticker"])


def test_backtest_top2_no_gate_returns_empty():
    prices, signals = _signals_with_one_gate()
    signals.monthly["gate"] = 0  # turn every gate off
    bt = analysis.backtest_top2(prices, signals)
    assert bt.n_trades == 0
    assert bt.n_rebalances == 0
    assert np.isnan(bt.hit_rate)
    # Baseline still computable.
    assert not np.isnan(bt.baseline_hit)


def test_anomaly_flags_detects_spike():
    weeks = pd.date_range("2023-01-01", periods=60, freq="W")
    rng = np.random.RandomState(0)
    calm = rng.normal(0, 1, 60)
    spiked = calm.copy()
    spiked[-1] = 8.0  # large terminal anomaly
    weekly = pd.DataFrame({"tsa_yoy_z": spiked, "job_openings_z": calm}, index=weeks)
    flags = analysis.anomaly_flags(Signals(pd.DataFrame(), weekly), z_threshold=1.5)

    by_signal = flags.set_index("signal")
    assert by_signal.loc["tsa_yoy", "alert"] == "YES"
    assert by_signal.loc["job_openings", "alert"] == ""


def test_upcoming_earnings_window():
    today = pd.Timestamp.now().normalize()
    earnings = pd.DataFrame(
        {
            "ticker": ["MAR", "HLT", "H"],
            "earnings": [
                today + pd.Timedelta(days=10),
                today + pd.Timedelta(days=60),
                today - pd.Timedelta(days=5),
            ],
        }
    )
    nxt = analysis.upcoming_earnings(earnings, within_days=30)
    assert list(nxt["ticker"]) == ["MAR"]
    assert int(nxt.iloc[0]["days_out"]) == 10


def test_pooled_validation_runs_and_is_bounded():
    # YoY + acceleration needs >24 months of history to produce any valid obs.
    months = pd.date_range("2021-01-31", periods=30, freq="ME")
    rng = np.random.RandomState(1)
    tsa = pd.Series(100.0 + np.arange(30) * 2 + rng.normal(0, 3, 30), index=months)
    prices = pd.DataFrame(
        {
            "A": 10.0 * (1 + rng.normal(0.01, 0.05, 30)).cumprod(),
            "B": 20.0 * (1 + rng.normal(0.01, 0.05, 30)).cumprod(),
        },
        index=months,
    )
    pv = analysis.pooled_validation(prices, tsa)
    assert not np.isnan(pv.pooled_r)
    assert -1.0 <= pv.pooled_r <= 1.0
    assert 0.0 <= pv.signal_on_hit <= 1.0
    assert pv.n_obs > 0

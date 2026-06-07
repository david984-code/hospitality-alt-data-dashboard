"""Unit tests for the analysis math, on synthetic data with known answers.

These target the most-broken-if-wrong code: the YoY/zscore transforms, the
lead-lag nowcast correlation, the top-2 backtest accounting, and the anomaly
and upcoming-earnings filters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
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
    assert nc.r_levels > 0.99
    assert nc.r_mom_growth > 0.99  # identical series correlate on changes too
    assert nc.r_mom_n > 0 and nc.r_mom_p < 0.05  # n and significance reported
    assert nc.best_lag_months == 0
    assert nc.best_r > 0.99
    assert "r" in nc.table.columns


def _signals_with_one_gate() -> tuple[pd.DataFrame, Signals]:
    """MAR +10%/mo, HLT +5%/mo, rest flat; one accelerating month (Jan). Config-driven
    so it stays valid as the traded universe (config.TICKERS) grows."""
    idx = pd.date_range("2022-01-31", periods=4, freq="ME")
    growth = {"MAR": 0.10, "HLT": 0.05}
    prices = pd.DataFrame(
        {t: 100.0 * (1 + growth.get(t, 0.0)) ** np.arange(4) for t in config.TICKERS},
        index=idx,
    )
    months = pd.PeriodIndex(["2022-01", "2022-02", "2022-03"], freq="M")
    monthly = pd.DataFrame(
        {"tsa_yoy": [5.0, 4.0, 3.0], "tsa_accel": [1.0, -1.0, -1.0], "gate": [1, 0, 0]},
        index=months,
    )
    brand_rank = {"MAR": 9.0, "HLT": 5.0}  # MAR & HLT are the top-2 by momentum
    for t in config.TICKERS:
        monthly[f"brand_{t}"] = [brand_rank.get(t, 1.0)] * 3
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


def _multi_month(n: int = 30) -> tuple[pd.DataFrame, Signals]:
    """Synthetic prices + monthly signals (config.TICKERS) with a mix of gate on/off."""
    idx = pd.date_range("2022-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(7)
    prices = pd.DataFrame(
        {t: 100 * (1 + rng.normal(0.01, 0.05, n)).cumprod() for t in config.TICKERS},
        index=idx,
    )
    monthly = pd.DataFrame(
        {
            "tsa_yoy": rng.normal(5, 2, n),
            "tsa_accel": rng.normal(0, 1, n),
            "gate": (rng.random(n) > 0.5).astype(int),
        },
        index=pd.PeriodIndex(idx, freq="M"),
    )
    for t in config.TICKERS:
        monthly[f"brand_{t}"] = rng.normal(0, 3, n)
    return prices, Signals(monthly=monthly, weekly=pd.DataFrame())


def test_strategy_significance_bounded():
    prices, signals = _multi_month()
    bt = analysis.backtest_top2(prices, signals)
    sig = analysis.strategy_significance(bt, prices, signals)
    assert sig.n_months >= 2
    assert 0.0 <= sig.naive_p <= 1.0
    assert 0.0 <= sig.clustered_p <= 1.0
    assert 0.0 <= sig.gate_p <= 1.0
    assert not np.isnan(sig.gate_on_mean)


def test_equity_curves_shape():
    prices, signals = _multi_month()
    bt = analysis.backtest_top2(prices, signals)
    eq = analysis.equity_curves(bt, prices)
    assert list(eq.columns) == ["strategy", "baseline"]
    assert isinstance(eq.index, pd.DatetimeIndex)
    assert (eq["strategy"] > 0).all() and (eq["baseline"] > 0).all()


def test_risk_metrics_table():
    prices, signals = _multi_month()
    bt = analysis.backtest_top2(prices, signals)
    rm = analysis.risk_metrics(bt, prices)
    assert set(rm.table.index) == {"Signal (demand-gated)", "Always-long"}
    for col in ["sharpe", "max_drawdown", "in_market", "ann_return", "ann_vol", "total_growth"]:
        assert col in rm.table.columns
    sig_inmkt = rm.table.loc["Signal (demand-gated)", "in_market"]
    assert 0.0 <= sig_inmkt <= 1.0
    # Always-long is invested every month; the gated overlay is invested no more than that.
    assert sig_inmkt <= rm.table.loc["Always-long", "in_market"]
    assert rm.table.loc["Signal (demand-gated)", "max_drawdown"] <= 0.0
    # deployed-capital stats are computed over the invested months only
    assert rm.deployed["n_invested"] >= 1
    assert {"mean_per_month", "ann_rate_deployed", "sharpe", "sharpe_lo", "sharpe_hi"} <= set(
        rm.deployed
    )
    # the Sharpe point estimate sits inside its own CI
    assert rm.deployed["sharpe_lo"] <= rm.deployed["sharpe"] <= rm.deployed["sharpe_hi"]


def test_stress_test_structure():
    rng = np.random.RandomState(5)
    days = pd.date_range("2020-01-01", periods=1400, freq="D")
    tsa = pd.Series(2_000_000 * (1 + rng.normal(0.0002, 0.01, 1400)).cumprod(), index=days)
    weeks = pd.date_range("2020-01-05", periods=200, freq="W")
    trends = pd.DataFrame({t: rng.uniform(40, 80, 200) for t in config.TICKERS}, index=weeks)
    months = pd.date_range("2020-01-31", periods=46, freq="ME")
    fred = pd.DataFrame(
        {"job_openings": rng.uniform(800, 1200, 46), "accom_emp": rng.uniform(1800, 2100, 46)},
        index=months,
    )
    prices = pd.DataFrame(
        {t: 100 * (1 + rng.normal(0.01, 0.05, 46)).cumprod() for t in config.TICKERS},
        index=months,
    )
    stt = analysis.stress_test(tsa, trends, fred, prices, "2020-01-01")
    assert set(stt.risk.table.index) == {"Signal (demand-gated)", "Always-long"}
    assert list(stt.equity.columns) == ["strategy", "baseline"]
    assert stt.start == "2020-01-01"


def test_earnings_search_study():
    weeks = pd.date_range("2022-01-02", periods=120, freq="W")
    rng = np.random.RandomState(3)
    trends = pd.DataFrame({"MAR": rng.normal(50, 5, 120)}, index=weeks)
    days = pd.date_range("2022-01-03", periods=600, freq="D")
    prices = pd.DataFrame({"MAR": 100 * (1 + rng.normal(0.0005, 0.01, 600)).cumprod()}, index=days)
    earnings = pd.DataFrame(
        {
            "ticker": ["MAR"] * 5,
            "earnings": pd.to_datetime(
                ["2023-02-15", "2023-05-15", "2023-08-15", "2023-11-15", "2024-02-15"]
            ),
        }
    )
    study = analysis.earnings_search_study(earnings, trends, prices)
    assert study.n >= 3
    assert -1.0 <= study.corr <= 1.0
    assert set(study.events.columns) >= {"ticker", "earnings", "pre_search_z", "reaction_pct"}

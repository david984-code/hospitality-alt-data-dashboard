"""Orchestrates fetch -> analyze and persists results to outputs/."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

import config
from src import analysis
from src.data import bls, prices, trends, tsa


@dataclass
class PipelineResult:
    tsa: pd.Series
    trends: pd.DataFrame
    fred: pd.DataFrame
    prices: pd.DataFrame
    universe_prices: pd.DataFrame
    earnings: pd.DataFrame
    signals: analysis.Signals
    nowcast: analysis.Nowcast
    backtest: analysis.Backtest
    validation: analysis.PooledValidation
    anomalies: pd.DataFrame
    upcoming: pd.DataFrame
    significance: analysis.Significance
    equity: pd.DataFrame
    earnings_study: analysis.EarningsStudy
    risk: analysis.RiskMetrics


def _trends_frame(tsa_s: pd.Series, force: bool, skip_trends: bool) -> pd.DataFrame:
    """Real Google Trends frame, or a flat stub when trends are skipped (CI)."""
    if skip_trends:
        idx = tsa_s.resample("W").mean().index
        return pd.DataFrame({t: 50.0 for t in config.TICKERS}, index=idx)
    return trends.fetch(force=force)


def _core_analysis(
    tsa_s: pd.Series,
    trends_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    price_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
) -> tuple:
    signals = analysis.build_signals(tsa_s, trends_df, fred_df)
    nowcast = analysis.demand_nowcast(tsa_s, fred_df["accom_emp"])
    bt = analysis.backtest_top2(price_df, signals)
    validation = analysis.pooled_validation(universe_df, tsa_s)
    anomalies = analysis.anomaly_flags(signals)
    upcoming = analysis.upcoming_earnings(earnings_df)
    return signals, nowcast, bt, validation, anomalies, upcoming


def _extra_analysis(bt, signals, trends_df, price_df, earnings_df) -> tuple:
    """Significance, equity curves, earnings event study, and risk metrics."""
    sig = analysis.strategy_significance(bt, price_df, signals)
    equity = analysis.equity_curves(bt, price_df)
    study = analysis.earnings_search_study(earnings_df, trends_df, price_df)
    risk = analysis.risk_metrics(bt, price_df)
    return sig, equity, study, risk


def _analyze(
    tsa_s: pd.Series,
    trends_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    price_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
) -> PipelineResult:
    signals, nowcast, bt, validation, anomalies, upcoming = _core_analysis(
        tsa_s, trends_df, fred_df, price_df, universe_df, earnings_df
    )
    sig, equity, study, risk = _extra_analysis(bt, signals, trends_df, price_df, earnings_df)

    _persist(nowcast, bt, validation, signals, anomalies, sig, study, risk)
    return PipelineResult(
        tsa_s,
        trends_df,
        fred_df,
        price_df,
        universe_df,
        earnings_df,
        signals,
        nowcast,
        bt,
        validation,
        anomalies,
        upcoming,
        sig,
        equity,
        study,
        risk,
    )


def run(force: bool = False, skip_trends: bool = False) -> PipelineResult:
    tsa_s = tsa.fetch(force=force)
    fred_df = bls.fetch(force=force)
    price_df = prices.fetch_prices(force=force)
    universe_df = prices.fetch_universe_prices(force=force)
    earnings_df = prices.fetch_earnings_dates(force=force)
    trends_df = _trends_frame(tsa_s, force, skip_trends)
    return _analyze(tsa_s, trends_df, fred_df, price_df, universe_df, earnings_df)


def _nowcast_summary(nowcast) -> dict:
    return {
        "r_coincident": round(nowcast.r_coincident, 3),
        "best_lag_months": nowcast.best_lag_months,
        "best_r": round(nowcast.best_r, 3),
        "note": "TSA YoY vs Accommodation-employment YoY (demand proxy)",
    }


def _strategy_summary(bt) -> dict:
    return {
        "rule": "Long top-2 lodging franchisor brands by brand-search momentum when TSA YoY is accelerating; hold 1 month",
        "hit_rate": round(bt.hit_rate, 3) if bt.n_trades else None,
        "mean_return_pct": round(bt.mean_return, 3) if bt.n_trades else None,
        "n_positions": bt.n_trades,
        "n_rebalances": bt.n_rebalances,
        "baseline_hit_rate": round(bt.baseline_hit, 3),
        "baseline_mean_pct": round(bt.baseline_mean, 3),
    }


def _validation_summary(validation) -> dict:
    return {
        "universe": config.UNIVERSE,
        "pooled_r": round(validation.pooled_r, 3),
        "signal_on_hit": round(validation.signal_on_hit, 3),
        "signal_on_mean_pct": round(validation.signal_on_mean, 3),
        "baseline_hit": round(validation.baseline_hit, 3),
        "n_obs": validation.n_obs,
    }


def _significance_summary(sig) -> dict:
    return {
        "n_positions": sig.n_positions,
        "n_signal_on_months": sig.n_months,
        "naive_p": round(sig.naive_p, 4),
        "clustered_p": round(sig.clustered_p, 4),
        "gate_on_mean_pct": round(sig.gate_on_mean, 3),
        "gate_off_mean_pct": round(sig.gate_off_mean, 3),
        "gate_p": round(sig.gate_p, 4),
        "note": "clustered_p is the honest test (1 obs per signal-on month); naive_p overstates.",
    }


def _build_summary(nowcast, bt, validation, sig, risk) -> dict:
    return {
        "demand_nowcast": _nowcast_summary(nowcast),
        "strategy": _strategy_summary(bt),
        "significance": _significance_summary(sig),
        "risk_overlay": risk.table.round(3).to_dict(orient="index"),
        "pooled_validation": _validation_summary(validation),
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
    }


def _persist(nowcast, bt, validation, signals, anomalies, sig, study, risk) -> None:
    summary = _build_summary(nowcast, bt, validation, sig, risk)
    (config.OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    nowcast.table.to_csv(config.OUTPUT_DIR / "nowcast_table.csv")
    if bt.n_trades:
        bt.trades.to_csv(config.OUTPUT_DIR / "backtest_trades.csv", index=False)
    signals.monthly.to_csv(config.OUTPUT_DIR / "signals_monthly.csv")
    anomalies.to_csv(config.OUTPUT_DIR / "anomalies.csv", index=False)
    if not study.events.empty:
        study.events.to_csv(config.OUTPUT_DIR / "earnings_study.csv", index=False)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run the hospitality alt-data pipeline")
    ap.add_argument("--force", action="store_true", help="ignore cache, re-fetch all")
    ap.add_argument("--skip-trends", action="store_true", help="stub Google Trends")
    args = ap.parse_args()

    res = run(force=args.force, skip_trends=args.skip_trends)
    n = res.nowcast
    print("\n=== DEMAND NOWCAST (TSA YoY vs accommodation-employment YoY) ===")
    print(n.table.round(3).to_string())
    print(
        f"Coincident r = {n.r_coincident:.3f} | best lag {n.best_lag_months}m, r = {n.best_r:.3f}"
    )

    bt = res.backtest
    print("\n=== STRATEGY: long top-2 franchisor brands on positive TSA-acceleration signal ===")
    print(f"Positions: {bt.n_trades} across {bt.n_rebalances} rebalances")
    print(
        f"Hit rate: {bt.hit_rate:.1%}  mean: {bt.mean_return:+.2f}%   "
        f"| baseline (always-long): {bt.baseline_hit:.1%}  {bt.baseline_mean:+.2f}%"
    )

    sig = res.significance
    print("\n=== SIGNIFICANCE (small sample — read honestly) ===")
    print(
        f"per-position p = {sig.naive_p:.4f} (naive)  |  clustered-by-month p = {sig.clustered_p:.4f}"
    )
    print(
        f"gate-ON {sig.gate_on_mean:+.2f}% vs gate-OFF {sig.gate_off_mean:+.2f}% / mo  "
        f"(Welch p = {sig.gate_p:.4f})"
    )

    print("\n=== RISK OVERLAY (signal vs always-long) ===")
    print(res.risk.table.round(3).to_string())

    v = res.validation
    print("\n=== POOLED VALIDATION (full lodging universe) ===")
    print(
        f"pooled r = {v.pooled_r:+.3f} | signal-on hit {v.signal_on_hit:.1%} ({v.signal_on_mean:+.2f}%) "
        f"vs baseline {v.baseline_hit:.1%} | n = {v.n_obs}"
    )

    print("\n=== ANOMALY FLAGS (latest week) ===")
    print(res.anomalies.to_string(index=False))
    print("\n=== UPCOMING EARNINGS (<=30d) ===")
    print(res.upcoming.to_string(index=False) if not res.upcoming.empty else "none within 30 days")

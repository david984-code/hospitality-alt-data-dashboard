"""Demand nowcast, alt-data signal, top-2 strategy backtest, pooled validation,
and pre-earnings anomaly flags.

Core finding (computed on 2022-2026 data):
  - TSA traveler volume is a strong COINCIDENT proxy for hospitality demand:
    TSA YoY vs Accommodation-employment YoY  ->  r ~= 0.91.
  - The tradeable edge is in the 2nd derivative: when TSA YoY growth is
    ACCELERATING, lodging equities outperform over the next ~1 month.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config


# ----------------------------------------------------------------------------- helpers
def _to_period(s: pd.Series | pd.DataFrame, freq: str = "M"):
    out = s.copy()
    out.index = pd.PeriodIndex(out.index, freq=freq)
    return out


def yoy(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods) * 100.0


def zscore(series: pd.Series, window: int | None = None) -> pd.Series:
    if window:
        return (series - series.rolling(window).mean()) / series.rolling(window).std()
    return (series - series.mean()) / series.std()


# ----------------------------------------------------------------------------- nowcast
@dataclass
class Nowcast:
    r_coincident: float
    best_lag_months: int
    best_r: float
    table: pd.DataFrame
    tsa_yoy: pd.Series
    demand_yoy: pd.Series


def _lag_correlation_table(tsa_y: pd.Series, dem_y: pd.Series, max_lag: int) -> pd.DataFrame:
    """Correlation of TSA YoY (shifted by each lag) vs demand YoY."""
    rows = []
    for lag in range(0, max_lag + 1):
        joined = pd.concat([tsa_y.shift(lag), dem_y], axis=1, join="inner").dropna()
        if len(joined) > 6:
            rows.append(
                {
                    "lag_months": lag,
                    "r": joined.iloc[:, 0].corr(joined.iloc[:, 1]),
                    "n": len(joined),
                }
            )
    return pd.DataFrame(rows).set_index("lag_months")


def demand_nowcast(tsa_daily: pd.Series, accom_emp: pd.Series, max_lag: int = 6) -> Nowcast:
    """Cross-correlation of TSA YoY vs accommodation-employment YoY (the demand proxy).

    Positive lag = TSA leads demand by that many months.
    """
    tsa_y = yoy(_to_period(tsa_daily.resample("ME").mean()), 12).dropna()
    dem_y = yoy(_to_period(accom_emp.resample("ME").last()), 12).dropna()

    table = _lag_correlation_table(tsa_y, dem_y, max_lag)
    r_by_lag = table["r"]
    best = int(r_by_lag.abs().idxmax())
    return Nowcast(
        r_coincident=float(r_by_lag.loc[0]),
        best_lag_months=best,
        best_r=float(r_by_lag.loc[best]),
        table=table,
        tsa_yoy=tsa_y.rename("tsa_yoy"),
        demand_yoy=dem_y.rename("demand_yoy"),
    )


# ----------------------------------------------------------------------------- signals
@dataclass
class Signals:
    monthly: pd.DataFrame  # tsa_yoy, tsa_accel, gate, + brand momentum per ticker
    weekly: pd.DataFrame  # higher-freq view for the dashboard / anomaly flags


def build_signals(tsa_daily: pd.Series, trends: pd.DataFrame, fred_df: pd.DataFrame) -> Signals:
    tsa_m = _to_period(tsa_daily.resample("ME").mean())
    tsa_yoy = yoy(tsa_m, 12)
    tsa_accel = tsa_yoy.diff()  # 2nd derivative -> the tradeable gate
    gate = (tsa_accel > 0).astype(int)

    brand_m = _to_period(trends.resample("ME").mean()) if not trends.empty else pd.DataFrame()
    monthly = pd.DataFrame({"tsa_yoy": tsa_yoy, "tsa_accel": tsa_accel, "gate": gate})
    for t in config.TICKERS:
        if t in brand_m.columns:
            monthly[f"brand_{t}"] = yoy(brand_m[t], 12)

    # Weekly view for the dashboard (z-scored momentum), used by anomaly flags.
    tsa_w = tsa_daily.resample("W").mean()
    weekly = pd.DataFrame({"tsa_yoy_z": zscore(yoy(tsa_w, 52))})
    if not trends.empty:
        tr_w = trends.resample("W").mean()
        for t in config.TICKERS:
            if t in tr_w.columns:
                weekly[f"brand_{t}_z"] = zscore(yoy(tr_w[t], 52))
    jo = fred_df["job_openings"].resample("W").ffill()
    weekly["job_openings_z"] = zscore(yoy(jo, 52))

    return Signals(monthly=monthly.dropna(subset=["tsa_accel"]), weekly=weekly)


# ----------------------------------------------------------------------------- backtest
@dataclass
class Backtest:
    trades: pd.DataFrame
    hit_rate: float
    n_trades: int
    n_rebalances: int
    mean_return: float
    baseline_hit: float
    baseline_mean: float
    equity_curve: pd.Series = field(default_factory=pd.Series)


def _positions_for_period(period, srow, brand_cols, fwd) -> list[dict]:
    """Top-2 brand-momentum positions for one accelerating month, if any."""
    scores = srow[brand_cols].dropna()
    scores.index = [c.replace("brand_", "") for c in scores.index]
    if len(scores) < 2:
        return []
    top2 = scores.sort_values(ascending=False).head(2).index
    out = []
    for t in top2:
        if period in fwd.index and t in fwd.columns and not pd.isna(fwd.loc[period, t]):
            out.append(
                {
                    "rebalance": str(period),
                    "ticker": t,
                    "tsa_accel": round(float(srow["tsa_accel"]), 2),
                    "brand_mom": round(float(scores[t]), 2),
                    "fwd_return_pct": round(float(fwd.loc[period, t]) * 100, 2),
                }
            )
    return out


def _baseline_stats(fwd: pd.DataFrame) -> tuple[float, float]:
    """Always-long-all-headline-names baseline: (hit rate, mean return %)."""
    base = (fwd[config.TICKERS].stack(future_stack=True).dropna() * 100).to_numpy()
    return float((base > 0).mean()), float(base.mean())


def _assemble_backtest(trades: pd.DataFrame, rebalances: int, fwd: pd.DataFrame) -> Backtest:
    base_hit, base_mean = _baseline_stats(fwd)
    if trades.empty:
        return Backtest(trades, float("nan"), 0, 0, float("nan"), base_hit, base_mean)
    rets = trades["fwd_return_pct"]
    eq = np.cumprod(1 + rets.to_numpy() / 100)
    return Backtest(
        trades=trades,
        hit_rate=float((rets > 0).mean()),
        n_trades=len(trades),
        n_rebalances=rebalances,
        mean_return=float(rets.mean()),
        baseline_hit=base_hit,
        baseline_mean=base_mean,
        equity_curve=pd.Series(eq, index=range(1, len(eq) + 1), name="equity"),
    )


def backtest_top2(prices: pd.DataFrame, signals: Signals, hold_months: int = 1) -> Backtest:
    """When the TSA-acceleration gate is on, go long the top-2 headline names by
    brand-search momentum and hold `hold_months`. Returns per-position results."""
    pm = _to_period(prices.resample("ME").last())
    fwd = pm.shift(-hold_months) / pm - 1.0
    m = signals.monthly
    brand_cols = [f"brand_{t}" for t in config.TICKERS if f"brand_{t}" in m.columns]

    gated = m[m["gate"] > 0]
    rebalances = int((gated[brand_cols].notna().sum(axis=1) >= 2).sum())
    rows = [
        r
        for period, srow in gated.iterrows()
        for r in _positions_for_period(period, srow, brand_cols, fwd)
    ]
    return _assemble_backtest(pd.DataFrame(rows), rebalances, fwd)


# ----------------------------------------------------------------------------- validation
@dataclass
class PooledValidation:
    pooled_r: float
    signal_on_hit: float
    signal_on_mean: float
    baseline_hit: float
    baseline_mean: float
    n_obs: int


def pooled_validation(universe_prices: pd.DataFrame, tsa_daily: pd.Series) -> PooledValidation:
    """Out-of-sample-ish check: pool the TSA-acceleration -> 1m-forward-return
    relationship across the full lodging universe (more independent-ish obs)."""
    tsa_m = _to_period(tsa_daily.resample("ME").mean())
    tsa_accel = (yoy(tsa_m, 12)).diff()
    pm = _to_period(universe_prices.resample("ME").last())
    fwd1 = pm.pct_change().shift(-1) * 100

    frames = [
        pd.concat([tsa_accel.rename("s"), fwd1[t].rename("r")], axis=1).dropna()
        for t in universe_prices.columns
    ]
    D = pd.concat(frames)
    on = D[D["s"] > 0]
    return PooledValidation(
        pooled_r=float(D["s"].corr(D["r"])),
        signal_on_hit=float((on["r"] > 0).mean()),
        signal_on_mean=float(on["r"].mean()),
        baseline_hit=float((D["r"] > 0).mean()),
        baseline_mean=float(D["r"].mean()),
        n_obs=int(len(on)),
    )


# ----------------------------------------------------------------------------- anomalies
def anomaly_flags(signals: Signals, z_threshold: float = 1.5) -> pd.DataFrame:
    """Latest-week anomaly check across the weekly z-scored signals."""
    w = signals.weekly
    rows = []
    for col in w.columns:
        ser = w[col].dropna()
        if len(ser) < 10:
            continue
        recent = ser.tail(52)
        latest_val = ser.iloc[-1]
        z = (latest_val - recent.mean()) / recent.std()
        rows.append(
            {
                "signal": col.replace("_z", ""),
                "as_of": ser.index[-1].date(),
                "latest_z": round(float(latest_val), 2),
                "z_vs_1y": round(float(z), 2),
                "alert": "YES" if abs(z) >= z_threshold else "",
            }
        )
    return pd.DataFrame(rows)


def upcoming_earnings(earnings: pd.DataFrame, within_days: int = 30) -> pd.DataFrame:
    """Next earnings date per ticker within `within_days` of today."""
    today = pd.Timestamp.now().normalize()
    fut = earnings[pd.to_datetime(earnings["earnings"]) >= today].copy()
    fut["earnings"] = pd.to_datetime(fut["earnings"])
    nxt = fut.sort_values("earnings").groupby("ticker").first().reset_index()
    nxt["days_out"] = (nxt["earnings"] - today).dt.days
    return nxt[nxt["days_out"] <= within_days].sort_values("days_out")

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
from scipy import stats

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
    r_levels: float  # corr of YoY levels — inflated by shared (co-trending) recovery
    r_mom_growth: float  # corr of raw MoM growth — inflated by shared seasonality
    r_deseason: float  # corr of DESEASONALIZED MoM growth — the honest headline read
    r_deseason_p: float  # p-value of the deseasonalized-MoM correlation
    r_deseason_n: int  # sample size for the deseasonalized-MoM correlation
    r_diff_yoy: float  # corr of differenced YoY — strictest (noisy) change-on-change measure
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


def _deseason(s: pd.Series) -> pd.Series:
    """MoM growth with the calendar-month seasonal mean removed."""
    g = s.pct_change()
    months = pd.PeriodIndex(g.index).month
    return g.groupby(months).transform(lambda x: x - x.mean())


def _robust_corrs(tsa_m, dem_m, tsa_y, dem_y) -> tuple[float, float, float, int, float]:
    """Co-movement on CHANGES, not co-trending levels.

    Returns (raw-MoM r, deseasonalized-MoM r, deseasonalized-MoM p, deseason n, differenced-YoY r).
    The deseasonalized read is the honest headline: both series peak in summer, so raw MoM
    is inflated by shared seasonality.
    """
    raw = pd.concat(
        [tsa_m.pct_change().rename("t"), dem_m.pct_change().rename("d")], axis=1
    ).dropna()
    ds = pd.concat([_deseason(tsa_m).rename("t"), _deseason(dem_m).rename("d")], axis=1).dropna()
    d = pd.concat([tsa_y.diff().rename("t"), dem_y.diff().rename("d")], axis=1).dropna()
    r_ds, p_ds = stats.pearsonr(ds["t"], ds["d"])
    return (
        float(raw["t"].corr(raw["d"])),
        float(r_ds),
        float(p_ds),
        int(len(ds)),
        float(d["t"].corr(d["d"])),
    )


def demand_nowcast(tsa_daily: pd.Series, accom_emp: pd.Series, max_lag: int = 6) -> Nowcast:
    """How closely TSA tracks the hotel-demand proxy (accommodation employment).

    Reports correlation three ways: YoY levels (co-trending, inflated), MoM growth, and
    differenced YoY (the honest change-on-change reads). Positive lag = TSA leads.
    """
    tsa_m = _to_period(tsa_daily.resample("ME").mean())
    dem_m = _to_period(accom_emp.resample("ME").last())
    tsa_y, dem_y = yoy(tsa_m, 12).dropna(), yoy(dem_m, 12).dropna()
    table = _lag_correlation_table(tsa_y, dem_y, max_lag)
    r_by_lag = table["r"]
    best = int(r_by_lag.abs().idxmax())
    r_mom, r_ds, p_ds, n_ds, r_dy = _robust_corrs(tsa_m, dem_m, tsa_y, dem_y)
    return Nowcast(
        r_levels=float(r_by_lag.loc[0]),
        r_mom_growth=r_mom,
        r_deseason=r_ds,
        r_deseason_p=p_ds,
        r_deseason_n=n_ds,
        r_diff_yoy=r_dy,
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
    hit_ci: tuple[float, float]  # Wilson 95% CI on signal_on_hit, computed on EFFECTIVE N
    n_eff: int  # effective N = distinct signal-on months (name-months are correlated)


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(center - half), float(center + half))


def pooled_validation(universe_prices: pd.DataFrame, tsa_daily: pd.Series) -> PooledValidation:
    """Out-of-sample-ish check: pool the TSA-acceleration -> 1m-forward-return relationship
    across the lodging universe. Name-months within a month are cross-sectionally correlated,
    so the Wilson CI is optimistic — effective N is closer to the # of distinct signal-on months."""
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
    hit = float((on["r"] > 0).mean())
    n_eff = int(on.index.nunique())  # distinct signal-on months (the independent unit)
    return PooledValidation(
        pooled_r=float(D["s"].corr(D["r"])),
        signal_on_hit=hit,
        signal_on_mean=float(on["r"].mean()),
        baseline_hit=float((D["r"] > 0).mean()),
        baseline_mean=float(D["r"].mean()),
        n_obs=int(len(on)),
        hit_ci=_wilson_ci(hit, n_eff),  # CI on effective N — the honest (wider) width
        n_eff=n_eff,
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


# ----------------------------------------------------------------------------- significance
@dataclass
class Significance:
    n_positions: int
    n_months: int
    naive_t: float  # per-position t-test vs 0 (overstates: ignores clustering)
    naive_p: float
    clustered_t: float  # 1 obs per signal-on month (the honest test)
    clustered_p: float
    gate_on_mean: float  # mean fwd return of the 3 names in gate-ON months
    gate_off_mean: float
    gate_t: float  # gate-ON vs gate-OFF (does the timing gate matter?)
    gate_p: float


def _headline_fwd_returns(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight 1-month forward return of the headline names, by month."""
    pm = _to_period(prices.resample("ME").last())
    fwd1 = pm[config.TICKERS].pct_change().shift(-1) * 100
    return fwd1.mean(axis=1)


def _gate_on_off_test(prices: pd.DataFrame, signals: Signals) -> tuple[float, float, float, float]:
    """Welch t-test of equal-weight forward return in gate-ON vs gate-OFF months."""
    fwd = _headline_fwd_returns(prices)
    gate = signals.monthly["gate"]
    on = fwd[gate.index[gate > 0]].dropna()
    off = fwd[gate.index[gate <= 0]].dropna()
    t, p = stats.ttest_ind(on, off, equal_var=False)
    return float(on.mean()), float(off.mean()), float(t), float(p)


def strategy_significance(
    backtest: Backtest, prices: pd.DataFrame, signals: Signals
) -> Significance:
    """Honest statistics on the strategy: naive vs month-clustered, plus the gate test."""
    tr = backtest.trades
    per_pos = tr["fwd_return_pct"].to_numpy()
    monthly = tr.groupby("rebalance")["fwd_return_pct"].mean().to_numpy()
    nt, npv = stats.ttest_1samp(per_pos, 0.0)
    ct, cpv = stats.ttest_1samp(monthly, 0.0)
    on_m, off_m, gt, gp = _gate_on_off_test(prices, signals)
    return Significance(
        n_positions=len(per_pos),
        n_months=len(monthly),
        naive_t=float(nt),
        naive_p=float(npv),
        clustered_t=float(ct),
        clustered_p=float(cpv),
        gate_on_mean=on_m,
        gate_off_mean=off_m,
        gate_t=gt,
        gate_p=gp,
    )


# ----------------------------------------------------------------------------- equity / risk
def _strategy_baseline_returns(
    backtest: Backtest, prices: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Aligned monthly returns: signal (long picks gate-ON, cash OFF) and always-long."""
    pm = _to_period(prices.resample("ME").last())
    base_m = pm[config.TICKERS].pct_change().shift(-1).mean(axis=1)
    idx = base_m.dropna().index
    strat_m = backtest.trades.groupby("rebalance")["fwd_return_pct"].mean() / 100.0
    strat_m.index = pd.PeriodIndex(strat_m.index, freq="M")
    strat = strat_m.reindex(idx).fillna(0.0)  # cash (0%) in gate-OFF months
    return strat, base_m.loc[idx].fillna(0.0)


def equity_curves(backtest: Backtest, prices: pd.DataFrame) -> pd.DataFrame:
    """Calendar-time cumulative growth of $1: signal (cash when OFF) vs always-long."""
    strat, base = _strategy_baseline_returns(backtest, prices)
    out = pd.DataFrame({"strategy": (1 + strat).cumprod(), "baseline": (1 + base).cumprod()})
    out.index = pd.PeriodIndex(out.index).to_timestamp()
    out.index.name = "date"
    return out


@dataclass
class RiskMetrics:
    table: pd.DataFrame  # rows: Signal / Always-long; cols: sharpe, max_dd, in_market, ...
    months: int
    deployed: dict = field(default_factory=dict)  # return-on-deployed-capital (invested months)


def _deployed_stats(strat: pd.Series) -> dict:
    """Return-on-deployed-capital: stats over only the invested (gate-ON) months.

    Sharpe carries a 95% CI (Lo 2002 i.i.d. SE) — small n makes the point estimate fragile.
    `ann_rate_deployed` is the per-invested-month rate annualized; it is NOT a realized
    annual return (capital is deployed only ~30% of the time).
    """
    dep = strat[strat != 0].to_numpy()
    n = len(dep)
    if n < 2:
        keys = ["mean_per_month", "ann_rate_deployed", "sharpe", "sharpe_lo", "sharpe_hi"]
        return dict.fromkeys(keys, float("nan")) | {"n_invested": n}
    sd = dep.std(ddof=1)
    sr_m = dep.mean() / sd if sd > 0 else float("nan")
    sr_a = sr_m * np.sqrt(12)
    se_a = np.sqrt((1 + 0.5 * sr_m**2) / n) * np.sqrt(12)  # SE of annualized Sharpe
    return {
        "n_invested": n,
        "mean_per_month": float(dep.mean()),
        "ann_rate_deployed": float((1 + dep.mean()) ** 12 - 1),
        "sharpe": float(sr_a),
        "sharpe_lo": float(sr_a - 1.96 * se_a),
        "sharpe_hi": float(sr_a + 1.96 * se_a),
    }


def _series_stats(r: pd.Series) -> dict:
    """Annualized risk/return stats for a monthly-return series (rf = 0)."""
    a = r.to_numpy()
    n = len(a)
    keys = ["total_growth", "ann_return", "ann_vol", "sharpe", "max_drawdown", "in_market"]
    if n == 0:
        return dict.fromkeys(keys, float("nan"))
    cum = np.cumprod(1 + a)
    vol = float(a.std(ddof=1) * np.sqrt(12))
    maxdd = float((cum / np.maximum.accumulate(cum) - 1).min())
    return {
        "total_growth": float(cum[-1]),
        "ann_return": float(cum[-1] ** (12 / n) - 1),
        "ann_vol": vol,
        "sharpe": float(a.mean() * 12 / vol) if vol > 0 else float("nan"),
        "max_drawdown": maxdd,
        "in_market": float((a != 0).mean()),
    }


def risk_metrics(backtest: Backtest, prices: pd.DataFrame) -> RiskMetrics:
    """Risk/exposure profile of the demand-gated overlay vs always-long buy-and-hold."""
    strat, base = _strategy_baseline_returns(backtest, prices)
    table = pd.DataFrame(
        {"Signal (demand-gated)": _series_stats(strat), "Always-long": _series_stats(base)}
    ).T
    return RiskMetrics(table=table, months=len(strat), deployed=_deployed_stats(strat))


@dataclass
class StressTest:
    risk: RiskMetrics
    equity: pd.DataFrame
    start: str


def stress_test(
    tsa_daily: pd.Series,
    trends: pd.DataFrame,
    fred_df: pd.DataFrame,
    prices: pd.DataFrame,
    start: str,
) -> StressTest:
    """Run the overlay over the full available history (incl. COVID) — the downturn test
    the 2022+ study window can't provide. Same logic, longer window."""
    signals = build_signals(tsa_daily, trends, fred_df)
    bt = backtest_top2(prices, signals)
    return StressTest(risk=risk_metrics(bt, prices), equity=equity_curves(bt, prices), start=start)


# ----------------------------------------------------------------------------- earnings study
@dataclass
class EarningsStudy:
    events: pd.DataFrame  # ticker, earnings, pre_search_z, reaction_pct
    corr: float  # corr(pre-earnings search z, post-earnings reaction)
    high_search_hit: float  # P(reaction > 0 | pre-search z > 0)
    n: int


def _presearch_z(search: pd.Series, date: pd.Timestamp, weeks: int) -> float | None:
    """Z-score of mean brand search in the `weeks` before earnings vs trailing year."""
    hist = search.dropna()
    hist = hist[hist.index < date]
    if len(hist) < 30:
        return None
    recent = hist.tail(weeks).mean()
    base = hist.tail(52)
    sd = base.std()
    if sd == 0 or np.isnan(sd):
        return None
    return float((recent - base.mean()) / sd)


def _reaction(px: pd.Series, date: pd.Timestamp, post_days: int) -> float | None:
    """Pct price change from the last close before earnings to `post_days` after."""
    s = px.dropna()
    before = s[s.index <= date]
    after = s[s.index > date]
    if before.empty or len(after) < post_days:
        return None
    return float((after.iloc[post_days - 1] / before.iloc[-1] - 1.0) * 100)


def _collect_events(
    earnings: pd.DataFrame,
    trends: pd.DataFrame,
    prices: pd.DataFrame,
    pre_weeks: int,
    post_days: int,
) -> pd.DataFrame:
    """One row per earnings date with pre-search z and the realized price reaction."""
    rows = []
    for _, e in earnings.iterrows():
        t = e["ticker"]
        d = pd.Timestamp(e["earnings"]).normalize()
        if t not in trends.columns or t not in prices.columns:
            continue
        z = _presearch_z(trends[t], d, pre_weeks)
        react = _reaction(prices[t], d, post_days)
        if z is None or react is None:
            continue
        rows.append(
            {
                "ticker": t,
                "earnings": d.date(),
                "pre_search_z": round(z, 2),
                "reaction_pct": round(react, 2),
            }
        )
    return pd.DataFrame(rows)


def earnings_search_study(
    earnings: pd.DataFrame,
    trends: pd.DataFrame,
    prices: pd.DataFrame,
    pre_weeks: int = 4,
    post_days: int = 3,
) -> EarningsStudy:
    """Event study: does elevated pre-earnings brand search precede the price reaction?"""
    ev = _collect_events(earnings, trends, prices, pre_weeks, post_days)
    if len(ev) < 3:
        return EarningsStudy(ev, float("nan"), float("nan"), len(ev))
    hi = ev[ev["pre_search_z"] > 0]
    return EarningsStudy(
        events=ev.sort_values("earnings"),
        corr=float(ev["pre_search_z"].corr(ev["reaction_pct"])),
        high_search_hit=float((hi["reaction_pct"] > 0).mean()) if len(hi) else float("nan"),
        n=len(ev),
    )

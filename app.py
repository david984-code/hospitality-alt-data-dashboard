"""Hospitality Alt-Data Dashboard.

TSA throughput + Google Trends brand search + BLS hospitality labor -> daily
demand signals and a pre-earnings monitoring tool for MAR / HLT / H.

Run:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src import pipeline

st.set_page_config(page_title="Hospitality Alt-Data Dashboard", layout="wide", page_icon=None)


@st.cache_data(ttl=3600, show_spinner="Fetching alt-data and computing signals...")
def load(force: bool = False) -> pipeline.PipelineResult:
    return pipeline.run(force=force)


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Hospitality Alt-Data")
st.sidebar.caption("TSA · Google Trends · BLS labor")
if st.sidebar.button("Refresh data now"):
    load.clear()
    st.cache_data.clear()
res = load()
st.sidebar.metric("TSA data through", res.tsa.index.max().strftime("%Y-%m-%d"))
st.sidebar.metric("Names tracked", ", ".join(config.TICKERS))
st.sidebar.caption(f"Validation universe: {len(config.UNIVERSE)} lodging names")
st.sidebar.caption(
    "Proxies: BLS Accommodation employment = demand; BLS Job Openings (L&H) = the "
    "Indeed-postings analog; PPI Traveler Accommodation = RevPAR-rate. True RevPAR (STR) is paid."
)

# ----------------------------------------------------------------------------- header
st.title("Hospitality Alt-Data Dashboard")
st.markdown(
    "Forecasting lodging demand for **Marriott (MAR)**, **Hilton (HLT)**, **Hyatt (H)** "
    "from alternative data, ahead of quarterly earnings."
)

# ----------------------------------------------------------------------------- today's signal
m = res.signals.monthly
latest = m.iloc[-1]
gate_on = latest["gate"] > 0
brand_cols = [f"brand_{t}" for t in config.TICKERS if f"brand_{t}" in m.columns]
scores = latest[brand_cols].dropna()
scores.index = [c.replace("brand_", "") for c in scores.index]
top2 = list(scores.sort_values(ascending=False).head(2).index) if len(scores) >= 2 else []

st.subheader("Today's signal")
c1, c2, c3, c4 = st.columns(4)
c1.metric("TSA traveler volume YoY", f"{latest['tsa_yoy']:+.1f}%")
c2.metric(
    "TSA acceleration (MoM Δ of YoY)",
    f"{latest['tsa_accel']:+.2f}",
    help="The tradeable gate: positive = travel-demand growth is accelerating.",
)
c3.metric("Signal gate", "ON — risk-on" if gate_on else "OFF — stand aside")
c4.metric(
    "Long top-2 picks",
    ", ".join(top2) if gate_on and top2 else "—",
    help="When the gate is ON, go long the 2 names with the strongest brand-search momentum.",
)

if gate_on and top2:
    st.success(
        f"Gate ON: TSA demand is accelerating. Strategy is long **{top2[0]}** and **{top2[1]}** "
        "(top-2 by Google-Trends brand momentum), 1-month hold."
    )
else:
    st.info("Gate OFF: TSA demand growth is not accelerating. Strategy stands aside.")

st.divider()

# ----------------------------------------------------------------------------- nowcast
left, right = st.columns([3, 2])
with left:
    st.subheader("TSA as a hospitality demand nowcast")
    nc = res.nowcast
    tsa_y = nc.tsa_yoy.copy()
    tsa_y.index = pd.PeriodIndex(nc.tsa_yoy.index).to_timestamp()
    dem_y = nc.demand_yoy.copy()
    dem_y.index = pd.PeriodIndex(nc.demand_yoy.index).to_timestamp()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=tsa_y.index, y=tsa_y.values, name="TSA volume YoY %", line=dict(width=2))
    )
    fig.add_trace(
        go.Scatter(
            x=dem_y.index,
            y=dem_y.values,
            name="Accommodation employment YoY %",
            line=dict(width=2, dash="dot"),
        )
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.1),
        yaxis_title="YoY %",
    )
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Lead-lag")
    st.metric(
        "Coincident correlation r",
        f"{res.nowcast.r_coincident:.2f}",
        help="TSA YoY vs Accommodation-employment YoY (demand proxy).",
    )
    st.caption(
        f"TSA traveler volume tracks hospitality demand almost one-for-one "
        f"(r = {res.nowcast.r_coincident:.2f}, contemporaneous). It is a strong real-time "
        "*nowcast* of sector demand. The cross-correlation by lag:"
    )
    st.dataframe(res.nowcast.table.round(3), width="stretch")

st.divider()

# ----------------------------------------------------------------------------- strategy
st.subheader("Strategy backtest — long top-2 on a positive signal")
bt = res.backtest
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Hit rate", f"{bt.hit_rate:.0%}", delta=f"{(bt.hit_rate - bt.baseline_hit):+.0%} vs baseline"
)
k2.metric(
    "Mean return / position",
    f"{bt.mean_return:+.1f}%",
    delta=f"{(bt.mean_return - bt.baseline_mean):+.1f}% vs baseline",
)
k3.metric("Positions", f"{bt.n_trades}", help=f"across {bt.n_rebalances} monthly rebalances")
k4.metric("Baseline (always-long)", f"{bt.baseline_hit:.0%}  /  {bt.baseline_mean:+.1f}%")

sc1, sc2 = st.columns([2, 3])
with sc1:
    if not bt.equity_curve.empty:
        eqfig = go.Figure()
        eqfig.add_trace(
            go.Scatter(
                x=bt.equity_curve.index, y=bt.equity_curve.values, name="Strategy", fill="tozeroy"
            )
        )
        eqfig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Cumulative growth of $1 (per-position, compounded)",
            xaxis_title="position #",
        )
        st.plotly_chart(eqfig, width="stretch")
with sc2:
    st.markdown("**Out-of-sample validation** (pooled across the 10-name lodging universe)")
    v = res.validation
    v1, v2, v3 = st.columns(3)
    v1.metric("Pooled signal r", f"{v.pooled_r:+.2f}")
    v2.metric(
        "Signal-on hit rate",
        f"{v.signal_on_hit:.0%}",
        delta=f"{(v.signal_on_hit - v.baseline_hit):+.0%}",
    )
    v3.metric("Observations", f"{v.n_obs}")
    st.caption(
        "The 3-name edge holds across a broader lodging universe (IHG, WH, CHH, and hotel "
        "REITs HST/PK/RHP/APLE), which argues it is not overfit to MAR/HLT/H. "
        "Caveat: cross-sectional observations within a month are correlated, so effective "
        "sample size is closer to the number of distinct signal-on months."
    )

with st.expander("Show all backtest positions"):
    st.dataframe(bt.trades, width="stretch", hide_index=True)

st.divider()

# ----------------------------------------------------------------------------- alerts
a1, a2 = st.columns(2)
with a1:
    st.subheader("Pre-earnings anomaly alerts")
    anom = res.anomalies.copy()
    st.dataframe(anom, width="stretch", hide_index=True)
    flagged = anom[anom["alert"] == "YES"] if "alert" in anom else pd.DataFrame()
    if not flagged.empty:
        for _, r in flagged.iterrows():
            st.warning(f"Anomaly: **{r['signal']}** at z = {r['z_vs_1y']} vs its 1-year range.")
    else:
        st.caption("No signal is currently more than 1.5σ from its trailing-year norm.")
with a2:
    st.subheader("Upcoming earnings (next 30 days)")
    if res.upcoming.empty:
        st.caption("No MAR/HLT/H earnings within the next 30 days.")
    else:
        up = res.upcoming.copy()
        up["earnings"] = pd.to_datetime(up["earnings"]).dt.strftime("%Y-%m-%d")
        st.dataframe(up[["ticker", "earnings", "days_out"]], width="stretch", hide_index=True)
    st.caption(
        "Watch the anomaly panel as these dates approach — a brand-search or travel-demand "
        "spike ahead of a print is the alert this dashboard is built to surface."
    )

st.divider()
st.caption(
    f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. Data: TSA.gov (passenger volumes), "
    "Google Trends (pytrends), BLS public API (JOLTS / CES / PPI), Yahoo Finance. "
    "Research tool only — not investment advice."
)

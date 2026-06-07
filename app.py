"""Hospitality Alt-Data Dashboard.

A real-time nowcast of US lodging demand built from public alternative data
(TSA throughput + Google Trends brand search + BLS hospitality labor), with an
exploratory pre-earnings timing signal for MAR / HLT / H.

Run:  uv run streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src import analysis, pipeline

st.set_page_config(page_title="Hospitality Alt-Data Dashboard", layout="wide", page_icon=None)


@st.cache_data(ttl=3600, show_spinner="Fetching alt-data and computing signals...")
def load(force: bool = False) -> pipeline.PipelineResult:
    return pipeline.run(force=force)


def pval_label(p: float, alpha: float = 0.05) -> str:
    if p != p:  # NaN
        return "n/a"
    verdict = "significant" if p < alpha else "not significant"
    return f"p = {p:.3f} ({verdict})"


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Hospitality Alt-Data")
st.sidebar.caption("TSA · Google Trends · BLS labor")
if st.sidebar.button("Refresh data now"):
    load.clear()
    st.cache_data.clear()
res = load()
st.sidebar.metric("TSA data through", res.tsa.index.max().strftime("%Y-%m-%d"))
st.sidebar.metric("Names tracked", ", ".join(config.TICKERS))
st.sidebar.caption(
    f"Validation universe: {len(config.FRANCHISORS)} franchisors + {len(config.REITS)} REITs"
)
with st.sidebar.expander("Methodology & data proxies"):
    st.markdown(
        "True RevPAR (revenue per available room) is paid **STR** data. This project "
        "reconstructs its *shape* from free public proxies:\n\n"
        "- **TSA throughput** → travel demand (daily, ~1–2 day lag)\n"
        "- **BLS Accommodation employment** → occupancy / demand (hotels staff to "
        "expected occupancy)\n"
        "- **BLS Job Openings, L&H** → forward hiring intent (the 'Indeed postings' "
        "analog; Indeed killed its public API)\n"
        "- **PPI Traveler Accommodation** → room-rate / ADR\n"
        "- **Google Trends** → per-brand booking intent\n\n"
        "Identity: **RevPAR = ADR × Occupancy**. This is a defensible RevPAR-*proxy* "
        "nowcast, not the paid number."
    )

# ----------------------------------------------------------------------------- header
st.title("Hospitality Alt-Data Dashboard")
st.markdown(
    "A real-time **nowcast of US lodging demand** from public alternative data, plus a "
    "demand-gated **risk overlay** across the major lodging franchisor brands "
    f"({', '.join(config.TICKERS)})."
)

# ----------------------------------------------------------------------------- 1. NOWCAST (lead)
st.header("1 · Travel demand nowcast")
left, right = st.columns([3, 2])
with left:
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
    st.metric(
        "Coincident correlation r",
        f"{nc.r_coincident:.2f}",
        help="TSA YoY vs BLS Accommodation-employment YoY (demand proxy).",
    )
    st.caption(
        f"TSA traveler volume tracks hospitality demand almost one-for-one "
        f"(r = {nc.r_coincident:.2f}, contemporaneous; best lag = {nc.best_lag_months}m). "
        "It doesn't *lead* the fundamentals — its value is **timeliness**: TSA prints in "
        "1–2 days, vs weeks for BLS and a quarter for company earnings. Lag table:"
    )
    st.dataframe(nc.table.round(3), width="stretch")

st.divider()

# ----------------------------------------------------------------------------- 2. TODAY'S SIGNAL
m = res.signals.monthly
latest = m.iloc[-1]
gate_on = latest["gate"] > 0
brand_cols = [f"brand_{t}" for t in config.TICKERS if f"brand_{t}" in m.columns]
scores = latest[brand_cols].dropna()
scores.index = [c.replace("brand_", "") for c in scores.index]
top2 = list(scores.sort_values(ascending=False).head(2).index) if len(scores) >= 2 else []

st.header("2 · Today's signal")
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
        f"Gate ON: TSA demand is accelerating. Exploratory signal is long **{top2[0]}** and "
        f"**{top2[1]}** (top-2 by Google-Trends brand momentum), 1-month hold."
    )
else:
    st.info("Gate OFF: TSA demand growth is not accelerating. Signal stands aside (holds cash).")

st.divider()

# ----------------------------------------------------------------------------- 3. RISK OVERLAY
bt = res.backtest
sig = res.significance
rt = res.risk.table
s_row = rt.loc["Signal (demand-gated)"]
b_row = rt.loc["Always-long"]

st.header("3 · Demand-gated risk overlay")
st.markdown(
    "Be long the franchisor brands **only when travel demand is accelerating**; hold cash "
    "otherwise. The pitch is *risk management*, not alpha: matching equity-like risk-adjusted "
    "returns with a fraction of the exposure and drawdown."
)

r1, r2, r3, r4 = st.columns(4)
r1.metric(
    "Sharpe (overlay)",
    f"{s_row['sharpe']:.2f}",
    delta=f"{(s_row['sharpe'] - b_row['sharpe']):+.2f} vs always-long",
    help="Annualized, rf = 0. Comparable Sharpe with far less time exposed.",
)
r2.metric(
    "Max drawdown",
    f"{s_row['max_drawdown']:.0%}",
    delta=f"{(s_row['max_drawdown'] - b_row['max_drawdown']):+.0%} vs always-long",
    help="Worst peak-to-trough. Smaller (less negative) is better.",
)
r3.metric("Time in market", f"{s_row['in_market']:.0%}", help="% of months actually invested.")
r4.metric(
    "Total growth",
    f"{s_row['total_growth']:.2f}x",
    delta=f"vs {b_row['total_growth']:.2f}x always-long",
    delta_color="off",
)

disp = pd.DataFrame(
    {
        "Sharpe": rt["sharpe"].round(2),
        "Ann. return": (rt["ann_return"] * 100).round(1).astype(str) + "%",
        "Ann. vol": (rt["ann_vol"] * 100).round(1).astype(str) + "%",
        "Max drawdown": (rt["max_drawdown"] * 100).round(1).astype(str) + "%",
        "Time in mkt": (rt["in_market"] * 100).round(0).astype(int).astype(str) + "%",
        "Growth": rt["total_growth"].round(2).astype(str) + "x",
    }
)
st.dataframe(disp, width="stretch")
st.caption(
    "The study window above (2022+) is deliberately post-COVID to avoid 2020-21 YoY "
    "base-effect distortions, so it is all bull market and total return trails buy-and-hold "
    "from cash drag. The **COVID stress test** below extends to 2019 to put the drawdown "
    "protection against a real crash."
)

# --- COVID stress test (full 2019+ history) ---
if res.stress is not None:
    srt = res.stress.risk.table
    ss = srt.loc["Signal (demand-gated)"]
    sb = srt.loc["Always-long"]
    st.markdown(f"**🦠 COVID stress test** — overlay over full history from {res.stress.start}")
    x1, x2, x3 = st.columns(3)
    x1.metric(
        "Overlay max drawdown",
        f"{ss['max_drawdown']:.0%}",
        delta=f"{(ss['max_drawdown'] - sb['max_drawdown']):+.0%} vs always-long",
    )
    x2.metric("Always-long max drawdown", f"{sb['max_drawdown']:.0%}")
    x3.metric(
        "Overlay Sharpe",
        f"{ss['sharpe']:.2f}",
        delta=f"{(ss['sharpe'] - sb['sharpe']):+.2f} vs always-long",
    )
    seq = res.stress.equity
    if not seq.empty:
        sfig = go.Figure()
        sfig.add_trace(go.Scatter(x=seq.index, y=seq["strategy"], name="Overlay (cash when OFF)"))
        sfig.add_trace(
            go.Scatter(x=seq.index, y=seq["baseline"], name="Always-long", line=dict(dash="dot"))
        )
        sfig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        )
        st.plotly_chart(sfig, width="stretch")
    st.caption(
        f"Across 2019–present — including the COVID crash — the demand gate cut the worst "
        f"drawdown from {sb['max_drawdown']:.0%} to {ss['max_drawdown']:.0%} by going to cash "
        "as travel collapsed. This is the downturn evidence the 2022+ window can't provide. "
        "Caveat: COVID is a single event and 2020-21 YoY math is noisy from base effects."
    )

st.divider()
ec1, ec2 = st.columns([3, 2])
with ec1:
    eq = res.equity
    if not eq.empty:
        st.markdown("**Cumulative growth of \\$1 (calendar time)**")
        eqfig = go.Figure()
        eqfig.add_trace(go.Scatter(x=eq.index, y=eq["strategy"], name="Overlay (cash when OFF)"))
        eqfig.add_trace(
            go.Scatter(x=eq.index, y=eq["baseline"], name="Always-long", line=dict(dash="dot"))
        )
        eqfig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        )
        st.plotly_chart(eqfig, width="stretch")
with ec2:
    st.markdown("**Is the timing itself real?** (small-sample significance)")
    st.caption(
        f"Naive per-position {pval_label(sig.naive_p)} overstates it — the names held in a "
        f"given month move together. Clustered by month: **{pval_label(sig.clustered_p)}**; "
        f"gate-ON vs gate-OFF **{pval_label(sig.gate_p)}** "
        f"({sig.gate_on_mean:+.1f}% vs {sig.gate_off_mean:+.1f}%/mo). "
        f"Over only {sig.n_months} signal-on months the edge is borderline — which is exactly "
        "why this is framed as a risk overlay, not an alpha claim."
    )

st.markdown("**Out-of-sample validation by business model**")
v1, v2 = st.columns(2)
fr_cols = [t for t in config.FRANCHISORS if t in res.universe_prices.columns]
reit_cols = [t for t in config.REITS if t in res.universe_prices.columns]
vf = analysis.pooled_validation(res.universe_prices[fr_cols], res.tsa)
vr = analysis.pooled_validation(res.universe_prices[reit_cols], res.tsa)
v1.metric(
    f"Franchisors ({len(fr_cols)})",
    f"{vf.signal_on_hit:.0%} hit",
    delta=f"{(vf.signal_on_hit - vf.baseline_hit):+.0%} vs base | r={vf.pooled_r:+.2f}",
    help="Asset-light brands: MAR/HLT/H/WH/CHH/IHG + timeshares HGV/VAC/TNL.",
)
v2.metric(
    f"Hotel REITs ({len(reit_cols)})",
    f"{vr.signal_on_hit:.0%} hit",
    delta=f"{(vr.signal_on_hit - vr.baseline_hit):+.0%} vs base | r={vr.pooled_r:+.2f}",
    help="Own the real estate: HST/PK/RHP/APLE/DRH/PEB/SHO/XHR/RLJ/INN.",
)
st.caption(
    "The TSA-acceleration → next-month-return effect shows up in **both** business models "
    "(franchisors and REITs), on names the signal was never tuned on — which argues it's a "
    "real sector-demand effect, not overfitting to the traded names. Splitting the two "
    "buckets matters because franchisors are fee/growth stories while REITs are "
    "rate/RevPAR stories."
)

with st.expander("Show all backtest positions"):
    st.dataframe(bt.trades, width="stretch", hide_index=True)

st.divider()

# ----------------------------------------------------------------------------- 4. EARNINGS SEARCH
st.header("4 · Do pre-earnings searches predict the print?")
study = res.earnings_study
if study.n < 3:
    st.info("Not enough earnings events with overlapping search history yet.")
else:
    e1, e2 = st.columns([3, 2])
    with e1:
        ev = study.events
        scat = go.Figure()
        scat.add_trace(
            go.Scatter(
                x=ev["pre_search_z"],
                y=ev["reaction_pct"],
                mode="markers",
                text=ev["ticker"] + " " + ev["earnings"].astype(str),
                marker=dict(size=9),
            )
        )
        scat.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Pre-earnings brand-search z vs price reaction",
            xaxis_title="Pre-earnings search z-score (4wk vs trailing year)",
            yaxis_title=f"Reaction % (close before → {3}d after)",
        )
        scat.add_hline(y=0, line_dash="dot", line_color="gray")
        scat.add_vline(x=0, line_dash="dot", line_color="gray")
        st.plotly_chart(scat, width="stretch")
    with e2:
        st.metric("Correlation", f"{study.corr:+.2f}", help="search z vs earnings reaction")
        st.metric("Up-reaction | high search", f"{study.high_search_hit:.0%}")
        st.metric("Events", f"{study.n}")
        st.caption(
            "Tests the premise behind the anomaly alerts: does a brand-search spike before a "
            f"print foreshadow the reaction? Across {study.n} events the relationship is "
            "weak — honestly, search intensity is not (yet) a reliable earnings predictor "
            "on this sample. Reported rather than hidden."
        )

st.divider()

# ----------------------------------------------------------------------------- 5. ALERTS
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
        st.caption(f"No earnings within the next 30 days for {', '.join(config.TICKERS)}.")
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
    "Research / monitoring tool only — not investment advice."
)

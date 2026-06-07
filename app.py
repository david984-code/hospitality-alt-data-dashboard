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
    "A real-time **nowcast of US lodging demand** from public alternative data, with an "
    "exploratory pre-earnings timing signal for **Marriott (MAR)**, **Hilton (HLT)**, "
    "**Hyatt (H)**."
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

# ----------------------------------------------------------------------------- 3. STRATEGY (exploratory)
bt = res.backtest
sig = res.significance
st.header("3 · Exploratory timing signal")
st.caption(
    "⚠️ Small sample — treat as a research hypothesis, not a track record. "
    f"Only {sig.n_months} months had the gate ON over a {len(m)}-month window."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Hit rate", f"{bt.hit_rate:.0%}", delta=f"{(bt.hit_rate - bt.baseline_hit):+.0%} vs baseline"
)
k2.metric(
    "Mean return / position",
    f"{bt.mean_return:+.1f}%",
    delta=f"{(bt.mean_return - bt.baseline_mean):+.1f}% vs baseline",
)
k3.metric("Signal-on months", f"{sig.n_months}", help=f"{bt.n_trades} positions (2 per month)")
k4.metric("Baseline (always-long)", f"{bt.baseline_hit:.0%}  /  {bt.baseline_mean:+.1f}%")

st.markdown("**Statistical significance** (does the edge survive honest testing?)")
s1, s2, s3 = st.columns(3)
s1.metric(
    "Per-position test",
    pval_label(sig.naive_p),
    help="Naive — treats 32 positions as independent. Overstated.",
)
s2.metric(
    "Clustered by month",
    pval_label(sig.clustered_p),
    help="1 observation per signal-on month — the honest test.",
)
s3.metric(
    "Gate ON vs OFF",
    pval_label(sig.gate_p),
    delta=f"{sig.gate_on_mean:+.1f}% vs {sig.gate_off_mean:+.1f}%/mo",
)
st.caption(
    "The naive per-position p-value looks strong, but the 2 names held in a given month move "
    "together, so the effective sample is ~the number of signal-on months, not the position "
    "count. Clustered by month, the edge is **borderline** and does not clear the bar versus "
    "simply owning hotels in this post-COVID bull market."
)

ec1, ec2 = st.columns([3, 2])
with ec1:
    eq = res.equity
    if not eq.empty:
        eqfig = go.Figure()
        eqfig.add_trace(go.Scatter(x=eq.index, y=eq["strategy"], name="Signal (cash when OFF)"))
        eqfig.add_trace(
            go.Scatter(
                x=eq.index, y=eq["baseline"], name="Always-long MAR/HLT/H", line=dict(dash="dot")
            )
        )
        eqfig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Cumulative growth of $1 (calendar time)",
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(eqfig, width="stretch")
with ec2:
    st.markdown("**Honest read of the curve**")
    st.caption(
        "The signal sits in cash ~60% of the time, so in a one-way bull market it **trails** "
        "buy-and-hold on total return — cash drag. Its value is *per-invested-month quality* "
        "(higher hit rate and mean return when it does hold), i.e. a risk-reduction overlay, "
        "not a return maximizer. Showing this is the point: the data doesn't support an "
        "alpha claim, and the dashboard says so."
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
    "real sector-demand effect, not overfitting to the three headline tickers. Splitting the "
    "two buckets matters because franchisors are fee/growth stories while REITs are "
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
    "Research / monitoring tool only — not investment advice."
)

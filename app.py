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
    "**A structured, reproducible read on US lodging demand from free alternative data — a "
    f"name-level monitoring tool for the hotel franchisors ({', '.join(config.TICKERS)}) between "
    "earnings.**"
)
st.caption(
    "TSA throughput (primary) + BLS hospitality labor + Google Trends. Not an information edge "
    "(TSA is widely watched) — a reproducible read. The systematic overlay in §3 is a "
    "risk-management study, not the headline. Research / monitoring tool, not investment advice."
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
    p_txt = "p < 0.001" if nc.r_deseason_p < 0.001 else f"p = {nc.r_deseason_p:.3f}"
    st.metric(
        "TSA–demand co-movement (deseasonalized)",
        f"r = {nc.r_deseason:.2f}",
        delta=f"n = {nc.r_deseason_n}, {p_txt}",
        delta_color="off",
        help="MoM growth with the calendar-month seasonal mean removed — the honest read.",
    )
    s1, s2, s3 = st.columns(3)
    s1.metric("Raw MoM", f"{nc.r_mom_growth:.2f}", help="Inflated by shared summer seasonality.")
    s2.metric("Levels YoY", f"{nc.r_levels:.2f}", help="Inflated — both co-trend out of COVID.")
    s3.metric("Diff'd YoY", f"{nc.r_diff_yoy:.2f}", help="Strictest but noisy; not significant.")
    st.caption(
        f"Both travel and hotel staffing peak in summer, so the honest read **strips seasonality**: "
        f"deseasonalized monthly co-movement **r = {nc.r_deseason:.2f}** ({p_txt}, n = "
        f"{nc.r_deseason_n}) — moderate. For contrast, raw MoM {nc.r_mom_growth:.2f} is seasonally "
        f"inflated, levels-of-YoY {nc.r_levels:.2f} is co-trend inflated, and differenced-YoY "
        f"{nc.r_diff_yoy:.2f} is noisier (not significant). The economic link is real; TSA's value "
        "is **timeliness** (1–2 day lag vs weeks/quarter), not a tight monthly fit."
    )
    with st.expander("Lead-lag table"):
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

dep = res.risk.deployed
st.header("3 · Risk-management study (secondary): demand-gated overlay")
st.markdown(
    "A study, not the headline. Be long the franchisor brands **only when travel demand is "
    "accelerating**, else hold cash — *risk management, not alpha*; the value is the **timing "
    "gate** (brand selection is secondary). Signal→execution: month-*t* TSA (known within ~2 "
    "days of month-end) sets exposure for month *t+1* — no look-ahead."
)

r1, r2, r3, r4 = st.columns(4)
r1.metric(
    "Return / invested month",
    f"{dep['mean_per_month']:+.1%}",
    help=(
        f"Mean over the {dep['n_invested']} gate-ON months only. Annualized per-invested-month "
        f"rate ≈ {dep['ann_rate_deployed']:.0%}, but that is NOT realized (deployed only "
        f"~{s_row['in_market']:.0%} of the time)."
    ),
)
r2.metric(
    "Deployed Sharpe",
    f"{dep['sharpe']:.1f}",
    delta=f"95% CI [{dep['sharpe_lo']:.1f}, {dep['sharpe_hi']:.1f}]",
    delta_color="off",
    help="Annualized, invested months only. Wide CI from small n — read as indicative, not precise.",
)
r3.metric(
    "Max drawdown",
    f"{s_row['max_drawdown']:.0%}",
    delta=f"{(s_row['max_drawdown'] - b_row['max_drawdown']):+.0%} vs always-long",
    help="Worst peak-to-trough (study window). Smaller (less negative) is better.",
)
r4.metric(
    "Time in market",
    f"{s_row['in_market']:.0%}",
    help="% of months invested; the rest in cash (the source of cash drag on total return).",
)
st.caption(
    f"Deployed-capital stats are conditional on the gate being ON ({dep['n_invested']} months) and "
    f"**small-sample**: the Sharpe's 95% CI is wide — **[{dep['sharpe_lo']:.1f}, {dep['sharpe_hi']:.1f}]** "
    "— so read it as indicative, not precise. The annualized rate is per-invested-month, **not "
    f"realized**: realized total-capital return ≈ {s_row['ann_return']:.0%}/yr, *below* always-long's "
    f"{b_row['ann_return']:.0%} (cash drag). Study window 2022+ is all bull market — the real "
    "downside test is the crash:"
)

# --- COVID stress test (full 2019+ history) — the headline proof ---
if res.stress is not None:
    srt = res.stress.risk.table
    ss = srt.loc["Signal (demand-gated)"]
    sb = srt.loc["Always-long"]
    st.markdown(f"#### 🦠 COVID stress test — full history from {res.stress.start}")
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
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        )
        st.plotly_chart(sfig, width="stretch")
    st.caption(
        f"Through COVID the **parameter-free** gate (fixed accel > 0, the same rule used "
        f"everywhere — not COVID-tuned) held the worst drawdown to {ss['max_drawdown']:.0%} vs "
        f"{sb['max_drawdown']:.0%}. **No look-ahead:** it was already OFF for Mar/Apr-2020 off "
        "*February's* decelerating TSA (accel ≈ −3.1), so it sat out the −34% crash on month-*t* "
        "data only. **Read it modestly:** going to cash when acceleration craters is near-mechanical "
        "and this is a single crash — it shows the gate fires *sensibly*, not predictive skill. "
        "(Overlay's own worst drawdown is Aug-2022, not COVID.)"
    )

st.caption(
    f"**Does the gate actually help?** The test that matters — gate-ON vs gate-OFF months — is "
    f"{pval_label(sig.gate_p)}: it **does not clear the 5% bar** ({sig.gate_on_mean:+.1f}% vs "
    f"{sig.gate_off_mean:+.1f}%/mo over {sig.n_months} signal-on months). The per-position "
    f"p≈{sig.naive_p:.2f} overstates it (positions within a month are correlated). The gate "
    "*threshold* is parameter-free, but choosing YoY-acceleration as the gate and top-2 brand "
    f"momentum as sizing are researcher choices — so on ~{sig.n_months} months treat p≈{sig.gate_p:.2f} "
    "as **'fails at 5%, likely worse out-of-sample'**, not a near-miss."
)

st.markdown("**Out-of-sample validation** — signal on names it was never tuned on")
v1, v2 = st.columns(2)
fr_cols = [t for t in config.FRANCHISORS if t in res.universe_prices.columns]
reit_cols = [t for t in config.REITS if t in res.universe_prices.columns]
vf = analysis.pooled_validation(res.universe_prices[fr_cols], res.tsa)
vr = analysis.pooled_validation(res.universe_prices[reit_cols], res.tsa)
v1.metric(
    f"Franchisors ({len(fr_cols)})",
    f"{vf.signal_on_hit:.0%} up",
    delta=f"{(vf.signal_on_hit - vf.baseline_hit):+.0%} vs base rate",
    help="Asset-light brands incl. MAR/HLT/H/WH/CHH/IHG + timeshares HGV/VAC/TNL.",
)
v2.metric(
    f"Hotel REITs ({len(reit_cols)})",
    f"{vr.signal_on_hit:.0%} up",
    delta=f"{(vr.signal_on_hit - vr.baseline_hit):+.0%} vs base rate",
    help="Own the real estate: HST/PK/RHP/APLE/DRH/PEB/SHO/XHR/RLJ/INN.",
)
st.caption(
    "**Metric:** hit rate = P(next-month total return > 0 | gate ON), pooled across each bucket's "
    "name-months, vs the unconditional base rate. The effect points the right way in **both** "
    "business models (franchisors are fee/growth, REITs rate/RevPAR), but the pooled linear "
    f"correlation is near-noise (r≈{vf.pooled_r:+.2f}/{vr.pooled_r:+.2f}) — directional, not linear."
)

with st.expander("📅 When was the gate ON? — signal history & realized returns", expanded=True):
    hist = (
        bt.trades.groupby("rebalance")
        .agg(picks=("ticker", lambda s: ", ".join(s)), avg_return_pct=("fwd_return_pct", "mean"))
        .reset_index()
        .sort_values("rebalance", ascending=False)
    )
    hist["avg_return_pct"] = hist["avg_return_pct"].round(2)
    hist["result"] = ["up" if v > 0 else "down" for v in hist["avg_return_pct"]]
    hist = hist.rename(columns={"rebalance": "month", "picks": "long picks"})
    st.caption(
        f"The gate was ON in {len(hist)} months. Each row: the 2 names held and the realized "
        "next-month return. Months not listed = gate OFF (in cash)."
    )
    st.dataframe(hist, width="stretch", hide_index=True)

with st.expander("Details — full risk table, study-window curve, all positions"):
    st.dataframe(
        pd.DataFrame(
            {
                "Sharpe": rt["sharpe"].round(2),
                "Ann. return": (rt["ann_return"] * 100).round(1).astype(str) + "%",
                "Ann. vol": (rt["ann_vol"] * 100).round(1).astype(str) + "%",
                "Max drawdown": (rt["max_drawdown"] * 100).round(1).astype(str) + "%",
                "Time in mkt": (rt["in_market"] * 100).round(0).astype(int).astype(str) + "%",
                "Growth": rt["total_growth"].round(2).astype(str) + "x",
            }
        ),
        width="stretch",
    )
    eq = res.equity
    if not eq.empty:
        st.caption("Study-window (2022+) cumulative growth — note the cash drag vs always-long.")
        eqfig = go.Figure()
        eqfig.add_trace(go.Scatter(x=eq.index, y=eq["strategy"], name="Overlay (cash when OFF)"))
        eqfig.add_trace(
            go.Scatter(x=eq.index, y=eq["baseline"], name="Always-long", line=dict(dash="dot"))
        )
        eqfig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        )
        st.plotly_chart(eqfig, width="stretch")
    st.dataframe(bt.trades, width="stretch", hide_index=True)

st.divider()

# ----------------------------------------------------------------------------- 4. EARNINGS SEARCH (null)
study = res.earnings_study
st.subheader("Bonus — do pre-earnings searches predict the print?")
if study.n < 3:
    st.caption("Not enough earnings events with overlapping search history yet.")
else:
    st.caption(
        f"Tested the anomaly-alert premise across {study.n} earnings events: correlation "
        f"{study.corr:+.2f} (weak). An honest **null result** — search intensity is not a "
        "reliable earnings predictor on this sample. Scatter in the expander."
    )
    with st.expander("Earnings-search event study (null result)"):
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
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Pre-earnings search z-score (4wk vs trailing year)",
            yaxis_title="Reaction % (close before → 3d after)",
        )
        scat.add_hline(y=0, line_dash="dot", line_color="gray")
        scat.add_vline(x=0, line_dash="dot", line_color="gray")
        st.plotly_chart(scat, width="stretch")

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

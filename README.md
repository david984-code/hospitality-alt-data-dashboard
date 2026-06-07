# Hospitality Alt-Data Dashboard

An alternative-data pipeline that aggregates **TSA checkpoint throughput**, **Google
Trends brand searches**, and **BLS hospitality labor data** to (1) **nowcast US lodging
demand** in near-real-time and (2) drive a demand-gated **risk overlay** across the major
lodging franchisor brands (MAR, HLT, H, WH, CHH, IHG). Served through a Streamlit
dashboard with pre-earnings anomaly alerts.

## Headline findings (computed on 2022-2026 data; as of 2026-06 snapshot)

| Result | Value |
|---|---|
| **Primary deliverable** — TSA volume as a demand nowcast (TSA YoY vs Accommodation-employment YoY, coincident) | **r = 0.92** |
| **Risk overlay** (study window, 2022+) — long top-2 franchisor brands when travel demand is accelerating, else cash | **Sharpe ~1.0 vs 0.73** always-long; **max drawdown −6% vs −23%**; invested only ~30% of months |
| **COVID stress test** (full history, 2019+) — same overlay through the crash | **max drawdown −11% vs −44%**; Sharpe 0.94 vs 0.66 — the gate went to cash as travel collapsed |
| Timing significance (clustered by month) | per-position p ≈ 0.005 (naive) → **clustered p ≈ 0.03**; gate-ON vs gate-OFF p ≈ 0.09 |
| Out-of-sample validation, pooled across a 20-name universe | franchisors 68% vs 58% (r = 0.12); REITs 63% vs 52% (r = 0.16) |

> Numbers are a nowcast and move with each data refresh; the figures above are the
> reproducible output of `uv run python -m src.pipeline` on the date noted.

**The honest takeaway:** the headline deliverable is the **r = 0.92 demand nowcast** — TSA
is a clean, timely (1–2 day lag) read on lodging demand. The trading angle is framed as a
**risk overlay, not alpha**: gating exposure on travel-demand *acceleration* delivered a
better Sharpe (~1.0 vs 0.73) and a fraction of the drawdown while invested only ~30% of the
time. The **COVID stress test** extends the overlay to 2019 (the limit of public TSA data)
and is the key downturn evidence: through the crash it held max drawdown to **−11% vs −44%**
for buy-and-hold. **Caveats kept loud:** COVID is a single event, 2020-21 YoY math is noisy
(why the headline study window starts in 2022), the timing edge is only borderline-significant
on ~16 months, and on total return the overlay trails buy-and-hold from cash drag. The
dashboard reports all of this rather than cherry-picking the flattering numbers.

## Data sources

| Signal | Source | Notes |
|---|---|---|
| TSA passenger throughput (daily) | TSA.gov passenger-volumes pages | Per-year archive URLs stitched into a 2022-present daily series. |
| Brand search interest (weekly) | Google Trends via `pytrends` | Marriott / Hilton / Hyatt / Wyndham / Choice / IHG, US, relative interest. |
| Job openings, Leisure & Hospitality (monthly) | BLS public API (JOLTS) | The **"Indeed job postings" analog** — Indeed killed its public API and blocks scraping, so this is the keyless, reproducible hiring-demand proxy. |
| PPI, Traveler Accommodation (monthly) | BLS public API | **RevPAR-rate / ADR proxy.** |
| Accommodation employment (monthly) | BLS public API (CES) | Hospitality **demand** proxy used for the nowcast. |
| Equity prices + earnings dates | Yahoo Finance via `yfinance` | MAR/HLT/H + a 20-name validation universe (9 asset-light franchisors + 10 hotel REITs, tested separately). |

### Honest caveats
- **True RevPAR is STR data (paid).** This project uses BLS Accommodation employment as
  a hospitality-demand proxy and PPI Traveler Accommodation as a room-rate proxy. The
  lead-lag headline is the TSA-vs-demand-proxy relationship, clearly labeled as such.
- **Small sample / correlated cross-section.** Three names over ~4.5 post-COVID years is
  low statistical power; observations within a month are correlated, so effective sample
  size is closer to the number of distinct signal-on months (16). The pooled 10-name test
  is the guard against reading too much into the 3-name result.
- This is a **research / monitoring tool, not investment advice.**

## Layout

```
config.py            tickers, universe, BLS series IDs, paths
src/data/            tsa.py  trends.py  bls.py  prices.py  cache.py  net.py (retry policy)
src/analysis.py      nowcast, signals, backtest, pooled validation, anomaly flags
src/pipeline.py      orchestrates fetch -> analyze -> outputs/
app.py               Streamlit dashboard
tests/               pytest unit tests for the analysis math
.github/workflows/   ci.yml (lint/type/test) + daily.yml (scheduled refresh)
outputs/             summary.json + CSVs (regenerated each run, gitignored)
data/                cached raw data (gitignored)
```

## Run it

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies (creates .venv from uv.lock):
uv sync

# Recompute everything from source and print the numbers:
uv run python -m src.pipeline --force

# Launch the dashboard:
uv run streamlit run app.py
```

The pipeline caches raw data under `data/` (12-24h TTL); the dashboard's
**Refresh data now** button forces a re-fetch. No API keys are required — TSA, BLS, and
Google Trends are all keyless (set `BLS_API_KEY` only if you want a higher BLS rate limit).

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy src app.py config.py   # type-check
uv run pytest -q         # tests
```

CI runs all four on every push/PR (`.github/workflows/ci.yml`); a scheduled job
(`daily.yml`) refreshes the data each morning and publishes the outputs as an artifact.

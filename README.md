# Hospitality Alt-Data Dashboard

An alternative-data pipeline that aggregates **TSA checkpoint throughput**, **Google
Trends brand searches**, and **BLS hospitality labor data** to nowcast lodging demand
and time a long strategy in **Marriott (MAR)**, **Hilton (HLT)**, and **Hyatt (H)**
ahead of quarterly earnings. Results are served through a daily-refreshing Streamlit
dashboard with pre-earnings anomaly alerts.

## Headline findings (computed on 2022-2026 data; as of 2026-06-04 snapshot)

| Result | Value |
|---|---|
| **Primary finding** — TSA volume as a demand nowcast (TSA YoY vs Accommodation-employment YoY, coincident) | **r = 0.92** |
| Exploratory timing signal: long top-2 of MAR/HLT/H on a positive TSA-acceleration signal, 1-month hold | 78% hit rate, +3.8% mean / position (16 signal-on months) vs 62% / +1.9% always-long baseline |
| Significance (honest, clustered by month) | per-position p = 0.008 (naive) → **clustered p = 0.06**; gate-ON vs gate-OFF p = 0.17 — **economically large, not statistically significant** at this sample |
| Out-of-sample validation, pooled across a 20-name universe | franchisors 68% vs 58% (r = 0.12, n = 144); REITs 63% vs 52% (r = 0.16, n = 160) |

> Numbers are a nowcast and move with each data refresh; the figures above are the
> reproducible output of `uv run python -m src.pipeline` on the date noted.

**The honest takeaway:** the headline deliverable is the **r = 0.92 demand nowcast** — TSA
is a clean, timely (1–2 day lag) read on lodging demand. The *timing signal* is a
research hypothesis: the edge is in the **second derivative** (when TSA YoY growth is
*accelerating*, lodging tends to outperform the next month), it shows up across both
asset-light franchisors and hotel REITs (so it isn't overfit to three tickers), but on
~16 signal-on months it is **not yet statistically significant** and a cash-when-OFF
overlay actually trails buy-and-hold in this bull market (cash drag). The dashboard
reports all of this rather than cherry-picking the flattering numbers.

## Data sources

| Signal | Source | Notes |
|---|---|---|
| TSA passenger throughput (daily) | TSA.gov passenger-volumes pages | Per-year archive URLs stitched into a 2022-present daily series. |
| Brand search interest (weekly) | Google Trends via `pytrends` | Marriott / Hilton / Hyatt, US, relative interest. |
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

# Hospitality Alt-Data Dashboard

An alternative-data pipeline that aggregates **TSA checkpoint throughput**, **Google
Trends brand searches**, and **BLS hospitality labor data** to nowcast lodging demand
and time a long strategy in **Marriott (MAR)**, **Hilton (HLT)**, and **Hyatt (H)**
ahead of quarterly earnings. Results are served through a daily-refreshing Streamlit
dashboard with pre-earnings anomaly alerts.

## Headline findings (computed on 2022-2026 data)

| Result | Value |
|---|---|
| TSA volume as a demand nowcast (TSA YoY vs Accommodation-employment YoY, coincident) | **r = 0.91** |
| Strategy: long top-2 of MAR/HLT/H on a positive TSA-acceleration signal, 1-month hold | **81% hit rate**, +4.2% mean / position, across 32 positions (16 rebalances) |
| Baseline (always-long the three names) | 61% hit rate, +1.9% mean |
| Out-of-sample validation, pooled across a 10-name lodging universe | hit 70% vs 58% baseline, pooled r = 0.16, n = 160 |

**The signal:** the tradeable edge is in the *second derivative* of travel demand.
When TSA year-over-year growth is **accelerating**, lodging equities outperform over
the next ~1 month. The level of TSA YoY by itself has no edge over a buy-and-hold in a
hotel bull market; the acceleration does. The relationship holds across a broader
lodging universe (IHG, WH, CHH and the hotel REITs HST/PK/RHP/APLE), which argues it
is not overfit to the three headline names.

## Data sources

| Signal | Source | Notes |
|---|---|---|
| TSA passenger throughput (daily) | TSA.gov passenger-volumes pages | Per-year archive URLs stitched into a 2022-present daily series. |
| Brand search interest (weekly) | Google Trends via `pytrends` | Marriott / Hilton / Hyatt, US, relative interest. |
| Job openings, Leisure & Hospitality (monthly) | BLS public API (JOLTS) | The **"Indeed job postings" analog** — Indeed killed its public API and blocks scraping, so this is the keyless, reproducible hiring-demand proxy. |
| PPI, Traveler Accommodation (monthly) | BLS public API | **RevPAR-rate / ADR proxy.** |
| Accommodation employment (monthly) | BLS public API (CES) | Hospitality **demand** proxy used for the nowcast. |
| Equity prices + earnings dates | Yahoo Finance via `yfinance` | MAR/HLT/H + the 10-name validation universe. |

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

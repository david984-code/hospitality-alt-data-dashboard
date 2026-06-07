"""Central configuration for the Hospitality Alt-Data Dashboard."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Headline traded names (the resume strategy: long top-2 of these).
TICKERS = ["MAR", "HLT", "H"]
TICKER_NAMES = {"MAR": "Marriott", "HLT": "Hilton", "H": "Hyatt"}

# Broader lodging universe used to validate the signal out-of-sample (pooled).
# Split by business model — the brand-momentum thesis applies to asset-light
# franchisors; the REITs that own the real estate are a separate, rate-sensitive
# comp set, so we validate the two buckets independently.
FRANCHISORS = ["MAR", "HLT", "H", "WH", "CHH", "IHG", "HGV", "VAC", "TNL"]
REITS = ["HST", "PK", "RHP", "APLE", "DRH", "PEB", "SHO", "XHR", "RLJ", "INN"]
UNIVERSE = FRANCHISORS + REITS

# Google Trends brand search terms (one per ticker, booking-intent flavored).
TREND_TERMS = {
    "MAR": "Marriott",
    "HLT": "Hilton",
    "H": "Hyatt",
}

# BLS series used as alt-data / fundamentals proxies (keyless public API).
#   JTS700000000000000JOL - Job Openings: Leisure & Hospitality (Indeed-postings analog)
#   PCU721110721110       - PPI: Traveler Accommodation (RevPAR-rate / ADR proxy)
#   CEU7072100001         - All Employees: Accommodation (hiring level, monthly)
BLS_JOB_OPENINGS = "JTS700000000000000JOL"
BLS_LODGING_PPI = "PCU721110721110"
BLS_ACCOM_EMP = "CEU7072100001"

# Optional BLS registration key (raises the rate limit). Keyless works for our use.
BLS_API_KEY = os.getenv("BLS_API_KEY", "")

# History window for the study.
START_DATE = "2022-01-01"

# Cache freshness in hours before a re-fetch is triggered.
CACHE_TTL_HOURS = 12

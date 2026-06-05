"""Google Trends brand search interest via pytrends (unofficial API).

Trends is rate-limited and returns *relative* interest (0-100) per request, so we
pull one term at a time over the full window and cache aggressively.
"""

from __future__ import annotations

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

import config

from . import cache

_NAME = "google_trends"


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=5, min=5, max=30))
def _interest(pytrends, term: str, timeframe: str) -> pd.Series:
    pytrends.build_payload([term], timeframe=timeframe, geo="US")
    df = pytrends.interest_over_time()
    if df.empty:
        raise RuntimeError(f"empty Trends response for {term}")
    return df[term]


def fetch(force: bool = False) -> pd.DataFrame:
    """Weekly search-interest DataFrame, one column per ticker."""
    if not force and cache.is_fresh(_NAME, ttl_hours=24):
        return cache.load(_NAME)

    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="en-US", tz=300)
    timeframe = f"{config.START_DATE} {pd.Timestamp.today():%Y-%m-%d}"
    cols = {}
    for ticker, term in config.TREND_TERMS.items():
        try:
            cols[ticker] = _interest(pytrends, term, timeframe)
        except Exception as exc:  # noqa: BLE001 - keep other terms if one fails
            print(f"[trends] {term} failed after retries: {exc}")

    out = pd.DataFrame(cols)
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    if not out.empty:
        cache.save(_NAME, out)
    return out


if __name__ == "__main__":
    print(fetch(force=True).tail())

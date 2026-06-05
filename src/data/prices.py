"""Hotel equity prices and historical earnings dates via yfinance."""

from __future__ import annotations

import pandas as pd

import config

from . import cache
from .net import flaky_retry

_PRICES = "prices"
_UNIVERSE = "universe_prices"
_EARNINGS = "earnings_dates"


@flaky_retry
def _download_close(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, start=config.START_DATE, progress=False, auto_adjust=True)
    close = raw["Close"][tickers]
    close.index = pd.to_datetime(close.index)
    close.index.name = "date"
    return close


def fetch_prices(force: bool = False) -> pd.DataFrame:
    """Daily adjusted close for the headline names (MAR/HLT/H)."""
    if not force and cache.is_fresh(_PRICES):
        return cache.load(_PRICES)
    close = _download_close(config.TICKERS)
    cache.save(_PRICES, close)
    return close


def fetch_universe_prices(force: bool = False) -> pd.DataFrame:
    """Daily adjusted close for the broader lodging universe (validation)."""
    if not force and cache.is_fresh(_UNIVERSE):
        return cache.load(_UNIVERSE)
    close = _download_close(config.UNIVERSE)
    cache.save(_UNIVERSE, close)
    return close


@flaky_retry
def _earnings_for(ticker: str) -> pd.DatetimeIndex:
    import yfinance as yf

    return yf.Ticker(ticker).get_earnings_dates(limit=24).index


def fetch_earnings_dates(force: bool = False) -> pd.DataFrame:
    """Historical earnings datetimes per ticker (long format: ticker, date)."""
    if not force and cache.is_fresh(_EARNINGS, ttl_hours=24):
        return cache.load(_EARNINGS)

    rows = []
    for ticker in config.TICKERS:
        try:
            for ts in _earnings_for(ticker):
                rows.append({"ticker": ticker, "earnings": pd.Timestamp(ts).tz_localize(None)})
        except Exception as exc:  # noqa: BLE001
            print(f"[prices] earnings dates failed for {ticker}: {exc}")

    df = pd.DataFrame(rows)
    df = df[df["earnings"] >= config.START_DATE].sort_values("earnings")
    df = df.reset_index(drop=True)
    cache.save(_EARNINGS, df)
    return df


if __name__ == "__main__":
    print(fetch_prices(force=True).tail())
    print(fetch_earnings_dates(force=True).tail(10))

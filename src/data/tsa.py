"""TSA daily checkpoint passenger volumes.

The live page (https://www.tsa.gov/travel/passenger-volumes) shows the current
year; prior years live at .../passenger-volumes/<YYYY>. We stitch them into one
daily series back to START_DATE.
"""

from __future__ import annotations

import datetime as dt
from io import StringIO

import pandas as pd
import requests

import config

from . import cache
from .net import network_retry

_BASE = "https://www.tsa.gov/travel/passenger-volumes"
_HEADERS = {"User-Agent": "Mozilla/5.0 (research; alt-data dashboard)"}
_NAME = "tsa_throughput"


@network_retry
def _fetch_year(year: int, current_year: int) -> pd.DataFrame:
    url = _BASE if year == current_year else f"{_BASE}/{year}"
    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    tbl = pd.read_html(StringIO(r.text))[0]
    tbl.columns = ["date", "pax"]
    tbl["date"] = pd.to_datetime(tbl["date"])
    tbl["pax"] = pd.to_numeric(tbl["pax"], errors="coerce")
    return tbl.dropna()


def fetch(force: bool = False) -> pd.Series:
    """Return a daily Series of TSA passengers indexed by date."""
    if not force and cache.is_fresh(_NAME):
        return cache.load(_NAME)["pax"]

    current_year = dt.date.today().year
    start_year = pd.Timestamp(config.START_DATE).year
    frames = []
    for year in range(start_year, current_year + 1):
        try:
            frames.append(_fetch_year(year, current_year))
        except Exception as exc:  # noqa: BLE001 - keep partial history on a bad year
            print(f"[tsa] warning: could not fetch {year}: {exc}")

    df = pd.concat(frames).drop_duplicates("date").sort_values("date")
    df = df.set_index("date")
    df = df[df.index >= config.START_DATE]
    cache.save(_NAME, df)
    return df["pax"]


if __name__ == "__main__":
    s = fetch(force=True)
    print(f"TSA: {len(s)} days, {s.index.min().date()} -> {s.index.max().date()}")
    print(s.tail())

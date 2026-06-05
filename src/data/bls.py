"""Hospitality labor & price series from the BLS public API (keyless).

  - Job openings, Leisure & Hospitality  -> "Indeed job postings" analog
  - PPI, Traveler Accommodation          -> RevPAR-rate / ADR proxy
  - All employees, Accommodation         -> hiring level

BLS returns monthly observations; we parse the M01-M12 periods into dates.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

import config

from . import cache
from .net import network_retry

_NAME = "bls_hospitality"
_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

_SERIES = {
    "job_openings": config.BLS_JOB_OPENINGS,
    "lodging_ppi": config.BLS_LODGING_PPI,
    "accom_emp": config.BLS_ACCOM_EMP,
}
_ID_TO_NAME = {v: k for k, v in _SERIES.items()}


@network_retry
def _request() -> dict:
    payload: dict = {
        "seriesid": list(_SERIES.values()),
        "startyear": pd.Timestamp(config.START_DATE).year,
        "endyear": dt.date.today().year,
    }
    if config.BLS_API_KEY:
        payload["registrationkey"] = config.BLS_API_KEY
    r = requests.post(_URL, json=payload, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS error: {body.get('message')}")
    return body


def _parse_series(series: dict) -> pd.Series:
    recs = {}
    for d in series["data"]:
        if not d["period"].startswith("M"):
            continue  # skip annual averages (M13)
        ts = pd.Timestamp(int(d["year"]), int(d["period"][1:]), 1)
        recs[ts] = float(d["value"])
    return pd.Series(recs)


def fetch(force: bool = False) -> pd.DataFrame:
    """Monthly DataFrame: job_openings, lodging_ppi, accom_emp."""
    if not force and cache.is_fresh(_NAME, ttl_hours=24):
        return cache.load(_NAME)

    body = _request()
    cols = {_ID_TO_NAME[s["seriesID"]]: _parse_series(s) for s in body["Results"]["series"]}
    df = pd.DataFrame(cols).sort_index()
    df = df[df.index >= config.START_DATE]
    df.index.name = "date"
    cache.save(_NAME, df)
    return df


if __name__ == "__main__":
    print(fetch(force=True).tail())

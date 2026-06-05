"""Tiny CSV-based cache so we don't hammer data sources on every run."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

import pandas as pd

import config


def _path(name: str) -> Path:
    return config.DATA_DIR / f"{name}.csv"


def is_fresh(name: str, ttl_hours: float | None = None) -> bool:
    """True if a cached file exists and is younger than the TTL."""
    ttl = config.CACHE_TTL_HOURS if ttl_hours is None else ttl_hours
    p = _path(name)
    if not p.exists():
        return False
    age_hours = (time.time() - p.stat().st_mtime) / 3600.0
    return age_hours < ttl


def save(name: str, df: pd.DataFrame) -> None:
    df.to_csv(_path(name))


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(_path(name), index_col=0)
    # Re-parse a datetime index if it looks like one.
    with contextlib.suppress(ValueError, TypeError):
        df.index = pd.to_datetime(df.index)
    return df

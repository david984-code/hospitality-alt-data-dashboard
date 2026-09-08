"""Reported system-wide RevPAR from the MAR / HLT / H earnings releases.

Unlike the rest of `src/data`, this module does not fetch anything. True RevPAR is
STR data and there is no free API for it, so `data/reported_revpar.csv` is filled in
by hand from the companies' quarterly releases (system-wide comparable RevPAR,
constant currency where the company reports it) and committed to the repo.

CSV schema — long format, one row per ticker-quarter:

    ticker           MAR | HLT | H
    fiscal_quarter   "Q1 2019" ... one space between the quarter and the year
    period_end       ISO date the fiscal quarter closed, e.g. 2019-03-31
    revpar_yoy_pct   reported system-wide comparable RevPAR growth, in percent
    source_url       link to the release the number was read off

Rows that have not been filled in yet carry the literal string TODO in
`revpar_yoy_pct`; the loader refuses to hand back a frame that still contains any,
so a half-populated file can never quietly become a chart.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import config

REPORTED_TICKERS = ("MAR", "HLT", "H")
COLUMNS = ("ticker", "fiscal_quarter", "period_end", "revpar_yoy_pct", "source_url")
TODO_MARKER = "TODO"
DEFAULT_PATH = config.DATA_DIR / "reported_revpar.csv"

_QUARTER_RE = re.compile(r"^Q([1-4]) (\d{4})$")
# A fiscal quarter may close a few days either side of the calendar quarter end.
_PERIOD_END_TOLERANCE_DAYS = 14


class ReportedRevparError(ValueError):
    """The reported-RevPAR CSV is missing, malformed, or incomplete."""


def _fail(path: Path, problem: str) -> None:
    raise ReportedRevparError(f"{path}: {problem}")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ReportedRevparError(
            f"{path} does not exist. It is hand-maintained — create it with the header "
            f"{','.join(COLUMNS)} and one row per ticker-quarter."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False).rename(columns=str.strip)
    if df.empty:
        _fail(path, "has a header but no rows")
    return df


def _check_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        _fail(path, f"missing required column(s) {missing}; expected exactly {list(COLUMNS)}")
    extra = [c for c in df.columns if c not in COLUMNS]
    if extra:
        _fail(path, f"unexpected column(s) {extra}; expected exactly {list(COLUMNS)}")


def _check_placeholders(df: pd.DataFrame, path: Path) -> None:
    todo = df[df["revpar_yoy_pct"].str.strip().str.upper() == TODO_MARKER]
    if not todo.empty:
        rows = ", ".join(f"{r.ticker} {r.fiscal_quarter}" for r in todo.itertuples())
        _fail(
            path,
            f"still has {len(todo)} placeholder {TODO_MARKER} row(s) ({rows}). Fill in the "
            "reported RevPAR growth and the source URL for each before using this file.",
        )


def _check_tickers(df: pd.DataFrame, path: Path) -> None:
    unknown = sorted(set(df["ticker"].str.strip()) - set(REPORTED_TICKERS))
    if unknown:
        _fail(path, f"unknown ticker(s) {unknown}; expected only {list(REPORTED_TICKERS)}")


def _parse_quarters(df: pd.DataFrame, path: Path) -> pd.Series:
    """Validate the `Qn YYYY` labels and return them as quarterly Periods."""
    bad = [q for q in df["fiscal_quarter"] if not _QUARTER_RE.match(q.strip())]
    if bad:
        _fail(
            path, f"malformed fiscal_quarter value(s) {sorted(set(bad))}; expected e.g. 'Q1 2019'"
        )
    parts = df["fiscal_quarter"].str.strip().str.extract(_QUARTER_RE)
    periods = [
        pd.Period(year=int(year), quarter=int(q), freq="Q")
        for q, year in parts.itertuples(index=False)
    ]
    return pd.Series(pd.PeriodIndex(periods, freq="Q"), index=df.index)


def _parse_period_end(df: pd.DataFrame, quarters: pd.Series, path: Path) -> pd.Series:
    """Parse period_end to dates and check each one closes the quarter it claims to."""
    parsed = pd.to_datetime(df["period_end"].str.strip(), errors="coerce", format="ISO8601")
    if parsed.isna().any():
        bad = df.loc[parsed.isna(), "period_end"].tolist()
        _fail(path, f"unparseable period_end value(s) {bad}; expected ISO dates like 2019-03-31")
    closes = pd.PeriodIndex(quarters).to_timestamp(how="end").normalize()
    drift = (parsed - pd.Series(closes, index=df.index)).dt.days.abs()
    off = df.loc[drift > _PERIOD_END_TOLERANCE_DAYS]
    if not off.empty:
        rows = ", ".join(
            f"{r.ticker} {r.fiscal_quarter} -> {r.period_end}" for r in off.itertuples()
        )
        _fail(
            path,
            f"period_end is more than {_PERIOD_END_TOLERANCE_DAYS} days from the quarter it "
            f"is labelled with ({rows})",
        )
    return parsed


def _check_values(df: pd.DataFrame, path: Path) -> pd.Series:
    values = pd.to_numeric(df["revpar_yoy_pct"].str.strip(), errors="coerce")
    if values.isna().any():
        bad = df.loc[values.isna(), "revpar_yoy_pct"].tolist()
        _fail(path, f"non-numeric revpar_yoy_pct value(s) {bad}")
    blank = df.loc[df["source_url"].str.strip() == ""]
    if not blank.empty:
        rows = ", ".join(f"{r.ticker} {r.fiscal_quarter}" for r in blank.itertuples())
        _fail(path, f"blank source_url for {rows}; every number needs its release link")
    return values


def _check_coverage(df: pd.DataFrame, quarters: pd.Series, path: Path) -> None:
    """Every ticker must cover every quarter between the file's first and last, with no gaps."""
    dupes = df[df.duplicated(subset=["ticker", "fiscal_quarter"], keep=False)]
    if not dupes.empty:
        rows = sorted({f"{r.ticker} {r.fiscal_quarter}" for r in dupes.itertuples()})
        _fail(path, f"duplicate ticker/quarter row(s) {rows}")

    absent = [t for t in REPORTED_TICKERS if t not in set(df["ticker"].str.strip())]
    if absent:
        _fail(path, f"has no rows for {absent}; all of {list(REPORTED_TICKERS)} are required")

    span = pd.period_range(quarters.min(), quarters.max(), freq="Q")
    gaps = [
        f"{ticker} Q{p.quarter} {p.year}"
        for ticker in REPORTED_TICKERS
        for p in sorted(set(span) - set(quarters[df["ticker"].str.strip() == ticker]))
    ]
    if gaps:
        _fail(
            path,
            f"missing {len(gaps)} quarter(s) between {span[0]} and {span[-1]}: {gaps}. "
            "Every ticker needs an unbroken quarterly history.",
        )


def load_reported_revpar(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the hand-maintained reported-RevPAR file.

    Returns the five schema columns with `period_end` parsed to dates and
    `revpar_yoy_pct` to floats, sorted by ticker then period_end.

    Raises `ReportedRevparError` — naming the offending rows — if the schema is wrong,
    a placeholder TODO row survives, a period_end does not parse or does not close the
    quarter it is labelled with, or any ticker is missing a quarter inside the file's
    span. The gap check is the important one: a hole in the history would silently
    shorten the comparison window rather than fail.
    """
    p = Path(path) if path is not None else DEFAULT_PATH
    df = _read(p)
    _check_columns(df, p)
    _check_placeholders(df, p)
    _check_tickers(df, p)
    quarters = _parse_quarters(df, p)
    _check_coverage(df, quarters, p)

    out = pd.DataFrame(
        {
            "ticker": df["ticker"].str.strip(),
            "fiscal_quarter": df["fiscal_quarter"].str.strip(),
            "period_end": _parse_period_end(df, quarters, p),
            "revpar_yoy_pct": _check_values(df, p),
            "source_url": df["source_url"].str.strip(),
        }
    )
    return out.sort_values(["ticker", "period_end"]).reset_index(drop=True)


if __name__ == "__main__":
    frame = load_reported_revpar()
    lo, hi = frame["period_end"].min().date(), frame["period_end"].max().date()
    print(f"{len(frame)} rows, {lo} -> {hi}")
    print(frame.tail())

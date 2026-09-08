"""Schema tests for the hand-maintained reported-RevPAR loader.

The file is typed in by hand from earnings releases, so every plausible typo — a
renamed column, a leftover TODO, a quarter skipped, a period_end that does not match
the quarter it is labelled with — has to fail loudly rather than shorten the sample.
"""

from __future__ import annotations

import pytest

from src.data import revpar_reported
from src.data.revpar_reported import ReportedRevparError, load_reported_revpar

HEADER = "ticker,fiscal_quarter,period_end,revpar_yoy_pct,source_url"
QUARTER_ENDS = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}


def _rows(quarters, tickers=revpar_reported.REPORTED_TICKERS, value="1.5"):
    """One row per ticker per (year, quarter) pair, all with the same placeholder value."""
    return [
        f"{t},Q{q} {y},{y}-{QUARTER_ENDS[q]},{value},https://example.com/{t}-{y}q{q}"
        for (y, q) in quarters
        for t in tickers
    ]


def _write(tmp_path, lines, name="reported_revpar.csv"):
    path = tmp_path / name
    path.write_text("\n".join([HEADER, *lines]) + "\n", encoding="utf-8")
    return path


def _four_quarters():
    return [(2019, 1), (2019, 2), (2019, 3), (2019, 4)]


def test_loads_a_well_formed_file(tmp_path):
    path = _write(tmp_path, _rows(_four_quarters()))
    df = load_reported_revpar(path)

    assert list(df.columns) == list(revpar_reported.COLUMNS)
    assert len(df) == 12  # 4 quarters x 3 tickers
    assert set(df["ticker"]) == set(revpar_reported.REPORTED_TICKERS)
    assert df["revpar_yoy_pct"].dtype.kind == "f"
    assert str(df["period_end"].dtype).startswith("datetime64")
    assert df["period_end"].min().date().isoformat() == "2019-03-31"
    # sorted by ticker then date
    assert df.equals(df.sort_values(["ticker", "period_end"]).reset_index(drop=True))


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(ReportedRevparError, match="does not exist"):
        load_reported_revpar(tmp_path / "nope.csv")


def test_missing_column_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("ticker,fiscal_quarter,period_end,revpar_yoy_pct\nMAR,Q1 2019,2019-03-31,1.5\n")
    with pytest.raises(ReportedRevparError, match="source_url"):
        load_reported_revpar(path)


def test_unexpected_column_is_rejected(tmp_path):
    path = tmp_path / "extra.csv"
    path.write_text(f"{HEADER},notes\nMAR,Q1 2019,2019-03-31,1.5,https://x,hi\n")
    with pytest.raises(ReportedRevparError, match="notes"):
        load_reported_revpar(path)


def test_placeholder_todo_rows_are_rejected(tmp_path):
    """The shipped file starts as TODO rows; it must not silently become a chart."""
    path = _write(tmp_path, _rows(_four_quarters(), value="TODO"))
    with pytest.raises(ReportedRevparError, match="placeholder TODO"):
        load_reported_revpar(path)


def test_the_shipped_file_has_the_documented_header():
    """data/reported_revpar.csv is hand-edited, so pin its header to the schema."""
    first_line = revpar_reported.DEFAULT_PATH.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == HEADER


def test_missing_quarter_is_named(tmp_path):
    quarters = [q for q in _four_quarters() if q != (2019, 3)]
    rows = _rows(quarters)
    # Reintroduce Q3 for two of the three tickers so the span still covers it.
    rows += _rows([(2019, 3)], tickers=("MAR", "HLT"))
    path = _write(tmp_path, rows)
    with pytest.raises(ReportedRevparError, match=r"missing 1 quarter\(s\).*H Q3 2019"):
        load_reported_revpar(path)


def test_gap_inside_a_ticker_history_is_rejected(tmp_path):
    quarters = [(2019, 1), (2019, 2), (2019, 4)]  # Q3 dropped for everyone
    path = _write(tmp_path, _rows(quarters))
    with pytest.raises(ReportedRevparError, match="Q3 2019"):
        load_reported_revpar(path)


def test_unknown_ticker_is_rejected(tmp_path):
    rows = _rows(_four_quarters()) + _rows([(2019, 1)], tickers=("WH",))
    path = _write(tmp_path, rows)
    with pytest.raises(ReportedRevparError, match="unknown ticker"):
        load_reported_revpar(path)


def test_absent_ticker_is_rejected(tmp_path):
    path = _write(tmp_path, _rows(_four_quarters(), tickers=("MAR", "HLT")))
    with pytest.raises(ReportedRevparError, match="no rows for"):
        load_reported_revpar(path)


def test_malformed_fiscal_quarter_is_rejected(tmp_path):
    path = _write(tmp_path, ["MAR,2019Q1,2019-03-31,1.5,https://x"])
    with pytest.raises(ReportedRevparError, match="malformed fiscal_quarter"):
        load_reported_revpar(path)


def test_duplicate_ticker_quarter_is_rejected(tmp_path):
    rows = _rows(_four_quarters()) + _rows([(2019, 2)], tickers=("MAR",))
    path = _write(tmp_path, rows)
    with pytest.raises(ReportedRevparError, match="duplicate"):
        load_reported_revpar(path)


def test_period_end_must_close_the_quarter_it_claims(tmp_path):
    rows = _rows(_four_quarters())
    rows[0] = "MAR,Q1 2019,2019-09-30,1.5,https://x"  # Q1 row carrying a Q3 date
    path = _write(tmp_path, rows)
    with pytest.raises(ReportedRevparError, match="days from the quarter"):
        load_reported_revpar(path)


def test_non_numeric_value_is_rejected(tmp_path):
    rows = _rows(_four_quarters())
    rows[0] = "MAR,Q1 2019,2019-03-31,up a bit,https://x"
    path = _write(tmp_path, rows)
    with pytest.raises(ReportedRevparError, match="non-numeric"):
        load_reported_revpar(path)


def test_blank_source_url_is_rejected(tmp_path):
    rows = _rows(_four_quarters())
    rows[0] = "MAR,Q1 2019,2019-03-31,1.5,"
    path = _write(tmp_path, rows)
    with pytest.raises(ReportedRevparError, match="blank source_url"):
        load_reported_revpar(path)


def test_negative_values_are_fine(tmp_path):
    """COVID quarters are large negatives — they are data, not errors."""
    path = _write(tmp_path, _rows(_four_quarters(), value="-72.4"))
    df = load_reported_revpar(path)
    assert (df["revpar_yoy_pct"] == -72.4).all()

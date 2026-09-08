"""Shared input assembly for the reported-RevPAR comparison chart and table.

Both scripts need the same thing: the hand-maintained reported-RevPAR file, the
composite demand index rebuilt from the cached alt-data sources, and the quarterly
comparison between them. Neither touches the pipeline or the backtest.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src import analysis
from src.data import bls, revpar_reported, trends, tsa


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--csv", default=None, help="reported-RevPAR file (default data/)")
    parser.add_argument("--force", action="store_true", help="ignore cache, re-fetch alt data")
    parser.add_argument(
        "--skip-trends",
        action="store_true",
        help="drop Google Trends from the index (rate-limited); TSA + hiring only",
    )
    return parser


def load_inputs(
    csv: str | None = None, force: bool = False, skip_trends: bool = False
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """(reported RevPAR, TSA daily, Trends, BLS) exactly as the comparison consumes them."""
    reported = revpar_reported.load_reported_revpar(csv)
    tsa_daily = tsa.fetch(force=force)
    fred = bls.fetch(force=force)
    if skip_trends:
        print("[revpar] --skip-trends: index is TSA + hospitality hiring only")
        trends_df = pd.DataFrame()
    else:
        trends_df = trends.fetch(force=force)
    return reported, tsa_daily, trends_df, fred


def build_comparison(
    csv: str | None = None, force: bool = False, skip_trends: bool = False
) -> analysis.RevparComparison:
    """Load reported RevPAR, rebuild the composite index, and compare them by quarter."""
    reported, tsa_daily, trends_df, fred = load_inputs(csv, force, skip_trends)
    index = analysis.composite_demand_index(tsa_daily, trends_df, fred)
    return analysis.compare_index_to_reported(index, reported)


def tsa_only_index(tsa_daily: pd.Series) -> pd.Series:
    """The repo's original demand read (TSA throughput alone) as a daily rebased level series.

    Same publication lag and base window as the composite, so the two are comparable.
    """
    return analysis._rebased(
        tsa_daily, analysis.PUBLICATION_LAG_DAYS["tsa"], analysis.INDEX_BASE_END
    ).rename("demand_index")


def build_both(
    csv: str | None = None, force: bool = False, skip_trends: bool = False
) -> tuple[analysis.RevparComparison, analysis.RevparComparison]:
    """(composite comparison, TSA-only comparison) from one load of the inputs."""
    reported, tsa_daily, trends_df, fred = load_inputs(csv, force, skip_trends)
    composite = analysis.composite_demand_index(tsa_daily, trends_df, fred)
    return (
        analysis.compare_index_to_reported(composite, reported),
        analysis.compare_index_to_reported(tsa_only_index(tsa_daily), reported),
    )


def summary_line(cmp: analysis.RevparComparison) -> str:
    """Three-line read of the comparison, printed by both scripts."""
    lo, hi = cmp.r_steady_ci
    return "\n".join(
        [
            f"full sample      r = {cmp.r:+.3f} (p = {cmp.p:.4f}, n = {cmp.n}) | "
            f"direction matched {cmp.hit_rate:.1%} of {cmp.n_direction} quarters",
            f"ex-COVID         r = {cmp.r_ex_covid:+.3f} (n = {cmp.n_ex_covid})",
            f"steady state     r = {cmp.r_steady:+.3f} (p = {cmp.p_steady:.4f}, n = {cmp.n_steady}, "
            f"95% CI [{lo:+.2f}, {hi:+.2f}]) | direction matched "
            f"{cmp.hit_rate_steady:.1%} of {cmp.n_direction_steady} quarters "
            f"(from {analysis.STEADY_START})",
        ]
    )

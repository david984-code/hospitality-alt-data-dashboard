"""Plot a public-data demand series' YoY against reported system-wide RevPAR growth.

Writes exactly one figure, outputs/index_vs_reported_revpar.png, at 1600x900.

    uv run python -m scripts.make_revpar_chart

The default layout is two panels in one figure. The left panel is the full history,
where the COVID collapse and rebound dwarf everything else. The right panel is the
steady-state window (analysis.STEADY_START onward), where reported RevPAR growth is back
in single digits and the index has to track ordinary-sized moves. That is the panel a
hotel analyst will judge the index on, so it gets the room and the statistics.

Both series are plotted in the same units (year-over-year percent) on one shared axis
per panel. A second axis would let the index be rescaled to look like a better fit than
it is. Pass --dual-axis to put the index on its own right-hand axis if the two series
live on very different scales. Pass --layout single for the earlier one-panel version.
"""

from __future__ import annotations

import argparse
import textwrap

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import config
from scripts.revpar_common import add_common_args, build_both, summary_line
from src import analysis
from src.data.revpar_reported import REPORTED_TICKERS

matplotlib.use("Agg")  # headless: this script never opens a window

OUT_PNG = config.OUTPUT_DIR / "index_vs_reported_revpar.png"
FIG_INCHES, DPI = (16.0, 9.0), 100  # 1600 x 900

TITLES = {
    "tsa": "TSA checkpoint throughput vs reported system-wide RevPAR growth",
    "composite": "Public data hotel demand index vs reported system-wide RevPAR growth",
}
SUBTITLES = {
    "tsa": (
        "Bold line: US TSA checkpoint throughput, quarterly mean of daily counts, lagged one day "
        "to its posting date. Dashed: an equal-weighted composite of TSA, Google brand search "
        "and BLS JOLTS leisure and hospitality job openings, shown because it was tried and it "
        "failed. RevPAR: system-wide comparable, constant currency, as reported in MAR, HLT and "
        "H earnings releases. All series are year-over-year percent."
    ),
    "composite": (
        "Bold line: US-only public data (TSA checkpoint throughput, Google brand search, BLS "
        "JOLTS leisure and hospitality job openings), equal-weighted, quarterly mean, each input "
        "lagged to its release date. Dashed: TSA throughput alone. RevPAR: system-wide "
        "comparable, constant currency, as reported in MAR, HLT and H earnings releases. All "
        "series are year-over-year percent."
    ),
}

# Three grays, light to dark, so the reported lines read as one family behind the index.
COMPANY_COLORS = {"MAR": "#b4b4b4", "HLT": "#8a8a8a", "H": "#5c5c5c"}
INDEX_COLOR = "#12507d"
TEXT_COLOR = "#3a3a3a"
MUTED = "#7a7a7a"


def _quarter_label(d: pd.Timestamp) -> str:
    return f"{d.year}Q{d.quarter}"


def _plot_reported(ax: Axes, table: pd.DataFrame) -> None:
    """Three thin gray lines: reported RevPAR YoY per company."""
    by_quarter = table.set_index("period_end")
    for ticker in REPORTED_TICKERS:
        series = by_quarter[ticker].dropna()
        if not series.empty:
            ax.plot(series.index, series.to_numpy(), lw=1.2, color=COMPANY_COLORS[ticker], zorder=2)


def _plot_index(ax: Axes, table: pd.DataFrame) -> None:
    """One bold line: the composite demand index's YoY."""
    series = table.set_index("period_end")["index_yoy"].dropna()
    if not series.empty:
        ax.plot(
            series.index,
            series.to_numpy(),
            lw=2.8,
            color=INDEX_COLOR,
            marker="o",
            ms=4,
            zorder=3,
        )


def _shade_covid(ax: Axes, label: bool = True) -> None:
    start, end = pd.Timestamp(analysis.COVID_START), pd.Timestamp(analysis.COVID_END)
    ax.axvspan(
        mdates.date2num(start), mdates.date2num(end), color="#3a3a3a", alpha=0.06, lw=0, zorder=0
    )
    if label:
        ax.annotate(
            "COVID quarters\n(excluded from the ex-COVID r)",
            xy=(start + (end - start) / 2, 0.97),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=8.5,
            color=MUTED,
        )


INDEX_LABELS = {
    "tsa": "TSA throughput (the repo's demand signal)",
    "composite": "Composite (TSA + brand search + JOLTS openings)",
}
OVERLAY_COLOR = "#c0392b"


def _stats_lines(
    main: analysis.RevparComparison,
    main_kind: str,
    overlay: analysis.RevparComparison | None,
    overlay_kind: str,
) -> list[str]:
    q = _quarter_label(pd.Timestamp(analysis.STEADY_START))
    lo, hi = main.r_steady_ci
    lines = [
        f"{INDEX_LABELS[main_kind].split(' (')[0]}, {q} onward:  r = {main.r_steady:+.2f}"
        f"   n = {main.n_steady}   95% CI [{lo:+.2f}, {hi:+.2f}]"
        f"   direction matched {main.hit_rate_steady:.0%}",
        f"{INDEX_LABELS[main_kind].split(' (')[0]}, ex-COVID:  r = {main.r_ex_covid:+.2f}"
        f"   n = {main.n_ex_covid}     full sample:  r = {main.r:+.2f}   n = {main.n}"
        f"   (COVID co-trend inflates both)",
    ]
    if overlay is not None:
        olo, ohi = overlay.r_steady_ci
        lines.append(
            f"{INDEX_LABELS[overlay_kind].split(' (')[0]} (dashed), {q} onward:  "
            f"r = {overlay.r_steady:+.2f}   n = {overlay.n_steady}   95% CI [{olo:+.2f}, {ohi:+.2f}]"
        )
    return lines


def _stats_box(fig: Figure, lines: list[str]) -> None:
    """Monospace block in the footer, lines padded to one width so the numbers line up."""
    width = max(len(line) for line in lines)
    fig.text(
        0.99,
        0.012,
        "\n".join(line.ljust(width) for line in lines),
        ha="right",
        va="bottom",
        fontsize=9,
        family="monospace",
        color=TEXT_COLOR,
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.5"},
    )


def _legend(fig: Figure, main_kind: str, overlay_kind: str | None) -> None:
    """One legend for the whole figure, in the footer strip, clear of the data."""
    handles = [
        Line2D([], [], color=INDEX_COLOR, lw=2.8, marker="o", ms=4, label=INDEX_LABELS[main_kind])
    ]
    if overlay_kind:
        handles.append(
            Line2D([], [], color=OVERLAY_COLOR, lw=1.4, ls="--", label=INDEX_LABELS[overlay_kind])
        )
    handles += [
        Line2D([], [], color=COMPANY_COLORS[t], lw=1.2, label=f"{t} reported RevPAR")
        for t in REPORTED_TICKERS
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.005),
        ncol=1,
        fontsize=9,
        frameon=False,
        labelcolor=TEXT_COLOR,
        handlelength=2.4,
    )


def _plot_overlay(ax: Axes, table: pd.DataFrame) -> None:
    series = table.set_index("period_end")["index_yoy"].dropna()
    if not series.empty:
        ax.plot(series.index, series.to_numpy(), lw=1.4, ls="--", color=OVERLAY_COLOR, zorder=2.5)


def _style(ax: Axes, ends: pd.DatetimeIndex, every: int = 1) -> None:
    ax.set_xticks(ends[::every])
    ax.set_xticklabels(
        [_quarter_label(d) for d in ends[::every]], rotation=45, ha="right", fontsize=8.5
    )
    pad = pd.Timedelta(days=40)
    ax.set_xlim(mdates.date2num(ends.min() - pad), mdates.date2num(ends.max() + pad))
    ax.axhline(0, color="#c8c8c8", lw=0.8, zorder=1)
    ax.grid(axis="y", color="#ececec", lw=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=9, colors=TEXT_COLOR)
    ax.set_ylabel("Year-over-year, %", fontsize=10, color=TEXT_COLOR)


def _header(fig: Figure, main_kind: str) -> None:
    fig.text(0.01, 0.975, TITLES[main_kind], fontsize=17, fontweight="bold", ha="left", va="top")
    fig.text(
        0.01,
        0.925,
        "\n".join(textwrap.wrap(SUBTITLES[main_kind], 185)),
        fontsize=9.5,
        color="#666666",
        ha="left",
        va="top",
    )


def make_chart(
    main: analysis.RevparComparison,
    main_kind: str,
    overlay: analysis.RevparComparison | None = None,
    overlay_kind: str | None = None,
) -> None:
    table = main.table
    ends = pd.DatetimeIndex(table["period_end"])
    steady_from = pd.Timestamp(analysis.STEADY_START)
    steady = table[ends >= steady_from]
    steady_ends = pd.DatetimeIndex(steady["period_end"])

    fig, (ax_full, ax_steady) = plt.subplots(
        1, 2, figsize=FIG_INCHES, dpi=DPI, width_ratios=[1.0, 1.55]
    )
    fig.subplots_adjust(left=0.045, right=0.99, top=0.815, bottom=0.235, wspace=0.16)

    # Left: full history for context. COVID and the rebound set the scale here.
    _shade_covid(ax_full)
    _plot_reported(ax_full, table)
    if overlay is not None:
        _plot_overlay(ax_full, overlay.table)
    _plot_index(ax_full, table)
    _style(ax_full, ends, every=2)
    ax_full.set_title(
        f"Full history, {_quarter_label(ends.min())} to {_quarter_label(ends.max())}",
        fontsize=11,
        loc="left",
        color=TEXT_COLOR,
        pad=8,
    )

    # Right: the steady-state window, where the index is judged on ordinary-sized moves.
    _plot_reported(ax_steady, steady)
    if overlay is not None:
        o_ends = pd.DatetimeIndex(overlay.table["period_end"])
        _plot_overlay(ax_steady, overlay.table[o_ends >= steady_from])
    _plot_index(ax_steady, steady)
    _style(ax_steady, steady_ends)
    ax_steady.set_title(
        f"Steady state, {_quarter_label(steady_from)} onward "
        "(no crash, no rebound, no base effects)",
        fontsize=11,
        loc="left",
        color=TEXT_COLOR,
        pad=8,
    )

    _header(fig, main_kind)
    _legend(fig, main_kind, overlay_kind)
    _stats_box(fig, _stats_lines(main, main_kind, overlay, overlay_kind or "composite"))

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG, dpi=DPI)
    plt.close(fig)


def main() -> None:
    parser = add_common_args(
        argparse.ArgumentParser(description="Chart the demand index against reported RevPAR")
    )
    parser.add_argument(
        "--index",
        choices=["tsa", "composite"],
        default="tsa",
        help="which series is the bold line; the other is drawn dashed for disclosure",
    )
    parser.add_argument(
        "--no-overlay", action="store_true", help="draw only the --index series, no dashed line"
    )
    args = parser.parse_args()
    composite, tsa_only = build_both(csv=args.csv, force=args.force, skip_trends=args.skip_trends)
    main, other = (tsa_only, composite) if args.index == "tsa" else (composite, tsa_only)
    other_kind = "composite" if args.index == "tsa" else "tsa"
    if args.no_overlay:
        make_chart(main, args.index)
    else:
        make_chart(main, args.index, other, other_kind)
    print(f"[{args.index}]")
    print(summary_line(main))
    print(f"[{other_kind}]")
    print(summary_line(other))
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()

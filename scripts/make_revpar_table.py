"""Write the per-quarter demand-index vs reported-RevPAR comparison as a CSV.

Writes outputs/index_vs_reported_revpar.csv — the numbers behind the chart.

    uv run python -m scripts.make_revpar_table
"""

from __future__ import annotations

import argparse

import pandas as pd

import config
from scripts.revpar_common import add_common_args, load_inputs, summary_line, tsa_only_index
from src import analysis

OUT_CSV = config.OUTPUT_DIR / "index_vs_reported_revpar.csv"
OUT_TSA_CSV = config.OUTPUT_DIR / "tsa_vs_reported_revpar.csv"
OUT_DIAG = config.OUTPUT_DIR / "index_components_vs_reported_revpar.csv"
COLUMNS = ["period_end", "index_yoy", "MAR", "HLT", "H", "average", "index_direction_correct"]


def build_table(cmp: analysis.RevparComparison) -> pd.DataFrame:
    out = cmp.table.reindex(columns=COLUMNS).copy()
    out["period_end"] = pd.to_datetime(out["period_end"]).dt.date
    numeric = [c for c in COLUMNS if c not in ("period_end", "index_direction_correct")]
    out[numeric] = out[numeric].round(2)
    return out


def main() -> None:
    parser = add_common_args(
        argparse.ArgumentParser(description="Tabulate the demand index against reported RevPAR")
    )
    args = parser.parse_args()
    reported, tsa_daily, trends_df, fred = load_inputs(args.csv, args.force, args.skip_trends)
    index = analysis.composite_demand_index(tsa_daily, trends_df, fred)
    cmp = analysis.compare_index_to_reported(index, reported)
    cmp_tsa = analysis.compare_index_to_reported(tsa_only_index(tsa_daily), reported)
    diag = analysis.component_diagnostics(tsa_daily, trends_df, fred, reported)

    table = build_table(cmp)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    build_table(cmp_tsa).to_csv(OUT_TSA_CSV, index=False)
    diag.round(3).to_csv(OUT_DIAG)

    print("composite index:")
    print(table.to_string(index=False))
    print(summary_line(cmp))
    print("\nTSA throughput alone:")
    print(summary_line(cmp_tsa))
    print("\nper-component diagnostic (each leg alone vs avg reported RevPAR YoY):")
    print(diag.round(3).to_string())
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_TSA_CSV}")
    print(f"wrote {OUT_DIAG}")


if __name__ == "__main__":
    main()

"""
Split-sample validation: fit on first half, validate on second half (and
vice versa) for all (region, model) pairs.

Tests whether generator performance degrades out-of-sample, which helps
identify overfitting in high-parameter models.

Library functions live in methods/split_sample.py.  This script provides
the interactive batch entry point (all-region or single-pair runs).

Usage:
  python scripts/split_sample.py                              # all regions/models
  python scripts/split_sample.py --region new_england
  python scripts/split_sample.py --region new_england --model kirsch

Output:
  outputs/split_sample/split_sample_results.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# Make project root importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import N_REALIZATIONS, SPLIT_SAMPLE_DIR, MODELS
from basins import CAMELS_REGIONS
from methods.split_sample import run_split_sample_for_pair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Split-sample validation")
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=N_REALIZATIONS,
        help=f"Realizations per split (default: {N_REALIZATIONS})",
    )
    args = parser.parse_args()

    regions = (
        {args.region: CAMELS_REGIONS[args.region]}
        if args.region
        else dict(CAMELS_REGIONS)
    )
    enabled = sorted(k for k, v in MODELS.items() if v.get("enabled", True))
    if args.model:
        enabled = [args.model]

    SPLIT_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for region_id in sorted(regions.keys()):
        for model_key in enabled:
            logger.info("Split-sample: %s / %s", region_id, model_key)
            rows = run_split_sample_for_pair(
                region_id,
                model_key,
                n_realizations=args.n_realizations,
            )
            all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
        out_path = SPLIT_SAMPLE_DIR / "split_sample_results.csv"
        df.to_csv(out_path, index=False)
        logger.info("Saved %s (%d rows)", out_path, len(df))

        summary = (
            df.groupby(["model", "validation"])["mare"].mean().unstack("validation")
        )
        if "in_sample" in summary.columns and "out_of_sample" in summary.columns:
            summary["degradation"] = summary["out_of_sample"] - summary["in_sample"]
            summary = summary.sort_values("degradation", ascending=False)
            print("\n" + "=" * 60)
            print("Split-Sample Summary (averaged across regions and splits)")
            print("=" * 60)
            print(summary.to_string(float_format="%.4f"))
            print()
    else:
        logger.warning("No results produced.")


if __name__ == "__main__":
    main()

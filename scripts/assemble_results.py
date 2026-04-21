"""
Serial assembly script: collects per-(region, model) metric CSVs into
cross-region summary files.

Run this after all analyze_single tasks have completed. By default,
calls check_stage_complete("analyze") first; aborts if any task is
missing. Use --skip-check to bypass (e.g. for single-region assembly
during a smoke test).

Usage:
  python scripts/assemble_results.py                    # all regions, checked
  python scripts/assemble_results.py --region new_england --skip-check
"""

import argparse
import sys
from pathlib import Path

# Make project root importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ACTIVE_REGIONS
from methods.assembly import assemble


def main():
    parser = argparse.ArgumentParser(
        description="Assemble per-region metric CSVs into cross-region summaries"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Single region to assemble (default: all)",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Do not run check_stage_complete --stage analyze before assembling",
    )
    args = parser.parse_args()

    region_filter = None
    if args.region:
        region_filter = [args.region]
    elif ACTIVE_REGIONS:
        region_filter = ACTIVE_REGIONS

    # When assembling the full set, fail fast if any (region, model) pair
    # is missing its metrics CSV. Single-region runs skip this by default
    # because they are typically smoke tests.
    if not args.skip_check and region_filter is None:
        from pipeline.check_stage_complete import check_stage

        if not check_stage("analyze"):
            print(
                "\nERROR: analyze stage incomplete. "
                "Re-run the missing tasks or pass --skip-check to proceed anyway.",
                file=sys.stderr,
            )
            sys.exit(1)

    print("=" * 70)
    print("SynHydro Model Comparison -- Assemble Results")
    print("=" * 70)

    assemble(region_filter=region_filter)

    print("\n" + "=" * 70)
    print("Assembly complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

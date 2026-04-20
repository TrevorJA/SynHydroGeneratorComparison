"""
Generate MAIN manuscript figures (Figures 1-10).

This is the thin orchestrator for the 10 main-text figures defined in
methods/plotting/manuscript.MANUSCRIPT_FIGURES. Figures that need
bespoke code (1, 4, 7, 10) and figures that need post-HPC CSV inputs
(3, 5, 9) ship as labelled placeholder PNGs until real producers are
wired in. See notes/figure_design.md and notes/deliverables_tracker.md
for status.

All output goes to config.MAIN_FIGURE_DIR (figures/main/). For the
much larger per-region diagnostic suite and cross-region summaries,
run 04b_si_figures.py.

Usage:
  python 04a_main_figures.py                          # all 10 main figs
  python 04a_main_figures.py --figure fig03_grand_mare_heatmap
  python 04a_main_figures.py --list-figures
"""

import argparse
import logging

from config import MAIN_FIGURE_DIR, OUTPUT_DIR
from methods.plotting import apply_rcparams
from methods.plotting.manuscript import (
    MANUSCRIPT_FIGURES,
    produce_manuscript_figures,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate main manuscript figures (Figures 1-10)"
    )
    parser.add_argument(
        "--figure",
        type=str,
        default=None,
        help=(
            "Produce only the named manuscript figure key "
            "(e.g. fig03_grand_mare_heatmap)."
        ),
    )
    parser.add_argument(
        "--list-figures",
        action="store_true",
        help="List manuscript figure registry keys and exit",
    )
    args = parser.parse_args()

    if args.list_figures:
        print("Manuscript figure registry:")
        for key, entry in MANUSCRIPT_FIGURES.items():
            print(f"  {key:40s}  {entry['title']}  [{entry['status']}]")
        return

    apply_rcparams()
    only_figures = [args.figure] if args.figure else None

    print("=" * 70)
    print("SynHydro Model Comparison -- MAIN MANUSCRIPT FIGURES")
    print(f"  Destination: {MAIN_FIGURE_DIR}")
    print("=" * 70)

    produced = produce_manuscript_figures(
        OUTPUT_DIR, MAIN_FIGURE_DIR, only=only_figures
    )
    for key, out_path in produced.items():
        print(f"  {key}: {out_path}")

    print("\n" + "=" * 70)
    print(f"Main figures complete ({len(produced)} produced).")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

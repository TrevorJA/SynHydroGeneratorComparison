"""
Generate SI convergence figures from pre-computed convergence CSVs.

Reads outputs/{region}/convergence_{model}.csv and
outputs/{region}/convergence_detail_{model}.csv for each region/model,
then produces per-region figures 22-28 and a cross-region comparison
series. All output goes to config.SI_FIGURE_DIR (figures/si/).

Requires: Run convergence_single.py (or run_convergence.sh SLURM array)
for all (region, model) pairs first.

Usage:
  python 04c_si_convergence_figures.py                       # all regions
  python 04c_si_convergence_figures.py --region new_england  # single region
"""

import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd

from config import OUTPUT_DIR, SI_FIGURE_DIR, MODELS, ACTIVE_REGIONS
from basins import CAMELS_REGIONS
from methods.analysis import assign_colors
from methods.plotting import (
    apply_rcparams,
    save_figure,
    fig_convergence_mare,
    fig_convergence_by_category,
    fig_convergence_heatmap,
    fig_convergence_spatial,
)

logger = logging.getLogger(__name__)

N_REALIZATIONS_SWEEP = [5, 10, 25, 50, 100, 200, 500]


def load_convergence_for_region(region_id):
    """Load and concatenate convergence CSVs for all models in a region.

    Parameters
    ----------
    region_id : str

    Returns
    -------
    summary_df : pd.DataFrame or None
    detail_df : pd.DataFrame or None
    """
    region_output = OUTPUT_DIR / region_id
    eligible_models = [
        k
        for k, v in MODELS.items()
        if v.get("enabled", True) and v["frequency"] != "annual"
    ]

    summary_parts = []
    detail_parts = []

    for model_key in eligible_models:
        summary_path = region_output / f"convergence_{model_key}.csv"
        detail_path = region_output / f"convergence_detail_{model_key}.csv"

        if summary_path.exists():
            summary_parts.append(pd.read_csv(summary_path))
        else:
            logger.warning("No convergence summary for %s / %s", region_id, model_key)

        if detail_path.exists():
            detail_parts.append(pd.read_csv(detail_path))

    summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else None
    detail_df = pd.concat(detail_parts, ignore_index=True) if detail_parts else None
    return summary_df, detail_df


def generate_convergence_figures(region_id, region_cfg, model_colors):
    """Produce convergence figures 22-28 for one region.

    Parameters
    ----------
    region_id : str
    region_cfg : dict
    model_colors : dict
    """
    print(f"\n{'='*70}")
    print(f"Convergence figures: {region_id} -- {region_cfg['description']}")
    print(f"{'='*70}")

    summary_df, detail_df = load_convergence_for_region(region_id)

    if summary_df is None or summary_df.empty:
        print(f"  No convergence data for {region_id} -- skipping")
        return None

    region_fig_dir = SI_FIGURE_DIR / region_id
    region_fig_dir.mkdir(parents=True, exist_ok=True)

    fig_convergence_mare(
        summary_df,
        N_REALIZATIONS_SWEEP,
        model_colors,
        region_fig_dir,
        "22_convergence_mare.png",
    )

    if detail_df is not None and not detail_df.empty:
        for cat, num in [
            ("marginal", "23"),
            ("temporal", "24"),
            ("drought", "25"),
            ("spectral", "26"),
        ]:
            fig_convergence_by_category(
                detail_df,
                cat,
                N_REALIZATIONS_SWEEP,
                model_colors,
                region_fig_dir,
                f"{num}_convergence_{cat}.png",
            )

    fig_convergence_heatmap(
        summary_df,
        model_colors,
        region_fig_dir,
        "27_convergence_heatmap.png",
    )
    fig_convergence_spatial(
        summary_df,
        N_REALIZATIONS_SWEEP,
        model_colors,
        region_fig_dir,
        "28_convergence_spatial.png",
    )

    print(f"  Figures saved to {region_fig_dir}")
    return summary_df


def cross_region_convergence(all_summaries, model_colors):
    """Compare convergence behavior across regions.

    Parameters
    ----------
    all_summaries : dict
        {region_id: summary_df}
    model_colors : dict
    """
    cross_dir = SI_FIGURE_DIR / "cross_region"
    cross_dir.mkdir(parents=True, exist_ok=True)

    all_models = sorted(
        {m for df in all_summaries.values() for m in df["model"].unique()}
    )

    for model_key in all_models:
        fig, ax = plt.subplots(figsize=(8, 5))
        region_palette = plt.cm.Set2.colors
        for i, (region_id, summary_df) in enumerate(all_summaries.items()):
            model_df = summary_df[summary_df["model"] == model_key]
            if model_df.empty:
                continue
            model_df = model_df.sort_values("n_realizations")
            ax.plot(
                model_df["n_realizations"],
                model_df["mare"],
                marker="o",
                markersize=5,
                linewidth=1.5,
                color=region_palette[i % len(region_palette)],
                label=region_id,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Number of Realizations")
        ax.set_ylabel("MARE")
        ax.set_title(f"Convergence by Region -- {model_key}")
        ax.set_xticks(N_REALIZATIONS_SWEEP)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, f"convergence_{model_key}_by_region.png", cross_dir)

    print(f"\n  Cross-region convergence figures saved to {cross_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate convergence figures from pre-computed CSVs"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Single region to process (default: all)",
    )
    args = parser.parse_args()

    apply_rcparams()

    region_filter = [args.region] if args.region else ACTIVE_REGIONS
    if region_filter:
        regions = {k: v for k, v in CAMELS_REGIONS.items() if k in region_filter}
    else:
        regions = dict(CAMELS_REGIONS)

    enabled_models = [k for k, v in MODELS.items() if v.get("enabled", True)]
    model_colors = assign_colors(enabled_models)

    print("=" * 70)
    print("SynHydro Model Comparison -- Convergence Figures")
    print(f"  Regions: {list(regions.keys())}")
    print("=" * 70)

    all_summaries = {}
    for region_id, region_cfg in regions.items():
        summary_df = generate_convergence_figures(region_id, region_cfg, model_colors)
        if summary_df is not None and not summary_df.empty:
            all_summaries[region_id] = summary_df

    if len(all_summaries) >= 2:
        print("\n\nGenerating cross-region convergence figures...")
        cross_region_convergence(all_summaries, model_colors)

    print("\n" + "=" * 70)
    print("Convergence figure generation complete.")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

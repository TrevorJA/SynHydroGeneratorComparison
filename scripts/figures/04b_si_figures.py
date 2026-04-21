"""
Generate SUPPORTING INFORMATION figures.

This is the destination for every figure that is not one of the 10
main-text manuscript figures. By default, any new exploratory figure
you add to this script (or to methods/plotting/) lands in the SI
bucket; main-text figures must be explicitly registered in
methods/plotting/manuscript.MANUSCRIPT_FIGURES.

Covers three layers, all writing under config.SI_FIGURE_DIR:
  (1) Per-region diagnostic figures (density, CDF, ACF, FDC, seasonal
      cycle, drought, PSD, Hurst, etc.)
      -> figures/si/{region}/
  (2) Per-region validation summaries and skill radar
      -> figures/si/{region}/
  (3) Cross-region summary heatmaps
      -> figures/si/cross_region/

For the 10 main manuscript figures, run 04a_main_figures.py.
For convergence figures, run 04c_si_convergence_figures.py (reads the
per-model CSVs produced by convergence_single.py / run_convergence.sh).
QA/QC basin verification figures are produced by 00_retrieve_data.py
into figures/si/basin_data_verification/.

Requires:
  analyze_single.py (Stage 2) for metrics CSVs
  assemble_results.py for cross-region figures

Usage:
  python 04b_si_figures.py                           # all regions
  python 04b_si_figures.py --region new_england
  python 04b_si_figures.py --skip-dist               # only validation/skill figs
"""

import argparse
import logging
import sys
from pathlib import Path

# Make project root importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    DATA_DIR,
    ANALYSIS_DIR,
    CROSS_REGION_DIR,
    SI_FIGURE_DIR,
    MODELS,
    ACTIVE_REGIONS,
    DIAGNOSTICS,
)
from basins import CAMELS_REGIONS
from methods.data import load_region_data, get_reference_site_index
from methods.io import load_ensembles_hdf5
from methods.analysis import assign_colors, build_model_data
from methods.metrics import load_metrics
from methods.plotting import (
    apply_rcparams,
    fig_density,
    fig_cdf,
    fig_acf,
    fig_fdc,
    fig_seasonal_cycle,
    fig_summary_stats,
    fig_qq_plots,
    fig_drought_duration,
    fig_drought_severity,
    fig_annual_extremes,
    fig_cross_site_correlation,
    fig_hurst_exponent,
    fig_seasonal_variance,
    fig_psd,
    fig_skill_radar,
    fig_validation_summary_from_csv,
    fig_cross_region_from_csv,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-region figure generation
# ---------------------------------------------------------------------------


def generate_region_figures(region_id, region_cfg, model_colors, skip_dist=False):
    """Generate all SI figures for one region.

    Parameters
    ----------
    region_id : str
    region_cfg : dict
    model_colors : dict
    skip_dist : bool
        If True, skip distribution figures (density, CDF, etc.) and only
        produce validation/skill figures from CSVs.
    """
    print(f"\n{'='*70}")
    print(f"SI figures: {region_id} -- {region_cfg['description']}")
    print(f"{'='*70}")

    Q_daily, Q_monthly, Q_annual = load_region_data(
        region_id, DATA_DIR, region_cfg["stations"]
    )

    region_output = ANALYSIS_DIR / region_id
    region_fig_dir = SI_FIGURE_DIR / region_id
    region_fig_dir.mkdir(parents=True, exist_ok=True)

    site_idx = get_reference_site_index(Q_monthly)
    print(f"  Reference site: {Q_monthly.columns[site_idx]} (index {site_idx})")

    # -----------------------------------------------------------------------
    # Distribution figures -- require ensemble files
    # -----------------------------------------------------------------------
    if not skip_dist:
        ensembles = load_ensembles_hdf5(region_output, MODELS)
        if not ensembles:
            print("  No ensembles found -- skipping distribution figures")
        else:
            print(f"  Loaded {len(ensembles)} ensembles: {list(ensembles.keys())}")
            _generate_distribution_figures(
                region_id,
                Q_daily,
                Q_monthly,
                Q_annual,
                ensembles,
                model_colors,
                region_fig_dir,
                site_idx,
            )

    # -----------------------------------------------------------------------
    # Validation summary figures -- from pre-computed CSVs
    # -----------------------------------------------------------------------
    print("  Generating validation summary figures from CSVs...")
    monthly_models = [
        k
        for k, v in MODELS.items()
        if v.get("enabled", True) and v["frequency"] in ("monthly", "daily")
    ]
    annual_models = [
        k
        for k, v in MODELS.items()
        if v.get("enabled", True) and v["frequency"] == "annual"
    ]

    for model_key in monthly_models:
        metrics_dict = load_metrics(region_output, model_key)
        if metrics_dict is None:
            print(f"    {model_key}: no metrics CSV -- skipping")
            continue
        fig_validation_summary_from_csv(
            metrics_dict,
            model_key,
            region_fig_dir,
            f"20_validation_summary_{model_key}.png",
        )

    for model_key in annual_models:
        metrics_dict = load_metrics(region_output, model_key)
        if metrics_dict is None:
            print(f"    {model_key}: no metrics CSV -- skipping")
            continue
        fig_validation_summary_from_csv(
            metrics_dict,
            model_key,
            region_fig_dir,
            f"20b_annual_validation_summary_{model_key}.png",
        )

    # -----------------------------------------------------------------------
    # Skill radar -- still requires ensemble objects (bootstrap CI)
    # -----------------------------------------------------------------------
    ensembles_for_radar = load_ensembles_hdf5(region_output, MODELS)
    if ensembles_for_radar:
        n_bootstrap = DIAGNOSTICS.get("skill_radar_n_bootstrap", 20)
        print("  Generating skill radar...")
        fig_skill_radar(
            Q_monthly,
            Q_annual,
            ensembles_for_radar,
            MODELS,
            model_colors,
            region_fig_dir,
            "21_skill_radar.png",
            site_index=site_idx,
            n_bootstrap=n_bootstrap,
        )
    else:
        print("  No ensembles for skill radar -- skipping")

    print(f"  SI figures saved to {region_fig_dir}")


def _generate_distribution_figures(
    region_id,
    Q_daily,
    Q_monthly,
    Q_annual,
    ensembles,
    model_colors,
    region_fig_dir,
    site_idx,
):
    """Generate all distribution-based figures for one region."""
    monthly_data = build_model_data(
        ensembles,
        "monthly",
        Q_monthly.iloc[:, site_idx].values,
        MODELS,
    )
    annual_data = build_model_data(
        ensembles,
        "annual",
        Q_annual.iloc[:, site_idx].values,
        MODELS,
    )

    acf_lag_m = DIAGNOSTICS.get("acf_max_lag_monthly", 24)
    acf_lag_a = DIAGNOSTICS.get("acf_max_lag_annual", 10)
    fdc_log = DIAGNOSTICS.get("fdc_log_scale", True)
    n_monthly = len(monthly_data) - 1
    n_annual = len(annual_data) - 1
    monthly_models_exist = any(
        MODELS[k]["frequency"] in ("monthly", "daily") for k in ensembles
    )

    if n_monthly > 0:
        fig_density(
            monthly_data,
            "Monthly",
            "01_monthly_density.png",
            model_colors,
            region_fig_dir,
        )
        fig_cdf(
            monthly_data, "Monthly", "03_monthly_cdf.png", model_colors, region_fig_dir
        )
        fig_acf(
            monthly_data,
            acf_lag_m,
            "months",
            "Monthly",
            "05_monthly_acf.png",
            model_colors,
            region_fig_dir,
        )
        fig_fdc(
            monthly_data,
            "Monthly",
            fdc_log,
            "07_monthly_fdc.png",
            model_colors,
            region_fig_dir,
        )
        fig_summary_stats(
            monthly_data,
            "Monthly",
            "10_monthly_summary_stats.png",
            model_colors,
            region_fig_dir,
        )
        fig_psd(
            monthly_data,
            "Monthly",
            "months",
            "18_monthly_psd.png",
            model_colors,
            region_fig_dir,
        )

    if n_annual > 0:
        fig_density(
            annual_data,
            "Annual",
            "02_annual_density.png",
            model_colors,
            region_fig_dir,
        )
        fig_cdf(
            annual_data, "Annual", "04_annual_cdf.png", model_colors, region_fig_dir
        )
        fig_acf(
            annual_data,
            acf_lag_a,
            "years",
            "Annual",
            "06_annual_acf.png",
            model_colors,
            region_fig_dir,
        )
        fig_fdc(
            annual_data,
            "Annual",
            fdc_log,
            "08_annual_fdc.png",
            model_colors,
            region_fig_dir,
        )
        fig_qq_plots(
            annual_data,
            "Annual",
            "11_annual_qq_plots.png",
            model_colors,
            region_fig_dir,
        )
        fig_hurst_exponent(
            annual_data,
            "Annual",
            "16_hurst_exponent.png",
            model_colors,
            region_fig_dir,
        )
        fig_psd(
            annual_data,
            "Annual",
            "years",
            "19_annual_psd.png",
            model_colors,
            region_fig_dir,
        )

    if monthly_models_exist:
        fig_seasonal_cycle(
            Q_monthly,
            ensembles,
            MODELS,
            model_colors,
            region_fig_dir,
            "09_seasonal_cycle.png",
            site_idx,
        )
        fig_drought_duration(
            Q_monthly,
            ensembles,
            MODELS,
            model_colors,
            region_fig_dir,
            "12_drought_duration.png",
            site_idx,
        )
        fig_drought_severity(
            Q_monthly,
            ensembles,
            MODELS,
            model_colors,
            region_fig_dir,
            "13_drought_severity.png",
            site_idx,
        )
        fig_annual_extremes(
            Q_monthly,
            ensembles,
            MODELS,
            model_colors,
            region_fig_dir,
            "14_annual_extremes.png",
            site_idx,
        )
        fig_seasonal_variance(
            Q_monthly,
            ensembles,
            MODELS,
            model_colors,
            region_fig_dir,
            "17_seasonal_variance.png",
            site_idx,
        )

    if Q_monthly.shape[1] >= 2:
        fig_cross_site_correlation(
            Q_monthly,
            ensembles,
            MODELS,
            region_fig_dir,
            "15_cross_site_correlation.png",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate SI (per-region + cross-region) diagnostic figures"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Single region to process (default: all enabled regions)",
    )
    parser.add_argument(
        "--skip-dist",
        action="store_true",
        help="Skip distribution figures (fast; only produces validation/skill figs)",
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
    print("SynHydro Model Comparison -- SUPPORTING INFORMATION FIGURES")
    print(f"  Regions: {list(regions.keys())}")
    print(f"  Destination: {SI_FIGURE_DIR}")
    print(f"  Skip distribution figures: {args.skip_dist}")
    print("=" * 70)

    for region_id, region_cfg in regions.items():
        generate_region_figures(
            region_id, region_cfg, model_colors, skip_dist=args.skip_dist
        )

    # Cross-region figures from mare.csv
    mare_csv = CROSS_REGION_DIR / "mare.csv"
    if mare_csv.exists():
        print("\n\nGenerating cross-region SI figures from CSV...")
        cross_dir = SI_FIGURE_DIR / "cross_region"
        cross_dir.mkdir(parents=True, exist_ok=True)
        fig_cross_region_from_csv(mare_csv, model_colors, cross_dir)
    else:
        print(
            "\nNo mare.csv found -- run assemble_results.py to generate cross-region figures"
        )

    print("\n" + "=" * 70)
    print("SI figure generation complete.")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

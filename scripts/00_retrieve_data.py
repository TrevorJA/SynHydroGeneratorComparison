"""
Download and cache CAMELS streamflow data for all configured regions.

Uses pygeohydro.get_camels() to download the full CAMELS dataset (671
stations, 1980-2014) in a single bulk operation, then extracts and caches
per-region subsets. Includes QA/QC checks and generates verification
figures in figures/basin_data_verification/.

Run once locally or on an HPC login node before submitting batch jobs.

Usage:
  python 00_retrieve_data.py                     # All regions
  python 00_retrieve_data.py --region new_england # Single region
  python 00_retrieve_data.py --verify             # Check cache status only
  python 00_retrieve_data.py --skip-figures       # Skip verification figures
"""

import argparse
import logging
import sys
from pathlib import Path

# Make project root importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_DIR, SI_FIGURE_DIR
from basins import CAMELS_REGIONS, ALL_STATION_IDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data retrieval
# ============================================================================


def retrieve_all_regions(regions, camels_qobs=None):
    """Download and cache data for specified regions.

    Parameters
    ----------
    regions : dict
        Subset of CAMELS_REGIONS to retrieve.
    camels_qobs : xarray.Dataset, optional
        Pre-downloaded CAMELS discharge dataset. If None, downloads it.

    Returns
    -------
    tuple of (dict, GeoDataFrame)
        region_data: mapping region_id -> (Q_daily, Q_monthly, Q_annual).
        camels_attrs: full CAMELS attributes GeoDataFrame (for spatial figures).
    """
    from methods.data import load_region_data

    camels_attrs = None
    if camels_qobs is None:
        from methods.data import download_camels

        print("\nDownloading full CAMELS dataset via pygeohydro.get_camels()...")
        camels_attrs, camels_qobs = download_camels()
        print(
            f"  {len(camels_attrs)} stations, discharge shape: {camels_qobs['discharge'].shape}"
        )

    region_data = {}
    for region_id, region_cfg in regions.items():
        stations = region_cfg["stations"]
        print(f"\n  {region_id}: {region_cfg['description']}")
        print(f"    Stations: {stations}")

        try:
            Q_daily, Q_monthly, Q_annual = load_region_data(
                region_id, DATA_DIR, stations, camels_qobs
            )
            region_data[region_id] = (Q_daily, Q_monthly, Q_annual)

            print(
                f"    Daily:   {Q_daily.shape} ({Q_daily.index[0].date()} to {Q_daily.index[-1].date()})"
            )
            print(f"    Monthly: {Q_monthly.shape}")
            print(f"    Annual:  {Q_annual.shape}")

        except Exception as e:
            logger.error("Failed to retrieve %s: %s", region_id, e)
            import traceback

            traceback.print_exc()

    return region_data, camels_attrs


# ============================================================================
# QA/QC checks
# ============================================================================


def run_qaqc(region_data):
    """Run QA/QC checks on all retrieved region data.

    Returns a dict of {region_id: qaqc_report}.
    """
    print("\n" + "=" * 70)
    print("QA/QC Report")
    print("=" * 70)

    reports = {}
    for region_id, (Q_daily, Q_monthly, Q_annual) in region_data.items():
        report = {}
        n_days = len(Q_daily)
        n_sites = Q_daily.shape[1]
        start = Q_daily.index[0].date()
        end = Q_daily.index[-1].date()
        n_years = (Q_daily.index[-1] - Q_daily.index[0]).days / 365.25

        report["n_days"] = n_days
        report["n_sites"] = n_sites
        report["start"] = start
        report["end"] = end
        report["n_years"] = n_years

        # Missing values
        n_nan = Q_daily.isna().sum()
        pct_nan = n_nan / n_days * 100
        report["missing_per_site"] = n_nan.to_dict()
        report["missing_pct_per_site"] = pct_nan.to_dict()

        # Zero/negative values
        n_zero = (Q_daily <= 0).sum()
        pct_zero = n_zero / n_days * 100
        report["zero_per_site"] = n_zero.to_dict()
        report["zero_pct_per_site"] = pct_zero.to_dict()

        # Basic statistics
        report["mean_cms"] = Q_daily.mean().to_dict()
        report["std_cms"] = Q_daily.std().to_dict()
        report["cv"] = (Q_daily.std() / Q_daily.mean()).to_dict()
        report["skewness"] = Q_daily.skew().to_dict()
        report["min_cms"] = Q_daily.min().to_dict()
        report["max_cms"] = Q_daily.max().to_dict()

        reports[region_id] = report

        # Print summary
        region_cfg = CAMELS_REGIONS[region_id]
        print(f"\n  {region_id} ({region_cfg['climate_type']})")
        print(f"    Period:   {start} to {end} ({n_years:.1f} years)")
        print(f"    Sites:    {n_sites}")

        any_issues = False
        for site in Q_daily.columns:
            issues = []
            if pct_nan[site] > 0:
                issues.append(f"{pct_nan[site]:.1f}% missing")
            if pct_zero[site] > 1:
                issues.append(f"{pct_zero[site]:.1f}% zero/neg")
            if n_years < 25:
                issues.append(f"short record ({n_years:.0f} yr)")

            status = "OK" if not issues else ", ".join(issues)
            mean_val = Q_daily[site].mean()
            cv_val = Q_daily[site].std() / mean_val if mean_val > 0 else float("inf")
            print(f"    {site}: mean={mean_val:.2f} m3/s, CV={cv_val:.2f}  [{status}]")
            if issues:
                any_issues = True

        if not any_issues:
            print(f"    All stations PASS QA/QC")

    return reports


# ============================================================================
# Nonstationarity screening
# ============================================================================


def mann_kendall_test(x):
    """Perform the Mann-Kendall trend test on a 1-D array.

    Parameters
    ----------
    x : array-like
        Time series values (assumed equally spaced).

    Returns
    -------
    dict
        Keys: tau, p_value, trend (str), sen_slope.
    """
    from scipy.stats import kendalltau

    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 10:
        return {
            "tau": np.nan,
            "p_value": np.nan,
            "trend": "insufficient",
            "sen_slope": np.nan,
        }

    t = np.arange(n)
    tau, p = kendalltau(t, x)

    # Sen's slope
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if j != i:
                slopes.append((x[j] - x[i]) / (j - i))
    sen_slope = float(np.median(slopes)) if slopes else np.nan

    if p < 0.05:
        trend = "increasing" if tau > 0 else "decreasing"
    else:
        trend = "none"

    return {
        "tau": float(tau),
        "p_value": float(p),
        "trend": trend,
        "sen_slope": sen_slope,
    }


def run_nonstationarity_screen(region_data):
    """Run Mann-Kendall trend test on annual flows for all basins.

    Saves results to outputs/nonstationarity_screen.csv.

    Parameters
    ----------
    region_data : dict
        {region_id: (Q_daily, Q_monthly, Q_annual)}
    """
    from config import OUTPUT_DIR

    rows = []
    print("\n" + "=" * 70)
    print("Nonstationarity Screen (Mann-Kendall on annual flows)")
    print("=" * 70)

    for region_id, (Q_daily, Q_monthly, Q_annual) in region_data.items():
        for site in Q_annual.columns:
            values = Q_annual[site].dropna().values
            result = mann_kendall_test(values)
            rows.append(
                {
                    "region": region_id,
                    "station": site,
                    "n_years": len(values),
                    "tau": result["tau"],
                    "p_value": result["p_value"],
                    "trend": result["trend"],
                    "sen_slope": result["sen_slope"],
                }
            )
            flag = "*" if result["p_value"] < 0.05 else " "
            print(
                f"  {flag} {region_id:25s} {site}  tau={result['tau']:+.3f}  "
                f"p={result['p_value']:.4f}  {result['trend']}"
            )

    if rows:
        df = pd.DataFrame(rows)
        out_dir = OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "nonstationarity_screen.csv"
        df.to_csv(out_path, index=False)
        print(f"\n  Saved {out_path}")

        n_sig = sum(1 for r in rows if r["p_value"] < 0.05)
        n_total = len(rows)
        print(f"  {n_sig}/{n_total} stations show significant trends (p < 0.05)")


# ============================================================================
# Verification figures
# ============================================================================


def create_verification_figures(region_data):
    """Create basin data verification figures."""
    fig_dir = SI_FIGURE_DIR / "basin_data_verification"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )

    all_regions = sorted(region_data.keys())
    n_regions = len(all_regions)
    region_palette = plt.cm.Set2.colors

    # --- Figure 1: Temporal coverage bar chart ---
    fig, ax = plt.subplots(
        figsize=(12, max(4, n_regions * 0.8 + len(ALL_STATION_IDS) * 0.25))
    )
    y_pos = 0
    y_ticks = []
    y_labels = []
    colors_list = []

    for i, region_id in enumerate(all_regions):
        Q_daily = region_data[region_id][0]
        color = region_palette[i % len(region_palette)]
        for site in Q_daily.columns:
            valid = Q_daily[site].dropna()
            if len(valid) > 0:
                ax.barh(
                    y_pos,
                    (valid.index[-1] - valid.index[0]).days,
                    left=valid.index[0].toordinal(),
                    height=0.7,
                    color=color,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.3,
                )
            y_ticks.append(y_pos)
            y_labels.append(f"{site}")
            y_pos += 1
        y_pos += 0.5  # gap between regions

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.invert_yaxis()

    # Convert x-axis to dates
    import matplotlib.dates as mdates

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))

    # Add region labels on right side
    y_pos = 0
    for i, region_id in enumerate(all_regions):
        n_sites = region_data[region_id][0].shape[1]
        mid_y = y_pos + (n_sites - 1) / 2
        ax.text(
            1.02,
            mid_y,
            region_id,
            transform=ax.get_yaxis_transform(),
            fontsize=8,
            fontweight="bold",
            color=region_palette[i % len(region_palette)],
            va="center",
        )
        y_pos += n_sites + 0.5

    ax.set_xlabel("Date")
    ax.set_title("CAMELS Station Temporal Coverage by Region")
    fig.tight_layout()
    fig.savefig(fig_dir / "temporal_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved temporal_coverage.png")

    # --- Figure 2: Monthly flow distributions per region (box plots) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, region_id in enumerate(all_regions[:6]):
        ax = axes[i]
        Q_monthly = region_data[region_id][1]
        color = region_palette[i % len(region_palette)]

        bp = ax.boxplot(
            [Q_monthly[col].dropna().values for col in Q_monthly.columns],
            labels=[col[-4:] for col in Q_monthly.columns],
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="red", markersize=4),
        )
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.6)

        ax.set_ylabel("Monthly Flow (m3/s)")
        ax.set_title(
            f"{region_id}\n({CAMELS_REGIONS[region_id]['climate_type']})", fontsize=10
        )
        ax.tick_params(axis="x", rotation=45)

    for i in range(n_regions, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Monthly Streamflow Distributions by Region", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "monthly_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved monthly_distributions.png")

    # --- Figure 3: Seasonal cycles per region ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    months = np.arange(1, 13)
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    for i, region_id in enumerate(all_regions[:6]):
        ax = axes[i]
        Q_monthly = region_data[region_id][1]

        for site in Q_monthly.columns:
            series = Q_monthly[site]
            monthly_means = series.groupby(series.index.month).mean()
            ax.plot(
                months,
                monthly_means.values,
                marker="o",
                markersize=3,
                linewidth=1.5,
                alpha=0.8,
                label=site[-4:],
            )

        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)
        ax.set_ylabel("Mean Monthly Flow (m3/s)")
        ax.set_title(f"{region_id}", fontsize=10)
        ax.legend(fontsize=7)

    for i in range(n_regions, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Mean Seasonal Cycle by Region", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "seasonal_cycles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved seasonal_cycles.png")

    # --- Figure 4: Annual time series per region ---
    fig, axes = plt.subplots(n_regions, 1, figsize=(14, 3 * n_regions), sharex=True)
    if n_regions == 1:
        axes = [axes]

    for i, region_id in enumerate(all_regions):
        ax = axes[i]
        Q_annual = region_data[region_id][2]
        color = region_palette[i % len(region_palette)]

        for site in Q_annual.columns:
            ax.plot(
                Q_annual.index,
                Q_annual[site],
                marker="o",
                markersize=3,
                linewidth=1,
                alpha=0.8,
                label=site[-4:],
            )

        ax.set_ylabel("Annual Flow (m3/s)")
        ax.set_title(
            f"{region_id} ({CAMELS_REGIONS[region_id]['climate_type']})", fontsize=10
        )
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Year")
    fig.suptitle(
        "Annual Streamflow Time Series by Region", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "annual_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved annual_timeseries.png")

    # --- Figure 5: Flow duration curves per region ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, region_id in enumerate(all_regions[:6]):
        ax = axes[i]
        Q_daily = region_data[region_id][0]

        for site in Q_daily.columns:
            values = Q_daily[site].dropna().values
            sorted_v = np.sort(values)[::-1]
            exceed = np.arange(1, len(sorted_v) + 1) / len(sorted_v) * 100
            ax.plot(exceed, sorted_v, linewidth=1, alpha=0.8, label=site[-4:])

        ax.set_yscale("log")
        ax.set_xlabel("Exceedance (%)")
        ax.set_ylabel("Daily Flow (m3/s)")
        ax.set_title(f"{region_id}", fontsize=10)
        ax.legend(fontsize=7)

    for i in range(n_regions, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Daily Flow Duration Curves by Region", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "flow_duration_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved flow_duration_curves.png")

    # --- Figure 6: Data quality summary heatmap ---
    all_sites = []
    all_regions_labels = []
    pct_valid = []
    mean_flow = []
    cv_flow = []

    for region_id in all_regions:
        Q_daily = region_data[region_id][0]
        for site in Q_daily.columns:
            all_sites.append(site)
            all_regions_labels.append(region_id)
            valid_pct = Q_daily[site].notna().mean() * 100
            pct_valid.append(valid_pct)
            mean_val = Q_daily[site].mean()
            mean_flow.append(mean_val)
            std_val = Q_daily[site].std()
            cv_flow.append(std_val / mean_val if mean_val > 0 else np.nan)

    summary_df = pd.DataFrame(
        {
            "station": all_sites,
            "region": all_regions_labels,
            "valid_pct": pct_valid,
            "mean_cms": mean_flow,
            "cv": cv_flow,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(all_sites) * 0.3)))

    for ax, col, title, cmap in [
        (axes[0], "valid_pct", "Data Completeness (%)", "RdYlGn"),
        (axes[1], "mean_cms", "Mean Flow (m3/s)", "Blues"),
        (axes[2], "cv", "CV (std/mean)", "YlOrRd"),
    ]:
        vals = summary_df[col].values.reshape(-1, 1)
        im = ax.imshow(vals, aspect="auto", cmap=cmap)
        ax.set_yticks(range(len(all_sites)))
        ax.set_yticklabels(
            [f"{r[:8]}:{s}" for r, s in zip(all_regions_labels, all_sites)],
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_title(title, fontsize=10)

        for j in range(len(all_sites)):
            v = vals[j, 0]
            if np.isfinite(v):
                fmt = f"{v:.0f}" if col == "valid_pct" else f"{v:.2f}"
                ax.text(0, j, fmt, ha="center", va="center", fontsize=7)

        fig.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle("Basin Data Quality Summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "data_quality_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved data_quality_summary.png")

    print(f"\n  All verification figures saved to {fig_dir}")


# ============================================================================
# Cache verification
# ============================================================================


def verify_cached_data():
    """Check that all regions have cached data and report status."""
    print("\nData cache verification:")
    print(f"  Cache directory: {DATA_DIR}")

    all_ok = True
    for region_id, region_cfg in CAMELS_REGIONS.items():
        daily_path = DATA_DIR / region_id / "daily.csv"
        monthly_path = DATA_DIR / region_id / "monthly.csv"
        annual_path = DATA_DIR / region_id / "annual.csv"

        if daily_path.exists() and monthly_path.exists() and annual_path.exists():
            Q = pd.read_csv(daily_path, index_col=0, parse_dates=True)
            n_stations = Q.shape[1]
            expected = len(region_cfg["stations"])
            status = (
                "OK"
                if n_stations == expected
                else f"WARN: {n_stations}/{expected} stations"
            )
            print(
                f"  {region_id:25s}  {status}  ({Q.shape[0]} days, {n_stations} sites)"
            )
        else:
            missing = []
            if not daily_path.exists():
                missing.append("daily")
            if not monthly_path.exists():
                missing.append("monthly")
            if not annual_path.exists():
                missing.append("annual")
            print(f"  {region_id:25s}  MISSING: {', '.join(missing)}")
            all_ok = False

    if all_ok:
        print("\nAll regions cached and ready.")
    else:
        print("\nSome regions missing. Run without --verify to download.")
    return all_ok


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download and cache CAMELS streamflow data"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Single region to download (default: all)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only check cached data status, don't download",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip generating verification figures",
    )
    args = parser.parse_args()

    if args.verify:
        ok = verify_cached_data()
        sys.exit(0 if ok else 1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.region:
        if args.region not in CAMELS_REGIONS:
            print(f"ERROR: Unknown region '{args.region}'")
            print(f"Available: {list(CAMELS_REGIONS.keys())}")
            sys.exit(1)
        regions = {args.region: CAMELS_REGIONS[args.region]}
    else:
        regions = CAMELS_REGIONS

    print("=" * 70)
    print("CAMELS Data Retrieval & QA/QC")
    print(f"  Regions:  {len(regions)}")
    print(f"  Stations: {sum(len(r['stations']) for r in regions.values())}")
    print(f"  Cache:    {DATA_DIR}")
    print("=" * 70)

    # Download CAMELS once, extract all regions
    region_data, camels_attrs = retrieve_all_regions(regions)

    if not region_data:
        print("No data retrieved successfully.")
        sys.exit(1)

    # QA/QC
    reports = run_qaqc(region_data)

    # Nonstationarity screen (Mann-Kendall on annual flows)
    run_nonstationarity_screen(region_data)

    # Verification figures
    if not args.skip_figures:
        print("\nGenerating verification figures...")
        create_verification_figures(region_data)

        # Spatial figures (require CAMELS attrs with geometry)
        if camels_attrs is not None:
            print("\nGenerating spatial figures...")
            from methods.plotting.spatial_overview import create_spatial_figures

            fig_dir = SI_FIGURE_DIR / "basin_data_verification"
            create_spatial_figures(camels_attrs, regions, fig_dir)

    # Summary
    succeeded = list(region_data.keys())
    failed = [r for r in regions if r not in region_data]

    print("\n" + "=" * 70)
    print("Retrieval Complete")
    print("=" * 70)
    print(f"  Succeeded: {len(succeeded)}/{len(regions)}  {succeeded}")
    if failed:
        print(f"  Failed:    {len(failed)}/{len(regions)}  {failed}")
    print(f"\n  Data cached in: {DATA_DIR}")
    print(f"  Figures in:     {SI_FIGURE_DIR / 'basin_data_verification'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

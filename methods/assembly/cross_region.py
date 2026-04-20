"""Cross-region assembly: collect per-(region, model) metric CSVs into summaries."""

import logging

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, MODELS
from basins import CAMELS_REGIONS
from methods.metrics import load_metrics
from .tier_concordance import assemble_tier_concordance

logger = logging.getLogger(__name__)


def assemble(region_filter=None) -> None:
    """Walk all (region, model) pairs and assemble cross-region CSVs.

    Parameters
    ----------
    region_filter : list of str or None
        If provided, restrict to these region IDs.
    """
    if region_filter:
        regions = {k: v for k, v in CAMELS_REGIONS.items() if k in region_filter}
    else:
        regions = dict(CAMELS_REGIONS)

    enabled_models = sorted(k for k, v in MODELS.items() if v.get("enabled", True))

    cross_dir = OUTPUT_DIR / "cross_region"
    cross_dir.mkdir(parents=True, exist_ok=True)

    mare_rows = []
    all_metrics_rows = []
    all_dist_rows = []
    missing_pairs = []

    for region_id in sorted(regions.keys()):
        region_output = OUTPUT_DIR / region_id
        for model_key in enabled_models:
            metrics_dict = load_metrics(region_output, model_key)
            if metrics_dict is None:
                missing_pairs.append((region_id, model_key))
                continue

            # -- all_metrics (tidy) --
            df = metrics_dict["metrics"].copy()
            df.insert(0, "model", model_key)
            df.insert(0, "region", region_id)
            all_metrics_rows.append(df)

            # -- all_distribution_stats --
            dist_df = metrics_dict["distribution_stats"].copy()
            dist_df.insert(0, "model", model_key)
            dist_df.insert(0, "region", region_id)
            all_dist_rows.append(dist_df)

            # -- MARE summary row --
            summary_df = metrics_dict["validation_summary"]
            summary_lookup = dict(zip(summary_df["metric_name"], summary_df["value"]))
            row = {
                "region": region_id,
                "model": model_key,
                "mare": summary_lookup.get("mean_absolute_relative_error"),
                "median_are": summary_lookup.get("median_absolute_relative_error"),
                "max_are": summary_lookup.get("max_absolute_relative_error"),
                "spatial_rmse": summary_lookup.get("spatial_correlation_rmse"),
                "annual_tier_mare": summary_lookup.get("annual_tier_mare"),
            }
            # Per-category MARE columns (monthly-tier and annual-tier)
            for key, val in summary_lookup.items():
                if key.startswith("mare_") or key.startswith("annual_mare_"):
                    row[key] = val
            # Alias: spatial is treated as a first-class metric category in
            # figure_design.md even though the underlying metric is a cross-
            # site correlation RMSE (not a per-metric MARE). Expose it as
            # mare_spatial for Fig 5's 6-panel decomposition.
            row.setdefault("mare_spatial", row["spatial_rmse"])
            mare_rows.append(row)

    # Save cross-region MARE table
    if mare_rows:
        mare_df = pd.DataFrame(mare_rows)
        mare_path = cross_dir / "mare.csv"
        mare_df.to_csv(mare_path, index=False)
        print(f"Saved {mare_path}  ({len(mare_df)} rows)")

        # Compute rank scores per region (lower MARE = better = rank 1)
        rank_rows = []
        for region_id, grp in mare_df.groupby("region"):
            grp_sorted = grp.dropna(subset=["mare"])
            if grp_sorted.empty:
                continue
            ranked = grp_sorted["mare"].rank(method="average")
            for idx, row in grp_sorted.iterrows():
                rank_rows.append(
                    {
                        "region": region_id,
                        "model": row["model"],
                        "mare": row["mare"],
                        "mare_rank": ranked.loc[idx],
                    }
                )
        if rank_rows:
            rank_df = pd.DataFrame(rank_rows)
            # Mean rank across regions
            mean_ranks = (
                rank_df.groupby("model")["mare_rank"]
                .agg(["mean", "std"])
                .rename(columns={"mean": "mean_rank", "std": "rank_std"})
                .sort_values("mean_rank")
            )
            rank_path = cross_dir / "rank_scores.csv"
            rank_df.to_csv(rank_path, index=False)
            print(f"Saved {rank_path}  ({len(rank_df)} rows)")

            mean_rank_path = cross_dir / "mean_rank_scores.csv"
            mean_ranks.to_csv(mean_rank_path)
            print(f"Saved {mean_rank_path}")

            # Spearman correlation between MARE and rank score
            from scipy.stats import spearmanr

            model_mare_mean = mare_df.groupby("model")["mare"].mean()
            common = mean_ranks.index.intersection(model_mare_mean.index)
            if len(common) >= 3:
                rho, pval = spearmanr(
                    model_mare_mean.loc[common],
                    mean_ranks.loc[common, "mean_rank"],
                )
                print(
                    f"MARE vs rank-score Spearman rho = {rho:.3f} " f"(p = {pval:.4f})"
                )
    else:
        print("WARNING: no MARE rows assembled -- check that analyze_single has run")

    # Save all_metrics
    if all_metrics_rows:
        all_metrics_df = pd.concat(all_metrics_rows, ignore_index=True)
        all_metrics_path = cross_dir / "all_metrics.csv"
        all_metrics_df.to_csv(all_metrics_path, index=False)
        print(f"Saved {all_metrics_path}  ({len(all_metrics_df)} rows)")

    # Save all_distribution_stats
    if all_dist_rows:
        all_dist_df = pd.concat(all_dist_rows, ignore_index=True)
        all_dist_path = cross_dir / "all_distribution_stats.csv"
        all_dist_df.to_csv(all_dist_path, index=False)
        print(f"Saved {all_dist_path}  ({len(all_dist_df)} rows)")

    # Tier 1 vs Tier 2 concordance artifact (see
    # notes/decisions_log.md 2026-04-17 tier-concordance-check-added)
    if mare_rows:
        assemble_tier_concordance(OUTPUT_DIR)

    # Report missing pairs
    if missing_pairs:
        print(
            f"\nMissing metrics for {len(missing_pairs)} (region, model) pair(s) "
            f"-- run analyze_single for these first:"
        )
        for region_id, model_key in missing_pairs:
            print(f"  {region_id:25s}  {model_key}")
    else:
        print("\nAll (region, model) pairs present.")

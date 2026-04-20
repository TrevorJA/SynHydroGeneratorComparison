"""Plotting functions that produce figures from pre-computed metric CSVs."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from methods.colors import MODEL_DISPLAY_LABELS
from basins import CAMELS_REGIONS


def fig_validation_summary_from_csv(metrics_dict, model_key, figure_dir, filename):
    """Produce a validation summary heatmap from a pre-loaded metrics dict.

    Reads the "metrics" DataFrame (columns: metric, category, value) and
    produces the same relative-error heatmap as fig_validation_summary, but
    without re-running validate_ensemble.

    Parameters
    ----------
    metrics_dict : dict
        Return value of load_metrics() with key "metrics".
    model_key : str
    figure_dir : Path
    filename : str
    """
    from methods.plotting import save_figure

    df = metrics_dict["metrics"]
    if df.empty:
        return

    pivot = df.pivot_table(
        index=lambda _: model_key,
        columns="metric",
        values="value",
        aggfunc="mean",
    )
    # pivot_table with index=callable gives a RangeIndex; name it explicitly
    pivot.index = [model_key]

    fig, ax = plt.subplots(
        figsize=(max(12, len(pivot.columns) * 0.9), max(4, len(pivot) * 0.7))
    )
    im = ax.imshow(
        pivot.values,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-0.5,
        vmax=0.5,
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(val) > 0.3 else "black",
                )

    fig.colorbar(im, ax=ax, label="Relative Error", shrink=0.8)
    ax.set_title(
        f"Validation Summary -- Relative Error by Metric\n(model: {model_key})",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, filename, figure_dir)


def fig_cross_region_from_csv(mare_csv_path, model_colors, figure_dir):
    """Produce cross-region heatmap and aridity scatter from mare.csv.

    Parameters
    ----------
    mare_csv_path : Path
    model_colors : dict
    figure_dir : Path
    """
    if not mare_csv_path.exists():
        print(f"  mare.csv not found at {mare_csv_path} -- skipping cross-region figs")
        return

    from methods.plotting import save_figure

    mare_df = pd.read_csv(mare_csv_path)
    all_models = sorted(mare_df["model"].unique())
    all_regions = sorted(mare_df["region"].unique())

    mare_matrix = mare_df.pivot_table(
        index="region", columns="model", values="mare", aggfunc="mean"
    )
    # Ensure all regions/models appear even if some are missing
    mare_matrix = mare_matrix.reindex(index=all_regions, columns=all_models)

    # Heatmap
    fig, ax = plt.subplots(
        figsize=(max(10, len(all_models) * 1.2), max(4, len(all_regions) * 0.8))
    )
    vals = mare_matrix.values.astype(float)
    im = ax.imshow(vals, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.5)
    model_labels = [MODEL_DISPLAY_LABELS.get(m, m) for m in all_models]
    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(all_regions)))
    ax.set_yticklabels(all_regions, fontsize=10)

    for i in range(len(all_regions)):
        for j in range(len(all_models)):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > 0.3 else "black",
                )

    fig.colorbar(im, ax=ax, label="MARE", shrink=0.8)
    ax.set_title(
        "Cross-Region Model Performance (MARE)\nLower is better",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, "region_model_mare_heatmap.png", figure_dir)

    # Aridity vs MARE scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_key in all_models:
        aridity_vals = []
        mare_vals = []
        for region_id in all_regions:
            region_cfg = CAMELS_REGIONS.get(region_id)
            if region_cfg is None:
                continue
            aridity_mid = np.mean(region_cfg["aridity_range"])
            row = (
                mare_matrix.loc[region_id, model_key]
                if region_id in mare_matrix.index
                else np.nan
            )
            if np.isfinite(float(row)) if not pd.isna(row) else False:
                aridity_vals.append(aridity_mid)
                mare_vals.append(float(row))
        if aridity_vals:
            ax.scatter(
                aridity_vals,
                mare_vals,
                color=model_colors.get(model_key, "gray"),
                label=MODEL_DISPLAY_LABELS.get(model_key, model_key),
                s=60,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
            )
    ax.set_xlabel("Aridity Index (midpoint of region range)")
    ax.set_ylabel("MARE")
    ax.set_title("Model Performance vs Basin Aridity")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "aridity_vs_mare.png", figure_dir)

    print(f"  Cross-region figures saved to {figure_dir}")

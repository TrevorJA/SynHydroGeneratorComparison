"""Heatmap of relative errors across all 12 metric categories."""

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from synhydro import validate_ensemble

from . import save_figure
from ..analysis import get_monthly_ensembles


def fig_extended_validation_heatmap(
    Q_monthly_hist: pd.DataFrame,
    ensembles: dict,
    models_config: dict,
    model_colors: dict,
    figure_dir: Path,
    filename: str,
    site_index: int = 0,
) -> None:
    monthly_ens = get_monthly_ensembles(ensembles, models_config)
    if not monthly_ens:
        return

    site = Q_monthly_hist.columns[site_index]

    all_results = {}
    for model_key, ensemble in monthly_ens.items():
        try:
            result = validate_ensemble(ensemble, Q_monthly_hist)
            all_results[model_key] = result
        except Exception:
            traceback.print_exc()

    if not all_results:
        return

    rows = []
    for model_key, result in all_results.items():
        df = result.to_dataframe()
        df_site = df[df["site"] == site]
        for _, row in df_site.iterrows():
            rows.append(
                {
                    "model": model_key,
                    "category": row["category"],
                    "metric": row["metric"],
                    "relative_error": row["relative_error"],
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df["label"] = metrics_df["category"] + ": " + metrics_df["metric"]

    pivot = metrics_df.pivot_table(
        index="model",
        columns="label",
        values="relative_error",
    )
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    fig, ax = plt.subplots(
        figsize=(max(14, len(pivot.columns) * 0.45), max(4, len(pivot) * 0.8))
    )
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=6)
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
                    fontsize=5,
                    color="white" if abs(val) > 0.3 else "black",
                )

    fig.colorbar(im, ax=ax, label="Relative Error", shrink=0.8)
    ax.set_title(
        f"Extended Validation: Relative Error by Model and Metric\n"
        f"(site: {site}, {len(pivot.columns)} metrics across "
        f"{len(set(metrics_df['category']))} categories)",
        fontsize=11,
    )
    fig.tight_layout()
    save_figure(fig, filename, figure_dir)

    print("\n  Category-level MARE:")
    cat_mare = (
        metrics_df.groupby(["model", "category"])["relative_error"]
        .apply(lambda x: np.nanmean(np.abs(x)))
        .unstack(fill_value=np.nan)
    )
    print(cat_mare.round(3).to_string())

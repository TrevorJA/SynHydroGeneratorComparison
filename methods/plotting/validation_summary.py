"""Validation summary heatmap of relative errors across metric categories."""

import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from synhydro import validate_ensemble

from . import save_figure

logger = logging.getLogger(__name__)


def fig_validation_summary(
    Q_monthly_hist: pd.DataFrame,
    ensembles: dict,
    models_config: dict,
    figure_dir: Path,
    filename: str,
    site_index: int = 0,
) -> None:
    if not ensembles:
        return

    site = Q_monthly_hist.columns[site_index]

    all_results = {}
    for model_key, ensemble in ensembles.items():
        try:
            if models_config[model_key]["frequency"] == "daily":
                ensemble = ensemble.resample("MS")
            result = validate_ensemble(ensemble, Q_monthly_hist)
            all_results[model_key] = result
        except Exception:
            traceback.print_exc()
            logger.warning("validate_ensemble failed for %s", model_key)

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

    if not rows:
        return

    metrics_df = pd.DataFrame(rows)
    pivot = metrics_df.pivot_table(
        index="model",
        columns="metric",
        values="relative_error",
    )

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
        f"Validation Summary -- Relative Error by Model and Metric\n(site: {site})",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, filename, figure_dir)

    print("\n  Validation summary scores:")
    for model_key, result in all_results.items():
        summary = result.summary
        mare = summary.get("mean_absolute_relative_error", float("nan"))
        spatial = summary.get("spatial_correlation_rmse", float("nan"))
        print(f"    {model_key:20s}  MARE={mare:.4f}  spatial_RMSE={spatial:.4f}")

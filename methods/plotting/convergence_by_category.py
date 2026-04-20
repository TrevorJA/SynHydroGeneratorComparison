"""Subplots per metric within a category, convergence curves per model."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import save_figure


def fig_convergence_by_category(
    detail_df: pd.DataFrame,
    category: str,
    n_realizations_sweep: list,
    model_colors: dict,
    figure_dir: Path,
    filename: str,
) -> None:
    cat_df = detail_df[detail_df["category"] == category]
    if cat_df.empty:
        return
    metrics = sorted(cat_df["metric"].unique())
    n_metrics = len(metrics)
    ncols = min(3, n_metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    for idx, metric in enumerate(metrics):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        metric_df = cat_df[cat_df["metric"] == metric]

        for model_key, grp in metric_df.groupby("model"):
            conv = (
                grp.groupby("n_realizations")["relative_error"]
                .apply(lambda x: np.median(np.abs(x.dropna())))
                .reset_index()
            )
            conv = conv.sort_values("n_realizations")
            ax.plot(
                conv["n_realizations"],
                conv["relative_error"],
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=model_colors.get(model_key, "gray"),
                label=model_key,
            )
        ax.set_xscale("log")
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("n_realizations")
        ax.set_ylabel("|Relative Error|")
        ax.set_xticks(n_realizations_sweep)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(axis="x", labelsize=7, rotation=45)

    for idx in range(n_metrics, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle(
        f"Convergence -- {category.title()} Metrics", fontsize=13, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    save_figure(fig, filename, figure_dir)

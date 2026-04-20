"""Side-by-side heatmaps: smallest vs largest n_realizations."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import save_figure


def fig_convergence_heatmap(
    summary_df: pd.DataFrame,
    model_colors: dict,
    figure_dir: Path,
    filename: str,
) -> None:
    n_min = summary_df["n_realizations"].min()
    n_max = summary_df["n_realizations"].max()
    cols = ["mare", "median_are", "max_are", "spatial_rmse"]

    df_min = summary_df[summary_df["n_realizations"] == n_min].set_index("model")[cols]
    df_max = summary_df[summary_df["n_realizations"] == n_max].set_index("model")[cols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(df_min) * 0.8)))

    for ax, df, title in [
        (ax1, df_min, f"n_realizations = {n_min}"),
        (ax2, df_max, f"n_realizations = {n_max}"),
    ]:
        vals = df.values.astype(float)
        im = ax.imshow(vals, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.5)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index, fontsize=10)
        ax.set_title(title)
        for i in range(len(df.index)):
            for j in range(len(cols)):
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

    fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label="Metric Value")
    fig.suptitle(
        "Validation Metrics: Few vs Many Realizations", fontsize=13, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    save_figure(fig, filename, figure_dir)

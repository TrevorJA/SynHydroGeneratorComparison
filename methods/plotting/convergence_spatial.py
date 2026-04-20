"""Spatial correlation RMSE convergence, multi-site models only."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from . import save_figure


def fig_convergence_spatial(
    summary_df: pd.DataFrame,
    n_realizations_sweep: list,
    model_colors: dict,
    figure_dir: Path,
    filename: str,
) -> None:
    spatial_df = summary_df.dropna(subset=["spatial_rmse"])
    if spatial_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_key, grp in spatial_df.groupby("model"):
        grp = grp.sort_values("n_realizations")
        ax.plot(
            grp["n_realizations"],
            grp["spatial_rmse"],
            marker="o",
            markersize=5,
            linewidth=1.5,
            color=model_colors.get(model_key, "gray"),
            label=model_key,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Number of Realizations")
    ax.set_ylabel("Spatial Correlation RMSE")
    ax.set_title("Convergence of Spatial Correlation RMSE")
    ax.set_xticks(n_realizations_sweep)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    save_figure(fig, filename, figure_dir)

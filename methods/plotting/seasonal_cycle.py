"""Seasonal cycle: monthly mean +/- 1 std for Historical and each model."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import HIST_STYLE, syn_style, save_figure


def fig_seasonal_cycle(
    Q_monthly_hist: pd.DataFrame,
    ensembles: dict,
    models_config: dict,
    model_colors: dict,
    figure_dir: Path,
    filename: str,
    site_index: int = 0,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    months = np.arange(1, 13)
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    hist_series = Q_monthly_hist.iloc[:, site_index]
    hist_means = hist_series.groupby(hist_series.index.month).mean().values
    hist_stds = hist_series.groupby(hist_series.index.month).std().values
    ax.plot(months, hist_means, **HIST_STYLE, marker="o", markersize=5)
    ax.fill_between(
        months,
        hist_means - hist_stds,
        hist_means + hist_stds,
        alpha=0.15,
        color="black",
    )

    for model_key, ensemble in ensembles.items():
        cfg = models_config[model_key]
        if cfg["frequency"] not in ("monthly", "daily"):
            continue
        all_monthly_vals = {m: [] for m in range(1, 13)}
        for i in range(ensemble.metadata.n_realizations):
            df = ensemble.data_by_realization[i]
            if cfg["frequency"] == "daily":
                df = df.resample("MS").sum()
            col = min(site_index, df.shape[1] - 1)
            series = df.iloc[:, col]
            for m in range(1, 13):
                all_monthly_vals[m].extend(series[series.index.month == m].values)
        syn_means = np.array([np.mean(all_monthly_vals[m]) for m in range(1, 13)])
        syn_stds = np.array([np.std(all_monthly_vals[m]) for m in range(1, 13)])
        style = syn_style(model_key, model_colors)
        ax.plot(months, syn_means, **style, marker="s", markersize=4)
        ax.fill_between(
            months,
            syn_means - syn_stds,
            syn_means + syn_stds,
            alpha=0.08,
            color=style["color"],
        )

    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly Flow (cms)")
    ax.set_title("Seasonal Cycle (Mean +/- 1 Std)")
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.legend()
    save_figure(fig, filename, figure_dir)

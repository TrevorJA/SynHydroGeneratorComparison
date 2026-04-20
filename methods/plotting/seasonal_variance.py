"""Coefficient of variation by month -- captures seasonal heteroscedasticity."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import HIST_STYLE, syn_style, save_figure


def fig_seasonal_variance(
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
    hist_cv = []
    for m in range(1, 13):
        vals = hist_series[hist_series.index.month == m].values
        mean_val = np.mean(vals)
        hist_cv.append(np.std(vals) / mean_val if abs(mean_val) > 1e-10 else 0.0)
    ax.plot(months, hist_cv, **HIST_STYLE, marker="o", markersize=5)

    for model_key, ensemble in ensembles.items():
        cfg = models_config[model_key]
        if cfg["frequency"] not in ("monthly", "daily"):
            continue
        monthly_vals = {m: [] for m in range(1, 13)}
        for i in range(ensemble.metadata.n_realizations):
            df = ensemble.data_by_realization[i]
            if cfg["frequency"] == "daily":
                df = df.resample("MS").sum()
            series = df.iloc[:, min(site_index, df.shape[1] - 1)]
            for m in range(1, 13):
                monthly_vals[m].extend(series[series.index.month == m].values)
        syn_cv = []
        for m in range(1, 13):
            vals = np.array(monthly_vals[m])
            mean_val = np.mean(vals)
            syn_cv.append(np.std(vals) / mean_val if abs(mean_val) > 1e-10 else 0.0)
        ax.plot(
            months,
            syn_cv,
            **syn_style(model_key, model_colors),
            marker="s",
            markersize=4,
        )

    ax.set_xlabel("Month")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Seasonal Variability (CV by Month)")
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.legend()
    save_figure(fig, filename, figure_dir)

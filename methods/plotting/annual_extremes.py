"""Compare distributions of annual maximum and minimum monthly flows."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import HIST_STYLE, syn_style, save_figure


def fig_annual_extremes(
    Q_monthly_hist: pd.DataFrame,
    ensembles: dict,
    models_config: dict,
    model_colors: dict,
    figure_dir: Path,
    filename: str,
    site_index: int = 0,
) -> None:
    hist_series = Q_monthly_hist.iloc[:, site_index]
    hist_annual_max = hist_series.resample("YS").max().values
    hist_annual_min = hist_series.resample("YS").min().values

    fig, (ax_max, ax_min) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, hist_vals in [
        (ax_max, hist_annual_max),
        (ax_min, hist_annual_min),
    ]:
        sorted_v = np.sort(hist_vals)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.plot(sorted_v, cdf, **HIST_STYLE)

    for model_key, ensemble in ensembles.items():
        cfg = models_config[model_key]
        if cfg["frequency"] not in ("monthly", "daily"):
            continue
        all_max, all_min = [], []
        for i in range(ensemble.metadata.n_realizations):
            df = ensemble.data_by_realization[i]
            col = min(site_index, df.shape[1] - 1)
            if cfg["frequency"] == "daily":
                series = df.iloc[:, col].resample("MS").sum()
            else:
                series = df.iloc[:, col]
            all_max.extend(series.resample("YS").max().values)
            all_min.extend(series.resample("YS").min().values)

        style = syn_style(model_key, model_colors)
        for ax, vals in [(ax_max, all_max), (ax_min, all_min)]:
            sorted_v = np.sort(vals)
            cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
            ax.plot(sorted_v, cdf, **style, linestyle="--")

    ax_max.set_xlabel("Annual Maximum Monthly Flow (cms)")
    ax_max.set_ylabel("Cumulative Probability")
    ax_max.set_title("Annual Maxima CDF")
    ax_max.legend()
    ax_min.set_xlabel("Annual Minimum Monthly Flow (cms)")
    ax_min.set_ylabel("Cumulative Probability")
    ax_min.set_title("Annual Minima CDF")
    ax_min.legend()
    fig.suptitle("Annual Extreme Flow Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, filename, figure_dir)

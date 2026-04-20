"""Autocorrelation function -- all models on one axis."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import HIST_STYLE, syn_style, save_figure
from ..analysis import compute_acf


def fig_acf(
    model_data: dict,
    max_lag: int,
    lag_unit: str,
    freq_label: str,
    filename: str,
    model_colors: dict,
    figure_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    lags = np.arange(max_lag + 1)
    for name, values in model_data.items():
        acf_vals = compute_acf(values, max_lag)
        if name == "Historical":
            ax.plot(lags, acf_vals, **HIST_STYLE, marker="o", markersize=4)
        else:
            ax.plot(
                lags,
                acf_vals,
                **syn_style(name, model_colors),
                marker="s",
                markersize=3,
            )
    ax.axhline(0, color="gray", linewidth=0.5)
    n_hist = len(model_data["Historical"])
    ci = 1.96 / np.sqrt(n_hist)
    ax.axhline(ci, color="gray", linewidth=0.5, linestyle=":")
    ax.axhline(-ci, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel(f"Lag ({lag_unit})")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"{freq_label} Autocorrelation Functions")
    ax.legend()
    save_figure(fig, filename, figure_dir)

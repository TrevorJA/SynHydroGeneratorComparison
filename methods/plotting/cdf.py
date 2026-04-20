"""Empirical CDF -- all models on one axis."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import HIST_STYLE, syn_style, save_figure


def fig_cdf(
    model_data: dict,
    freq_label: str,
    filename: str,
    model_colors: dict,
    figure_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in model_data.items():
        sorted_v = np.sort(values)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        if name == "Historical":
            ax.plot(sorted_v, cdf, **HIST_STYLE)
        else:
            ax.plot(sorted_v, cdf, **syn_style(name, model_colors), linestyle="--")
    ax.set_xlabel(f"{freq_label} Flow (cms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"{freq_label} Cumulative Distribution Functions")
    ax.legend()
    save_figure(fig, filename, figure_dir)

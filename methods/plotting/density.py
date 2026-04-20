"""KDE density plot -- all models on one axis."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from . import HIST_STYLE, syn_style, save_figure


def fig_density(
    model_data: dict,
    freq_label: str,
    filename: str,
    model_colors: dict,
    figure_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in model_data.items():
        kde = sp_stats.gaussian_kde(values)
        x = np.linspace(values.min(), np.percentile(values, 99.5), 300)
        if name == "Historical":
            ax.plot(x, kde(x), **HIST_STYLE)
            ax.fill_between(x, kde(x), alpha=0.15, color="black")
        else:
            ax.plot(x, kde(x), **syn_style(name, model_colors))
    ax.set_xlabel(f"{freq_label} Flow (cms)")
    ax.set_ylabel("Density")
    ax.set_title(f"{freq_label} Flow Distributions")
    ax.legend()
    save_figure(fig, filename, figure_dir)

"""Flow duration curve -- all models on one axis."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import HIST_STYLE, syn_style, save_figure


def fig_fdc(
    model_data: dict,
    freq_label: str,
    log_scale: bool,
    filename: str,
    model_colors: dict,
    figure_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in model_data.items():
        sorted_v = np.sort(values)[::-1]
        exceed = np.arange(1, len(sorted_v) + 1) / len(sorted_v) * 100
        if name == "Historical":
            ax.plot(exceed, sorted_v, **HIST_STYLE)
        else:
            ax.plot(exceed, sorted_v, **syn_style(name, model_colors), linestyle="--")
    ax.set_xlabel("Exceedance Probability (%)")
    ax.set_ylabel(f"{freq_label} Flow (cms)")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(f"{freq_label} Flow Duration Curves")
    ax.legend()
    save_figure(fig, filename, figure_dir)

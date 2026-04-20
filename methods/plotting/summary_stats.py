"""Bar chart of relative errors in key statistics vs Historical."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from . import save_figure


def fig_summary_stats(
    model_data: dict,
    freq_label: str,
    filename: str,
    model_colors: dict,
    figure_dir: Path,
) -> None:
    hist = model_data["Historical"]
    hist_stats = {
        "Mean": np.mean(hist),
        "Std": np.std(hist),
        "Skewness": sp_stats.skew(hist),
        "Lag-1 ACF": np.corrcoef(hist[:-1], hist[1:])[0, 1],
    }
    syn_keys = [k for k in model_data if k != "Historical"]
    stat_names = list(hist_stats.keys())
    n_stats = len(stat_names)
    n_models = len(syn_keys)
    if n_models == 0:
        return

    rel_errors = np.zeros((n_models, n_stats))
    for j, stat_name in enumerate(stat_names):
        h_val = hist_stats[stat_name]
        for i, mk in enumerate(syn_keys):
            vals = model_data[mk]
            if stat_name == "Mean":
                s_val = np.mean(vals)
            elif stat_name == "Std":
                s_val = np.std(vals)
            elif stat_name == "Skewness":
                s_val = sp_stats.skew(vals)
            elif stat_name == "Lag-1 ACF":
                s_val = np.corrcoef(vals[:-1], vals[1:])[0, 1]
            else:
                s_val = 0
            if abs(h_val) > 1e-6:
                rel_errors[i, j] = (s_val - h_val) / abs(h_val) * 100
            else:
                rel_errors[i, j] = 0

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_stats)
    width = 0.8 / n_models
    for i, mk in enumerate(syn_keys):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(
            x + offset,
            rel_errors[i],
            width,
            color=model_colors.get(mk, "gray"),
            label=mk,
            alpha=0.85,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stat_names)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(f"{freq_label} Summary Statistics -- Relative Error vs Historical")
    ax.legend(loc="best")
    save_figure(fig, filename, figure_dir)

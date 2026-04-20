"""QQ plots: each model vs Historical in its own panel."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import save_figure


def fig_qq_plots(
    model_data: dict,
    freq_label: str,
    filename: str,
    model_colors: dict,
    figure_dir: Path,
) -> None:
    syn_keys = [k for k in model_data if k != "Historical"]
    n = len(syn_keys)
    if n == 0:
        return
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False
    )
    hist_sorted = np.sort(model_data["Historical"])
    hist_quantiles = np.linspace(0, 1, len(hist_sorted))
    for idx, mk in enumerate(syn_keys):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        syn_sorted = np.sort(model_data[mk])
        syn_quantiles = np.linspace(0, 1, len(syn_sorted))
        common_q = np.linspace(0.01, 0.99, 200)
        h_vals = np.interp(common_q, hist_quantiles, hist_sorted)
        s_vals = np.interp(common_q, syn_quantiles, syn_sorted)
        ax.scatter(
            h_vals,
            s_vals,
            s=8,
            alpha=0.6,
            color=model_colors.get(mk, "gray"),
            edgecolors="none",
        )
        lo = min(h_vals.min(), s_vals.min())
        hi = max(h_vals.max(), s_vals.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Historical")
        ax.set_ylabel("Synthetic")
        ax.set_title(mk)
        ax.set_aspect("equal", adjustable="datalim")
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)
    fig.suptitle(f"{freq_label} QQ Plots vs Historical", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, filename, figure_dir)

"""Bar chart comparing Hurst exponent across models."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from synhydro.core.statistics import compute_hurst_exponent

from . import save_figure


def fig_hurst_exponent(
    model_data: dict,
    freq_label: str,
    filename: str,
    model_colors: dict,
    figure_dir: Path,
) -> None:
    hurst_values = {}
    for name, values in model_data.items():
        try:
            result = compute_hurst_exponent(values, method="rs")
            h = float(result["H"])
            if np.isfinite(h):
                hurst_values[name] = h
        except Exception:
            pass

    if len(hurst_values) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(hurst_values.keys())
    h_vals = [hurst_values[n] for n in names]
    colors = [
        "lightgray" if n == "Historical" else model_colors.get(n, "gray") for n in names
    ]

    bars = ax.bar(
        names, h_vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5
    )
    ax.axhline(
        0.5, color="red", linewidth=1, linestyle="--", label="H=0.5 (no persistence)"
    )
    ax.set_ylabel("Hurst Exponent (H)")
    ax.set_title(f"{freq_label} Hurst Exponent (R/S Analysis)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    for bar, val in zip(bars, h_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure(fig, filename, figure_dir)

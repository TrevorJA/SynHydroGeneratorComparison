"""Bar chart comparing CRPS and CRPSS across methods."""

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from synhydro import validate_ensemble

from . import save_figure
from ..analysis import get_monthly_ensembles


def fig_crps_comparison(
    Q_monthly_hist: pd.DataFrame,
    ensembles: dict,
    models_config: dict,
    model_colors: dict,
    figure_dir: Path,
    filename: str,
    site_index: int = 0,
) -> None:
    monthly_ens = get_monthly_ensembles(ensembles, models_config)
    if not monthly_ens:
        return

    site = Q_monthly_hist.columns[site_index]
    colors = model_colors

    crps_data = {}
    for model_key, ensemble in monthly_ens.items():
        try:
            result = validate_ensemble(ensemble, Q_monthly_hist, metrics=["crps"])
            if site in result.crps:
                crps_data[model_key] = result.crps[site]
        except Exception:
            traceback.print_exc()

    if not crps_data:
        return

    model_keys = list(crps_data.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    crps_means = [crps_data[k]["crps_mean"]["synthetic_median"] for k in model_keys]
    bar_colors = [colors.get(k, "gray") for k in model_keys]
    bars = ax1.bar(
        model_keys,
        crps_means,
        color=bar_colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("CRPS (lower is better)")
    ax1.set_title("Mean CRPS by Method")
    ax1.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, crps_means):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    crpss_vals = []
    for k in model_keys:
        if "crpss" in crps_data[k]:
            crpss_vals.append(crps_data[k]["crpss"]["synthetic_median"])
        else:
            crpss_vals.append(np.nan)

    bars = ax2.bar(
        model_keys,
        crpss_vals,
        color=bar_colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.axhline(
        0,
        color="red",
        linewidth=1,
        linestyle="--",
        label="CRPSS=0 (no skill vs climatology)",
    )
    ax2.set_ylabel("CRPSS (higher is better)")
    ax2.set_title("CRPS Skill Score vs Climatology")
    ax2.set_ylim(-0.5, 1.0)
    ax2.legend(fontsize=8)
    ax2.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, crpss_vals):
        if np.isfinite(val):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(f"CRPS Comparison (site: {site})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, filename, figure_dir)

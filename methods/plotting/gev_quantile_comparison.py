"""Bar chart comparing GEV return-period quantiles across methods."""

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from synhydro import validate_ensemble

from . import save_figure
from ..analysis import get_monthly_ensembles


def fig_gev_quantile_comparison(
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

    gev_data = {}
    for model_key, ensemble in monthly_ens.items():
        try:
            result = validate_ensemble(ensemble, Q_monthly_hist, metrics=["extremes"])
            if site in result.extremes:
                gev_data[model_key] = result.extremes[site]
        except Exception:
            traceback.print_exc()

    if not gev_data:
        return

    model_keys = list(gev_data.keys())
    return_periods = ["gev_q10", "gev_q50", "gev_q100"]
    rp_labels = ["10-year", "50-year", "100-year"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(return_periods))
    width = 0.8 / (len(model_keys) + 1)

    first_model = list(gev_data.values())[0]
    obs_vals = [first_model[rp]["observed"] for rp in return_periods]
    ax.bar(
        x - 0.4 + width * 0.5,
        obs_vals,
        width,
        color="black",
        alpha=0.7,
        label="Observed",
        edgecolor="black",
    )

    for i, model_key in enumerate(model_keys):
        syn_vals = []
        syn_lo = []
        syn_hi = []
        for rp in return_periods:
            if rp in gev_data[model_key]:
                entry = gev_data[model_key][rp]
                syn_vals.append(entry["synthetic_median"])
                syn_lo.append(entry["synthetic_p10"])
                syn_hi.append(entry["synthetic_p90"])
            else:
                syn_vals.append(0)
                syn_lo.append(0)
                syn_hi.append(0)

        offset = x - 0.4 + width * (i + 1.5)
        yerr_lo = [v - lo for v, lo in zip(syn_vals, syn_lo)]
        yerr_hi = [hi - v for v, hi in zip(syn_vals, syn_hi)]

        ax.bar(
            offset,
            syn_vals,
            width,
            color=colors.get(model_key, "gray"),
            alpha=0.85,
            label=model_key,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.errorbar(
            offset,
            syn_vals,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            color="black",
            capsize=3,
            linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(rp_labels)
    ax.set_xlabel("Return Period")
    ax.set_ylabel("Flow (cms)")
    ax.set_title(
        f"GEV Return Period Quantiles (site: {site})\n"
        f"Error bars: P10-P90 across realizations",
        fontsize=11,
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, filename, figure_dir)

"""L-moment ratio diagram comparing observed and synthetic L-CV vs L-skewness."""

import traceback
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from synhydro import validate_ensemble

from . import save_figure
from ..analysis import get_monthly_ensembles


def fig_lmoment_comparison(
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

    fig, ax = plt.subplots(figsize=(8, 7))

    obs_lcv = obs_lskew = None
    for model_key, ensemble in monthly_ens.items():
        try:
            result = validate_ensemble(ensemble, Q_monthly_hist, metrics=["lmoments"])
            if site not in result.lmoments:
                continue
            lm = result.lmoments[site]
            lcv_syn = lm["l_cv"]["synthetic_median"]
            lskew_syn = lm["l_skewness"]["synthetic_median"]
            lcv_p10 = lm["l_cv"]["synthetic_p10"]
            lcv_p90 = lm["l_cv"]["synthetic_p90"]
            lskew_p10 = lm["l_skewness"]["synthetic_p10"]
            lskew_p90 = lm["l_skewness"]["synthetic_p90"]

            ax.errorbar(
                lcv_syn,
                lskew_syn,
                xerr=[[lcv_syn - lcv_p10], [lcv_p90 - lcv_syn]],
                yerr=[[lskew_syn - lskew_p10], [lskew_p90 - lskew_syn]],
                fmt="o",
                color=colors.get(model_key, "gray"),
                markersize=8,
                capsize=4,
                linewidth=1.5,
                label=model_key,
            )

            obs_lcv = lm["l_cv"]["observed"]
            obs_lskew = lm["l_skewness"]["observed"]
        except Exception:
            traceback.print_exc()
            continue

    if obs_lcv is not None:
        ax.plot(obs_lcv, obs_lskew, "k*", markersize=15, label="Observed", zorder=10)

    ax.set_xlabel("L-CV (L-scale / L-location)")
    ax.set_ylabel("L-Skewness")
    ax.set_title(
        f"L-Moment Ratio Diagram (site: {site})\n"
        f"Error bars: P10-P90 across realizations",
        fontsize=11,
    )
    ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, filename, figure_dir)

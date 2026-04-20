"""Per-model validation panel figures."""

import logging
import traceback
from pathlib import Path

import pandas as pd

from synhydro.plotting import plot_validation_panel

from . import save_figure

logger = logging.getLogger(__name__)


def fig_validation_panels(
    Q_monthly_hist: pd.DataFrame,
    ensembles: dict,
    models_config: dict,
    figure_dir: Path,
    filename_prefix: str,
    site_index: int = 0,
) -> None:
    site = Q_monthly_hist.columns[site_index]
    obs_series = Q_monthly_hist[site]

    monthly_ensembles = {
        k: v
        for k, v in ensembles.items()
        if models_config[k]["frequency"] in ("monthly", "daily")
    }

    for model_key, ensemble in monthly_ensembles.items():
        try:
            ens = (
                ensemble.resample("MS")
                if models_config[model_key]["frequency"] == "daily"
                else ensemble
            )
            fig, axes = plot_validation_panel(
                ens,
                observed=obs_series,
                site=site,
                timestep="monthly",
            )
            fig.suptitle(
                f"Validation Panel -- {model_key}\n(site: {site})",
                fontsize=13,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            save_figure(fig, f"{filename_prefix}_{model_key}.png", figure_dir)
        except Exception:
            traceback.print_exc()
            logger.warning("plot_validation_panel failed for %s", model_key)

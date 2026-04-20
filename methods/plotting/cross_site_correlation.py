"""Compare cross-site correlation matrices using synhydro statistics."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from synhydro.core.statistics import compute_spatial_correlation

from . import save_figure


def fig_cross_site_correlation(
    Q_monthly_hist: pd.DataFrame,
    ensembles: dict,
    models_config: dict,
    figure_dir: Path,
    filename: str,
) -> None:
    hist_corr = compute_spatial_correlation(Q_monthly_hist)
    n_sites = hist_corr.shape[0]
    if n_sites < 2:
        return

    multisite_models = {
        k: v
        for k, v in ensembles.items()
        if models_config[k].get("multisite", False)
        and models_config[k]["frequency"] in ("monthly", "daily")
    }
    if not multisite_models:
        return

    n_panels = 1 + len(multisite_models)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), squeeze=False)
    axes = axes[0]

    im = axes[0].imshow(
        hist_corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal"
    )
    axes[0].set_title("Historical")
    axes[0].set_xticks(range(n_sites))
    axes[0].set_yticks(range(n_sites))

    for idx, (model_key, ensemble) in enumerate(multisite_models.items(), 1):
        cfg = models_config[model_key]
        corr_sum = np.zeros((n_sites, n_sites))
        count = 0
        for i in range(ensemble.metadata.n_realizations):
            df = ensemble.data_by_realization[i]
            if cfg["frequency"] == "daily":
                df = df.resample("MS").sum()
            sub = df.iloc[:, :n_sites]
            corr = compute_spatial_correlation(sub)
            corr_sum += corr.values
            count += 1
        avg_corr = corr_sum / count

        axes[idx].imshow(avg_corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
        axes[idx].set_title(model_key)
        axes[idx].set_xticks(range(n_sites))
        axes[idx].set_yticks(range(n_sites))

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="Correlation")
    fig.suptitle("Cross-Site Correlation Matrices", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, filename, figure_dir)

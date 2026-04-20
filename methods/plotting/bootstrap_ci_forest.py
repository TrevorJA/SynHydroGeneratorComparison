"""Forest plot showing bootstrap CIs for each method and metric."""

import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from synhydro import bootstrap_metric_ci

from . import save_figure
from ..analysis import get_monthly_ensembles


def fig_bootstrap_ci_forest(
    Q_monthly_hist: pd.DataFrame,
    ensembles: dict,
    models_config: dict,
    model_colors: dict,
    figure_dir: Path,
    filename: str,
    site_index: int = 0,
    comparison_metrics: Optional[List[str]] = None,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
) -> None:
    monthly_ens = get_monthly_ensembles(ensembles, models_config)
    if not monthly_ens:
        return

    if comparison_metrics is None:
        comparison_metrics = [
            "mean",
            "std",
            "skewness",
            "cv",
            "lag1_acf",
            "lag2_acf",
            "p10",
            "p50",
            "p90",
            "annual_variance",
        ]

    site = Q_monthly_hist.columns[site_index]

    all_ci = {}
    for model_key, ensemble in monthly_ens.items():
        try:
            ci = bootstrap_metric_ci(
                ensemble,
                Q_monthly_hist,
                sites=[site],
                metrics=comparison_metrics,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
            )
            all_ci[model_key] = ci
        except Exception:
            traceback.print_exc()

    if not all_ci:
        return

    metrics = comparison_metrics
    n_metrics = len(metrics)
    model_keys = list(all_ci.keys())
    n_models = len(model_keys)

    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(2.2 * n_metrics, max(3, n_models * 0.6 + 1)),
        sharey=True,
    )
    if n_metrics == 1:
        axes = [axes]

    y_positions = np.arange(n_models)

    for ax, metric in zip(axes, metrics):
        for i, model_key in enumerate(model_keys):
            ci_df = all_ci[model_key]
            row = ci_df[ci_df["metric"] == metric]
            if row.empty:
                continue
            row = row.iloc[0]

            re = row["relative_error"]
            re_lo = row["re_ci_lower"]
            re_hi = row["re_ci_upper"]

            color = model_colors.get(model_key, "gray")
            ax.errorbar(
                re,
                i,
                xerr=[[re - re_lo], [re_hi - re]],
                fmt="o",
                color=color,
                markersize=5,
                capsize=3,
                linewidth=1.5,
            )

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Rel. Error")
        ax.set_title(metric, fontsize=9)
        ax.set_xlim(-0.6, 0.6)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(model_keys, fontsize=9)

    fig.suptitle(
        f"Bootstrap 95% CIs on Relative Error (site: {site}, B={n_bootstrap})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, filename, figure_dir)

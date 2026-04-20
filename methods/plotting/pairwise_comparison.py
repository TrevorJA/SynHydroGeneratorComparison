"""Matrix showing which method is significantly better for each metric pair."""

import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from synhydro import compare_methods

from . import save_figure
from ..analysis import get_monthly_ensembles


def fig_pairwise_comparison(
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
    if len(monthly_ens) < 2:
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
    model_keys = list(monthly_ens.keys())
    n_models = len(model_keys)

    comparison_results = {}
    for i in range(n_models):
        for j in range(i + 1, n_models):
            key_a = model_keys[i]
            key_b = model_keys[j]
            try:
                comp = compare_methods(
                    monthly_ens[key_a],
                    monthly_ens[key_b],
                    Q_monthly_hist,
                    sites=[site],
                    metrics=comparison_metrics,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed,
                )
                comparison_results[(key_a, key_b)] = comp
            except Exception:
                traceback.print_exc()

    if not comparison_results:
        return

    wins = np.zeros((n_models, n_models), dtype=int)
    total_metrics = len(comparison_metrics)

    for (key_a, key_b), comp in comparison_results.items():
        i = model_keys.index(key_a)
        j = model_keys.index(key_b)
        sig = comp[comp["significant"]]
        a_wins = (sig["better_method"] == "A").sum()
        b_wins = (sig["better_method"] == "B").sum()
        wins[i, j] = a_wins
        wins[j, i] = b_wins

    fig, ax = plt.subplots(figsize=(max(6, n_models * 1.2), max(5, n_models * 1.0)))
    im = ax.imshow(wins, cmap="YlGn", vmin=0, vmax=total_metrics, aspect="equal")

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_keys, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_keys, fontsize=10)

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=11, color="gray")
            else:
                ax.text(
                    j,
                    i,
                    str(wins[i, j]),
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white" if wins[i, j] > total_metrics * 0.6 else "black",
                )

    fig.colorbar(
        im, ax=ax, label=f"Significant wins (out of {total_metrics})", shrink=0.8
    )
    ax.set_xlabel("Column method")
    ax.set_ylabel("Row method")
    ax.set_title(
        f"Pairwise Method Comparison\n"
        f"Cell (row, col) = metrics where row is significantly better than col\n"
        f"(paired bootstrap, B={n_bootstrap}, alpha=0.05, site: {site})",
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, filename, figure_dir)

    print("\n  Pairwise comparison details:")
    for (key_a, key_b), comp in comparison_results.items():
        sig = comp[comp["significant"]]
        print(f"\n    {key_a} vs {key_b}:")
        if len(sig) == 0:
            print("      No significant differences")
        for _, row in sig.iterrows():
            print(
                f"      {row['metric']:18s} -> {row['better_method']} is better "
                f"(diff={row['diff_estimate']:.4f}, "
                f"CI=[{row['diff_ci_lower']:.4f}, {row['diff_ci_upper']:.4f}])"
            )

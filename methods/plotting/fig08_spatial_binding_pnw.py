"""Figure 8: Spatial correlation is the binding constraint in Pacific NW.

Reframed from a standalone spatial-fidelity figure to an RQ2 instance,
per notes/decisions_log.md 2026-04-17 fig8-reframed-as-rq2-instance.
The claim: in the Pacific Northwest, the spatial category dominates
the MARE ranking ordering -- a generator can look good on marginal
and temporal metrics yet lose to simpler methods because it fails
to reproduce inter-site correlation.

Layout: 2 rows x 2 columns.
  (a) Observed cross-site correlation matrix for Pacific NW (reference)
  (b) Spatial MARE heatmap: 9 multisite generators x 6 regions
  (c) Per-regime bar chart: how much does spatial dominate the
      overall MARE ranking for each region?
  (d) Scatter: spatial MARE vs overall MARE across all (generator,
      region), showing the Pacific NW subset standing out

Data sources:
  - outputs/cross_region/mare.csv (spatial_rmse, mare columns)
  - outputs/cross_region/all_metrics.csv (for observed correlation,
    if present; otherwise skip panel (a) gracefully)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fig03_grand_mare_heatmap import (
    FAMILY_GROUPS,
    MODEL_LABELS,
    REGION_LABELS,
    REGION_ORDER,
    _tier_matrix,
)

logger = logging.getLogger(__name__)

# Multisite generators only (univariate generators do not claim spatial
# capability; evaluating them for spatial fidelity would be unfair).
MULTISITE_MODELS: tuple[str, ...] = (
    "matalas",
    "gaussian_copula",
    "t_copula",
    "vine_copula",
    "hmm",
    "hmm_knn",
    "knn_bootstrap",
    "kirsch",
    "multisite_phase_randomization",
)

FOCUS_REGION = "pacific_northwest"


def _ordered_multisite_models() -> list[str]:
    """Multisite models sorted by family (consistent with other figures)."""
    out: list[str] = []
    for _, _, members in FAMILY_GROUPS:
        out.extend(m for m in members if m in MULTISITE_MODELS)
    return out


def _draw_spatial_heatmap(
    ax: plt.Axes,
    mare_df: pd.DataFrame,
    models: list[str],
) -> plt.cm.ScalarMappable:
    """Panel (b): spatial MARE heatmap across 9 x 6."""
    matrix = _tier_matrix(mare_df, "spatial_rmse", models, REGION_ORDER)
    arr = matrix.to_numpy(dtype=float)
    n_rows, n_cols = arr.shape

    finite = arr[np.isfinite(arr)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0

    im = ax.imshow(arr, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            v = arr[i, j]
            if np.isnan(v):
                continue
            rel = (v - vmin) / ((vmax - vmin) or 1)
            color = "white" if rel > 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)

    # Highlight the Pacific NW column
    pnw_idx = list(REGION_ORDER).index(FOCUS_REGION)
    ax.add_patch(
        plt.Rectangle(
            (pnw_idx - 0.5, -0.5),
            1,
            n_rows,
            fill=False,
            edgecolor="#1b4f72",
            linewidth=2.0,
        )
    )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [REGION_LABELS.get(r, r) for r in REGION_ORDER],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=8)
    ax.set_title("(b) Spatial correlation RMSE (9 multisite generators)", fontsize=10)
    ax.tick_params(axis="both", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    return im


def _draw_observed_correlation(ax: plt.Axes, output_base: Path) -> None:
    """Panel (a): observed 4x4 correlation matrix for PNW (if available).

    Reads the cached historical monthly data for PNW and computes
    Pearson correlation between sites. Falls back to a placeholder if
    the cache is missing.
    """
    try:
        from config import DATA_DIR

        monthly_path = DATA_DIR / FOCUS_REGION / "monthly.csv"
        if not monthly_path.exists():
            raise FileNotFoundError(monthly_path)

        Q = pd.read_csv(monthly_path, index_col=0, parse_dates=True)
        corr = Q.corr()
        im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{corr.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if abs(corr.iloc[i, j]) < 0.6 else "white",
                )
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=7)
        ax.set_title(
            "(a) Observed cross-site correlation (Pacific NW)",
            fontsize=10,
        )
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.04)
    except Exception as exc:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"Observed correlation matrix unavailable\n({exc})",
            ha="center",
            va="center",
            fontsize=9,
            color="0.4",
        )


def _draw_spatial_dominance(
    ax: plt.Axes, mare_df: pd.DataFrame, models: list[str]
) -> None:
    """Panel (c): per-region bar chart of spatial MARE normalized by overall MARE.

    Plots the median (across multisite generators) of
    (spatial_rmse / mare) per region. Values > 1 indicate that spatial
    errors exceed the overall MARE -- the category drives the ranking.
    """
    sub = mare_df[mare_df["model"].isin(models)].copy()
    if "mare" not in sub.columns or "spatial_rmse" not in sub.columns:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "mare.csv lacks required columns for panel (c)",
            ha="center",
            va="center",
            fontsize=9,
            color="0.4",
        )
        return

    sub["ratio"] = sub["spatial_rmse"] / sub["mare"]
    summary = (
        sub.groupby("region")["ratio"]
        .median()
        .reindex(list(REGION_ORDER))
        .to_frame("median_ratio")
    )

    xs = np.arange(len(summary))
    colors = ["#1b4f72" if r == FOCUS_REGION else "#888888" for r in summary.index]
    ax.bar(
        xs,
        summary["median_ratio"].values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(1.0, color="0.3", linewidth=0.7, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [REGION_LABELS.get(r, r) for r in summary.index],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Median spatial RMSE / overall MARE", fontsize=9)
    ax.set_title(
        "(c) Spatial dominance of MARE by region (>1 = spatial drives ranking)",
        fontsize=10,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_scatter(ax: plt.Axes, mare_df: pd.DataFrame, models: list[str]) -> None:
    """Panel (d): spatial MARE vs overall MARE, highlighting PNW subset."""
    sub = mare_df[mare_df["model"].isin(models)].dropna(subset=["mare", "spatial_rmse"])

    if sub.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Not enough data for scatter",
            ha="center",
            va="center",
            fontsize=9,
            color="0.4",
        )
        return

    mask_pnw = sub["region"] == FOCUS_REGION
    ax.scatter(
        sub.loc[~mask_pnw, "mare"],
        sub.loc[~mask_pnw, "spatial_rmse"],
        s=30,
        color="#888888",
        alpha=0.7,
        label="Other regions",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.scatter(
        sub.loc[mask_pnw, "mare"],
        sub.loc[mask_pnw, "spatial_rmse"],
        s=55,
        color="#1b4f72",
        label="Pacific NW",
        edgecolor="white",
        linewidth=0.8,
    )

    # 1:1 reference line
    lim = max(sub["mare"].max(), sub["spatial_rmse"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="0.5", linewidth=0.7, linestyle="--")

    ax.set_xlabel("Overall MARE", fontsize=9)
    ax.set_ylabel("Spatial correlation RMSE", fontsize=9)
    ax.set_title(
        "(d) Spatial RMSE vs overall MARE (PNW standing out = binding)",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def produce(output_base: Path, main_figure_dir: Path) -> Path:
    """Produce Figure 8 from outputs/cross_region/mare.csv (+ observed data)."""
    mare_path = Path(output_base) / "cross_region" / "mare.csv"
    out_path = Path(main_figure_dir) / "fig08_spatial_binding_pnw.png"

    if not mare_path.exists():
        raise FileNotFoundError(
            f"fig08 requires {mare_path}; run assemble_results.py first"
        )

    mare_df = pd.read_csv(mare_path)
    models = _ordered_multisite_models()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    _draw_observed_correlation(axes[0, 0], output_base)
    im_b = _draw_spatial_heatmap(axes[0, 1], mare_df, models)
    _draw_spatial_dominance(axes[1, 0], mare_df, models)
    _draw_scatter(axes[1, 1], mare_df, models)

    # Colorbar for panel (b)
    cbar = fig.colorbar(im_b, ax=axes[0, 1], shrink=0.8, pad=0.04)
    cbar.set_label("Spatial RMSE", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Figure 8. Spatial correlation is the binding constraint in "
        "the Pacific Northwest.",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Figure 8 written: %s", out_path)
    return out_path

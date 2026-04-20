"""Figure 5: Metric-category decomposition.

Answers RQ2: what statistical properties are hardest to preserve,
and does difficulty depend on regime?

Layout: 2 rows x 3 columns = 6 sub-heatmaps, one per metric category.
Each sub-heatmap is (10 generators, Tier 1) x (6 regions). CRITICAL:
all six panels share a single color scale so difficulty is comparable
across categories, not just within.

Data source: outputs/cross_region/mare.csv. Uses the `mare_<category>`
columns written by cross_region.assemble().
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fig03_grand_mare_heatmap import (
    ANNUAL_ONLY_MODELS,
    FAMILY_GROUPS,
    MODEL_LABELS,
    REGION_LABELS,
    REGION_ORDER,
    _tier_matrix,
)

logger = logging.getLogger(__name__)

# Panel order follows figure_design.md (2 rows x 3 cols).
CATEGORY_ORDER: tuple[str, ...] = (
    "marginal",
    "temporal",
    "spatial",
    "drought",
    "spectral",
    "extremes",
)

CATEGORY_TITLES: dict[str, str] = {
    "marginal": "(a) Marginal",
    "temporal": "(b) Temporal",
    "spatial": "(c) Spatial",
    "drought": "(d) Drought",
    "spectral": "(e) Spectral",
    "extremes": "(f) Extremes",
}


def _tier1_models() -> list[str]:
    """Return Tier 1 (non-annual) models in canonical family order."""
    models: list[str] = []
    for _, _, members in FAMILY_GROUPS:
        models.extend(m for m in members if m not in ANNUAL_ONLY_MODELS)
    return models


def _draw_category_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    model_order: list[str],
    region_order: tuple[str, ...],
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    univariate_models: set[str],
    show_ylabels: bool,
) -> plt.cm.ScalarMappable:
    """Draw one category sub-heatmap."""
    arr = matrix.to_numpy(dtype=float)
    n_rows, n_cols = arr.shape

    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            v = arr[i, j]
            if np.isnan(v):
                continue
            rel = (v - vmin) / ((vmax - vmin) or 1)
            color = "white" if rel > 0.6 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6, color=color)

    # Hatch univariate rows for the spatial panel (metric not applicable)
    if "spatial" in title.lower():
        for i, m in enumerate(model_order):
            if m in univariate_models:
                ax.add_patch(
                    plt.Rectangle(
                        (-0.5, i - 0.5),
                        n_cols,
                        1,
                        facecolor="none",
                        hatch="//",
                        edgecolor="0.5",
                        linewidth=0,
                    )
                )

    # Family separator lines
    y = -0.5
    for _, _, fam_models in FAMILY_GROUPS:
        present = [m for m in fam_models if m in model_order]
        if not present:
            continue
        n = len(present)
        sep_y = y + n
        if sep_y < n_rows - 0.5:
            ax.axhline(sep_y, color="0.3", linewidth=0.4)
        y = sep_y

    # Axes
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [REGION_LABELS.get(r, r) for r in region_order],
        rotation=30,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(range(n_rows))
    if show_ylabels:
        ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in model_order], fontsize=7)
    else:
        ax.set_yticklabels([])

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    return im


def produce(output_base: Path, main_figure_dir: Path) -> Path:
    """Produce Figure 5 from outputs/cross_region/mare.csv.

    Parameters
    ----------
    output_base : Path
        Root of outputs/.
    main_figure_dir : Path
        Destination (typically config.MAIN_FIGURE_DIR).

    Returns
    -------
    Path
        Path to written figure.
    """
    mare_path = Path(output_base) / "cross_region" / "mare.csv"
    out_path = Path(main_figure_dir) / "fig05_category_decomposition.png"

    if not mare_path.exists():
        raise FileNotFoundError(
            f"fig05 requires {mare_path}; run assemble_results.py first"
        )

    mare_df = pd.read_csv(mare_path)
    models = _tier1_models()

    # Determine which models are univariate (for hatching spatial panel)
    univariate = set(models) - {
        "matalas",
        "gaussian_copula",
        "t_copula",
        "vine_copula",
        "knn_bootstrap",
        "kirsch",
        "multisite_phase_randomization",
    }

    # Build matrices per category. Missing columns produce all-NaN matrices
    # (reported in the log) rather than erroring.
    matrices: dict[str, pd.DataFrame] = {}
    missing_cats: list[str] = []
    for cat in CATEGORY_ORDER:
        col = f"mare_{cat}"
        if col not in mare_df.columns:
            missing_cats.append(cat)
            matrices[cat] = pd.DataFrame(
                np.nan, index=models, columns=list(REGION_ORDER)
            )
            continue
        matrices[cat] = _tier_matrix(mare_df, col, models, REGION_ORDER)

    if missing_cats:
        logger.warning(
            "fig05: mare.csv lacks per-category columns for: %s "
            "(these panels will be blank)",
            missing_cats,
        )

    # Shared color scale across all 6 panels (the critical design choice).
    all_vals = np.concatenate(
        [m.to_numpy(dtype=float).ravel() for m in matrices.values()]
    )
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=False)
    cmap = "YlOrRd"

    im = None
    for idx, cat in enumerate(CATEGORY_ORDER):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        im = _draw_category_panel(
            ax,
            matrices[cat],
            models,
            REGION_ORDER,
            CATEGORY_TITLES[cat],
            cmap,
            vmin,
            vmax,
            univariate,
            show_ylabels=(c == 0),
        )

    # Single shared colorbar on the right
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02, aspect=40)
    cbar.set_label("MARE (shared scale across categories)", fontsize=9)

    fig.suptitle(
        "Figure 5. Metric-category decomposition: which properties "
        "are hardest to preserve?",
        fontsize=12,
        y=0.995,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Figure 5 written: %s", out_path)
    return out_path

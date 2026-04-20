"""Figure 3: Grand MARE heatmap (two-tier).

Central result of the paper. Side-by-side panels:
  Panel (a) Tier 1 -- Monthly-scale comparison, 10 generators x 6 regions.
  Panel (b) Tier 2 -- Annual-scale comparison, 13 generators x 6 regions.

Rows are grouped by methodological family (Classical, Copula, Regime/
Wavelet, Nonparametric) with thin separators and left-margin labels.
Regions are ordered by aridity (wettest to driest). Cells display MARE
to one decimal place; outliers (> 2x row or column median) get a bold
border. Row/column marginal medians are drawn as bars alongside the
heatmaps.

Data source: outputs/cross_region/mare.csv (produced by
assemble_results.py). Input columns used: region, model, mare,
annual_tier_mare.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)

# Ordering conventions ------------------------------------------------------

# Regions sorted wettest to driest by aridity index (see figure_design.md).
REGION_ORDER: tuple[str, ...] = (
    "pacific_northwest",
    "new_england",
    "mid_atlantic",
    "southern_appalachians",
    "central_plains",
    "central_rockies",
)

# Short labels shown on the column axis.
REGION_LABELS: dict[str, str] = {
    "pacific_northwest": "Pac NW",
    "new_england": "New Eng",
    "mid_atlantic": "Mid-Atl",
    "southern_appalachians": "S App",
    "central_plains": "Central Pl",
    "central_rockies": "Central Rk",
}

# Family assignment and ordering (see publication_plan.md Section 3.2).
# Each entry: (family_key, family_label, [model_key, ...]).
FAMILY_GROUPS: list[tuple[str, str, list[str]]] = [
    ("A", "Classical", ["thomas_fiering", "matalas", "arfima"]),
    ("B", "Copula", ["gaussian_copula", "t_copula", "vine_copula"]),
    ("C", "Regime/Wavelet", ["hmm", "hmm_knn", "warm"]),
    (
        "D",
        "Nonparametric",
        [
            "knn_bootstrap",
            "kirsch",
            "phase_randomization",
            "multisite_phase_randomization",
        ],
    ),
]

# Short labels shown on the row axis.
MODEL_LABELS: dict[str, str] = {
    "thomas_fiering": "Thomas-Fiering",
    "matalas": "Matalas",
    "arfima": "ARFIMA",
    "gaussian_copula": "Gaussian cop.",
    "t_copula": "t-copula",
    "vine_copula": "Vine copula",
    "hmm": "Multi-site HMM",
    "hmm_knn": "HMM-KNN",
    "warm": "WARM",
    "knn_bootstrap": "KNN bootstrap",
    "kirsch": "Kirsch-Nowak",
    "phase_randomization": "Phase rand.",
    "multisite_phase_randomization": "Multi-site phase rand.",
}

# Models that are annual-only (excluded from Tier 1 / Panel a).
ANNUAL_ONLY_MODELS: frozenset[str] = frozenset({"hmm", "hmm_knn", "warm"})


def _tier_matrix(
    mare_df: pd.DataFrame,
    value_col: str,
    model_order: Sequence[str],
    region_order: Sequence[str],
) -> pd.DataFrame:
    """Pivot mare_df into a (model x region) matrix for one tier.

    Missing (region, model) pairs become NaN. Models or regions not
    present in mare_df are preserved as all-NaN rows/columns so the
    heatmap layout stays fixed.
    """
    pivot = (
        mare_df.pivot_table(
            index="model", columns="region", values=value_col, aggfunc="first"
        )
        .reindex(index=list(model_order))
        .reindex(columns=list(region_order))
    )
    return pivot


def _outlier_mask(matrix: pd.DataFrame) -> np.ndarray:
    """Return boolean mask marking cells > 2x row median OR > 2x column median.

    NaN cells are never marked.
    """
    arr = matrix.to_numpy(dtype=float)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=bool)

    with np.errstate(invalid="ignore"):
        row_med = np.nanmedian(arr, axis=1, keepdims=True)
        col_med = np.nanmedian(arr, axis=0, keepdims=True)
        gt_row = arr > (2.0 * row_med)
        gt_col = arr > (2.0 * col_med)

    mask = np.where(np.isnan(arr), False, gt_row | gt_col)
    return mask


def _draw_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    model_order: Sequence[str],
    region_order: Sequence[str],
    title: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    annual_models: Iterable[str] = (),
) -> plt.cm.ScalarMappable:
    """Draw one heatmap panel (Tier 1 or Tier 2).

    Parameters
    ----------
    matrix : pd.DataFrame
        Rows indexed by model_order, columns by region_order.
    annual_models : iterable of str
        Models whose row labels should be bolded (Tier 2 annual-native).
    """
    arr = matrix.to_numpy(dtype=float)
    n_rows, n_cols = arr.shape

    im = ax.imshow(
        arr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    # Numeric annotations
    for i in range(n_rows):
        for j in range(n_cols):
            v = arr[i, j]
            if np.isnan(v):
                continue
            # Text colour flips to white on dark cells for legibility.
            rel = (v - (vmin or np.nanmin(arr))) / (
                ((vmax or np.nanmax(arr)) - (vmin or np.nanmin(arr))) or 1
            )
            color = "white" if rel > 0.6 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color=color)

    # Outlier borders
    mask = _outlier_mask(matrix)
    for i, j in zip(*np.where(mask)):
        ax.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.8,
            )
        )

    # Axes labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [REGION_LABELS.get(r, r) for r in region_order], rotation=30, ha="right"
    )
    ax.set_yticks(range(n_rows))
    row_labels = [MODEL_LABELS.get(m, m) for m in model_order]
    ax.set_yticklabels(row_labels, fontsize=8)

    # Bold row labels for annual-native models
    annual_set = set(annual_models)
    for tick, m in zip(ax.get_yticklabels(), model_order):
        if m in annual_set:
            tick.set_fontweight("bold")

    # Family separator lines and labels in the left margin
    y = -0.5
    for fam_key, fam_label, fam_models in FAMILY_GROUPS:
        present = [m for m in fam_models if m in model_order]
        if not present:
            continue
        n = len(present)
        # Separator below the last row of this family (if not final family)
        sep_y = y + n
        if sep_y < n_rows - 0.5:
            ax.axhline(sep_y, color="0.3", linewidth=0.5)
        # Family label on the far left (outside the axes)
        ax.text(
            -0.02,
            (y + sep_y) / 2,
            fam_label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
            rotation=90,
        )
        y = sep_y

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Region (wettest to driest)", fontsize=9)
    ax.tick_params(axis="both", which="both", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return im


def _marginal_medians(matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (row_medians, col_medians) ignoring NaN."""
    arr = matrix.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        row_med = np.nanmedian(arr, axis=1)
        col_med = np.nanmedian(arr, axis=0)
    return row_med, col_med


def produce(output_base: Path, main_figure_dir: Path) -> Path:
    """Produce Figure 3 from outputs/cross_region/mare.csv.

    Parameters
    ----------
    output_base : Path
        Root of the outputs/ directory.
    main_figure_dir : Path
        Destination directory (typically config.MAIN_FIGURE_DIR).

    Returns
    -------
    Path
        Path to the written figure.
    """
    mare_path = Path(output_base) / "cross_region" / "mare.csv"
    out_path = Path(main_figure_dir) / "fig03_grand_mare_heatmap.png"

    if not mare_path.exists():
        raise FileNotFoundError(
            f"fig03 requires {mare_path}; run assemble_results.py first"
        )

    mare_df = pd.read_csv(mare_path)

    # Tier 1 (monthly, 10 generators): value column is `mare`.
    tier1_models: list[str] = []
    for _, _, members in FAMILY_GROUPS:
        tier1_models.extend(m for m in members if m not in ANNUAL_ONLY_MODELS)
    tier1_matrix = _tier_matrix(mare_df, "mare", tier1_models, REGION_ORDER)

    # Tier 2 (annual, all 13 generators): value column is `annual_tier_mare`.
    tier2_models: list[str] = []
    for _, _, members in FAMILY_GROUPS:
        tier2_models.extend(members)
    tier2_matrix = _tier_matrix(mare_df, "annual_tier_mare", tier2_models, REGION_ORDER)

    # Shared color scale per tier (each tier has its own min/max).
    def _vminmax(m: pd.DataFrame) -> tuple[float | None, float | None]:
        arr = m.to_numpy(dtype=float)
        if np.all(np.isnan(arr)):
            return None, None
        return float(np.nanmin(arr)), float(np.nanmax(arr))

    t1_vmin, t1_vmax = _vminmax(tier1_matrix)
    t2_vmin, t2_vmax = _vminmax(tier2_matrix)

    # Figure layout: two panels side-by-side at full WRR width (7 in).
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5.5),
        gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.45},
    )

    cmap = "YlOrRd"

    im1 = _draw_panel(
        axes[0],
        tier1_matrix,
        tier1_models,
        REGION_ORDER,
        "(a) Tier 1 -- Monthly scale (10 generators)",
        cmap,
        t1_vmin,
        t1_vmax,
    )
    im2 = _draw_panel(
        axes[1],
        tier2_matrix,
        tier2_models,
        REGION_ORDER,
        "(b) Tier 2 -- Annual scale (all 13 generators)",
        cmap,
        t2_vmin,
        t2_vmax,
        annual_models=ANNUAL_ONLY_MODELS,
    )

    # Colorbars per panel (different scales).
    for ax, im, label in (
        (axes[0], im1, "MARE (monthly)"),
        (axes[1], im2, "MARE (annual)"),
    ):
        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.04)
        cbar.set_label(label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # Marginal medians printed as a tight caption row below the panels.
    t1_row_med, t1_col_med = _marginal_medians(tier1_matrix)
    t2_row_med, t2_col_med = _marginal_medians(tier2_matrix)

    def _fmt(medians: np.ndarray) -> str:
        return ", ".join("-" if np.isnan(m) else f"{m:.2f}" for m in medians)

    caption = (
        "Tier 1 column medians (Pac NW ... Central Rk): "
        f"{_fmt(t1_col_med)}\n"
        "Tier 2 column medians (Pac NW ... Central Rk): "
        f"{_fmt(t2_col_med)}"
    )
    fig.text(
        0.5,
        0.02,
        caption,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="0.3",
    )

    fig.suptitle(
        "Figure 3. Grand MARE heatmap: when do stochastic generators fail?",
        fontsize=12,
        y=0.995,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Figure 3 written: %s", out_path)
    return out_path

"""Figure 9: Ensemble-size convergence (RQ3).

Answers: how many realizations are needed for stable diagnostic
assessment?

Layout: 2 panels.
  (a) Convergence curves per generator. X = N_REALIZATIONS (log),
      Y = Overall MARE. One curve per Tier-1 generator, colored by
      family. Each curve is the median across regions with a shaded
      band showing the inter-region IQR.
  (b) Convergence threshold by metric category. Bars show the
      minimum N for each category to settle within 5% of its asymptote,
      medianed across (region, generator).

Data source: outputs/{region}/convergence_{model}.csv (per-model
summary CSVs produced by convergence_single.py). Expected columns:
  model, n_realizations, draw, mare, median_are, max_are,
  spatial_rmse, elapsed_s
Detail CSV (optional): outputs/{region}/convergence_detail_{model}.csv
with per-metric relative_error; used for panel (b).
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
    REGION_ORDER,
)

logger = logging.getLogger(__name__)

FAMILY_COLORS: dict[str, str] = {
    "A": "#1f77b4",  # Classical (blue)
    "B": "#ff7f0e",  # Copula (orange)
    "C": "#2ca02c",  # Regime/Wavelet (green)
    "D": "#d62728",  # Nonparametric (red)
}

CATEGORY_ORDER: tuple[str, ...] = (
    "marginal",
    "temporal",
    "spatial",
    "drought",
    "spectral",
    "extremes",
)

# Threshold rule: N at which category MARE drops within 5% of its
# value at the maximum swept N.
CONVERGENCE_TOLERANCE = 0.05


def _family_of(model_key: str) -> str:
    for fam_key, _, members in FAMILY_GROUPS:
        if model_key in members:
            return fam_key
    return "?"


def _load_convergence_data(output_base: Path) -> pd.DataFrame:
    """Load all convergence_<model>.csv files across regions into one frame.

    Returns a long-format DataFrame with columns
    `region, model, n_realizations, draw, mare, ...`.
    """
    frames: list[pd.DataFrame] = []
    for region in REGION_ORDER:
        region_dir = Path(output_base) / region
        if not region_dir.exists():
            continue
        for path in sorted(region_dir.glob("convergence_*.csv")):
            name = path.stem  # convergence_<model>
            if name.startswith("convergence_detail_"):
                continue  # handled separately
            model = name.removeprefix("convergence_")
            df = pd.read_csv(path)
            df["region"] = region
            df["model"] = df.get("model", model)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_convergence_detail(output_base: Path) -> pd.DataFrame:
    """Load convergence_detail_<model>.csv files. Returns empty if none."""
    frames: list[pd.DataFrame] = []
    for region in REGION_ORDER:
        region_dir = Path(output_base) / region
        if not region_dir.exists():
            continue
        for path in sorted(region_dir.glob("convergence_detail_*.csv")):
            model = path.stem.removeprefix("convergence_detail_")
            df = pd.read_csv(path)
            df["region"] = region
            df["model"] = df.get("model", model)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _draw_panel_a(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Per-model median-across-regions MARE vs N_realizations."""
    tier1_models = [
        m
        for _, _, members in FAMILY_GROUPS
        for m in members
        if m not in ANNUAL_ONLY_MODELS
    ]

    # Aggregate across draws first (mean), then across regions (median + IQR)
    agg_draws = (
        summary.groupby(["region", "model", "n_realizations"])["mare"]
        .mean()
        .reset_index()
    )

    for model in tier1_models:
        sub = agg_draws[agg_draws["model"] == model]
        if sub.empty:
            continue

        grouped = (
            sub.groupby("n_realizations")["mare"]
            .agg(
                [
                    "median",
                    lambda x: np.percentile(x, 25),
                    lambda x: np.percentile(x, 75),
                ]
            )
            .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
        )
        grouped = grouped.sort_index()

        fam = _family_of(model)
        color = FAMILY_COLORS.get(fam, "#333333")
        ns = grouped.index.values

        ax.fill_between(
            ns,
            grouped["q25"].values,
            grouped["q75"].values,
            alpha=0.12,
            color=color,
            linewidth=0,
        )
        ax.plot(
            ns,
            grouped["median"].values,
            marker="o",
            markersize=4,
            linewidth=1.3,
            color=color,
            label=MODEL_LABELS.get(model, model),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of realizations", fontsize=10)
    ax.set_ylabel("Overall MARE (median across regions)", fontsize=10)
    ax.set_title("(a) MARE convergence by generator", fontsize=11)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="best")
    ax.grid(True, which="both", axis="both", linewidth=0.3, alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _compute_category_thresholds(detail: pd.DataFrame) -> pd.DataFrame:
    """For each (model, region, category), find the smallest n at which
    per-category MARE is within CONVERGENCE_TOLERANCE of its asymptote.

    Returns long-format DataFrame: category, model, region, threshold_n.
    """
    if detail.empty:
        return pd.DataFrame(columns=["category", "threshold_n"])

    # Aggregate per-category relative_error at each (region, model, n)
    if "category" not in detail.columns or "relative_error" not in detail.columns:
        return pd.DataFrame(columns=["category", "threshold_n"])

    agg = (
        detail.groupby(["region", "model", "n_realizations", "category"])[
            "relative_error"
        ]
        .apply(lambda x: float(np.nanmean(np.abs(x))))
        .reset_index(name="cat_mare")
    )

    rows: list[dict] = []
    for (region, model, cat), grp in agg.groupby(["region", "model", "category"]):
        grp = grp.sort_values("n_realizations")
        if grp.empty:
            continue
        n_max = grp["n_realizations"].max()
        asymptote = float(grp.loc[grp["n_realizations"] == n_max, "cat_mare"].iloc[0])
        if not np.isfinite(asymptote) or asymptote == 0:
            threshold_n = n_max
        else:
            within = grp[
                np.abs(grp["cat_mare"] - asymptote) / asymptote <= CONVERGENCE_TOLERANCE
            ]
            threshold_n = (
                int(within["n_realizations"].min()) if not within.empty else int(n_max)
            )
        rows.append(
            {
                "region": region,
                "model": model,
                "category": cat,
                "threshold_n": threshold_n,
            }
        )

    return pd.DataFrame(rows)


def _draw_panel_b(ax: plt.Axes, detail: pd.DataFrame) -> None:
    """Per-category minimum-N-to-converge, median across (region, model)."""
    thresholds = _compute_category_thresholds(detail)

    if thresholds.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "convergence_detail_*.csv not available\n(panel skipped)",
            ha="center",
            va="center",
            fontsize=10,
            color="0.4",
        )
        return

    summary = (
        thresholds.groupby("category")["threshold_n"]
        .agg(["median", "min", "max"])
        .reindex(list(CATEGORY_ORDER))
    )

    xs = np.arange(len(summary))
    ax.bar(
        xs,
        summary["median"].values,
        color="#666666",
        edgecolor="black",
        linewidth=0.5,
    )
    # Error whiskers from min to max
    for i, (_, row) in enumerate(summary.iterrows()):
        if np.isnan(row["min"]):
            continue
        ax.vlines(i, row["min"], row["max"], color="0.3", linewidth=1.0)

    # Reference lines
    for ref in (50, 100, 200):
        ax.axhline(ref, color="0.6", linewidth=0.6, linestyle="--")
        ax.text(
            len(summary) - 0.5,
            ref,
            f"N={ref}",
            fontsize=7,
            color="0.5",
            va="bottom",
            ha="right",
        )

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([c.capitalize() for c in summary.index], rotation=20)
    ax.set_ylabel("Minimum N (for MARE within 5% of asymptote)", fontsize=10)
    ax.set_title("(b) Convergence threshold by metric category", fontsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def produce(output_base: Path, main_figure_dir: Path) -> Path:
    """Produce Figure 9 from per-model convergence CSVs."""
    output_base = Path(output_base)
    out_path = Path(main_figure_dir) / "fig09_convergence.png"

    summary = _load_convergence_data(output_base)
    if summary.empty:
        raise FileNotFoundError(
            f"fig09 requires convergence_*.csv under {output_base}/<region>/; "
            "run convergence_single.py first"
        )

    detail = _load_convergence_detail(output_base)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _draw_panel_a(axes[0], summary)
    _draw_panel_b(axes[1], detail)

    fig.suptitle(
        "Figure 9. Convergence of MARE with ensemble size. "
        "How many realizations are needed?",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Figure 9 written: %s", out_path)
    return out_path

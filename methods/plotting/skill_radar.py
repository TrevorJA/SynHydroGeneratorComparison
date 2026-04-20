"""Skill radar (spider) plot comparing generator performance across six axes.

Each axis represents a distinct hydrologic skill dimension derived from the
ValidationResult categories returned by synhydro.validate_ensemble().  Scores
are median absolute relative errors (MdARE) clipped at 2.0 and normalised to
[0, 1], so a lower score is better and zero is perfect.

Bootstrap confidence bands (90 %) are computed by resampling realizations with
replacement and show within-model stochastic variability.
"""

import logging
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from synhydro import validate_ensemble
from synhydro.core.ensemble import Ensemble

from . import save_figure
from ..colors import MODEL_DISPLAY_LABELS, FAMILY_COLORS, get_model_family

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Radar axis definitions
# ---------------------------------------------------------------------------

RADAR_AXES = [
    "Marginal",
    "Temporal",
    "Seasonal",
    "Extremes",
    "Drought",
    "Spatial",
]

# Maps radar axis name -> ValidationResult category names whose metrics
# contribute to that axis score.
AXIS_CATEGORIES = {
    "Marginal": ["marginal", "lmoments"],
    "Temporal": ["temporal", "spectral"],
    "Seasonal": ["seasonal"],
    "Extremes": ["extremes", "fdc"],
    "Drought": ["drought", "ssi_drought"],
    "Spatial": ["spatial"],
}

# Score ceiling used for normalisation.  Errors above this are treated equally.
SCORE_CLIP = 2.0

# Number of bootstrap resamples for CI estimation.
# Set lower (e.g. 50) for development runs; increase to 500+ for publication.
N_BOOTSTRAP = 5

# Confidence interval level.
CI_ALPHA = 0.10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mdare_from_df(df: pd.DataFrame, categories: list) -> float:
    """Compute MdARE for a set of ValidationResult categories from a tidy df."""
    mask = df["category"].isin(categories)
    vals = df.loc[mask, "relative_error"].dropna()
    if len(vals) == 0:
        return np.nan
    return float(np.median(np.abs(vals)))


def _spatial_score_from_result(result) -> float:
    """Extract spatial correlation RMSE from a ValidationResult summary."""
    return float(result.summary.get("spatial_correlation_rmse", np.nan))


def _score_from_result(result, site: str) -> dict:
    """Compute the 6 axis scores from a single ValidationResult.

    Parameters
    ----------
    result : ValidationResult
    site : str
        Reference site column name (used to filter the tidy dataframe).

    Returns
    -------
    dict
        {axis_name: score_in_[0,1]} for all 6 axes.
    """
    df = result.to_dataframe()
    df_site = df[df["site"] == site]

    scores = {}
    for axis, cats in AXIS_CATEGORIES.items():
        if axis == "Spatial":
            raw = _spatial_score_from_result(result)
        else:
            raw = _mdare_from_df(df_site, cats)
        if np.isfinite(raw):
            scores[axis] = min(raw, SCORE_CLIP) / SCORE_CLIP
        else:
            scores[axis] = np.nan
    return scores


def _bootstrap_scores(
    ensemble: Ensemble,
    Q_ref: pd.DataFrame,
    site: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    """Draw bootstrap resamples and return per-axis score arrays.

    Parameters
    ----------
    ensemble : Ensemble
    Q_ref : pd.DataFrame
        Reference historical data passed to validate_ensemble.
    site : str
        Reference site name.
    n_bootstrap : int
    rng : np.random.Generator

    Returns
    -------
    dict
        {axis_name: np.ndarray of shape (n_bootstrap,)} -- may contain NaNs.
    """
    n_real = ensemble.metadata.n_realizations
    ids = list(range(n_real))
    boot_scores = {ax: [] for ax in RADAR_AXES}

    for _ in range(n_bootstrap):
        sampled_ids = rng.choice(ids, size=n_real, replace=True)
        boot_data = {
            i: ensemble.data_by_realization[sid] for i, sid in enumerate(sampled_ids)
        }
        boot_ens = Ensemble(boot_data)
        try:
            res = validate_ensemble(boot_ens, Q_ref)
            s = _score_from_result(res, site)
        except Exception:
            s = {ax: np.nan for ax in RADAR_AXES}
        for ax in RADAR_AXES:
            boot_scores[ax].append(s.get(ax, np.nan))

    return {ax: np.array(vals) for ax, vals in boot_scores.items()}


# ---------------------------------------------------------------------------
# Radar drawing helpers
# ---------------------------------------------------------------------------


def _radar_coordinates(scores: list, n_axes: int) -> tuple:
    """Convert a list of axis scores to (x, y) polygon coordinates."""
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    xs = [s * np.cos(a) for s, a in zip(scores, angles)]
    ys = [s * np.sin(a) for s, a in zip(scores, angles)]
    xs.append(xs[0])
    ys.append(ys[0])
    return np.array(xs), np.array(ys)


def _draw_radar_grid(ax, n_axes: int, n_rings: int = 4) -> None:
    """Draw the background grid (rings and spokes) on a polar-like cartesian ax."""
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    ring_vals = np.linspace(0, 1, n_rings + 1)[1:]

    for r in ring_vals:
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(
            r * np.cos(theta),
            r * np.sin(theta),
            color="#CCCCCC",
            linewidth=0.5,
            zorder=0,
        )

    for a in angles:
        ax.plot(
            [0, np.cos(a)], [0, np.sin(a)], color="#CCCCCC", linewidth=0.5, zorder=0
        )

    # Score labels on one spoke
    for r in ring_vals:
        ax.text(
            r * np.cos(np.pi / 2 + 0.05),
            r * np.sin(np.pi / 2 + 0.05),
            f"{r:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#888888",
        )


def _draw_axis_labels(ax, axes_names: list) -> None:
    """Place axis labels at the tips of each spoke."""
    n = len(axes_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pad = 1.18
    for name, a in zip(axes_names, angles):
        x, y = pad * np.cos(a), pad * np.sin(a)
        ha = "center"
        if np.cos(a) > 0.1:
            ha = "left"
        elif np.cos(a) < -0.1:
            ha = "right"
        ax.text(
            x,
            y,
            name,
            ha=ha,
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )


# ---------------------------------------------------------------------------
# Public figure function
# ---------------------------------------------------------------------------


def fig_skill_radar(
    Q_monthly: pd.DataFrame,
    Q_annual: pd.DataFrame,
    ensembles: dict,
    models_config: dict,
    model_colors: dict,
    figure_dir: Path,
    filename: str,
    site_index: int = 0,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: Optional[int] = 0,
) -> None:
    """Radar (spider) skill plot comparing all generators on six hydrologic axes.

    Parameters
    ----------
    Q_monthly, Q_annual : pd.DataFrame
        Historical streamflow at monthly and annual resolution.
    ensembles : dict
        {model_key: Ensemble}
    models_config : dict
        The MODELS config dict.
    model_colors : dict
        {model_key: hex_color}
    figure_dir : Path
        Output directory.
    filename : str
        Output filename (e.g. "21_skill_radar.png").
    site_index : int
        Column index of the reference site.
    n_bootstrap : int
        Number of bootstrap resamples for CI bands.
    seed : int, optional
        Seed for the bootstrap RNG.
    """
    site = Q_monthly.columns[site_index]
    rng = np.random.default_rng(seed)

    monthly_ens = {
        k: v
        for k, v in ensembles.items()
        if models_config[k]["frequency"] in ("monthly", "daily")
    }
    annual_ens = {
        k: v for k, v in ensembles.items() if models_config[k]["frequency"] == "annual"
    }

    # Determine which Q_ref to use per model
    def _q_ref(model_key):
        if models_config[model_key]["frequency"] == "annual":
            return Q_annual
        ens = ensembles[model_key]
        if models_config[model_key]["frequency"] == "daily":
            return Q_monthly
        return Q_monthly

    n_axes = len(RADAR_AXES)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)

    _draw_radar_grid(ax, n_axes)
    _draw_axis_labels(ax, RADAR_AXES)

    legend_handles = []
    printed_header = False

    for model_key, ensemble in {**monthly_ens, **annual_ens}.items():
        color = model_colors.get(model_key, "#888888")
        label = MODEL_DISPLAY_LABELS.get(model_key, model_key)

        q_ref = _q_ref(model_key)
        if models_config[model_key]["frequency"] == "daily":
            try:
                ensemble_for_val = ensemble.resample("MS")
            except Exception:
                ensemble_for_val = ensemble
        else:
            ensemble_for_val = ensemble

        # Central scores
        try:
            result = validate_ensemble(ensemble_for_val, q_ref)
            central = _score_from_result(result, site)
        except Exception:
            traceback.print_exc()
            logger.warning(
                "validate_ensemble failed for %s -- skipping radar", model_key
            )
            continue

        scores = [central.get(ax, np.nan) for ax in RADAR_AXES]
        if all(np.isnan(scores)):
            continue

        # Replace NaN with 0 for plotting (NaN axis = unknown)
        scores_plot = [s if np.isfinite(s) else 0.0 for s in scores]

        xs, ys = _radar_coordinates(scores_plot, n_axes)
        ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.9, zorder=3)
        ax.fill(xs, ys, color=color, alpha=0.08, zorder=2)

        # Bootstrap CI
        try:
            boot = _bootstrap_scores(ensemble_for_val, q_ref, site, n_bootstrap, rng)
            lo_scores = [
                np.nanpercentile(boot[a], CI_ALPHA / 2 * 100) for a in RADAR_AXES
            ]
            hi_scores = [
                np.nanpercentile(boot[a], (1 - CI_ALPHA / 2) * 100) for a in RADAR_AXES
            ]

            # Draw CI band as a shaded envelope between lo and hi polygons
            angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
            lo_plot = [s if np.isfinite(s) else 0.0 for s in lo_scores]
            hi_plot = [s if np.isfinite(s) else 0.0 for s in hi_scores]

            # Build closed polygon for hi, then reversed lo, for fill_between
            angles_closed = np.append(angles, angles[0])
            lo_r = np.append(lo_plot, lo_plot[0])
            hi_r = np.append(hi_plot, hi_plot[0])

            xs_hi = hi_r * np.cos(angles_closed)
            ys_hi = hi_r * np.sin(angles_closed)
            xs_lo = lo_r * np.cos(angles_closed)
            ys_lo = lo_r * np.sin(angles_closed)

            band_x = np.concatenate([xs_hi, xs_lo[::-1]])
            band_y = np.concatenate([ys_hi, ys_lo[::-1]])
            ax.fill(band_x, band_y, color=color, alpha=0.12, zorder=1)

        except Exception:
            pass

        if not printed_header:
            printed_header = True

        handle = mpatches.Patch(color=color, label=label, alpha=0.85)
        legend_handles.append(handle)

    # Score annotation legend note
    ax.text(
        0,
        -1.35,
        "Score: 0 = perfect, 1 = poor  |  MdARE clipped at 2.0",
        ha="center",
        va="center",
        fontsize=7,
        color="#888888",
        style="italic",
    )

    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.38, 1.08),
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        title="Generator",
        title_fontsize=8,
        ncol=1,
    )

    site_label = Q_monthly.columns[site_index]
    ax.set_title(
        f"Skill Radar  (reference site: {site_label})\n"
        f"Shaded band: 90% bootstrap CI  ({n_bootstrap} resamples)",
        fontsize=10,
        fontweight="bold",
        pad=16,
    )

    fig.tight_layout()
    save_figure(fig, filename, figure_dir)

    # Print a quick summary table
    print("\n  Skill scores (MdARE normalised, lower = better):")
    header = f"  {'Model':20s}" + "".join(f"  {a[:5]:>6}" for a in RADAR_AXES)
    print(header)
    for model_key, ensemble in {**monthly_ens, **annual_ens}.items():
        q_ref = _q_ref(model_key)
        if models_config[model_key]["frequency"] == "daily":
            try:
                ensemble_for_val = ensemble.resample("MS")
            except Exception:
                ensemble_for_val = ensemble
        else:
            ensemble_for_val = ensemble
        try:
            result = validate_ensemble(ensemble_for_val, q_ref)
            central = _score_from_result(result, site)
            row = f"  {MODEL_DISPLAY_LABELS.get(model_key, model_key):20s}"
            for a in RADAR_AXES:
                v = central.get(a, np.nan)
                row += f"  {v:6.3f}" if np.isfinite(v) else f"  {'nan':>6}"
            print(row)
        except Exception:
            pass

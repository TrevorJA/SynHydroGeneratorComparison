"""Shared plotting utilities and figure function re-exports.

Each figure function lives in its own module within this package.
Shared constants (HIST_STYLE, SYN_ALPHA, PLOT_RCPARAMS) and helpers
(syn_style, save_figure, apply_rcparams) are defined here and imported
by the individual figure modules.
"""

from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================================
# Shared style constants
# ============================================================================

PLOT_RCPARAMS = {
    # Display resolution (save_figure already writes at 300 dpi)
    "figure.dpi": 150,
    "figure.facecolor": "white",
    # Typography
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "axes.linewidth": 0.8,
    "axes.facecolor": "white",
    # Grid -- subtle dotted lines so data is primary
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.color": "#BBBBBB",
    # Legend
    "legend.fontsize": 8,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "#CCCCCC",
    "legend.borderpad": 0.4,
    # Ticks -- outward for cleaner axis separation
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Lines
    "lines.linewidth": 1.5,
    "patch.linewidth": 0.5,
}

HIST_STYLE = dict(color="black", linewidth=2.5, label="Historical", zorder=10)
SYN_ALPHA = 0.85


# ============================================================================
# Shared helper functions
# ============================================================================


def apply_rcparams() -> None:
    """Apply shared matplotlib rcParams."""
    plt.rcParams.update(PLOT_RCPARAMS)


def syn_style(model_key: str, model_colors: dict) -> dict:
    """Return plot kwargs for a synthetic model.

    Uses the canonical display label from :mod:`methods.colors` so that
    all figures show clean academic names rather than internal key strings.
    """
    from ..colors import MODEL_DISPLAY_LABELS

    label = MODEL_DISPLAY_LABELS.get(model_key, model_key)
    return dict(
        color=model_colors.get(model_key, "#888888"),
        linewidth=1.5,
        alpha=SYN_ALPHA,
        label=label,
    )


def save_figure(fig, name: str, figure_dir: Path) -> None:
    """Save figure to *figure_dir/name* and close."""
    path = Path(figure_dir) / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ============================================================================
# Re-exports (one per figure module)
# ============================================================================

from .density import fig_density
from .cdf import fig_cdf
from .acf import fig_acf
from .fdc import fig_fdc
from .seasonal_cycle import fig_seasonal_cycle
from .summary_stats import fig_summary_stats
from .qq_plots import fig_qq_plots
from .drought_duration import fig_drought_duration
from .drought_severity import fig_drought_severity
from .annual_extremes import fig_annual_extremes
from .cross_site_correlation import fig_cross_site_correlation
from .hurst_exponent import fig_hurst_exponent
from .seasonal_variance import fig_seasonal_variance
from .psd import fig_psd
from .validation_summary import fig_validation_summary
from .validation_panels import fig_validation_panels
from .convergence_mare import fig_convergence_mare
from .convergence_by_category import fig_convergence_by_category
from .convergence_heatmap import fig_convergence_heatmap
from .convergence_spatial import fig_convergence_spatial
from .extended_validation_heatmap import fig_extended_validation_heatmap
from .bootstrap_ci_forest import fig_bootstrap_ci_forest
from .pairwise_comparison import fig_pairwise_comparison
from .crps_comparison import fig_crps_comparison
from .lmoment_comparison import fig_lmoment_comparison
from .gev_quantile_comparison import fig_gev_quantile_comparison
from .skill_radar import fig_skill_radar
from .spatial_overview import (
    fig_conus_overview,
    fig_region_flowlines,
    create_spatial_figures,
)
from .validation_csv import fig_validation_summary_from_csv, fig_cross_region_from_csv

__all__ = [
    # Utilities
    "PLOT_RCPARAMS",
    "HIST_STYLE",
    "SYN_ALPHA",
    "apply_rcparams",
    "syn_style",
    "save_figure",
    # Distribution figures
    "fig_density",
    "fig_cdf",
    "fig_acf",
    "fig_fdc",
    "fig_seasonal_cycle",
    "fig_summary_stats",
    "fig_qq_plots",
    "fig_drought_duration",
    "fig_drought_severity",
    "fig_annual_extremes",
    "fig_cross_site_correlation",
    "fig_hurst_exponent",
    "fig_seasonal_variance",
    "fig_psd",
    "fig_validation_summary",
    "fig_validation_panels",
    # Convergence figures
    "fig_convergence_mare",
    "fig_convergence_by_category",
    "fig_convergence_heatmap",
    "fig_convergence_spatial",
    # Skill radar
    "fig_skill_radar",
    # CSV-based validation figures
    "fig_validation_summary_from_csv",
    "fig_cross_region_from_csv",
    # Statistical testing figures
    "fig_extended_validation_heatmap",
    "fig_bootstrap_ci_forest",
    "fig_pairwise_comparison",
    "fig_crps_comparison",
    "fig_lmoment_comparison",
    "fig_gev_quantile_comparison",
]

"""Manuscript figure registry and placeholder producers.

Maps each of the 10 main-text figures (and the central supplementary
figures) in notes/figure_design.md to a producer function. Figures
that need new schematic/design code (1, 4, 7, 10) ship as labelled
placeholder PNGs so the pipeline runs end-to-end from the first
commit. Real producers can be substituted as manuscript writing
proceeds.

See notes/deliverables_tracker.md for status of each figure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Placeholder producer
# ---------------------------------------------------------------------------


def placeholder(
    fig_name: str,
    title: str,
    reason: str,
    out_path: Path,
) -> None:
    """Write a placeholder PNG for a manuscript figure not yet implemented.

    The placeholder is rendered with matplotlib and labelled with the
    figure name, title, and a short reason. Allows the pipeline to run
    end-to-end before all figure producers exist.

    Parameters
    ----------
    fig_name : str
        Registry key (e.g. "fig01_design_overview").
    title : str
        Human-readable title from figure_design.md.
    reason : str
        Short explanation of why this is a placeholder.
    out_path : Path
        Destination PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_axis_off()
    ax.text(
        0.5,
        0.72,
        "PLACEHOLDER",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#b22222",
    )
    ax.text(0.5, 0.55, fig_name, ha="center", va="center", fontsize=14)
    ax.text(0.5, 0.45, title, ha="center", va="center", fontsize=12, style="italic")
    ax.text(
        0.5,
        0.25,
        reason,
        ha="center",
        va="center",
        fontsize=10,
        wrap=True,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    logger.info("Placeholder written: %s", out_path)


# ---------------------------------------------------------------------------
# Producers. All producers accept (output_base, manuscript_dir). Placeholder
# producers ignore output_base; real producers read from it.
# ---------------------------------------------------------------------------


def _produce_fig01_placeholder(output_base: Path, manuscript_dir: Path) -> None:
    placeholder(
        "fig01_design_overview",
        "Experimental design overview",
        "Schematic (map + Budyko stratification + generator taxonomy) "
        "requires bespoke code; see figure_design.md Figure 1.",
        manuscript_dir / "fig01_design_overview.png",
    )


def _produce_fig02_placeholder(output_base: Path, manuscript_dir: Path) -> None:
    placeholder(
        "fig02_hydroclimatic_diversity",
        "Observed hydroclimatic diversity + raw-ensemble showcase",
        "Composite layout (per-region seasonal cycle + FDC + raw observed "
        "vs best/failure synthetic traces) pending. See decisions_log.md "
        "raw-ensemble-showcase-in-fig2.",
        manuscript_dir / "fig02_hydroclimatic_diversity.png",
    )


def _produce_fig03(output_base: Path, manuscript_dir: Path) -> None:
    """Real producer for Figure 3 (Grand MARE heatmap, two-tier).

    Delegates to methods.plotting.fig03_grand_mare_heatmap.produce.
    """
    from .fig03_grand_mare_heatmap import produce as _produce

    _produce(output_base, manuscript_dir)


def _produce_fig04_placeholder(output_base: Path, manuscript_dir: Path) -> None:
    placeholder(
        "fig04_mare_rank_consistency",
        "MARE vs rank-score consistency",
        "Producer pending Phase 2 results. See decisions_log.md "
        "fig4-demotion-rule for the rule governing main/supplement placement.",
        manuscript_dir / "fig04_mare_rank_consistency.png",
    )


def _produce_fig05(output_base: Path, manuscript_dir: Path) -> None:
    """Real producer for Figure 5 (metric-category decomposition)."""
    from .fig05_category_decomposition import produce as _produce

    _produce(output_base, manuscript_dir)


def _produce_fig06_placeholder(output_base: Path, manuscript_dir: Path) -> None:
    placeholder(
        "fig06_regime_deep_dives",
        "Regime diagnostic deep dives",
        "Composite of existing per-region figures (09, 07, 05, 12) for "
        "2 representative regimes. Implement via subplot_mosaic post-HPC.",
        manuscript_dir / "fig06_regime_deep_dives.png",
    )


def _produce_fig07_placeholder(output_base: Path, manuscript_dir: Path) -> None:
    placeholder(
        "fig07_failure_mode_taxonomy",
        "Failure mode taxonomy",
        "Schematic produced after Phase 2 results; severity thresholds "
        "per decisions_log.md failure-mode-severity-thresholds.",
        manuscript_dir / "fig07_failure_mode_taxonomy.png",
    )


def _produce_fig08(output_base: Path, manuscript_dir: Path) -> None:
    """Real producer for Figure 8 (spatial binding constraint in PNW)."""
    from .fig08_spatial_binding_pnw import produce as _produce

    _produce(output_base, manuscript_dir)


def _produce_fig09(output_base: Path, manuscript_dir: Path) -> None:
    """Real producer for Figure 9 (convergence curves + thresholds)."""
    from .fig09_convergence import produce as _produce

    _produce(output_base, manuscript_dir)


def _produce_fig10_placeholder(output_base: Path, manuscript_dir: Path) -> None:
    placeholder(
        "fig10_practitioner_guide",
        "Practitioner selection guide",
        "Decision table conditional on Phase 2 results; see "
        "figure_design.md Figure 10.",
        manuscript_dir / "fig10_practitioner_guide.png",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MANUSCRIPT_FIGURES: dict[str, dict] = {
    "fig01_design_overview": {
        "title": "Experimental design overview",
        "producer": _produce_fig01_placeholder,
        "inputs": [],
        "status": "placeholder",
    },
    "fig02_hydroclimatic_diversity": {
        "title": "Observed hydroclimatic diversity",
        "producer": _produce_fig02_placeholder,
        "inputs": ["data/*/monthly.csv"],
        "status": "placeholder",
    },
    "fig03_grand_mare_heatmap": {
        "title": "Grand MARE heatmap (two-tier)",
        "producer": _produce_fig03,
        "inputs": [
            "outputs/cross_region/mare.csv",
            "outputs/cross_region/tier_concordance.csv",
        ],
        "status": "code-ready",
    },
    "fig04_mare_rank_consistency": {
        "title": "MARE vs rank-score consistency",
        "producer": _produce_fig04_placeholder,
        "inputs": [
            "outputs/cross_region/mean_rank_scores.csv",
            "outputs/cross_region/mare.csv",
        ],
        "status": "placeholder",
    },
    "fig05_category_decomposition": {
        "title": "Metric-category decomposition",
        "producer": _produce_fig05,
        "inputs": ["outputs/cross_region/mare.csv"],
        "status": "code-ready",
    },
    "fig06_regime_deep_dives": {
        "title": "Regime diagnostic deep dives",
        "producer": _produce_fig06_placeholder,
        "inputs": [],
        "status": "placeholder",
    },
    "fig07_failure_mode_taxonomy": {
        "title": "Failure mode taxonomy",
        "producer": _produce_fig07_placeholder,
        "inputs": ["outputs/cross_region/all_metrics.csv"],
        "status": "placeholder",
    },
    "fig08_spatial_binding_pnw": {
        "title": "Spatial correlation binding in PNW",
        "producer": _produce_fig08,
        "inputs": ["outputs/cross_region/mare.csv"],
        "status": "code-ready",
    },
    "fig09_convergence": {
        "title": "Convergence analysis",
        "producer": _produce_fig09,
        "inputs": ["outputs/convergence/new_england/convergence_kirsch.csv"],
        "status": "code-ready",
    },
    "fig10_practitioner_guide": {
        "title": "Practitioner selection guide",
        "producer": _produce_fig10_placeholder,
        "inputs": [],
        "status": "placeholder",
    },
}


def produce_manuscript_figures(
    output_base: Path,
    main_figure_dir: Path,
    only: list[str] | None = None,
) -> dict[str, Path]:
    """Produce all registered manuscript figures.

    Parameters
    ----------
    output_base : Path
        Root of the outputs/ directory (used to validate input artifacts
        listed in the registry).
    main_figure_dir : Path
        Absolute destination for main-text figures. Caller is expected
        to pass config.MAIN_FIGURE_DIR here.
    only : list of str, optional
        If given, produce only these registry keys.

    Returns
    -------
    dict
        Mapping figure key -> output path.
    """
    output_base = Path(output_base)
    manuscript_dir = Path(main_figure_dir)
    manuscript_dir.mkdir(parents=True, exist_ok=True)

    # Inputs declared in the registry are anchored at the experiment
    # directory (_EXPT_DIR = output_base.parent). "outputs/..." paths
    # live under output_base; "data/..." paths live in a sibling
    # directory. Resolve both forms explicitly so the warning is correct
    # regardless of cwd.
    expt_root = output_base.parent

    def _resolve(rel: str) -> Path:
        return (expt_root / rel).resolve()

    produced = {}
    keys = only if only else list(MANUSCRIPT_FIGURES.keys())

    for key in keys:
        if key not in MANUSCRIPT_FIGURES:
            logger.warning("Unknown manuscript figure key: %s", key)
            continue
        entry = MANUSCRIPT_FIGURES[key]

        # Advisory missing-input log. Placeholders still render; real
        # producers raise a clear FileNotFoundError when inputs are
        # missing, which is surfaced to the caller.
        missing_inputs = []
        for rel in entry.get("inputs", []):
            if "*" in rel:
                if not list(expt_root.glob(rel)):
                    missing_inputs.append(rel)
            elif not _resolve(rel).exists():
                missing_inputs.append(rel)

        if missing_inputs:
            level = (
                logging.ERROR if entry["status"] == "code-ready" else logging.WARNING
            )
            logger.log(
                level,
                "%s [%s]: missing inputs %s",
                key,
                entry["status"],
                missing_inputs,
            )

        producer: Callable[[Path, Path], None] = entry["producer"]
        try:
            producer(output_base, manuscript_dir)
        except Exception as exc:
            logger.error("%s producer failed: %s", key, exc)
            continue
        produced[key] = manuscript_dir / f"{key}.png"

    logger.info("Produced %d manuscript figures in %s", len(produced), manuscript_dir)
    return produced

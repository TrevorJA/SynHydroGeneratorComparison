"""Convergence sweep logic and per-model convergence helpers.

Strategy: generate one large ensemble at N_MAX realizations, then
evaluate nested subsets at each sweep level.  At each level we draw
``N_BOOTSTRAP_DRAWS`` random subsets (without replacement) of size *n*
from the full ensemble and compute MARE for each draw, yielding a
distribution of MARE values that quantifies subsampling variability.

The subsets are seeded deterministically so results are reproducible.
"""

import logging
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from synhydro import validate_ensemble

from config import DATA_DIR, OUTPUT_DIR, N_YEARS, SEED, MODELS
from basins import CAMELS_REGIONS
from methods.data import load_region_data, select_input_data, get_reference_site_index
from methods.io import GENERATOR_CLASSES

logger = logging.getLogger(__name__)

N_REALIZATIONS_SWEEP = [5, 10, 25, 50, 100, 200, 500]
N_MAX = max(N_REALIZATIONS_SWEEP)
N_BOOTSTRAP_DRAWS = 10  # replicated subsamples per sweep level


def convergence_exists(region_output, model_key: str) -> bool:
    """Return True if convergence_<model_key>.csv exists in region_output.

    Parameters
    ----------
    region_output : path-like
    model_key : str

    Returns
    -------
    bool
    """
    return (Path(region_output) / f"convergence_{model_key}.csv").exists()


def _subsample_ensemble(ensemble, indices):
    """Return a new Ensemble containing only the given realization indices.

    Parameters
    ----------
    ensemble : Ensemble
        Full ensemble with N_MAX realizations.
    indices : array-like of int
        Which realizations to keep (0-based).

    Returns
    -------
    Ensemble
        Subset ensemble.
    """
    return ensemble.subset(indices)


def run_convergence_for_region_model(
    region_id: str,
    model_key: str,
    n_years: int = N_YEARS,
    seed: int = SEED,
    force: bool = False,
    n_max: int | None = None,
) -> None:
    """Fit one generator and sweep ensemble sizes with replicated subsampling.

    Generates one large ensemble at ``n_max`` realizations, then for each
    sweep level <= n_max draws ``N_BOOTSTRAP_DRAWS`` random subsets of
    that size, computes MARE for each, and records the mean, std, and
    individual draw values.

    Parameters
    ----------
    region_id : str
    model_key : str
    n_years : int
    seed : int
    force : bool
        If True, recompute even if output already exists.
    n_max : int, optional
        Cap the sweep at this realization count. Defaults to module-level
        ``N_MAX`` (publication scale). The sweep levels are the subset
        of ``N_REALIZATIONS_SWEEP`` with value <= n_max, plus n_max
        itself if it is not already in the list. Use a small value
        (e.g. 20) for smoke tests -- sampling more realizations than
        were generated is nonsensical.
    """
    region_cfg = CAMELS_REGIONS[region_id]
    region_output = OUTPUT_DIR / region_id
    region_output.mkdir(parents=True, exist_ok=True)

    summary_path = region_output / f"convergence_{model_key}.csv"
    detail_path = region_output / f"convergence_detail_{model_key}.csv"

    if not force and convergence_exists(region_output, model_key):
        logger.info(
            "Convergence CSV already exists for %s / %s -- skipping",
            region_id,
            model_key,
        )
        return

    effective_n_max = int(n_max) if n_max is not None else N_MAX
    sweep_levels = sorted({n for n in N_REALIZATIONS_SWEEP if n <= effective_n_max})
    if effective_n_max not in sweep_levels:
        sweep_levels.append(effective_n_max)
        sweep_levels.sort()
    if not sweep_levels:
        sweep_levels = [effective_n_max]

    Q_daily, Q_monthly, Q_annual = load_region_data(
        region_id, DATA_DIR, region_cfg["stations"]
    )

    cfg = MODELS[model_key]
    class_name = cfg["class_name"]
    gen_cls = GENERATOR_CLASSES[class_name]

    ref_site_idx = get_reference_site_index(Q_monthly)
    Q_input = select_input_data(Q_daily, Q_monthly, Q_annual, cfg, ref_site_idx)

    # --- Fit ---
    logger.info("Fitting %s for %s...", model_key, region_id)
    try:
        gen = gen_cls(**cfg.get("init_kwargs", {}))
        fit_kwargs = dict(cfg.get("fit_kwargs", {}))
        if class_name == "MultiSiteHMMGenerator":
            fit_kwargs.setdefault("random_state", seed)
        gen.fit(Q_input, **fit_kwargs)
    except Exception:
        logger.error("Failed to fit %s for %s", model_key, region_id)
        traceback.print_exc()
        return

    # --- Generate full ensemble at effective_n_max ---
    logger.info(
        "Generating %d realizations for %s (sweep levels: %s)...",
        effective_n_max,
        model_key,
        sweep_levels,
    )
    t0 = time.time()
    try:
        gen_kwargs = dict(cfg.get("gen_kwargs", {}))
        full_ensemble = gen.generate(
            n_realizations=effective_n_max,
            n_years=n_years,
            seed=seed,
            **gen_kwargs,
        )
    except Exception:
        logger.error(
            "Failed to generate full ensemble for %s / %s", model_key, region_id
        )
        traceback.print_exc()
        return
    gen_elapsed = time.time() - t0
    logger.info("  Generated %d realizations in %.1fs", effective_n_max, gen_elapsed)

    # Prepare monthly-resampled version if needed
    if cfg["frequency"] == "daily":
        full_monthly = full_ensemble.resample("MS")
    elif cfg["frequency"] == "annual":
        full_monthly = None
    else:
        full_monthly = full_ensemble

    # Choose validation reference
    Q_ref = Q_annual if cfg["frequency"] == "annual" else Q_monthly
    ens_for_val = full_ensemble if full_monthly is None else full_monthly

    summary_rows = []
    detail_rows = []
    rng = np.random.default_rng(seed)

    for n_real in sweep_levels:
        if n_real >= effective_n_max:
            # At the cap, use the full ensemble (single draw).
            draws = [np.arange(effective_n_max)]
        else:
            draws = [
                rng.choice(effective_n_max, size=n_real, replace=False)
                for _ in range(N_BOOTSTRAP_DRAWS)
            ]

        draw_mares = []
        for draw_idx, indices in enumerate(draws):
            t1 = time.time()
            try:
                subset = _subsample_ensemble(ens_for_val, indices)
                result = validate_ensemble(subset, Q_ref)
                elapsed = time.time() - t1

                mare = result.summary.get("mean_absolute_relative_error", np.nan)
                draw_mares.append(mare)

                summary_rows.append(
                    {
                        "model": model_key,
                        "n_realizations": n_real,
                        "draw": draw_idx,
                        "mare": mare,
                        "median_are": result.summary.get(
                            "median_absolute_relative_error", np.nan
                        ),
                        "max_are": result.summary.get(
                            "max_absolute_relative_error", np.nan
                        ),
                        "spatial_rmse": result.summary.get(
                            "spatial_correlation_rmse", np.nan
                        ),
                        "elapsed_s": elapsed,
                    }
                )

                # Detail rows only for draw 0 (to keep file size manageable)
                if draw_idx == 0:
                    df = result.to_dataframe()
                    df["model"] = model_key
                    df["n_realizations"] = n_real
                    detail_rows.append(df)

            except Exception:
                logger.error("  n_real=%4d draw=%d FAILED", n_real, draw_idx)
                traceback.print_exc()

        if draw_mares:
            mean_mare = np.nanmean(draw_mares)
            std_mare = np.nanstd(draw_mares)
            logger.info(
                "  n_real=%4d  MARE=%.4f +/- %.4f  (%d draws)",
                n_real,
                mean_mare,
                std_mare,
                len(draw_mares),
            )

    if not summary_rows:
        logger.warning("No results produced for %s / %s", region_id, model_key)
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved %s", summary_path)

    if detail_rows:
        detail_df = pd.concat(detail_rows, ignore_index=True)
        keep = ["model", "n_realizations", "metric", "category", "relative_error"]
        keep = [c for c in keep if c in detail_df.columns]
        detail_df[keep].to_csv(detail_path, index=False)
        logger.info("Saved %s", detail_path)

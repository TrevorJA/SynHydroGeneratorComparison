"""Per-ensemble metric computation and CSV cache helpers."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from synhydro import validate_ensemble

from methods.analysis import (
    pool_realization_values,
    aggregate_to_annual,
    compute_acf,
)

logger = logging.getLogger(__name__)


def _per_category_mare(result) -> dict[str, float]:
    """Extract MARE for each metric category from a ValidationResult.

    Returns
    -------
    dict
        Mapping ``category_name -> mean absolute relative error`` for
        that category only.
    """
    category_map = {
        "marginal": result.marginal,
        "temporal": result.temporal,
        "drought": result.drought,
        "spectral": result.spectral,
        "seasonal": result.seasonal,
        "annual": result.annual,
        "fdc": result.fdc,
        "lmoments": result.lmoments,
        "extremes": result.extremes,
        "crps": result.crps,
        "ssi_drought": result.ssi_drought,
    }
    out: dict[str, float] = {}
    for cat_name, cat_data in category_map.items():
        errors = []
        for site_metrics in cat_data.values():
            for values in site_metrics.values():
                if isinstance(values, dict):
                    re = values.get("relative_error")
                    if re is not None and np.isfinite(re):
                        errors.append(abs(re))
        out[cat_name] = float(np.mean(errors)) if errors else np.nan
    return out


def compute_metrics_for_ensemble(
    ensemble,
    Q_monthly: pd.DataFrame,
    Q_annual: pd.DataFrame,
    model_key: str,
    models_config: dict,
    site_idx: int = 0,
) -> dict:
    """Compute all metrics for one ensemble.

    Produces three tiers of output:
      1. **Monthly-tier** validation (primary; skipped for annual-only models)
      2. **Annual-tier** validation (all models, aggregated to annual)
      3. Distribution statistics at both frequencies

    Parameters
    ----------
    ensemble : Ensemble
        The synthetic ensemble to evaluate.
    Q_monthly : pd.DataFrame
        Historical monthly streamflow.
    Q_annual : pd.DataFrame
        Historical annual streamflow.
    model_key : str
        Key identifying the model in models_config.
    models_config : dict
        The MODELS config dict.
    site_idx : int
        Column index of the reference site.

    Returns
    -------
    dict
        Keys: "metrics" (DataFrame), "validation_summary" (DataFrame),
        "distribution_stats" (DataFrame).
    """
    cfg = models_config[model_key]
    freq = cfg["frequency"]

    # Univariate generators (multisite=False) produce single-column
    # ensembles regardless of how many historical sites exist. The
    # caller's site_idx is measured against the multi-site historical
    # DataFrame, so we collapse it to 0 when indexing the ensemble.
    ens_site_idx = site_idx if cfg.get("multisite", False) else 0

    # ------------------------------------------------------------------
    # Tier 1: Monthly-scale validation (skip for annual-only generators)
    # ------------------------------------------------------------------
    if freq == "daily":
        ens_for_monthly = ensemble.resample("MS")
    elif freq == "annual":
        ens_for_monthly = None
    else:
        ens_for_monthly = ensemble

    if ens_for_monthly is not None:
        result = validate_ensemble(ens_for_monthly, Q_monthly)
    else:
        # Annual-only: validate at native annual resolution
        result = validate_ensemble(ensemble, Q_annual)

    # --- metrics DataFrame (tidy) ---
    metrics_df = result.to_dataframe()[["metric", "category", "relative_error"]].rename(
        columns={"relative_error": "value"}
    )

    # --- validation_summary DataFrame ---
    summary_rows = [{"metric_name": k, "value": v} for k, v in result.summary.items()]

    # Per-category MARE
    cat_mare = _per_category_mare(result)
    for cat_name, cat_val in cat_mare.items():
        summary_rows.append({"metric_name": f"mare_{cat_name}", "value": cat_val})

    # ------------------------------------------------------------------
    # Tier 2: Annual-scale validation (all generators, for cross-tier
    #         comparison). Annual generators validate natively; monthly
    #         and daily generators aggregate to annual first.
    # ------------------------------------------------------------------
    if freq == "annual":
        annual_result = result  # already validated at annual
    else:
        if freq == "daily":
            ens_annual = ensemble.resample("YS")
        else:
            ens_annual = ens_for_monthly.resample("YS")
        annual_result = validate_ensemble(
            ens_annual,
            Q_annual,
            metrics=["marginal", "temporal", "drought", "extremes"],
        )

    annual_mare = annual_result.summary.get("mean_absolute_relative_error", np.nan)
    summary_rows.append({"metric_name": "annual_tier_mare", "value": annual_mare})

    # Per-category MARE for annual tier
    annual_cat_mare = _per_category_mare(annual_result)
    for cat_name, cat_val in annual_cat_mare.items():
        summary_rows.append(
            {"metric_name": f"annual_mare_{cat_name}", "value": cat_val}
        )

    summary_df = pd.DataFrame(summary_rows)

    # --- distribution_stats DataFrame ---
    stats_rows = []

    # Monthly pooled values (skip annual-only models)
    if ens_for_monthly is not None:
        monthly_values = pool_realization_values(ens_for_monthly, ens_site_idx)
        for stat_name, stat_val in _compute_distribution_stats(monthly_values):
            stats_rows.append(
                {"frequency": "monthly", "statistic": stat_name, "value": stat_val}
            )

    # Annual pooled values
    if freq == "annual":
        annual_values = pool_realization_values(ensemble, ens_site_idx)
    elif freq == "daily":
        annual_values = aggregate_to_annual(ensemble, ens_site_idx)
    else:
        annual_values = aggregate_to_annual(ens_for_monthly, ens_site_idx)

    for stat_name, stat_val in _compute_distribution_stats(annual_values):
        stats_rows.append(
            {"frequency": "annual", "statistic": stat_name, "value": stat_val}
        )

    dist_df = pd.DataFrame(stats_rows)

    return {
        "metrics": metrics_df,
        "validation_summary": summary_df,
        "distribution_stats": dist_df,
    }


def _compute_distribution_stats(values: np.ndarray) -> list:
    """Compute summary statistics for a pooled array.

    Parameters
    ----------
    values : np.ndarray
        Pooled 1-D array of flow values.

    Returns
    -------
    list of (str, float) tuples
    """
    import scipy.stats as sp_stats

    clean = values[np.isfinite(values)]
    if len(clean) < 4:
        return []

    acf = compute_acf(clean, 2)
    return [
        ("mean", float(np.mean(clean))),
        ("std", float(np.std(clean))),
        ("skewness", float(sp_stats.skew(clean))),
        ("kurtosis", float(sp_stats.kurtosis(clean))),
        ("lag1_acf", float(acf[1]) if len(acf) > 1 else np.nan),
        ("lag2_acf", float(acf[2]) if len(acf) > 2 else np.nan),
    ]


def save_metrics(output_dir, model_key: str, metrics_dict: dict) -> None:
    """Save metrics dict to CSV files.

    Parameters
    ----------
    output_dir : path-like
        Directory where CSV files are written.
    model_key : str
        Model identifier used in filenames.
    metrics_dict : dict
        Return value of compute_metrics_for_ensemble().
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_dict["metrics"].to_csv(output_dir / f"metrics_{model_key}.csv", index=False)
    metrics_dict["validation_summary"].to_csv(
        output_dir / f"validation_summary_{model_key}.csv", index=False
    )
    metrics_dict["distribution_stats"].to_csv(
        output_dir / f"distribution_stats_{model_key}.csv", index=False
    )
    logger.info("Saved metrics CSVs for %s in %s", model_key, output_dir)


def load_metrics(output_dir, model_key: str) -> dict:
    """Load saved metrics CSVs.

    Parameters
    ----------
    output_dir : path-like
        Directory containing CSV files.
    model_key : str
        Model identifier.

    Returns
    -------
    dict or None
        Dict with "metrics", "validation_summary", "distribution_stats" keys,
        or None if any file is missing.
    """
    output_dir = Path(output_dir)
    paths = {
        "metrics": output_dir / f"metrics_{model_key}.csv",
        "validation_summary": output_dir / f"validation_summary_{model_key}.csv",
        "distribution_stats": output_dir / f"distribution_stats_{model_key}.csv",
    }
    if not all(p.exists() for p in paths.values()):
        return None
    return {k: pd.read_csv(v) for k, v in paths.items()}


def metrics_exist(output_dir, model_key: str) -> bool:
    """Return True if all three metric CSVs exist for this model.

    Parameters
    ----------
    output_dir : path-like
    model_key : str

    Returns
    -------
    bool
    """
    output_dir = Path(output_dir)
    return all(
        (output_dir / fname).exists()
        for fname in [
            f"metrics_{model_key}.csv",
            f"validation_summary_{model_key}.csv",
            f"distribution_stats_{model_key}.csv",
        ]
    )

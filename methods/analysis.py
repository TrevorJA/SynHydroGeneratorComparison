"""Shared analysis utilities: pooling, aggregation, color assignment."""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from synhydro.core.ensemble import Ensemble

logger = logging.getLogger(__name__)


def assign_colors(model_keys: list) -> dict:
    """Assign semantic family colors to each model key.

    Colors are defined in :mod:`methods.colors` and grouped by generation
    type (bootstrap, spectral, AR, state-based, copula).  Unregistered
    keys receive medium gray.
    """
    from .colors import MODEL_COLORS

    return {key: MODEL_COLORS.get(key, "#888888") for key in model_keys}


def pool_realization_values(ensemble: Ensemble, site_idx: int = 0) -> np.ndarray:
    """Concatenate first-site values across all realizations."""
    arrays = []
    for i in range(ensemble.metadata.n_realizations):
        df = ensemble.data_by_realization[i]
        arrays.append(df.iloc[:, site_idx].values)
    return np.concatenate(arrays)


def aggregate_to_monthly(ensemble: Ensemble, site_idx: int = 0) -> np.ndarray:
    """Aggregate daily ensemble to monthly totals, pool across realizations."""
    arrays = []
    for i in range(ensemble.metadata.n_realizations):
        df = ensemble.data_by_realization[i]
        monthly = df.resample("MS").sum()
        arrays.append(monthly.iloc[:, site_idx].values)
    return np.concatenate(arrays)


def aggregate_to_annual(ensemble: Ensemble, site_idx: int = 0) -> np.ndarray:
    """Aggregate to annual totals, pool across realizations."""
    arrays = []
    for i in range(ensemble.metadata.n_realizations):
        df = ensemble.data_by_realization[i]
        annual = df.resample("YS").sum()
        arrays.append(annual.iloc[:, site_idx].values)
    return np.concatenate(arrays)


def compute_acf(data: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute sample autocorrelation function."""
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        if len(data) > lag:
            acf.append(np.corrcoef(data[:-lag], data[lag:])[0, 1])
        else:
            acf.append(np.nan)
    return np.array(acf)


FREQ_RANK = {"daily": 0, "monthly": 1, "annual": 2}


def build_model_data(
    ensembles: dict,
    target_freq: str,
    Q_hist_values: np.ndarray,
    models_config: dict,
    site_idx: int = 0,
) -> Dict[str, np.ndarray]:
    """Build {label: 1-D array} for all models at a target frequency.

    Parameters
    ----------
    ensembles : dict
        Mapping model_key -> Ensemble.
    target_freq : str
        "daily", "monthly", or "annual".
    Q_hist_values : np.ndarray
        Historical 1-D values at the target frequency.
    models_config : dict
        The MODELS config dict.
    site_idx : int
        Site column index.

    Returns
    -------
    dict
        {label: 1-D numpy array} including "Historical".
    """
    target_rank = FREQ_RANK[target_freq]
    data = {"Historical": Q_hist_values}

    for model_key, ensemble in ensembles.items():
        cfg = models_config[model_key]
        native_rank = FREQ_RANK[cfg["frequency"]]
        if native_rank > target_rank:
            continue
        # Univariate generators produce single-column ensembles; clamp
        # to column 0 when the caller passes a multi-site historical
        # reference site index.
        ens_site_idx = site_idx if cfg.get("multisite", False) else 0
        if native_rank == target_rank:
            values = pool_realization_values(ensemble, ens_site_idx)
        elif target_freq == "monthly" and cfg["frequency"] == "daily":
            values = aggregate_to_monthly(ensemble, ens_site_idx)
        elif target_freq == "annual":
            values = aggregate_to_annual(ensemble, ens_site_idx)
        else:
            continue
        data[model_key] = values

    return data


def get_monthly_ensembles(
    ensembles: dict,
    models_config: dict,
) -> dict:
    """Filter to monthly-frequency models only."""
    return {
        k: v
        for k, v in ensembles.items()
        if models_config[k]["frequency"] in ("monthly",)
    }

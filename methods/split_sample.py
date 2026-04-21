"""Split-sample validation logic: shared library functions.

The HPC array entry point (pipeline/split_sample_single.py) and the
interactive batch script (scripts/split_sample.py) both import from here.
Keeping the core logic here removes the importlib hack previously needed
because the source file had a numeric prefix (06_split_sample.py).
"""

import logging
import traceback

import numpy as np
import pandas as pd
from synhydro import validate_ensemble

from config import DATA_DIR, N_REALIZATIONS, N_YEARS, SEED, MODELS
from basins import CAMELS_REGIONS
from methods.data import (
    load_region_data,
    select_input_data,
    get_reference_site_index,
)
from methods.io import GENERATOR_CLASSES

logger = logging.getLogger(__name__)

# Water-year split point: WY 1981-1997 / WY 1998-2014
SPLIT_DATE = "1997-10-01"


def split_data(Q_daily, Q_monthly, Q_annual, split_date=SPLIT_DATE):
    """Split all three frequency DataFrames at the given date.

    Parameters
    ----------
    Q_daily, Q_monthly, Q_annual : pd.DataFrame
    split_date : str
        Date at which to split.

    Returns
    -------
    tuple of (first_half, second_half) for each frequency
    """
    sd = pd.Timestamp(split_date)
    return (
        (Q_daily[Q_daily.index < sd], Q_daily[Q_daily.index >= sd]),
        (Q_monthly[Q_monthly.index < sd], Q_monthly[Q_monthly.index >= sd]),
        (Q_annual[Q_annual.index < sd], Q_annual[Q_annual.index >= sd]),
    )


def run_split_sample_for_pair(
    region_id,
    model_key,
    n_realizations=N_REALIZATIONS,
    n_years=N_YEARS,
    seed=SEED,
):
    """Run split-sample validation for one (region, model) pair.

    Fits on first half, validates on both halves, then repeats with
    the halves reversed.

    Parameters
    ----------
    region_id : str
    model_key : str
    n_realizations : int
    n_years : int
    seed : int

    Returns
    -------
    list of dict
        One row per (split_direction, validation_half) combination.
    """
    region_cfg = CAMELS_REGIONS[region_id]
    Q_daily, Q_monthly, Q_annual = load_region_data(
        region_id, DATA_DIR, region_cfg["stations"]
    )

    cfg = MODELS[model_key]
    class_name = cfg["class_name"]
    gen_cls = GENERATOR_CLASSES[class_name]
    ref_site_idx = get_reference_site_index(Q_monthly)

    (d1, d2), (m1, m2), (a1, a2) = split_data(Q_daily, Q_monthly, Q_annual)

    rows = []
    for split_name, train_d, train_m, train_a, val_m, val_a, val_label in [
        ("first_half", d1, m1, a1, m2, a2, "out_of_sample"),
        ("first_half", d1, m1, a1, m1, a1, "in_sample"),
        ("second_half", d2, m2, a2, m1, a1, "out_of_sample"),
        ("second_half", d2, m2, a2, m2, a2, "in_sample"),
    ]:
        try:
            Q_input = select_input_data(train_d, train_m, train_a, cfg, ref_site_idx)
            gen = gen_cls(**cfg.get("init_kwargs", {}))
            fit_kwargs = dict(cfg.get("fit_kwargs", {}))
            if class_name == "MultiSiteHMMGenerator":
                fit_kwargs.setdefault("random_state", seed)
            gen.fit(Q_input, **fit_kwargs)

            gen_kwargs = dict(cfg.get("gen_kwargs", {}))
            ensemble = gen.generate(
                n_realizations=n_realizations,
                n_years=n_years,
                seed=seed,
                **gen_kwargs,
            )

            if cfg["frequency"] == "daily":
                ensemble = ensemble.resample("MS")

            Q_ref = val_a if cfg["frequency"] == "annual" else val_m
            result = validate_ensemble(ensemble, Q_ref)

            mare = result.summary.get("mean_absolute_relative_error", np.nan)
            spatial = result.summary.get("spatial_correlation_rmse", np.nan)

            rows.append(
                {
                    "region": region_id,
                    "model": model_key,
                    "train_half": split_name,
                    "validation": val_label,
                    "mare": mare,
                    "spatial_rmse": spatial,
                }
            )
            logger.info(
                "  %s / %s  train=%s  val=%s  MARE=%.4f",
                region_id,
                model_key,
                split_name,
                val_label,
                mare,
            )
        except Exception:
            logger.error(
                "  FAILED: %s / %s  train=%s  val=%s",
                region_id,
                model_key,
                split_name,
                val_label,
            )
            traceback.print_exc()
            rows.append(
                {
                    "region": region_id,
                    "model": model_key,
                    "train_half": split_name,
                    "validation": val_label,
                    "mare": np.nan,
                    "spatial_rmse": np.nan,
                }
            )

    return rows

"""Ensemble I/O: loading, saving, and generator class resolution."""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from synhydro.core.ensemble import Ensemble
from synhydro.methods.generation import (
    ARFIMAGenerator,
    GaussianCopulaGenerator,
    HMMKNNGenerator,
    KirschGenerator,
    KNNBootstrapGenerator,
    MatalasGenerator,
    MultiSiteHMMGenerator,
    MultisitePhaseRandomizationGenerator,
    PhaseRandomizationGenerator,
    ThomasFieringGenerator,
    VineCopulaGenerator,
    WARMGenerator,
)

logger = logging.getLogger(__name__)

GENERATOR_CLASSES = {
    "ARFIMAGenerator": ARFIMAGenerator,
    "GaussianCopulaGenerator": GaussianCopulaGenerator,
    "HMMKNNGenerator": HMMKNNGenerator,
    "KirschGenerator": KirschGenerator,
    "KNNBootstrapGenerator": KNNBootstrapGenerator,
    "MatalasGenerator": MatalasGenerator,
    "MultiSiteHMMGenerator": MultiSiteHMMGenerator,
    "MultisitePhaseRandomizationGenerator": MultisitePhaseRandomizationGenerator,
    "PhaseRandomizationGenerator": PhaseRandomizationGenerator,
    "ThomasFieringGenerator": ThomasFieringGenerator,
    "VineCopulaGenerator": VineCopulaGenerator,
    "WARMGenerator": WARMGenerator,
}


def load_ensembles_pickle(
    output_dir: Path,
    models_config: dict,
) -> Dict[str, Ensemble]:
    """Load all available ensemble pickle files based on config.

    Parameters
    ----------
    output_dir : Path
        Directory containing pickle files.
    models_config : dict
        The MODELS config dict mapping model_key -> config.

    Returns
    -------
    dict
        Mapping of model_key -> Ensemble for each found pickle.
    """
    ensembles = {}
    for model_key, cfg in models_config.items():
        if not cfg.get("enabled", True):
            continue
        pkl_path = output_dir / cfg["output_file"]
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                ensembles[model_key] = pickle.load(f)
        else:
            logger.warning(
                "Skipped %s (file not found: %s)", model_key, cfg["output_file"]
            )
    return ensembles


def load_ensembles_hdf5(
    region_dir: Path,
    models_config: dict,
) -> Dict[str, Ensemble]:
    """Load HDF5 ensembles for a region.

    Parameters
    ----------
    region_dir : Path
        Directory containing HDF5 files for a region
        (e.g. outputs/new_england/).
    models_config : dict
        The MODELS config dict.

    Returns
    -------
    dict
        Mapping of model_key -> Ensemble.
    """
    ensembles = {}
    for model_key, cfg in models_config.items():
        if not cfg.get("enabled", True):
            continue
        h5_path = region_dir / f"ensemble_{model_key}.h5"
        if h5_path.exists():
            ensembles[model_key] = Ensemble.from_hdf5(str(h5_path))
        else:
            logger.debug("No HDF5 for %s at %s", model_key, h5_path)
    return ensembles


def load_ensemble(region_output: Path, model_key: str) -> Optional[Ensemble]:
    """Load ensemble from HDF5, falling back to pickle. Returns None if not found.

    Parameters
    ----------
    region_output : Path
        Directory containing ensemble files for this region.
    model_key : str
        Model identifier.

    Returns
    -------
    Ensemble or None
        Returns None if no file is found.
    """
    import pickle as _pickle

    h5_path = region_output / f"ensemble_{model_key}.h5"
    pkl_path = region_output / f"ensemble_{model_key}.pkl"

    if h5_path.exists():
        logger.info("Loading ensemble from %s", h5_path)
        return Ensemble.from_hdf5(str(h5_path))
    if pkl_path.exists():
        logger.info("Loading ensemble from %s", pkl_path)
        with open(pkl_path, "rb") as f:
            return _pickle.load(f)
    return None


def convergence_exists(region_output: Path, model_key: str) -> bool:
    """Return True if convergence_<model_key>.csv exists in region_output.

    Parameters
    ----------
    region_output : Path
    model_key : str

    Returns
    -------
    bool
    """
    return (region_output / f"convergence_{model_key}.csv").exists()


def split_sample_output_path(output_dir: Path, region_id: str, model_key: str) -> Path:
    """Return the canonical CSV path for a (region, model) split-sample result.

    Parameters
    ----------
    output_dir : Path
        Base OUTPUT_DIR (contains the `split_sample/` subdirectory).
    region_id : str
    model_key : str

    Returns
    -------
    Path
    """
    return output_dir / "split_sample" / f"{region_id}__{model_key}.csv"


def split_sample_exists(output_dir: Path, region_id: str, model_key: str) -> bool:
    """Return True if the split-sample CSV for this pair exists.

    Parameters
    ----------
    output_dir : Path
    region_id : str
    model_key : str

    Returns
    -------
    bool
    """
    return split_sample_output_path(output_dir, region_id, model_key).exists()


def save_historical_csvs(
    Q_daily: pd.DataFrame,
    Q_monthly: pd.DataFrame,
    Q_annual: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save historical reference data as CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    Q_daily.to_csv(output_dir / "historical_daily.csv")
    Q_monthly.to_csv(output_dir / "historical_monthly.csv")
    Q_annual.to_csv(output_dir / "historical_annual.csv")
    print(f"  Saved historical data to {output_dir}")

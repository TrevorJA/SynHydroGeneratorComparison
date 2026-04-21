"""Task mapping functions for all three pipeline stages."""

import zlib

from basins import CAMELS_REGIONS
from config import MODELS


def derive_task_seed(base_seed: int, region_id: str, model_key: str) -> int:
    """Deterministic, task-unique seed for (region, model) pair.

    Global `np.random.seed(SEED)` across 78 array tasks leaves every
    (region, model) starting from an identical RNG state. Mixing the
    (region, model) key into the seed via crc32 gives each task its
    own stream while staying reproducible across re-runs (crc32 is
    stable, unlike Python's salted hash()). Range-clipped to uint32.
    """
    key = f"{region_id}|{model_key}".encode()
    return (base_seed + zlib.crc32(key)) % (2**32)


def get_generation_tasks() -> list:
    """All (region, model) pairs for generation stage. Used by generate_single.py.

    Returns
    -------
    list of (str, str)
        Each element is (region_id, model_key).
    """
    regions = sorted(CAMELS_REGIONS.keys())
    models = sorted(k for k, v in MODELS.items() if v.get("enabled", True))
    tasks = []
    for region_id in regions:
        for model_key in models:
            tasks.append((region_id, model_key))
    return tasks


def get_analysis_tasks() -> list:
    """All (region, model) pairs for analysis stage. Same as generation tasks.

    Returns
    -------
    list of (str, str)
        Each element is (region_id, model_key).
    """
    regions = sorted(CAMELS_REGIONS.keys())
    models = sorted(k for k, v in MODELS.items() if v.get("enabled", True))
    tasks = []
    for region_id in regions:
        for model_key in models:
            tasks.append((region_id, model_key))
    return tasks


def get_convergence_tasks() -> list:
    """(region, model) pairs for convergence stage -- non-annual models only.

    Only non-annual models are eligible because validate_ensemble is called
    against Q_monthly.

    Returns
    -------
    list of (str, str)
        Each element is (region_id, model_key).
    """
    regions = sorted(CAMELS_REGIONS.keys())
    models = sorted(
        k
        for k, v in MODELS.items()
        if v.get("enabled", True) and v["frequency"] != "annual"
    )
    tasks = []
    for region_id in regions:
        for model_key in models:
            tasks.append((region_id, model_key))
    return tasks


def get_split_sample_tasks() -> list:
    """All (region, model) pairs for split-sample stage.

    Same decomposition as generation/analysis. Each task runs four
    fit/validate scenarios (first_half train x in/out-of-sample, then
    second_half train x in/out-of-sample) internally.

    Returns
    -------
    list of (str, str)
        Each element is (region_id, model_key).
    """
    regions = sorted(CAMELS_REGIONS.keys())
    models = sorted(k for k, v in MODELS.items() if v.get("enabled", True))
    tasks = []
    for region_id in regions:
        for model_key in models:
            tasks.append((region_id, model_key))
    return tasks

"""Task mapping functions for all three pipeline stages."""

from basins import CAMELS_REGIONS
from config import MODELS


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

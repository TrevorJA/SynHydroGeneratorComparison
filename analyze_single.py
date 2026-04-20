"""
HPC entry point: compute metrics for a single (region, model) pair.

Designed for SLURM array jobs. Maps a task ID to a specific region and
model combination, loads the pre-generated ensemble, computes validation
metrics, and saves them as CSVs. Skips if CSVs already exist unless
--force is given.

Usage:
  python analyze_single.py --task-id 0
  python analyze_single.py --region new_england --model kirsch
  python analyze_single.py --list-tasks
  python analyze_single.py --region new_england --model kirsch --force

From SLURM (see run_analyze.sh):
  python analyze_single.py --task-id $SLURM_ARRAY_TASK_ID
"""

import argparse
import logging
import os
import sys

# pyvinecopulib must be imported before pandas/pyarrow on Windows
try:
    import pyvinecopulib  # noqa: F401
except ImportError:
    pass

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    MODELS,
    ACTIVE_REGIONS,
    SINGLE_SITE_INDEX,
)
from basins import CAMELS_REGIONS
from methods.data import load_region_data, get_reference_site_index
from methods.io import load_ensemble
from methods.metrics import compute_metrics_for_ensemble, save_metrics, metrics_exist
from methods.tasks import get_analysis_tasks

logger = logging.getLogger(__name__)


def analyze_for_region_model(region_id, model_key, force=False):
    """Compute and save metrics for one (region, model) pair.

    Parameters
    ----------
    region_id : str
    model_key : str
    force : bool
        If True, recompute even if CSVs already exist.
    """
    region_cfg = CAMELS_REGIONS[region_id]
    region_output = OUTPUT_DIR / region_id

    if not force and metrics_exist(region_output, model_key):
        logger.info(
            "Metrics already exist for %s / %s -- skipping (use --force to override)",
            region_id,
            model_key,
        )
        return

    ensemble = load_ensemble(region_output, model_key)
    if ensemble is None:
        logger.warning(
            "No ensemble file found for %s / %s -- skipping",
            region_id,
            model_key,
        )
        return

    Q_daily, Q_monthly, Q_annual = load_region_data(
        region_id, DATA_DIR, region_cfg["stations"]
    )

    site_idx = get_reference_site_index(Q_monthly)
    logger.info(
        "Computing metrics for %s / %s (site_idx=%d)",
        region_id,
        model_key,
        site_idx,
    )

    metrics_dict = compute_metrics_for_ensemble(
        ensemble,
        Q_monthly,
        Q_annual,
        model_key,
        MODELS,
        site_idx=site_idx,
    )

    save_metrics(region_output, model_key, metrics_dict)
    logger.info("Done: %s / %s", region_id, model_key)


def main():
    parser = argparse.ArgumentParser(description="HPC single-task metric computation")
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="SLURM array task ID (0-indexed)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Region ID (use with --model for direct invocation)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model key (use with --region for direct invocation)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute metrics even if CSVs already exist",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print task mapping and exit",
    )
    args = parser.parse_args()

    tasks = get_analysis_tasks()

    if args.list_tasks:
        print(f"Total tasks: {len(tasks)}")
        print(f"Use --array=0-{len(tasks) - 1} in SLURM submission\n")
        for i, (region_id, model_key) in enumerate(tasks):
            print(f"  {i:3d}  {region_id:25s}  {model_key}")
        return

    # Resolve (region, model) from either --task-id or --region/--model
    if args.region is not None and args.model is not None:
        region_id = args.region
        model_key = args.model
    elif args.task_id is not None:
        if args.task_id < 0 or args.task_id >= len(tasks):
            print(
                f"ERROR: task-id {args.task_id} out of range [0, {len(tasks) - 1}]",
                file=sys.stderr,
            )
            sys.exit(1)
        region_id, model_key = tasks[args.task_id]
    else:
        print(
            "ERROR: provide --task-id or both --region and --model (or use --list-tasks)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Configure per-task log file
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"{region_id}_{model_key}_analyze.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logger.info(
        "analyze_single: region=%s, model=%s, force=%s",
        region_id,
        model_key,
        args.force,
    )

    analyze_for_region_model(region_id, model_key, force=args.force)


if __name__ == "__main__":
    main()

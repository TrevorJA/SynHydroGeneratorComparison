"""
HPC entry point: split-sample validation for a single (region, model) pair.

Designed for SLURM array jobs. Runs the full 4-scenario split-sample
protocol (first_half train x {in, out}-of-sample, second_half train
x {in, out}-of-sample) for one (region, model) pair and saves the
result as a per-task CSV under outputs/split_sample/.

Skips if the output CSV already exists unless --force is given.

Usage:
  python split_sample_single.py --task-id 0
  python split_sample_single.py --region new_england --model kirsch
  python split_sample_single.py --list-tasks
  python split_sample_single.py --region new_england --model kirsch --force

From SLURM (see run_split_sample.sh):
  python split_sample_single.py --task-id $SLURM_ARRAY_TASK_ID
"""

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

# pyvinecopulib must be imported before pandas/pyarrow on Windows
try:
    import pyvinecopulib  # noqa: F401
except ImportError:
    pass

import pandas as pd

from config import OUTPUT_DIR, N_REALIZATIONS, N_YEARS, SEED
from methods.io import split_sample_exists, split_sample_output_path
from methods.tasks import get_split_sample_tasks

logger = logging.getLogger(__name__)


def _load_run_split_sample_for_pair():
    """Import run_split_sample_for_pair from 06_split_sample.py.

    The source file has a numeric-prefixed name that cannot be imported
    as a regular Python module, so we load it via importlib.
    """
    src = Path(__file__).resolve().parent / "06_split_sample.py"
    spec = importlib.util.spec_from_file_location("_split_sample_src", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_split_sample_for_pair


def main():
    parser = argparse.ArgumentParser(
        description="HPC single-task split-sample validation"
    )
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
        "--n-realizations",
        type=int,
        default=N_REALIZATIONS,
        help=f"Realizations per fit/validate scenario (default: {N_REALIZATIONS})",
    )
    parser.add_argument(
        "--n-years",
        type=int,
        default=N_YEARS,
        help=f"Years per realization (default: {N_YEARS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if output CSV already exists",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print task mapping and exit",
    )
    args = parser.parse_args()

    tasks = get_split_sample_tasks()

    if args.list_tasks:
        print(f"Total tasks: {len(tasks)}")
        print(f"Use --array=0-{len(tasks) - 1} in SLURM submission\n")
        for i, (region_id, model_key) in enumerate(tasks):
            print(f"  {i:3d}  {region_id:25s}  {model_key}")
        return

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

    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"{region_id}_{model_key}_split_sample.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    out_path = split_sample_output_path(OUTPUT_DIR, region_id, model_key)
    if split_sample_exists(OUTPUT_DIR, region_id, model_key) and not args.force:
        logger.info("SKIP: %s already exists (use --force to overwrite)", out_path)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "split_sample_single: region=%s, model=%s, n_real=%d, force=%s",
        region_id,
        model_key,
        args.n_realizations,
        args.force,
    )

    run_split_sample_for_pair = _load_run_split_sample_for_pair()
    rows = run_split_sample_for_pair(
        region_id,
        model_key,
        n_realizations=args.n_realizations,
        n_years=args.n_years,
        seed=args.seed,
    )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("Saved %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()

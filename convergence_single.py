"""
HPC entry point: convergence sweep for a single (region, model) pair.

Designed for SLURM array jobs. Fits one generator, then sweeps
N_REALIZATIONS_SWEEP ensemble sizes and runs validate_ensemble at each
level, saving summary and detail CSVs. Skips if output already exists
unless --force is given.

Only non-annual models are included (annual models have no monthly
validate_ensemble target).

Usage:
  python convergence_single.py --task-id 0
  python convergence_single.py --region new_england --model kirsch
  python convergence_single.py --list-tasks
  python convergence_single.py --region new_england --model kirsch --force

From SLURM (see run_convergence.sh):
  python convergence_single.py --task-id $SLURM_ARRAY_TASK_ID
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

import numpy as np

from config import N_YEARS, SEED
from methods.io import convergence_exists
from methods.metrics import run_convergence_for_region_model
from methods.tasks import get_convergence_tasks

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="HPC single-task convergence sweep")
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
        help="Recompute even if output CSVs already exist",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=None,
        help=(
            "Cap the convergence sweep at this realization count. "
            "Defaults to the module-level N_MAX (500). For smoke tests "
            "pass the same value as --n-realizations used for generation."
        ),
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print task mapping and exit",
    )
    args = parser.parse_args()

    tasks = get_convergence_tasks()

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
    log_file = os.path.join("logs", f"{region_id}_{model_key}_convergence.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logger.info(
        "convergence_single: region=%s, model=%s, force=%s",
        region_id,
        model_key,
        args.force,
    )

    np.random.seed(args.seed)

    run_convergence_for_region_model(
        region_id,
        model_key,
        n_years=args.n_years,
        seed=args.seed,
        force=args.force,
        n_max=args.n_max,
    )


if __name__ == "__main__":
    main()

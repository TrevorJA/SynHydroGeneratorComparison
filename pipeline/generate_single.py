"""
HPC entry point: generate ensembles for a single (region, model) pair.

Designed for SLURM array jobs. Maps a task ID to a specific region and
model combination, then runs generation for that pair only.

Usage:
  # Direct invocation
  python pipeline/generate_single.py --task-id 0
  python pipeline/generate_single.py --list-tasks

  # From SLURM (see pipeline/slurm/run_generate.sh)
  python pipeline/generate_single.py --task-id $SLURM_ARRAY_TASK_ID
"""

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

# Make project root importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pyvinecopulib must be imported before pandas/pyarrow on Windows
try:
    import pyvinecopulib  # noqa: F401
except ImportError:
    pass

import numpy as np

from config import (
    DATA_DIR,
    GENERATION_DIR,
    N_REALIZATIONS,
    N_YEARS,
    SEED,
    OUTPUT_FORMAT,
    SINGLE_SITE_INDEX,
    MODELS,
    DIAGNOSTICS,
    LOG_DIR,
)
from basins import CAMELS_REGIONS
from methods.data import load_region_data, select_input_data, get_reference_site_index
from methods.io import GENERATOR_CLASSES, save_historical_csvs
from methods.tasks import derive_task_seed, get_generation_tasks

logger = logging.getLogger(__name__)


def generate_for_region_model(
    region_id, model_key, n_realizations, n_years, seed, output_format
):
    """Generate ensemble for one (region, model) pair."""
    region_cfg = CAMELS_REGIONS[region_id]

    Q_daily, Q_monthly, Q_annual = load_region_data(
        region_id, DATA_DIR, region_cfg["stations"]
    )

    region_output = GENERATION_DIR / region_id
    region_output.mkdir(parents=True, exist_ok=True)

    if DIAGNOSTICS.get("save_historical", True):
        hist_path = region_output / "historical_monthly.csv"
        if not hist_path.exists():
            save_historical_csvs(Q_daily, Q_monthly, Q_annual, region_output)

    # Skip if output already exists
    h5_path = region_output / f"ensemble_{model_key}.h5"
    pkl_path = region_output / f"ensemble_{model_key}.pkl"
    if h5_path.exists() or pkl_path.exists():
        existing = h5_path if h5_path.exists() else pkl_path
        logger.info(
            "Ensemble already exists for %s / %s at %s -- skipping",
            region_id,
            model_key,
            existing,
        )
        return

    cfg = MODELS[model_key]
    class_name = cfg["class_name"]
    gen_cls = GENERATOR_CLASSES[class_name]

    logger.info("Fitting %s for %s...", model_key, region_id)

    ref_site_idx = get_reference_site_index(Q_monthly)
    Q_input = select_input_data(Q_daily, Q_monthly, Q_annual, cfg, ref_site_idx)

    # Zero-flow check for log-transform models
    if hasattr(Q_input, "values"):
        data_vals = Q_input.values
    else:
        data_vals = Q_input
    if np.any(data_vals <= 0):
        init_kwargs = cfg.get("init_kwargs", {})
        if init_kwargs.get("generate_using_log_flow", False):
            logger.warning(
                "Skipping %s for %s: data has zero/negative values "
                "incompatible with log-transform",
                model_key,
                region_id,
            )
            return

    t0 = time.time()
    gen = gen_cls(**cfg.get("init_kwargs", {}))

    fit_kwargs = dict(cfg.get("fit_kwargs", {}))
    if class_name == "MultiSiteHMMGenerator":
        fit_kwargs.setdefault("random_state", seed)
    gen.fit(Q_input, **fit_kwargs)

    gen_kwargs = dict(cfg.get("gen_kwargs", {}))
    ensemble = gen.generate(
        n_years=n_years,
        n_realizations=n_realizations,
        seed=seed,
        **gen_kwargs,
    )

    elapsed = time.time() - t0
    shape = ensemble.data_by_realization[0].shape
    logger.info(
        "Generated %d realizations, shape=%s in %.1fs",
        ensemble.metadata.n_realizations,
        shape,
        elapsed,
    )

    if output_format == "hdf5":
        out_path = region_output / f"ensemble_{model_key}.h5"
        ensemble.to_hdf5(str(out_path))
    else:
        out_path = region_output / f"ensemble_{model_key}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(ensemble, f)

    # Post-save verification. Guards against partial writes / corrupted
    # HDF5 that would otherwise pass a bare `.exists()` gate. On failure
    # raise so the SLURM task exits nonzero.
    if output_format == "hdf5":
        import h5py
        try:
            with h5py.File(out_path, "r") as f:
                _ = list(f.keys())
        except Exception as exc:
            raise RuntimeError(
                f"Ensemble HDF5 at {out_path} failed post-save verification: {exc}"
            ) from exc
    else:
        if out_path.stat().st_size == 0:
            raise RuntimeError(f"Ensemble pickle at {out_path} is zero bytes")

    logger.info("Saved: %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="HPC single-task ensemble generation")
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="SLURM array task ID (0-indexed)",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=N_REALIZATIONS,
        help=f"Number of realizations (default: {N_REALIZATIONS})",
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
        "--format",
        type=str,
        default=OUTPUT_FORMAT,
        choices=["hdf5", "pickle"],
        help=f"Output format (default: {OUTPUT_FORMAT})",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print task mapping and exit",
    )
    args = parser.parse_args()

    tasks = get_generation_tasks()

    if args.list_tasks:
        print(f"Total tasks: {len(tasks)}")
        print(f"Use --array=0-{len(tasks) - 1} in SLURM submission\n")
        for i, (region_id, model_key) in enumerate(tasks):
            print(f"  {i:3d}  {region_id:25s}  {model_key}")
        return

    if args.task_id is None:
        print("ERROR: --task-id is required (or use --list-tasks)", file=sys.stderr)
        sys.exit(1)

    if args.task_id < 0 or args.task_id >= len(tasks):
        print(
            f"ERROR: task-id {args.task_id} out of range [0, {len(tasks) - 1}]",
            file=sys.stderr,
        )
        sys.exit(1)

    region_id, model_key = tasks[args.task_id]

    # Configure logging to file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{region_id}_{model_key}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    task_seed = derive_task_seed(args.seed, region_id, model_key)

    logger.info(
        "Task %d: region=%s, model=%s, n_real=%d, n_years=%d, base_seed=%d, task_seed=%d",
        args.task_id,
        region_id,
        model_key,
        args.n_realizations,
        args.n_years,
        args.seed,
        task_seed,
    )

    np.random.seed(task_seed)

    generate_for_region_model(
        region_id,
        model_key,
        args.n_realizations,
        args.n_years,
        task_seed,
        args.format,
    )


if __name__ == "__main__":
    main()

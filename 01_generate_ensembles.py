"""
Generate synthetic streamflow ensembles for all configured CAMELS regions.

Reads region definitions from basins.py and model configurations from
config.py. For each (region, model) pair, fits the generator on observed
data and produces an ensemble of synthetic realizations.

Requires: Run 00_retrieve_data.py first to cache CAMELS streamflow data.

Usage:
  python 01_generate_ensembles.py                          # All regions, all models
  python 01_generate_ensembles.py --region new_england     # Single region
  python 01_generate_ensembles.py --region new_england --model kirsch  # Single pair
  python 01_generate_ensembles.py --n-realizations 1000    # Override ensemble size
"""

import argparse
import logging
import time
import traceback

# pyvinecopulib must be imported before pandas/pyarrow on Windows to avoid
# a C++ runtime DLL conflict.  Keep this import first.
try:
    import pyvinecopulib  # noqa: F401
except ImportError:
    pass

import numpy as np
import pickle

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    N_REALIZATIONS,
    N_YEARS,
    SEED,
    OUTPUT_FORMAT,
    SINGLE_SITE_INDEX,
    MODELS,
    DIAGNOSTICS,
    ACTIVE_REGIONS,
    ACTIVE_MODELS,
)
from basins import CAMELS_REGIONS
from methods.data import load_region_data, select_input_data, get_reference_site_index
from methods.io import GENERATOR_CLASSES, save_historical_csvs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def get_enabled_models(model_filter=None):
    """Return the subset of MODELS that are enabled and pass the filter."""
    enabled = {k: v for k, v in MODELS.items() if v.get("enabled", True)}
    if model_filter:
        enabled = {k: v for k, v in enabled.items() if k in model_filter}
    return enabled


def get_active_regions(region_filter=None):
    """Return the subset of CAMELS_REGIONS to process."""
    if region_filter:
        return {k: v for k, v in CAMELS_REGIONS.items() if k in region_filter}
    return dict(CAMELS_REGIONS)


def generate_for_region(
    region_id,
    region_cfg,
    models_to_run,
    n_realizations,
    n_years,
    seed,
    output_format,
):
    """Generate ensembles for one region across specified models."""
    print(f"\n{'='*70}")
    print(f"Region: {region_id} -- {region_cfg['description']}")
    print(f"  Stations: {region_cfg['stations']}")
    print(f"  Climate:  {region_cfg['climate_type']}")
    print(f"{'='*70}")

    Q_daily, Q_monthly, Q_annual = load_region_data(
        region_id, DATA_DIR, region_cfg["stations"]
    )
    print(f"  Data: {Q_daily.shape[0]} days, {Q_daily.shape[1]} sites")
    print(f"  Period: {Q_daily.index[0].date()} to {Q_daily.index[-1].date()}")

    ref_site_idx = get_reference_site_index(Q_monthly)
    print(
        f"  Reference site (highest mean flow): {Q_monthly.columns[ref_site_idx]} (index {ref_site_idx})"
    )

    region_output = OUTPUT_DIR / region_id
    region_output.mkdir(parents=True, exist_ok=True)

    if DIAGNOSTICS.get("save_historical", True):
        save_historical_csvs(Q_daily, Q_monthly, Q_annual, region_output)

    results = {}
    n_models = len(models_to_run)

    for i, (model_key, cfg) in enumerate(models_to_run.items(), 1):
        class_name = cfg["class_name"]
        freq = cfg["frequency"]
        sites_label = "multi-site" if cfg["multisite"] else "single-site"

        print(f"\n  [{i}/{n_models}] {model_key}: {cfg['description']}")
        print(f"    Class: {class_name} | Frequency: {freq} | {sites_label}")

        try:
            gen_cls = GENERATOR_CLASSES[class_name]

            Q_input = select_input_data(Q_daily, Q_monthly, Q_annual, cfg, ref_site_idx)

            # Check for zero-flow issues with log-transform models
            if hasattr(Q_input, "values"):
                data_vals = Q_input.values
            else:
                data_vals = Q_input
            if np.any(data_vals <= 0):
                init_kwargs = cfg.get("init_kwargs", {})
                if init_kwargs.get("generate_using_log_flow", False):
                    print(
                        f"    WARNING: Data contains zero/negative values; "
                        f"skipping {model_key} (requires log-transform)"
                    )
                    results[model_key] = None
                    continue

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

            if DIAGNOSTICS.get("print_summary", True):
                shape = ensemble.data_by_realization[0].shape
                print(
                    f"    Generated {ensemble.metadata.n_realizations} "
                    f"realizations, shape={shape}  ({elapsed:.1f}s)"
                )

            if output_format == "hdf5":
                out_path = region_output / f"ensemble_{model_key}.h5"
                ensemble.to_hdf5(str(out_path))
            else:
                out_path = region_output / f"ensemble_{model_key}.pkl"
                with open(out_path, "wb") as f:
                    pickle.dump(ensemble, f)
            print(f"    Saved: {out_path}")

            results[model_key] = ensemble

        except Exception:
            print(f"    FAILED -- skipping {model_key}")
            traceback.print_exc()
            results[model_key] = None

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic ensembles for CAMELS regions"
    )
    parser.add_argument(
        "--region", type=str, default=None, help="Single region to process"
    )
    parser.add_argument("--model", type=str, default=None, help="Single model to run")
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=None,
        help=f"Override n_realizations (default: {N_REALIZATIONS})",
    )
    parser.add_argument(
        "--n-years",
        type=int,
        default=None,
        help=f"Override n_years (default: {N_YEARS})",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help=f"Override seed (default: {SEED})"
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["hdf5", "pickle"],
        help=f"Output format (default: {OUTPUT_FORMAT})",
    )
    args = parser.parse_args()

    n_realizations = args.n_realizations or N_REALIZATIONS
    n_years = args.n_years or N_YEARS
    seed = args.seed or SEED
    output_format = args.format or OUTPUT_FORMAT

    np.random.seed(seed)

    region_filter = [args.region] if args.region else ACTIVE_REGIONS
    model_filter = [args.model] if args.model else ACTIVE_MODELS

    regions = get_active_regions(region_filter)
    models = get_enabled_models(model_filter)

    print("=" * 70)
    print("SynHydro Model Comparison -- Ensemble Generation")
    print(f"  Regions:        {len(regions)} ({list(regions.keys())})")
    print(f"  Models:         {len(models)} ({list(models.keys())})")
    print(f"  Realizations:   {n_realizations}")
    print(f"  Years:          {n_years}")
    print(f"  Seed:           {seed}")
    print(f"  Output format:  {output_format}")
    print("=" * 70)

    all_results = {}
    for region_id, region_cfg in regions.items():
        results = generate_for_region(
            region_id,
            region_cfg,
            models,
            n_realizations,
            n_years,
            seed,
            output_format,
        )
        all_results[region_id] = results

    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    for region_id, results in all_results.items():
        succeeded = [k for k, v in results.items() if v is not None]
        failed = [k for k, v in results.items() if v is None]
        print(f"\n  {region_id}: {len(succeeded)}/{len(results)} succeeded")
        if failed:
            print(f"    Failed: {failed}")
    print(f"\nOutputs saved under: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

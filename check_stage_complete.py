"""
Stage checkpoint gate.

Walks outputs/ to determine which expected (region, model) artifacts
exist for a given pipeline stage, writes a status CSV under
outputs/status/, and exits 0 only if all tasks have completed.

Used as a barrier between HPC stages in run_pipeline.sh, and called
from assemble_results.py before running cross-region assembly.

Usage:
  python check_stage_complete.py --stage generate
  python check_stage_complete.py --stage analyze
  python check_stage_complete.py --stage convergence
  python check_stage_complete.py --stage split_sample
  python check_stage_complete.py --stage all       # all stages in one run
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR
from methods.io import (
    convergence_exists,
    split_sample_exists,
)
from methods.metrics import metrics_exist
from methods.tasks import (
    get_analysis_tasks,
    get_convergence_tasks,
    get_generation_tasks,
    get_split_sample_tasks,
)

logger = logging.getLogger(__name__)


def _ensemble_artifact(region_output: Path, model_key: str) -> Path:
    """Preferred ensemble artifact path (hdf5 first, then pickle)."""
    h5 = region_output / f"ensemble_{model_key}.h5"
    pkl = region_output / f"ensemble_{model_key}.pkl"
    if h5.exists():
        return h5
    return pkl


def _file_stats(path: Path) -> tuple:
    """Return (size_mb, mtime_iso) for a path, or ('', '') if missing."""
    if not path.exists():
        return ("", "")
    size_mb = round(path.stat().st_size / (1024 * 1024), 3)
    mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    return (size_mb, mtime)


def check_generate_stage() -> pd.DataFrame:
    """Return status DataFrame for the generate stage."""
    rows = []
    for region_id, model_key in get_generation_tasks():
        region_output = OUTPUT_DIR / region_id
        artifact = _ensemble_artifact(region_output, model_key)
        exists = artifact.exists()
        size_mb, mtime = _file_stats(artifact)
        rows.append(
            {
                "stage": "generate",
                "region": region_id,
                "model": model_key,
                "status": "ok" if exists else "missing",
                "artifact_path": str(artifact) if exists else "",
                "size_mb": size_mb,
                "mtime": mtime,
            }
        )
    return pd.DataFrame(rows)


def check_analyze_stage() -> pd.DataFrame:
    """Return status DataFrame for the analyze stage."""
    rows = []
    for region_id, model_key in get_analysis_tasks():
        region_output = OUTPUT_DIR / region_id
        exists = metrics_exist(region_output, model_key)
        # Canonical representative artifact: the tidy metrics CSV.
        artifact = region_output / f"metrics_{model_key}.csv"
        if not artifact.exists():
            # Fall back to any metrics_*.csv for this model.
            candidates = list(region_output.glob(f"*{model_key}*metrics*.csv"))
            artifact = candidates[0] if candidates else artifact
        size_mb, mtime = _file_stats(artifact)
        rows.append(
            {
                "stage": "analyze",
                "region": region_id,
                "model": model_key,
                "status": "ok" if exists else "missing",
                "artifact_path": str(artifact) if artifact.exists() else "",
                "size_mb": size_mb,
                "mtime": mtime,
            }
        )
    return pd.DataFrame(rows)


def check_convergence_stage() -> pd.DataFrame:
    """Return status DataFrame for the convergence stage."""
    rows = []
    for region_id, model_key in get_convergence_tasks():
        region_output = OUTPUT_DIR / region_id
        exists = convergence_exists(region_output, model_key)
        artifact = region_output / f"convergence_{model_key}.csv"
        size_mb, mtime = _file_stats(artifact)
        rows.append(
            {
                "stage": "convergence",
                "region": region_id,
                "model": model_key,
                "status": "ok" if exists else "missing",
                "artifact_path": str(artifact) if exists else "",
                "size_mb": size_mb,
                "mtime": mtime,
            }
        )
    return pd.DataFrame(rows)


def check_split_sample_stage() -> pd.DataFrame:
    """Return status DataFrame for the split_sample stage."""
    rows = []
    for region_id, model_key in get_split_sample_tasks():
        exists = split_sample_exists(OUTPUT_DIR, region_id, model_key)
        artifact = OUTPUT_DIR / "split_sample" / f"{region_id}__{model_key}.csv"
        size_mb, mtime = _file_stats(artifact)
        rows.append(
            {
                "stage": "split_sample",
                "region": region_id,
                "model": model_key,
                "status": "ok" if exists else "missing",
                "artifact_path": str(artifact) if exists else "",
                "size_mb": size_mb,
                "mtime": mtime,
            }
        )
    return pd.DataFrame(rows)


STAGE_CHECKERS = {
    "generate": check_generate_stage,
    "analyze": check_analyze_stage,
    "convergence": check_convergence_stage,
    "split_sample": check_split_sample_stage,
}


def check_stage(stage: str, write_csv: bool = True) -> bool:
    """Run the checker for one stage. Return True iff all tasks complete."""
    if stage not in STAGE_CHECKERS:
        raise ValueError(f"Unknown stage: {stage}. Valid: {list(STAGE_CHECKERS)}")

    df = STAGE_CHECKERS[stage]()

    status_dir = OUTPUT_DIR / "status"
    if write_csv:
        status_dir.mkdir(parents=True, exist_ok=True)
        out_path = status_dir / f"{stage}_status.csv"
        df.to_csv(out_path, index=False)
        logger.info("Wrote %s", out_path)

    n_total = len(df)
    n_ok = int((df["status"] == "ok").sum())
    n_missing = n_total - n_ok

    print(f"\n=== Stage: {stage} ===")
    print(f"  complete: {n_ok} / {n_total}")
    if n_missing > 0:
        print(f"  MISSING:  {n_missing}")
        missing_df = df[df["status"] == "missing"].sort_values(["region", "model"])
        for _, r in missing_df.iterrows():
            print(f"    {r['region']:25s}  {r['model']}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Check pipeline stage completeness")
    parser.add_argument(
        "--stage",
        choices=list(STAGE_CHECKERS) + ["all"],
        required=True,
        help="Stage to check, or 'all' for every stage.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write status CSV (console output only)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.stage == "all":
        all_ok = True
        for stage in STAGE_CHECKERS:
            ok = check_stage(stage, write_csv=not args.no_write)
            all_ok = all_ok and ok
        sys.exit(0 if all_ok else 1)

    ok = check_stage(args.stage, write_csv=not args.no_write)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

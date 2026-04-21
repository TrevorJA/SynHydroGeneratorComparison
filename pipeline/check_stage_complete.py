"""
Stage checkpoint gate.

Walks stage-specific output directories to determine which expected
(region, model) artifacts exist for a given pipeline stage, writes a
status CSV under outputs/status/, and exits 0 only if all tasks have
completed.

Used as a barrier between HPC stages in pipeline/slurm/run_pipeline.sh,
and called from scripts/assemble_results.py before cross-region assembly.

Usage:
  python pipeline/check_stage_complete.py --stage generate
  python pipeline/check_stage_complete.py --stage analyze
  python pipeline/check_stage_complete.py --stage convergence
  python pipeline/check_stage_complete.py --stage split_sample
  python pipeline/check_stage_complete.py --stage all
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Make project root importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    OUTPUT_DIR,
    GENERATION_DIR,
    ANALYSIS_DIR,
    CONVERGENCE_DIR,
    SPLIT_SAMPLE_DIR,
)
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


def _hdf5_readable(path: Path) -> bool:
    """Return True if path opens cleanly as an HDF5 file.

    Catches the "file exists but truncated/corrupted mid-write" failure
    mode that `.exists()` alone cannot detect. If the file is not an
    HDF5 artifact (e.g. .pkl), we fall back to a size check -- any file
    larger than 1 KB is presumed non-empty; zero-byte pickles fail.
    """
    if not path.exists():
        return False
    if path.suffix == ".h5":
        try:
            import h5py  # local import: only needed during gating
            with h5py.File(path, "r") as f:
                # Force metadata read so truncated files raise.
                _ = list(f.keys())
            return True
        except Exception as exc:
            logger.warning("HDF5 unreadable at %s: %s", path, exc)
            return False
    # Non-HDF5 artifacts (pickle): treat nonzero-size as readable.
    return path.stat().st_size > 0


def check_generate_stage() -> pd.DataFrame:
    """Return status DataFrame for the generate stage.

    Beyond file existence, HDF5 artifacts are validated by opening them
    and listing root keys. Partial writes or corrupted files are
    flagged as "corrupt" so they do not sneak past the gate.
    """
    rows = []
    for region_id, model_key in get_generation_tasks():
        region_output = GENERATION_DIR / region_id
        artifact = _ensemble_artifact(region_output, model_key)
        if not artifact.exists():
            status = "missing"
        elif not _hdf5_readable(artifact):
            status = "corrupt"
        else:
            status = "ok"
        size_mb, mtime = _file_stats(artifact)
        rows.append(
            {
                "stage": "generate",
                "region": region_id,
                "model": model_key,
                "status": status,
                "artifact_path": str(artifact) if artifact.exists() else "",
                "size_mb": size_mb,
                "mtime": mtime,
            }
        )
    return pd.DataFrame(rows)


def check_analyze_stage() -> pd.DataFrame:
    """Return status DataFrame for the analyze stage."""
    rows = []
    for region_id, model_key in get_analysis_tasks():
        region_output = ANALYSIS_DIR / region_id
        exists = metrics_exist(region_output, model_key)
        artifact = region_output / f"metrics_{model_key}.csv"
        if not artifact.exists():
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
        region_output = CONVERGENCE_DIR / region_id
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
        exists = split_sample_exists(SPLIT_SAMPLE_DIR, region_id, model_key)
        artifact = SPLIT_SAMPLE_DIR / f"{region_id}__{model_key}.csv"
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
    n_bad = n_total - n_ok

    print(f"\n=== Stage: {stage} ===")
    print(f"  complete: {n_ok} / {n_total}")
    if n_bad > 0:
        for bad_status in ("missing", "corrupt"):
            bad_df = df[df["status"] == bad_status].sort_values(["region", "model"])
            if not bad_df.empty:
                print(f"  {bad_status.upper()}: {len(bad_df)}")
                for _, r in bad_df.iterrows():
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

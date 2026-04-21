"""
Memory profiling wrapper for generate_single.py.

Runs the three heaviest generators (vine_copula, arfima, hmm) against
the largest-sample region (new_england by default) at publication
N_REALIZATIONS, recording peak RSS. Used to validate the 8 GB SLURM
allocation in run_hpc_array.sh.

Writes outputs/profiling/memory_profile.csv with columns:
  (model, region, n_realizations, peak_rss_mb, elapsed_sec, status)

Usage:
  python profile_memory.py                   # default 3 models x 1 region
  python profile_memory.py --models vine_copula,arfima
  python profile_memory.py --regions new_england,central_rockies

Requires the tracemalloc-based measurement (standard library); does not
require the external memory_profiler package.
"""

import argparse
import gc
import logging
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

# Make project root importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import OUTPUT_DIR, N_REALIZATIONS, N_YEARS, SEED

logger = logging.getLogger(__name__)

DEFAULT_MODELS = ["vine_copula", "arfima", "hmm"]
DEFAULT_REGIONS = ["new_england"]


def _subprocess_peak_rss_mb(cmd: list[str]) -> tuple[int, float, float]:
    """Run cmd as a subprocess and return (exit_code, elapsed_sec, peak_rss_mb).

    Uses Python's tracemalloc for cross-platform peak tracking of the
    parent process. For subprocess RSS we fall back to psutil if
    available; otherwise we return NaN for peak_rss_mb and rely on
    SLURM's seff/sacct to provide the authoritative number.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None

    start = time.time()
    if psutil is None:
        proc = subprocess.Popen(cmd)
        peak = float("nan")
        exit_code = proc.wait()
    else:
        proc = subprocess.Popen(cmd)
        try:
            p = psutil.Process(proc.pid)
            peak = 0.0
            while proc.poll() is None:
                try:
                    rss = p.memory_info().rss / (1024 * 1024)
                    # Include child processes
                    for child in p.children(recursive=True):
                        try:
                            rss += child.memory_info().rss / (1024 * 1024)
                        except psutil.NoSuchProcess:
                            pass
                    if rss > peak:
                        peak = rss
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.2)
            exit_code = proc.wait()
        except Exception:
            proc.wait()
            exit_code = proc.returncode
            peak = float("nan")

    elapsed = time.time() - start
    return exit_code, elapsed, peak


def profile_one(
    model_key: str, region_id: str, n_realizations: int, n_years: int, seed: int
) -> dict:
    """Invoke generate_single.py for a single (region, model) and record peak RSS."""
    logger.info(
        "Profiling %s / %s at N=%d ...",
        region_id,
        model_key,
        n_realizations,
    )

    cmd = [
        sys.executable,
        "generate_single.py",
        "--region",
        region_id,
        "--model",
        model_key,
        "--n-realizations",
        str(n_realizations),
        "--n-years",
        str(n_years),
        "--seed",
        str(seed),
        "--force",
    ]

    exit_code, elapsed, peak_rss_mb = _subprocess_peak_rss_mb(cmd)

    return {
        "model": model_key,
        "region": region_id,
        "n_realizations": n_realizations,
        "peak_rss_mb": (
            round(peak_rss_mb, 1)
            if peak_rss_mb == peak_rss_mb  # NaN check
            else float("nan")
        ),
        "elapsed_sec": round(elapsed, 1),
        "status": "ok" if exit_code == 0 else f"exit={exit_code}",
    }


def main():
    parser = argparse.ArgumentParser(description="Memory profiling for generate_single")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model keys (default: {','.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default=",".join(DEFAULT_REGIONS),
        help=f"Comma-separated region IDs (default: {','.join(DEFAULT_REGIONS)})",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=N_REALIZATIONS,
        help=f"Realizations per profile (default: {N_REALIZATIONS})",
    )
    parser.add_argument(
        "--n-years",
        type=int,
        default=N_YEARS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]

    rows = []
    for region_id in regions:
        for model_key in models:
            row = profile_one(
                model_key,
                region_id,
                args.n_realizations,
                args.n_years,
                args.seed,
            )
            rows.append(row)
            gc.collect()

    out_dir = OUTPUT_DIR / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "memory_profile.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(df.to_string(index=False))

    # Flag any result exceeding 7000 MB (< 8 GB SLURM allocation minus overhead).
    if not df.empty:
        overage = df[df["peak_rss_mb"].notna() & (df["peak_rss_mb"] > 7000)]
        if not overage.empty:
            print(
                "\nWARNING: the following runs approached or exceeded the 8 GB allocation:"
            )
            print(overage.to_string(index=False))


if __name__ == "__main__":
    main()

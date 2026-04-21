#!/bin/bash
#
# SLURM array job for ensemble generation (Stage 1).
#
# Each array task generates ensembles for one (region, model) pair.
# Run `python pipeline/generate_single.py --list-tasks` to see the full mapping.
#
# Typical submission (from project root):
#   sbatch pipeline/slurm/run_generate.sh
#
# To run a subset (e.g. first 11 tasks = first region):
#   sbatch --array=0-10 pipeline/slurm/run_generate.sh
#
# Prerequisites:
#   1. CAMELS data must be cached first:
#        python scripts/00_retrieve_data.py
#   2. Python environment with synhydro installed must be available.
#      Update PYTHON_CMD below to point to your environment.

#SBATCH --job-name=synhydro
#SBATCH --array=0-77
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# ============================================================================
# Configuration -- edit these for your cluster
# ============================================================================
#
# N_REALIZATIONS, N_YEARS, SEED, and OUTPUT_FORMAT are NOT redefined here --
# generate_single.py reads them from config.py as argparse defaults.

PYTHON_CMD="python"

# ============================================================================
# Execution
# ============================================================================

# Change to project root regardless of where sbatch was submitted from.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs

# Echo the effective config so it is visible in the SLURM log.
CONFIG_SUMMARY=$(${PYTHON_CMD} -c \
    "from config import N_REALIZATIONS, N_YEARS, SEED, OUTPUT_FORMAT; \
     print(f'N_REAL={N_REALIZATIONS}  N_YEARS={N_YEARS}  SEED={SEED}  FORMAT={OUTPUT_FORMAT}')")

echo "============================================"
echo "SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Array Task ID:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:            $(hostname)"
echo "Start time:      $(date)"
echo "Working dir:     $(pwd)"
echo "Config (config.py): ${CONFIG_SUMMARY}"
echo "============================================"

${PYTHON_CMD} pipeline/generate_single.py \
    --task-id "${SLURM_ARRAY_TASK_ID}"

EXIT_CODE=$?

echo "============================================"
echo "End time:        $(date)"
echo "Exit code:       ${EXIT_CODE}"
echo "============================================"

exit ${EXIT_CODE}

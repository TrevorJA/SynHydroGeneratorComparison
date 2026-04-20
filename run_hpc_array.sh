#!/bin/bash
#
# SLURM array job template for ensemble generation.
#
# Each array task generates ensembles for one (region, model) pair.
# Run `python generate_single.py --list-tasks` to see the full mapping.
#
# Typical submission:
#   sbatch run_hpc_array.sh
#
# To run a subset (e.g. first 11 tasks = first region):
#   sbatch --array=0-10 run_hpc_array.sh
#
# Prerequisites:
#   1. CAMELS data must be cached first:
#        python 00_retrieve_data.py
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

PYTHON_CMD="python"
N_REALIZATIONS=1000
N_YEARS=50
SEED=42
FORMAT="hdf5"

# ============================================================================
# Execution
# ============================================================================

cd "$(dirname "$0")" || exit 1
mkdir -p logs

echo "============================================"
echo "SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Array Task ID:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:            $(hostname)"
echo "Start time:      $(date)"
echo "Working dir:     $(pwd)"
echo "============================================"

${PYTHON_CMD} generate_single.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --n-realizations "${N_REALIZATIONS}" \
    --n-years "${N_YEARS}" \
    --seed "${SEED}" \
    --format "${FORMAT}"

EXIT_CODE=$?

echo "============================================"
echo "End time:        $(date)"
echo "Exit code:       ${EXIT_CODE}"
echo "============================================"

exit ${EXIT_CODE}

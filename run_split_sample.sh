#!/bin/bash
#
# SLURM array job for split-sample validation (Stage 4).
#
# Each array task runs the 4-scenario split-sample protocol for one
# (region, model) pair and writes one CSV to outputs/split_sample/.
#
# Run `python split_sample_single.py --list-tasks` to see the full mapping.
#
# Typical submission:
#   sbatch run_split_sample.sh
#
# To run a subset (e.g. first region only):
#   sbatch --array=0-12 run_split_sample.sh
#
# Prerequisites:
#   1. CAMELS data must be cached:
#        python 00_retrieve_data.py
#   2. Python environment with synhydro installed must be available.
#      Update PYTHON_CMD below to point to your environment.

#SBATCH --job-name=synhydro_split
#SBATCH --array=0-77
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_split_%A_%a.out
#SBATCH --error=logs/slurm_split_%A_%a.err

# ============================================================================
# Configuration -- edit these for your cluster
# ============================================================================

PYTHON_CMD="python"
N_REALIZATIONS=500
N_YEARS=50
SEED=42

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

${PYTHON_CMD} split_sample_single.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --n-realizations "${N_REALIZATIONS}" \
    --n-years "${N_YEARS}" \
    --seed "${SEED}"

EXIT_CODE=$?

echo "============================================"
echo "End time:        $(date)"
echo "Exit code:       ${EXIT_CODE}"
echo "============================================"

exit ${EXIT_CODE}

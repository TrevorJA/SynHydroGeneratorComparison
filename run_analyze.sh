#!/bin/bash
#
# SLURM array job for metric computation (Stage 2).
#
# Each array task computes validation metrics for one (region, model) pair
# and saves three CSVs under outputs/{region}/.
#
# Run `python analyze_single.py --list-tasks` to see the full mapping.
#
# Typical submission:
#   sbatch run_analyze.sh
#
# To run a subset (e.g. first region only):
#   sbatch --array=0-12 run_analyze.sh
#
# Prerequisites:
#   1. Stage 1 (generate_single.py) must have completed for these tasks.
#   2. Python environment with synhydro installed must be available.
#      Update PYTHON_CMD below to point to your environment.

#SBATCH --job-name=synhydro_analyze
#SBATCH --array=0-77
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_analyze_%A_%a.out
#SBATCH --error=logs/slurm_analyze_%A_%a.err

# ============================================================================
# Configuration -- edit these for your cluster
# ============================================================================

PYTHON_CMD="python"

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

${PYTHON_CMD} analyze_single.py --task-id "${SLURM_ARRAY_TASK_ID}"

EXIT_CODE=$?

echo "============================================"
echo "End time:        $(date)"
echo "Exit code:       ${EXIT_CODE}"
echo "============================================"

exit ${EXIT_CODE}

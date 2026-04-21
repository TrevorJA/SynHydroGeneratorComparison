#!/bin/bash
#
# SLURM array job for convergence sweeps (Stage 2b).
#
# Each array task fits one generator for one region and sweeps
# N_REALIZATIONS_SWEEP ensemble sizes, saving convergence CSVs.
# Only non-annual models are included.
#
# Run `python convergence_single.py --list-tasks` to see the full mapping.
#
# Typical submission:
#   sbatch run_convergence.sh
#
# Prerequisites:
#   1. CAMELS data must be cached first: python 00_retrieve_data.py
#   2. Python environment with synhydro installed must be available.
#      Update PYTHON_CMD below to point to your environment.
#
# NOTE: convergence sweeps fit and generate repeatedly at different sizes,
# so they are more compute-intensive than analyze_single. The slow path
# at publication scale (N_MAX=500, 10 replicated subsets per sweep level,
# 7 sweep levels) is vine copula and ARFIMA; these can approach 10 hrs
# per task. Adjust --time and --mem downward for development runs.

#SBATCH --job-name=synhydro_conv
#SBATCH --array=0-59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_conv_%A_%a.out
#SBATCH --error=logs/slurm_conv_%A_%a.err

# ============================================================================
# Configuration -- edit these for your cluster
# ============================================================================
#
# N_YEARS and SEED come from config.py defaults (see run_hpc_array.sh note).

PYTHON_CMD="python"

# ============================================================================
# Execution
# ============================================================================

cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs

CONFIG_SUMMARY=$(${PYTHON_CMD} -c \
    "from config import N_REALIZATIONS, N_YEARS, SEED; \
     print(f'N_REAL={N_REALIZATIONS}  N_YEARS={N_YEARS}  SEED={SEED}')")

echo "============================================"
echo "SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Array Task ID:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:            $(hostname)"
echo "Start time:      $(date)"
echo "Working dir:     $(pwd)"
echo "Config (config.py): ${CONFIG_SUMMARY}"
echo "============================================"

${PYTHON_CMD} pipeline/convergence_single.py \
    --task-id "${SLURM_ARRAY_TASK_ID}"

EXIT_CODE=$?

echo "============================================"
echo "End time:        $(date)"
echo "Exit code:       ${EXIT_CODE}"
echo "============================================"

exit ${EXIT_CODE}

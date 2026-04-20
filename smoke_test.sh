#!/bin/bash
#
# End-to-end dev-config smoke test.
#
# Runs the full pipeline on a single region (new_england) at
# N_REALIZATIONS=10 / N_YEARS=10 to validate that every stage
# works before committing HPC time. These values are intentionally
# tiny -- the goal is to surface import errors and workflow bugs,
# not to produce meaningful scientific results. Override with the
# env vars N_REAL, N_YEARS, REGION, SEED if needed.
#
# Usage:
#   ./smoke_test.sh                    # new_england
#   REGION=central_plains ./smoke_test.sh
#
# Exit status:
#   0 -- all stages passed
#   nonzero -- a stage failed or produced incomplete artifacts

set -e

cd "$(dirname "$0")" || exit 1

# ============================================================================
# Configuration
# ============================================================================

REGION="${REGION:-new_england}"
N_REAL="${N_REAL:-10}"
N_YEARS="${N_YEARS:-10}"
SEED="${SEED:-42}"
PYTHON_CMD="${PYTHON_CMD:-python}"

echo "============================================"
echo "SynHydro model_comparison smoke test"
echo "  Region:          ${REGION}"
echo "  N_REALIZATIONS:  ${N_REAL}"
echo "  N_YEARS:         ${N_YEARS}"
echo "  Seed:            ${SEED}"
echo "============================================"

# ============================================================================
# Step 0 -- verify CAMELS data cache
# ============================================================================

if [ ! -d "data/${REGION}" ]; then
    echo ""
    echo "ERROR: data/${REGION} missing. Run:"
    echo "  ${PYTHON_CMD} 00_retrieve_data.py"
    exit 1
fi
echo ""
echo "[0/6] Data cache present: data/${REGION}"

# ============================================================================
# Step 1 -- generation (serial, all 13 models for this region)
# ============================================================================

echo ""
echo "[1/6] Generation (serial, 13 models)"
${PYTHON_CMD} 01_generate_ensembles.py \
    --region "${REGION}" \
    --n-realizations "${N_REAL}" \
    --n-years "${N_YEARS}" \
    --seed "${SEED}"

# ============================================================================
# Step 2 -- analysis (serial loop over models via analyze_single.py)
#
# Uses the same per-(region, model) entry point as the HPC array so
# the output layout matches exactly (metrics_*.csv per model, required
# by assemble_results and by the analyze-stage gate below).
# ============================================================================

echo ""
echo "[2/6] Analysis (serial loop over models)"
ANA_MODELS=$(${PYTHON_CMD} -c "
import sys
sys.path.insert(0, '.')
from methods.tasks import get_analysis_tasks
models = sorted({m for r, m in get_analysis_tasks() if r == '${REGION}'})
print(' '.join(models))
")
echo "  Models: ${ANA_MODELS}"
for model in ${ANA_MODELS}; do
    echo "  -- ${model}"
    ${PYTHON_CMD} analyze_single.py --region "${REGION}" --model "${model}"
done

# ============================================================================
# Step 3 -- assembly
# ============================================================================

echo ""
echo "[3/6] Assembly"
${PYTHON_CMD} assemble_results.py --region "${REGION}"

# ============================================================================
# Step 4 -- split-sample (serial, single region)
# ============================================================================

echo ""
echo "[4/6] Split-sample (serial, single region)"
${PYTHON_CMD} 06_split_sample.py \
    --region "${REGION}" \
    --n-realizations "${N_REAL}"

# ============================================================================
# Step 5 -- convergence (serial loop over non-annual models for this region)
#
# Smoke test caps the sweep at N_REAL so we never sample more realizations
# than were generated. Uses convergence_single.py to match the HPC flow.
# ============================================================================

echo ""
echo "[5/6] Convergence (serial, single region, non-annual, sweep cap=${N_REAL})"
CONV_MODELS=$(${PYTHON_CMD} -c "
import sys
sys.path.insert(0, '.')
from methods.tasks import get_convergence_tasks
models = sorted({m for r, m in get_convergence_tasks() if r == '${REGION}'})
print(' '.join(models))
")
echo "  Models: ${CONV_MODELS}"
for model in ${CONV_MODELS}; do
    echo "  -- ${model}"
    ${PYTHON_CMD} convergence_single.py \
        --region "${REGION}" \
        --model "${model}" \
        --n-years "${N_YEARS}" \
        --n-max "${N_REAL}" \
        --seed "${SEED}"
done

# ============================================================================
# Step 6 -- figures (SI + main manuscript)
# ============================================================================

echo ""
echo "[6a] SI figures (per-region + cross-region)"
${PYTHON_CMD} 04b_si_figures.py --region "${REGION}"

echo ""
echo "[6b] SI convergence figures (per-region + cross-region overlay)"
${PYTHON_CMD} 04c_si_convergence_figures.py --region "${REGION}"

echo ""
echo "[6c] Main manuscript figures (registry + placeholders)"
${PYTHON_CMD} 04a_main_figures.py

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================"
echo "Smoke test complete."
echo ""
echo "Artifact summary:"
echo "  Ensembles:   outputs/${REGION}/ensemble_*.h5"
echo "  Metrics:     outputs/${REGION}/metrics_*.csv"
echo "  Assembly:    outputs/cross_region/*.csv"
echo "  Split:       outputs/split_sample/${REGION}__*.csv"
echo "  Convergence: outputs/${REGION}/convergence_*.csv"
echo "  Concordance: outputs/cross_region/tier_concordance.csv"
echo "  SI figs:     figures/si/${REGION}/*.png, figures/si/cross_region/*.png"
echo "  Main figs:   figures/main/fig*.png"
echo "============================================"

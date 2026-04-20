#!/bin/bash
#
# Full-pipeline orchestrator with SLURM dependency chaining.
#
# Submits all stages as SLURM array jobs and chains them with
# --dependency=afterok: so a failing upstream stage blocks downstream
# work. After each array completes, a checkpoint job runs
# `check_stage_complete.py` to produce an audit CSV.
#
# DAG:
#   generate (78 tasks)
#     -> analyze  (78 tasks)
#       -> assemble (serial)
#     -> convergence  (60 tasks, non-annual models only)
#     -> split_sample (78 tasks)
#       -> figures (serial)
#
# Assemble only depends on analyze. Figures depend on assemble +
# convergence + split_sample so that all cross-region CSVs exist
# when figures run.
#
# Usage:
#   ./run_pipeline.sh
#   SKIP=generate ./run_pipeline.sh         # assume generate already done
#   SKIP=generate,analyze ./run_pipeline.sh # start from assemble
#
# Prerequisites:
#   1. CAMELS data cached: python 00_retrieve_data.py
#   2. SLURM available in PATH.

set -e

cd "$(dirname "$0")" || exit 1
mkdir -p logs outputs/status

PYTHON_CMD="${PYTHON_CMD:-python}"
SKIP="${SKIP:-}"

should_skip() {
    [[ ",${SKIP}," == *",$1,"* ]]
}

submit_array() {
    local script="$1"
    local deps="$2"   # comma-joined job IDs (empty string if none)
    local args
    if [ -n "${deps}" ]; then
        args="--parsable --dependency=afterok:${deps}"
    else
        args="--parsable"
    fi
    # shellcheck disable=SC2086
    sbatch ${args} "${script}"
}

# Submit a serial checkpoint job that runs check_stage_complete.
# Exits nonzero (fails the dependency chain) if any task is missing.
submit_check() {
    local stage="$1"
    local deps="$2"
    sbatch --parsable \
        --dependency=afterok:"${deps}" \
        --job-name="check_${stage}" \
        --time=00:10:00 --mem=4G \
        --output="logs/check_${stage}_%j.out" \
        --wrap="${PYTHON_CMD} check_stage_complete.py --stage ${stage}"
}

submit_serial() {
    local name="$1"
    local cmd="$2"
    local deps="$3"
    sbatch --parsable \
        --dependency=afterok:"${deps}" \
        --job-name="${name}" \
        --time=01:00:00 --mem=8G \
        --output="logs/${name}_%j.out" \
        --wrap="${cmd}"
}

GEN_JOB=""
ANA_JOB=""
ASM_JOB=""
CONV_JOB=""
SPLIT_JOB=""

echo "============================================"
echo "SynHydro model_comparison pipeline"
echo "============================================"

if should_skip generate; then
    echo "[skip] generate stage"
else
    GEN_JOB=$(submit_array run_hpc_array.sh "")
    echo "submitted generate:    ${GEN_JOB}"
    GEN_CHECK=$(submit_check generate "${GEN_JOB}")
    echo "submitted gen check:   ${GEN_CHECK}"
fi

if should_skip analyze; then
    echo "[skip] analyze stage"
else
    ANA_DEPS="${GEN_CHECK:-}"
    ANA_JOB=$(submit_array run_analyze.sh "${ANA_DEPS}")
    echo "submitted analyze:     ${ANA_JOB}"
    ANA_CHECK=$(submit_check analyze "${ANA_JOB}")
    echo "submitted ana check:   ${ANA_CHECK}"
fi

if should_skip assemble; then
    echo "[skip] assemble stage"
else
    ASM_DEPS="${ANA_CHECK:-}"
    ASM_JOB=$(submit_serial assemble \
        "${PYTHON_CMD} assemble_results.py" \
        "${ASM_DEPS}")
    echo "submitted assemble:    ${ASM_JOB}"
fi

if should_skip convergence; then
    echo "[skip] convergence stage"
else
    CONV_DEPS="${ANA_CHECK:-}"
    CONV_JOB=$(submit_array run_convergence.sh "${CONV_DEPS}")
    echo "submitted convergence: ${CONV_JOB}"
    CONV_CHECK=$(submit_check convergence "${CONV_JOB}")
    echo "submitted conv check:  ${CONV_CHECK}"
fi

if should_skip split_sample; then
    echo "[skip] split_sample stage"
else
    SPLIT_DEPS="${ANA_CHECK:-}"
    SPLIT_JOB=$(submit_array run_split_sample.sh "${SPLIT_DEPS}")
    echo "submitted split:       ${SPLIT_JOB}"
    SPLIT_CHECK=$(submit_check split_sample "${SPLIT_JOB}")
    echo "submitted split check: ${SPLIT_CHECK}"
fi

if should_skip figures; then
    echo "[skip] figures stage"
else
    # Figures depend on assemble + convergence + split_sample checkpoints.
    FIG_DEPS=""
    for d in "${ASM_JOB}" "${CONV_CHECK:-}" "${SPLIT_CHECK:-}"; do
        if [ -n "$d" ]; then
            if [ -n "${FIG_DEPS}" ]; then FIG_DEPS="${FIG_DEPS}:$d"; else FIG_DEPS="$d"; fi
        fi
    done
    # SI first (diagnostic + convergence), then main manuscript figures.
    SI_FIG_JOB=$(submit_serial si_figures \
        "${PYTHON_CMD} 04b_si_figures.py" \
        "${FIG_DEPS}")
    echo "submitted SI figures:  ${SI_FIG_JOB}"
    SI_CONV_FIG_JOB=$(submit_serial si_conv_figures \
        "${PYTHON_CMD} 04c_si_convergence_figures.py" \
        "${FIG_DEPS}")
    echo "submitted SI conv figs: ${SI_CONV_FIG_JOB}"
    MAIN_FIG_JOB=$(submit_serial main_figures \
        "${PYTHON_CMD} 04a_main_figures.py" \
        "${SI_FIG_JOB}:${SI_CONV_FIG_JOB}")
    echo "submitted main figs:   ${MAIN_FIG_JOB}"
fi

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Status CSVs written to outputs/status/*.csv as checkpoints run."

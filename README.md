# Model Comparison Experiment

Regime-stratified benchmark of 13 stochastic streamflow generators across
6 hydroclimatic regimes and 24 CAMELS basins. Target journal: Water
Resources Research. See [notes/publication_plan.md](notes/publication_plan.md)
for the full research plan.

## Canonical references

All design, figure specs, open decisions, and conventions live in
[notes/](notes/). This README is a pointer, not a duplicate.

- Research plan and thesis: [notes/publication_plan.md](notes/publication_plan.md)
- Figure specs (10 main + 13 supplementary): [notes/figure_design.md](notes/figure_design.md)
- Generator coverage justification: [notes/library_readiness.md](notes/library_readiness.md)
- Open decisions (append-only): [notes/decisions_log.md](notes/decisions_log.md)
- Manuscript deliverables tracker: [notes/deliverables_tracker.md](notes/deliverables_tracker.md)
- Notes conventions: [notes/STANDARDS.md](notes/STANDARDS.md)

Internal review documents: [notes/review_round2.md](notes/review_round2.md),
[notes/reviewer_critique.md](notes/reviewer_critique.md).

---

## Repository layout

```
pipeline/               HPC array entry points (called by SLURM)
  generate_single.py
  analyze_single.py
  convergence_single.py
  split_sample_single.py
  check_stage_complete.py
  slurm/                SLURM submission scripts
    run_generate.sh
    run_analyze.sh
    run_convergence.sh
    run_split_sample.sh
    run_pipeline.sh     full DAG orchestrator

scripts/                interactive / one-shot scripts
  00_retrieve_data.py
  01_generate_ensembles.py
  assemble_results.py
  split_sample.py
  evaluate_fig4_rule.py
  profile_memory.py
  figures/
    04a_main_figures.py
    04b_si_figures.py
    04c_si_convergence_figures.py

methods/                library modules (metrics, assembly, plotting, I/O)
config.py               all runtime parameters and stage directory constants
basins.py               CAMELS_REGIONS metadata
smoke_test.sh           end-to-end dev-config validation (single region)
```

---

## Pipeline stages

| #  | Stage           | Interactive entry point                | HPC entry point                         | Tasks | Output directory                       |
|----|-----------------|----------------------------------------|-----------------------------------------|-------|----------------------------------------|
| 0  | Retrieve        | `scripts/00_retrieve_data.py`          | N/A (runs once, locally)                | 1     | `data/{region}/`                       |
| 1  | Generate        | `scripts/01_generate_ensembles.py`     | `pipeline/slurm/run_generate.sh`        | 78    | `outputs/generation/{region}/`         |
| 2  | Analyze         | loop over `pipeline/analyze_single.py` | `pipeline/slurm/run_analyze.sh`         | 78    | `outputs/analysis/{region}/`           |
| 3  | Convergence     | loop over `pipeline/convergence_single.py` | `pipeline/slurm/run_convergence.sh` | 60    | `outputs/convergence/{region}/`        |
| 4  | Split-sample    | loop over `pipeline/split_sample_single.py` | `pipeline/slurm/run_split_sample.sh` | 78   | `outputs/split_sample/`                |
| A  | Assemble        | `scripts/assemble_results.py`          | (serial, after Stage 2)                 | 1     | `outputs/cross_region/`                |
| F1 | SI figures      | `scripts/figures/04b_si_figures.py`    | (serial)                                | 1     | `figures/si/{region}/`                 |
| F2 | SI convergence  | `scripts/figures/04c_si_convergence_figures.py` | (serial)                       | 1     | `figures/si/`                          |
| F3 | Main figures    | `scripts/figures/04a_main_figures.py`  | (serial)                                | 1     | `figures/main/`                        |

Stage 3 (convergence) excludes annual-native models; 60 = 10 non-annual models × 6 regions.

Figures: `figures/main/` holds the 10 manuscript main-text figures (registry:
`methods/plotting/manuscript.py::MANUSCRIPT_FIGURES`). `figures/si/` holds
per-region diagnostics, convergence detail, and QA/QC verification.

---

## Reproducing the experiment

### Environment setup on Hopper

```bash
# 1. Load Python module (Hopper)
module load python/3.11.5

# Required for the Python interpreter to find libpython at runtime:
export LD_LIBRARY_PATH=/opt/ohpc/pub/utils/python/3.11.5/lib:$LD_LIBRARY_PATH

# 2. Create and activate a project-local venv
cd /path/to/SynHydroGeneratorComparison
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the internal SynHydro library (editable)
pip install -e /path/to/SynHydro

# 5. (Optional) Verify pinned versions match the publication run
pip install -r requirements-lock.txt
```

Add the `LD_LIBRARY_PATH` export to your `~/.bashrc` or prepend it to
every SLURM `--wrap` command (see the SLURM scripts in `pipeline/slurm/`).

### Data retrieval (once per cluster)

```bash
# Runs on the login node — fast (~30 s per region)
python scripts/00_retrieve_data.py
```

Caches CAMELS data to `data/{region}/` and writes
`outputs/nonstationarity_screen.csv`.

### Smoke test (dev config, single region)

**Submit to SLURM** — the test takes ~5 minutes; do not run on the login node.

```bash
sbatch --job-name=smoke_test \
       --time=00:30:00 --mem=16G \
       --output=logs/smoke_test_%j.out \
       --wrap="export LD_LIBRARY_PATH=/opt/ohpc/pub/utils/python/3.11.5/lib:\$LD_LIBRARY_PATH && \
               PYTHON_CMD=venv/bin/python3.11 bash smoke_test.sh"
```

Or for a specific region:
```bash
sbatch ... --wrap="... REGION=central_plains PYTHON_CMD=venv/bin/python3.11 bash smoke_test.sh"
```

### Full pipeline (production, N=500)

```bash
# Submit all stages with SLURM afterok dependency chaining:
PYTHON_CMD=venv/bin/python3.11 bash pipeline/slurm/run_pipeline.sh

# Monitor:
squeue -u $USER
# Checkpoint status (run after a stage array completes):
venv/bin/python3.11 pipeline/check_stage_complete.py --stage generate
```

To skip already-completed stages:
```bash
SKIP=generate,analyze PYTHON_CMD=venv/bin/python3.11 bash pipeline/slurm/run_pipeline.sh
```

### Stage-by-stage (manual)

```bash
sbatch pipeline/slurm/run_generate.sh                             # Stage 1: 78 tasks
sbatch --dependency=afterok:$GEN_JID pipeline/slurm/run_analyze.sh  # Stage 2: 78 tasks
venv/bin/python3.11 pipeline/check_stage_complete.py --stage analyze
venv/bin/python3.11 scripts/assemble_results.py
sbatch pipeline/slurm/run_convergence.sh                          # Stage 3: 60 tasks
sbatch pipeline/slurm/run_split_sample.sh                         # Stage 4: 78 tasks
# After arrays complete:
venv/bin/python3.11 scripts/figures/04b_si_figures.py
venv/bin/python3.11 scripts/figures/04c_si_convergence_figures.py
venv/bin/python3.11 scripts/figures/04a_main_figures.py
```

---

## Configuration

All runtime parameters live in [config.py](config.py). Key values:

- `N_REALIZATIONS` — 500 for publication, 10–20 for development.
- `N_YEARS` — 50 years per realization.
- `SEED` — base random seed; each (region, model) task derives its own
  seed via `methods/tasks.py::derive_task_seed`.
- `GENERATION_DIR`, `ANALYSIS_DIR`, `CONVERGENCE_DIR`, `SPLIT_SAMPLE_DIR`,
  `CROSS_REGION_DIR` — stage-specific output directories. Always import
  these constants; never construct paths from `OUTPUT_DIR` directly.
- `MODELS` — enable/disable, init/fit/gen kwargs per generator.
- `ACTIVE_REGIONS` / `ACTIVE_MODELS` — subset control.

Basin metadata and selection rationale: [basins.py](basins.py).

---

## Workflow infrastructure

- `pipeline/check_stage_complete.py` — stage checkpoint gate. Walks
  stage output directories, validates HDF5 files are openable (not just
  present), writes `outputs/status/{stage}_status.csv`, exits nonzero
  if any (region, model) artifact is missing or corrupt.
- `pipeline/slurm/run_pipeline.sh` — full DAG orchestrator with SLURM
  `--dependency=afterok:` chaining.
- `smoke_test.sh` — end-to-end dev-config test on one region (N=10,
  10 years). Submit via SLURM; see above.
- `scripts/profile_memory.py` — validates the 8 GB SLURM allocation
  for the heaviest models at publication scale. Submit via SLURM.

---

## Status

Current phase: Pre-HPC pipeline hardening (Phase 0).
See [notes/deliverables_tracker.md](notes/deliverables_tracker.md) for
per-deliverable status and [notes/2026-04-21_handoff.md](notes/2026-04-21_handoff.md)
for current blockers.

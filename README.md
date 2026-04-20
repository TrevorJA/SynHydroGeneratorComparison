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

Internal review documents (for context): [notes/review_round2.md](notes/review_round2.md),
[notes/reviewer_critique.md](notes/reviewer_critique.md).

## Pipeline stages

Four HPC-parallelizable stages plus assembly and figures. All HPC array
sizes are derived from [methods/tasks.py](methods/tasks.py).

| #  | Stage           | Entry (serial, single region)     | Entry (HPC)                 | Tasks | Output location                |
|----|-----------------|-----------------------------------|-----------------------------|-------|--------------------------------|
| 0  | Retrieve        | `00_retrieve_data.py`             | N/A (runs once, locally)    | 1     | `data/{region}/*.csv`          |
| 1  | Generate        | `01_generate_ensembles.py`        | `run_hpc_array.sh`          | 78    | `outputs/{region}/ensemble_*.h5` |
| 2  | Analyze         | `analyze_single.py` (loop)        | `run_analyze.sh`            | 78    | `outputs/{region}/metrics_*.csv` |
| 3  | Convergence     | `convergence_single.py` (loop)    | `run_convergence.sh`        | 60    | `outputs/{region}/convergence_*.csv` |
| 4  | Split-sample    | `split_sample_single.py` (loop)   | `run_split_sample.sh`       | 78    | `outputs/split_sample/*.csv`   |
| A  | Assemble        | `assemble_results.py`             | (serial, after Stage 2)     | 1     | `outputs/cross_region/*.csv`   |
| F1 | SI figures      | `04b_si_figures.py`               | (serial)                    | 1     | `figures/si/{region}/`, `figures/si/cross_region/` |
| F2 | SI convergence  | `04c_si_convergence_figures.py`   | (serial, after Stage 3)     | 1     | `figures/si/{region}/22-28_*.png` |
| F3 | Main figures    | `04a_main_figures.py`             | (serial)                    | 1     | `figures/main/`                |

Figures are split by audience. `figures/main/` holds the 10 manuscript
main-text figures (registry: `methods/plotting/manuscript.py::MANUSCRIPT_FIGURES`);
`figures/si/` is the default destination for everything else, including
per-region diagnostic suites, cross-region summaries, convergence detail,
and QA/QC basin verification figures. Any new exploratory figure goes
to SI unless explicitly registered in `MANUSCRIPT_FIGURES`.

Stage 3 (convergence) excludes annual-native models since they have no
monthly validation target; 60 = 10 non-annual models x 6 regions.

## Reproducing the experiment

### Smoke test (minutes, local)
```bash
cd experiments/model_comparison
./smoke_test.sh                       # dev config: N=20, 1 region, end-to-end
```

### Full pipeline (HPC)
```bash
cd experiments/model_comparison
python 00_retrieve_data.py             # one-time data caching
./run_pipeline.sh                      # chains all stages with afterok barriers
```

Or stage by stage:
```bash
sbatch run_hpc_array.sh                # Stage 1: generation (78 tasks)
sbatch --dependency=afterok:$JOB_1 run_analyze.sh    # Stage 2
python check_stage_complete.py --stage analyze        # gate
python assemble_results.py
sbatch run_convergence.sh              # Stage 3
sbatch run_split_sample.sh             # Stage 4
python 04b_si_figures.py               # SI diagnostics (per-region + cross-region)
python 04c_si_convergence_figures.py   # SI convergence figures
python 04a_main_figures.py             # main manuscript figures
```

## Configuration

All runtime parameters live in [config.py](config.py). Key values:

- `N_REALIZATIONS` -- 500 for publication, 20 for development.
- `N_YEARS` -- 50 years per realization.
- `MODELS` -- enable/disable, init/fit/gen kwargs per generator.
- `ACTIVE_REGIONS` / `ACTIVE_MODELS` -- subset control.

Basin metadata and selection rationale: [basins.py](basins.py).

## Workflow infrastructure

- `check_stage_complete.py` -- stage checkpoint gate. Walks `outputs/`
  for expected artifacts, writes `outputs/status/{stage}_status.csv`,
  exits nonzero if any (region, model) missing.
- `run_pipeline.sh` -- full DAG with SLURM `--dependency=afterok:`
  chaining.
- `smoke_test.sh` -- end-to-end dev-config test on one region.
- `profile_memory.py` -- validates 8 GB SLURM allocation for the 3
  slowest models at publication scale.

## Status

Current phase: Pre-HPC pipeline hardening.
See [notes/deliverables_tracker.md](notes/deliverables_tracker.md) for
per-deliverable status.

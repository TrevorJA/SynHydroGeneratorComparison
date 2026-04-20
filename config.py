"""
Configuration for SynHydro model comparison experiments.

Controls which models are run, their parameters, ensemble settings,
and diagnostic options. Edit this file to customize experiments.
"""

from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
# Anchored to this file's directory so paths are stable regardless of
# the caller's cwd. Prior versions used cwd-relative paths which
# nested when shell scripts cd'd into model_comparison before running
# Python.
_EXPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = _EXPT_DIR / "data"
OUTPUT_DIR = _EXPT_DIR / "outputs"
FIGURE_DIR = _EXPT_DIR / "figures"

# Figures are split into two roots by audience:
#   - MAIN_FIGURE_DIR: the 10 manuscript main-text figures
#       (produced by 04a_main_figures.py; registry in
#       methods/plotting/manuscript.py)
#   - SI_FIGURE_DIR: everything else -- per-region diagnostic suite,
#       cross-region summaries, convergence detail, QA/QC figures.
#       Default destination for any new exploratory figure.
MAIN_FIGURE_DIR = FIGURE_DIR / "main"
SI_FIGURE_DIR = FIGURE_DIR / "si"

# ============================================================================
# Ensemble Settings
# ============================================================================
# Set to 500 for publication-quality runs; 20 for development/testing.
N_REALIZATIONS = 500
N_YEARS = 50
SEED = 42
OUTPUT_FORMAT = "hdf5"  # "hdf5" or "pickle"

# ============================================================================
# Site selection
# ============================================================================
# Deprecated: scripts now use get_reference_site_index(Q_monthly) to select
# the site with the highest long-run mean flow.  This constant is retained
# only as a fallback for external callers that have not been updated.
SINGLE_SITE_INDEX = 0

# ============================================================================
# Subset control
# ============================================================================
# Set to None for all, or a list to restrict.
# e.g. ACTIVE_REGIONS = ["new_england", "pacific_northwest"]
# e.g. ACTIVE_MODELS = ["kirsch", "matalas"]
ACTIVE_REGIONS = None
ACTIVE_MODELS = None

# ============================================================================
# Model Registry
# ============================================================================
# Each entry maps a short model key to its configuration.
#
# Fields:
#   enabled       -- set False to skip this model
#   class_name    -- generator class name (importable from synhydro.methods.generation)
#   category      -- "nonparametric" or "parametric"
#   frequency     -- native output frequency: "daily", "monthly", or "annual"
#   multisite     -- True if the model supports multi-site generation
#   description   -- one-line summary shown in console output
#   init_kwargs   -- dict of keyword arguments passed to the generator constructor
#   fit_kwargs    -- dict of keyword arguments passed to .fit()
#   gen_kwargs    -- dict of extra keyword arguments passed to .generate()
#   data_prep     -- optional callable name for special data preparation
#   output_file   -- filename stem for the ensemble output (extension added by script)

MODELS = {
    # ------------------------------------------------------------------
    # Nonparametric generators
    # ------------------------------------------------------------------
    "kirsch": {
        "enabled": True,
        "class_name": "KirschGenerator",
        "category": "nonparametric",
        "frequency": "monthly",
        "multisite": True,
        "description": "Kirsch-Nowak bootstrap with K-NN resampling",
        "init_kwargs": {
            "generate_using_log_flow": True,
            "matrix_repair_method": "spectral",
        },
        "fit_kwargs": {},
        "gen_kwargs": {},
    },
    "knn_bootstrap": {
        "enabled": True,
        "class_name": "KNNBootstrapGenerator",
        "category": "nonparametric",
        "frequency": "monthly",
        "multisite": True,
        "description": "K-Nearest Neighbor bootstrap (Lall & Sharma 1996)",
        "init_kwargs": {
            "n_neighbors": None,
            "block_size": 1,
        },
        "fit_kwargs": {},
        "gen_kwargs": {},
    },
    "phase_randomization": {
        "enabled": True,
        "class_name": "PhaseRandomizationGenerator",
        "category": "nonparametric",
        "frequency": "daily",
        "multisite": False,
        "description": "Fourier phase randomization (Brunner et al. 2019)",
        "init_kwargs": {
            "marginal": "kappa",
            "win_h_length": 15,
        },
        "fit_kwargs": {},
        "gen_kwargs": {},
        "data_prep": "trim_daily_to_complete_years",
    },
    # ------------------------------------------------------------------
    # Parametric generators
    # ------------------------------------------------------------------
    "thomas_fiering": {
        "enabled": True,
        "class_name": "ThomasFieringGenerator",
        "category": "parametric",
        "frequency": "monthly",
        "multisite": False,
        "description": "Thomas-Fiering AR(1) with Stedinger normalization",
        "init_kwargs": {},
        "fit_kwargs": {},
        "gen_kwargs": {},
    },
    "matalas": {
        "enabled": True,
        "class_name": "MatalasGenerator",
        "category": "parametric",
        "frequency": "monthly",
        "multisite": True,
        "description": "Matalas multivariate AR(1) (MAR(1), Matalas 1967)",
        "init_kwargs": {
            "log_transform": True,
        },
        "fit_kwargs": {},
        "gen_kwargs": {},
    },
    "hmm": {
        "enabled": True,
        "class_name": "MultiSiteHMMGenerator",
        "category": "parametric",
        "frequency": "annual",
        "multisite": True,
        "description": "Multi-site Hidden Markov Model (Gold et al. 2024)",
        "init_kwargs": {
            "n_states": 2,
            "covariance_type": "full",
        },
        "fit_kwargs": {},
        "gen_kwargs": {},
    },
    "warm": {
        "enabled": True,
        "class_name": "WARMGenerator",
        "category": "parametric",
        "frequency": "annual",
        "multisite": False,
        "description": "Wavelet Auto-Regressive Method (Nowak et al. 2011)",
        "init_kwargs": {
            "wavelet": "morl",
            "scales": 16,
            "ar_order": 1,
        },
        "fit_kwargs": {},
        "gen_kwargs": {},
    },
    "arfima": {
        "enabled": True,
        "class_name": "ARFIMAGenerator",
        "category": "parametric",
        "frequency": "monthly",
        "multisite": False,
        "description": "ARFIMA(p,d,q) with BIC order selection (Hosking 1984)",
        "init_kwargs": {
            "p": 1,
            "q": 0,
            "d_method": "whittle",
            "truncation_lag": 100,
            "deseasonalize": True,
            "auto_order": True,
        },
        "fit_kwargs": {},
        "gen_kwargs": {},
    },
    # ------------------------------------------------------------------
    # Copula-based generators
    # ------------------------------------------------------------------
    "gaussian_copula": {
        "enabled": True,
        "class_name": "GaussianCopulaGenerator",
        "category": "parametric",
        "frequency": "monthly",
        "multisite": True,
        "description": "Gaussian copula with parametric marginals (Genest & Favre 2007)",
        "init_kwargs": {
            "copula_type": "gaussian",
            "marginal_method": "parametric",
        },
        "fit_kwargs": {},
        "gen_kwargs": {},
    },
    "t_copula": {
        "enabled": True,
        "class_name": "GaussianCopulaGenerator",
        "category": "parametric",
        "frequency": "monthly",
        "multisite": True,
        "description": "Student-t copula with parametric marginals (Tootoonchi et al. 2022)",
        "init_kwargs": {
            "copula_type": "t",
            "marginal_method": "parametric",
        },
        "fit_kwargs": {},
        "gen_kwargs": {},
    },
    "vine_copula": {
        "enabled": True,
        "class_name": "VineCopulaGenerator",
        "category": "parametric",
        "frequency": "monthly",
        "multisite": True,
        "description": "R-vine copula with heterogeneous pairwise dependence (Bedford & Cooke 2002)",
        "init_kwargs": {
            "vine_type": "rvine",
            "family_set": "all",
            "selection_criterion": "aic",
            "marginal_method": "parametric",
        },
        "fit_kwargs": {},
        "gen_kwargs": {},
    },
    # ------------------------------------------------------------------
    # New generators
    # ------------------------------------------------------------------
    "hmm_knn": {
        "enabled": True,
        "class_name": "HMMKNNGenerator",
        "category": "parametric",
        "frequency": "annual",
        "multisite": True,
        "description": "HMM-KNN regime-conditioned bootstrap (Prairie et al. 2008)",
        "init_kwargs": {
            "n_states": 2,
            "covariance_type": "full",
            "n_init": 10,
        },
        "fit_kwargs": {},
        "gen_kwargs": {},
    },
    "multisite_phase_randomization": {
        "enabled": True,
        "class_name": "MultisitePhaseRandomizationGenerator",
        "category": "nonparametric",
        "frequency": "daily",
        "multisite": True,
        "description": "Multisite wavelet phase randomization (Brunner & Gilleland 2020)",
        "init_kwargs": {
            "wavelet": "cmor1.5-1.0",
            "n_scales": 100,
            "window_days": 15,
        },
        "fit_kwargs": {},
        "gen_kwargs": {},
        "data_prep": "trim_daily_df_to_complete_years",
    },
}

# ============================================================================
# Diagnostics (used by downstream analysis scripts)
# ============================================================================
# Auto-scale bootstrap resamples so smoke tests (small N_REALIZATIONS)
# do not spend hours inside fig_skill_radar. Publication runs (N >= 100)
# use the full 500 resamples; dev runs (N < 100) drop to 20.
_SKILL_RADAR_N_BOOTSTRAP = 500 if N_REALIZATIONS >= 100 else 20

DIAGNOSTICS = {
    "print_summary": True,
    "save_historical": True,
    "acf_max_lag_monthly": 24,
    "acf_max_lag_annual": 10,
    "fdc_log_scale": True,
    "hist_bins_monthly": 30,
    "hist_bins_annual": 15,
    # Skill radar bootstrap resamples. Scaled by N_REALIZATIONS above.
    "skill_radar_n_bootstrap": _SKILL_RADAR_N_BOOTSTRAP,
}

# ============================================================================
# Drought definition
# ============================================================================
# Two drought metrics are computed by the validation framework:
#
# 1. Threshold-based drought (category "drought"):
#    A drought event begins when monthly flow drops below the 20th
#    percentile of observed flows at each site (Q20). Duration is the
#    number of consecutive months below Q20; severity is the cumulative
#    deficit below Q20. The threshold is derived from the observed
#    (training) record and applied identically to synthetic realizations.
#    Reference: Van Loon & Laaha (2015), doi:10.5194/hess-19-3173-2015
#
# 2. SSI-based drought (category "ssi_drought"):
#    The Standardized Streamflow Index (SSI) is computed at a 12-month
#    timescale by fitting gamma distributions to observed monthly flows,
#    then transforming both observed and synthetic series to standard
#    normal space. A drought event begins when SSI < -1 and ends when
#    SSI >= 0. Duration and severity are computed analogously.
#    Reference: McKee et al. (1993), Proc. 8th Conf. Applied Climatology.
#
# Both metrics appear in the validation output. The manuscript uses
# threshold-based drought in main-text Figure 9 and SSI-based drought
# in supplementary material.
DROUGHT_THRESHOLD = None  # None = 20th percentile (default)

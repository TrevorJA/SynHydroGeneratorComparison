"""Model color scheme and display labels for the model comparison experiment.

Organizes the 11 generators into five generation-type families, each
assigned a distinct hue family so that visual groupings reflect methodological
relationships across all figures.

  bootstrap  -- nonparametric resampling       (orange)
  spectral   -- Fourier/spectral nonparametric (amber)
  ar         -- parametric autoregressive      (purple)
  state      -- parametric state-based         (teal-green)
  copula     -- copula-based                   (blue)
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Model families
# ---------------------------------------------------------------------------

MODEL_FAMILIES: dict[str, list[str]] = {
    "bootstrap": ["kirsch", "knn_bootstrap"],
    "spectral": ["phase_randomization"],
    "ar": ["thomas_fiering", "matalas", "arfima"],
    "state": ["hmm", "warm"],
    "copula": ["gaussian_copula", "t_copula", "vine_copula"],
}

# Representative color per family (used for family-level legends or annotations)
FAMILY_COLORS: dict[str, str] = {
    "bootstrap": "#E05C00",
    "spectral": "#D4AC0D",
    "ar": "#8E44AD",
    "state": "#117A65",
    "copula": "#2980B9",
}

# ---------------------------------------------------------------------------
# Per-model colors
# ---------------------------------------------------------------------------

MODEL_COLORS: dict[str, str] = {
    # Bootstrap (warm orange tones)
    "kirsch": "#E05C00",
    "knn_bootstrap": "#F5A623",
    # Spectral (golden amber -- warm, visually distinct from orange)
    "phase_randomization": "#D4AC0D",
    # Parametric AR (purple-violet)
    "thomas_fiering": "#5B2C8D",
    "matalas": "#8E44AD",
    "arfima": "#BB8FCE",
    # State-based parametric (teal-green)
    "hmm": "#117A65",
    "warm": "#1E8449",
    # Copula-based (blue)
    "gaussian_copula": "#1A5276",
    "t_copula": "#2980B9",
    "vine_copula": "#7FB3D3",
}

# ---------------------------------------------------------------------------
# Display labels (clean academic names for figure axes and legends)
# ---------------------------------------------------------------------------

MODEL_DISPLAY_LABELS: dict[str, str] = {
    "kirsch": "Kirsch-Nowak",
    "knn_bootstrap": "KNN Bootstrap",
    "phase_randomization": "Phase Rand.",
    "thomas_fiering": "Thomas-Fiering",
    "matalas": "MATALAS",
    "arfima": "ARFIMA",
    "hmm": "HMM",
    "warm": "WARM",
    "gaussian_copula": "Gaussian Copula",
    "t_copula": "Student-t Copula",
    "vine_copula": "Vine Copula",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_model_family(model_key: str) -> Optional[str]:
    """Return the family name for *model_key*, or None if not registered."""
    for family, members in MODEL_FAMILIES.items():
        if model_key in members:
            return family
    return None


def get_model_colors(model_keys: list) -> dict:
    """Return a {model_key: hex_color} dict for the given keys.

    Parameters
    ----------
    model_keys : list
        Ordered list of model identifier strings.

    Returns
    -------
    dict
        Maps each key to its semantic hex color.  Unregistered keys fall
        back to medium gray ``"#888888"``.
    """
    return {key: MODEL_COLORS.get(key, "#888888") for key in model_keys}

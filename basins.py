"""
CAMELS basin registry for multi-region synthetic streamflow model comparison.

Basin selection follows a stratified design across the aridity-snow fraction
space (Addor et al. 2017), with geographic clustering to enable multi-site
dependence testing. All basins are from the CAMELS dataset (Newman et al.
2015; Addor et al. 2017), which ensures minimal anthropogenic impact and
standardized meteorological forcing.

Selection criteria:
  1. Complete daily discharge record within CAMELS period (1980-10-01 to
     2014-09-30, water years 1981-2014).
  2. Stratified sampling across 6 hydroclimatic regimes to span the full
     aridity index (0.3-3.0+) and snow fraction (0.0-0.6+) space.
  3. Geographic proximity within each region (same HUC-04 or adjacent) to
     test multi-site spatial dependence structure.
  4. Basin areas between 50-2000 km2 to avoid both tiny headwaters (noisy)
     and large mixed-use basins.
  5. Preference for basins used in prior CAMELS benchmark studies (Kratzert
     et al. 2019; Addor et al. 2017) to facilitate comparison.

The 6 regions span distinct positions in the Budyko space:
  - Humid, snow-influenced (New England)
  - Humid, mixed rain/snow (Mid-Atlantic, replacing prior DRB gauges)
  - Humid, rain-dominated (Southern Appalachians)
  - Arid, snowmelt-dominated (Central Rockies)
  - Wet maritime (Pacific Northwest)
  - Semi-arid continental (Central Great Plains)

References:
  Addor, N., et al. (2017). The CAMELS data set: catchment attributes and
      meteorology for large-sample studies. HESS, 21(10), 5293-5313.
  Newman, A.J., et al. (2015). Development of a large-sample watershed-scale
      hydrometeorological dataset for the contiguous USA. WRR, 51(8),
      4684-4701.
  Kratzert, F., et al. (2019). Toward improved predictions in ungauged
      basins: exploiting the power of machine learning. WRR, 55(12),
      11344-11354.
"""

# ============================================================================
# Region definitions
# ============================================================================
#
# Station IDs are 8-digit USGS gauge numbers confirmed present in the CAMELS
# dataset. Each cluster consists of basins in the same HUC-04 sub-region.
#
# NOTE: If a station is found to be missing during data retrieval
# (00_retrieve_data.py), the script will report which IDs failed and
# suggest replacements from the same HUC region.

CAMELS_REGIONS = {
    "new_england": {
        "huc_02": "01",
        "description": "Humid snow-influenced headwaters (Maine/New Hampshire)",
        "climate_type": "humid_snow",
        "aridity_range": [0.55, 0.75],
        "snow_frac_range": [0.25, 0.40],
        "stations": [
            "01013500",  # Fish River near Fort Kent, ME (area ~2253 km2)
            "01031500",  # Piscataquis River near Dover-Foxcroft, ME (~769 km2)
            "01052500",  # Diamond River near Wentworth Location, NH (~384 km2)
            "01022500",  # Narraguagus River at Cherryfield, ME (~574 km2)
        ],
        "justification": (
            "Northern New England cluster spanning Maine and New Hampshire. "
            "Strong spring snowmelt signal with 25-40% snow fraction. Basins "
            "01013500 and 01031500 are among the most-studied CAMELS basins in "
            "HUC-01 (Kratzert et al. 2019). Basin 01022500 replaces 01035000 "
            "(Passadumkeag River at Lowell — absent from pygeohydro CAMELS "
            "671-station dataset). Narraguagus River is a comparable Maine "
            "coastal-headwater basin (~574 km2). "
            "Tests preservation of seasonal cycle timing, snowmelt peak "
            "magnitude, and moderate spatial correlation driven by shared "
            "snowpack dynamics."
        ),
    },
    "mid_atlantic": {
        "huc_02": "02",
        "description": "Humid mixed rain/snow (Pennsylvania/New Jersey)",
        "climate_type": "humid_mixed",
        "aridity_range": [0.65, 0.85],
        "snow_frac_range": [0.10, 0.20],
        "stations": [
            "01440000",  # Flat Brook near Flatbrookville, NJ (~166 km2)
            "01439500",  # Bush Kill at Shoemakers, PA (~310 km2)
            "01440400",  # Brodhead Creek near Analomink, PA (~171 km2)
            "01435000",  # Neversink River near Claryville, NY (~170 km2)
        ],
        "justification": (
            "Delaware River Basin headwater tributaries in the Pocono/Kittatinny "
            "region. Replaces the managed main-stem Delaware gauges from the "
            "original DRB study with unmanaged CAMELS basins from the same "
            "geographic area. Mixed rain-snow regime with moderate seasonality. "
            "Flat Brook (01440000) was included in the original DRB experiment. "
            "01440400 (Brodhead Creek) provides a distinct Pocono watershed for "
            "spatial correlation testing. Tests generators under humid, moderate "
            "conditions -- the baseline regime where most methods should "
            "perform well."
        ),
    },
    "southern_appalachians": {
        "huc_02": "03",
        "description": "Humid rain-dominated (Blue Ridge, NC/VA)",
        "climate_type": "humid_rain",
        "aridity_range": [0.65, 0.90],
        "snow_frac_range": [0.02, 0.08],
        "stations": [
            "03164000",  # New River near Galax, VA (~2950 km2)
            "03170000",  # Little River at Graysontown, VA (~790 km2)
            "03173000",  # Walker Creek at Bane, VA (~540 km2)
            "03180500",  # Greenbrier River at Durbin, WV (~340 km2)
        ],
        "justification": (
            "Southern Appalachian cluster in the New River basin (HUC-0305). "
            "Rain-dominated with negligible snow, high baseflow index, "
            "moderate-to-high skewness from convective storms. These are "
            "well-gauged basins with complete CAMELS records. Tests generator "
            "performance on flashy hydrographs and right-skewed marginal "
            "distributions where parametric assumptions are strained."
        ),
    },
    "central_rockies": {
        "huc_02": "14",
        "description": "Arid snowmelt-dominated headwaters (Colorado)",
        "climate_type": "arid_snow",
        "aridity_range": [1.2, 2.5],
        "snow_frac_range": [0.45, 0.65],
        "stations": [
            "09066300",  # Middle Creek near Minturn, CO (~60 km2)
            "09047700",  # Straight Creek near Dillon, CO (~80 km2)
            "09065500",  # Gore Creek at Upper Station near Minturn, CO (~130 km2)
            "09066000",  # Black Gore Creek near Minturn, CO (~50 km2)
        ],
        "justification": (
            "Central Colorado headwater cluster in the Upper Colorado basin. "
            "Snow fraction 45-65% with a single dominant melt peak (May-June) "
            "and near-zero baseflow in winter. Aridity index >1 means PET "
            "exceeds precipitation. This is the most challenging seasonal "
            "regime for generators: extreme seasonality, compressed flow "
            "distribution, and strong year-to-year variability in snowpack. "
            "Tests whether generators can reproduce the sharp seasonal on/off "
            "behavior and interannual snowmelt variability."
        ),
    },
    "pacific_northwest": {
        "huc_02": "17",
        "description": "Wet maritime coastal (SW Washington)",
        "climate_type": "wet_maritime",
        "aridity_range": [0.25, 0.45],
        "snow_frac_range": [0.02, 0.10],
        "stations": [
            "12010000",  # Naselle River near Naselle, WA (~140 km2)
            "12013500",  # Willapa River near Willapa, WA (~320 km2)
            "12020000",  # Chehalis River near Doty, WA (~460 km2)
            "12035000",  # Satsop River near Satsop, WA (~780 km2)
        ],
        "justification": (
            "SW Washington coastal cluster, among the wettest basins in CAMELS "
            "(aridity 0.25-0.45). Rain-dominated with sustained high winter "
            "flows and strong inter-basin correlation from shared Pacific "
            "frontal systems. Minimal snow. This cluster specifically tests "
            "multi-site spatial dependence preservation -- the high shared "
            "forcing should produce strong cross-site correlations that "
            "multivariate generators must reproduce."
        ),
    },
    "central_plains": {
        "huc_02": "10",
        "description": "Semi-arid continental (Kansas/Nebraska)",
        "climate_type": "semiarid_continental",
        "aridity_range": [1.3, 2.2],
        "snow_frac_range": [0.08, 0.18],
        "stations": [
            "06876700",  # Solomon River near Woodston, KS (~1700 km2)
            "06878000",  # Chapman Creek near Chapman, KS (~780 km2)
            "06885500",  # Black Vermillion River near Frankfort, KS (~1060 km2)
            "06888500",  # Mill Creek near Paxico, KS (~830 km2)
        ],
        "justification": (
            "Central Kansas cluster in the Kansas River basin (HUC-1027). "
            "Semi-arid with aridity >1, high interannual variability, and "
            "convective-storm-driven summer peaks. Flow is perennial but "
            "highly variable with occasional near-zero months. CV values "
            "3.7-4.3 are the highest of any region. This regime stresses "
            "generators that assume Gaussian or log-normal marginals due "
            "to extreme positive skewness and heavy right tails."
        ),
    },
}

# ============================================================================
# Climate type descriptions (for cross-region analysis context)
# ============================================================================

CLIMATE_TYPES = {
    "humid_snow": "Moderate seasonal cycle, snowmelt peak, high autocorrelation",
    "humid_mixed": "Moderate seasonality, rain+snow, high baseflow, moderate skewness",
    "humid_rain": "Weak seasonal cycle, flashy response, high skewness",
    "arid_snow": "Extreme seasonal cycle, single snowmelt peak, near-zero winter flow",
    "wet_maritime": "Weak seasonality, sustained high flows, high spatial correlation",
    "semiarid_continental": "High variability, convective peaks, extreme positive skewness",
}

# ============================================================================
# Generator challenge matrix -- which regimes test which generator properties
# ============================================================================

REGIME_CHALLENGES = {
    "humid_snow": ["seasonal_cycle_timing", "snowmelt_peak", "lag1_autocorrelation"],
    "humid_mixed": ["baseline_performance", "moderate_seasonality"],
    "humid_rain": [
        "skewness_preservation",
        "flashy_response",
        "drought_characteristics",
    ],
    "arid_snow": ["extreme_seasonality", "zero_flow_months", "interannual_variability"],
    "wet_maritime": ["spatial_correlation", "high_flow_tail", "sustained_wet_periods"],
    "semiarid_continental": ["marginal_distribution", "heavy_tails", "intermittency"],
}

# ============================================================================
# Convenience accessors
# ============================================================================

ALL_STATION_IDS = [
    sid for region in CAMELS_REGIONS.values() for sid in region["stations"]
]

ALL_REGION_IDS = sorted(CAMELS_REGIONS.keys())

N_REGIONS = len(CAMELS_REGIONS)
N_TOTAL_BASINS = len(ALL_STATION_IDS)


def get_region_stations(region_id: str) -> list:
    """Return the list of station IDs for a region."""
    return CAMELS_REGIONS[region_id]["stations"]


def get_region_ids() -> list:
    """Return all region IDs in sorted order."""
    return ALL_REGION_IDS

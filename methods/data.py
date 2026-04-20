"""Data loading, frequency conversion, preparation, and CAMELS retrieval."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CFS_TO_CMS = 0.028316846592


# ============================================================================
# Frequency conversion
# ============================================================================


def prepare_frequencies(
    Q_daily: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive monthly and annual DataFrames from daily streamflow.

    Parameters
    ----------
    Q_daily : pd.DataFrame
        Daily streamflow with DatetimeIndex.

    Returns
    -------
    Q_daily, Q_monthly, Q_annual : tuple of pd.DataFrame
    """
    Q_monthly = Q_daily.resample("MS").sum()
    Q_annual = Q_monthly.resample("YS").sum()
    return Q_daily, Q_monthly, Q_annual


# ============================================================================
# Data preparation helpers
# ============================================================================


def trim_daily_to_complete_years(Q_daily_series: pd.Series) -> pd.Series:
    """Trim a daily series to complete 365-day years (remove leap days first).

    Phase Randomization requires data length to be a multiple of 365.
    """
    Q_no_leap = Q_daily_series[
        ~((Q_daily_series.index.month == 2) & (Q_daily_series.index.day == 29))
    ]
    n_years = len(Q_no_leap) // 365
    trimmed = Q_no_leap.iloc[: n_years * 365]
    print(f"    Trimmed to {n_years} complete years ({len(trimmed)} days)")
    return trimmed


def trim_daily_df_to_complete_years(
    Q_daily_df: Union[pd.DataFrame, pd.Series],
) -> Union[pd.DataFrame, pd.Series]:
    """Trim a daily DataFrame to complete 365-day years (remove leap days first).

    MultisitePhaseRandomizationGenerator requires data length to be a multiple
    of 365 with no leap days present.
    """
    Q_no_leap = Q_daily_df[
        ~((Q_daily_df.index.month == 2) & (Q_daily_df.index.day == 29))
    ]
    n_years = len(Q_no_leap) // 365
    trimmed = Q_no_leap.iloc[: n_years * 365]
    print(f"    Trimmed to {n_years} complete years ({len(trimmed)} days)")
    return trimmed


DATA_PREP_FUNCTIONS = {
    "trim_daily_to_complete_years": trim_daily_to_complete_years,
    "trim_daily_df_to_complete_years": trim_daily_df_to_complete_years,
}


def get_reference_site_index(Q_monthly: pd.DataFrame) -> int:
    """Return the column index of the site with the largest long-run mean flow.

    Using the highest-flow gauge minimises the risk that near-zero monthly
    values dominate relative-error metrics for univariate models.

    Parameters
    ----------
    Q_monthly : pd.DataFrame
        Monthly streamflow with one column per gauging station.

    Returns
    -------
    int
        Zero-based column index of the reference site.
    """
    return int(Q_monthly.mean(axis=0).values.argmax())


def select_input_data(
    Q_daily: pd.DataFrame,
    Q_monthly: pd.DataFrame,
    Q_annual: pd.DataFrame,
    model_cfg: dict,
    single_site_index: int = 0,
) -> Union[pd.DataFrame, pd.Series]:
    """Select and shape the correct input data for a model.

    Parameters
    ----------
    Q_daily, Q_monthly, Q_annual : pd.DataFrame
        Streamflow at each frequency.
    model_cfg : dict
        Model configuration dict (must have "frequency", "multisite" keys).
    single_site_index : int
        Column index used for univariate models.

    Returns
    -------
    pd.DataFrame or pd.Series
    """
    freq = model_cfg["frequency"]
    multisite = model_cfg["multisite"]

    if freq == "daily":
        data = Q_daily
    elif freq == "monthly":
        data = Q_monthly
    elif freq == "annual":
        data = Q_annual
    else:
        raise ValueError(f"Unknown frequency: {freq}")

    if not multisite:
        data = data.iloc[:, single_site_index]

    prep_name = model_cfg.get("data_prep")
    if prep_name:
        data = DATA_PREP_FUNCTIONS[prep_name](data)

    return data


# ============================================================================
# CAMELS data retrieval via pygeohydro
# ============================================================================


def download_camels():
    """Download the full CAMELS dataset using pygeohydro.

    Returns
    -------
    attrs : geopandas.GeoDataFrame
        Basin attributes for all 671 CAMELS stations.
    qobs : xarray.Dataset
        Daily discharge (cfs) for all 671 stations, 1980-2014.
    """
    from pygeohydro import get_camels

    logger.info("Downloading CAMELS dataset via pygeohydro.get_camels()...")
    attrs, qobs = get_camels()
    logger.info(
        "CAMELS downloaded: %d stations, %s to %s",
        len(attrs),
        str(qobs.time.values[0])[:10],
        str(qobs.time.values[-1])[:10],
    )
    return attrs, qobs


def extract_region_from_camels(
    qobs,
    station_ids: list,
) -> pd.DataFrame:
    """Extract daily streamflow for specific stations from CAMELS xarray Dataset.

    Parameters
    ----------
    qobs : xarray.Dataset
        Full CAMELS discharge dataset (from get_camels()).
    station_ids : list of str
        USGS station IDs to extract.

    Returns
    -------
    pd.DataFrame
        Daily streamflow in m3/s with DatetimeIndex and one column per station.
    """
    # Verify all station IDs exist in CAMELS
    available = set(qobs.station_id.values.astype(str))
    missing = [sid for sid in station_ids if sid not in available]
    if missing:
        raise ValueError(
            f"Station(s) {missing} not found in CAMELS dataset. "
            f"CAMELS contains {len(available)} stations."
        )

    # Extract discharge for the requested stations
    subset = qobs["discharge"].sel(station_id=station_ids)

    # Convert xarray -> pandas DataFrame with time as index, stations as columns
    df = subset.to_dataframe("discharge").reset_index()
    df = df.pivot(index="time", columns="station_id", values="discharge")
    df.index = pd.to_datetime(df.index)
    df.index.name = "datetime"
    df.columns = [str(c) for c in df.columns]
    df.columns.name = None

    # Convert cfs to m3/s
    df = df * CFS_TO_CMS

    return df


def get_camels_streamflow(
    station_ids: list,
    cache_dir: Path,
    region_id: Optional[str] = None,
    camels_qobs=None,
) -> pd.DataFrame:
    """Get daily streamflow for a set of CAMELS stations.

    If cached CSV exists, loads from cache. Otherwise extracts from the
    CAMELS xarray Dataset (pass via camels_qobs) or falls back to the
    NWIS REST API.

    Parameters
    ----------
    station_ids : list of str
        USGS station IDs (8-digit strings).
    cache_dir : Path
        Directory for cached CSV files.
    region_id : str, optional
        If provided, cache is stored under ``cache_dir/region_id/daily.csv``.
    camels_qobs : xarray.Dataset, optional
        Pre-downloaded CAMELS dataset. If None and cache misses, downloads
        via NWIS REST API as fallback.

    Returns
    -------
    pd.DataFrame
        Daily streamflow (m3/s) with DatetimeIndex and one column per station.
    """
    if region_id:
        cache_subdir = Path(cache_dir) / region_id
    else:
        cache_subdir = Path(cache_dir)
    cache_subdir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_subdir / "daily.csv"

    if csv_path.exists():
        logger.info("Loading cached data from %s", csv_path)
        Q = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return Q

    # Extract from CAMELS xarray if available
    if camels_qobs is not None:
        logger.info("Extracting stations %s from CAMELS dataset", station_ids)
        Q = extract_region_from_camels(camels_qobs, station_ids)
    else:
        # Fallback: direct NWIS REST API
        logger.info("Downloading streamflow via NWIS for stations %s", station_ids)
        Q = _fetch_nwis_multiple(station_ids)

    # Filter to longest contiguous block where all stations have valid data
    valid_mask = Q.notna().all(axis=1)
    if valid_mask.sum() > 0:
        valid_groups = (valid_mask != valid_mask.shift()).cumsum()
        valid_groups_filtered = valid_groups[valid_mask]
        group_sizes = valid_groups_filtered.value_counts()
        longest_group_id = group_sizes.idxmax()
        longest_mask = valid_groups_filtered == longest_group_id
        valid_indices = longest_mask[longest_mask].index
        Q = Q.loc[valid_indices]
        logger.info(
            "Filtered to longest contiguous period: %d days (%s to %s)",
            len(Q),
            Q.index[0].date(),
            Q.index[-1].date(),
        )
    else:
        logger.warning(
            "No overlapping valid data for stations %s; keeping all rows",
            station_ids,
        )

    Q.to_csv(csv_path)
    logger.info("Cached daily data to %s", csv_path)
    return Q


def _fetch_nwis_multiple(station_ids: list) -> pd.DataFrame:
    """Fetch daily streamflow for multiple stations via NWIS REST API."""
    import io
    import urllib.request

    series_list = []
    for sid in station_ids:
        url = (
            "https://waterservices.usgs.gov/nwis/dv/"
            f"?format=rdb&sites={sid}"
            "&startDT=1980-01-01&endDT=2014-12-31"
            "&parameterCd=00060&statCd=00003"
        )
        logger.info("Fetching %s from NWIS...", sid)
        with urllib.request.urlopen(url, timeout=120) as resp:
            text = resp.read().decode("utf-8")

        lines = []
        header = None
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if header is None:
                header = line.split("\t")
                continue
            if line.startswith("5s") or line.startswith("5d"):
                continue
            lines.append(line.split("\t"))

        if not lines:
            raise ValueError(f"No data returned for station {sid}")

        df = pd.DataFrame(lines, columns=header)
        val_col = None
        for col in df.columns:
            if "00060" in col and "00003" in col and "_cd" not in col:
                val_col = col
                break
        if val_col is None:
            raise ValueError(f"No discharge column for {sid}: {list(df.columns)}")

        dates = pd.to_datetime(df["datetime"])
        values = pd.to_numeric(df[val_col], errors="coerce") * CFS_TO_CMS
        series_list.append(pd.Series(values.values, index=dates, name=sid))

    Q = pd.concat(series_list, axis=1)
    Q.index.name = "datetime"
    return Q


def load_region_data(
    region_id: str,
    cache_dir: Path,
    stations: list,
    camels_qobs=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load (or download) CAMELS data for a region at all 3 frequencies.

    Parameters
    ----------
    region_id : str
        Region identifier (e.g. "new_england").
    cache_dir : Path
        Root cache directory.
    stations : list of str
        USGS station IDs for the region.
    camels_qobs : xarray.Dataset, optional
        Pre-downloaded CAMELS dataset for extraction.

    Returns
    -------
    Q_daily, Q_monthly, Q_annual : tuple of pd.DataFrame
    """
    cache_subdir = Path(cache_dir) / region_id
    monthly_path = cache_subdir / "monthly.csv"
    annual_path = cache_subdir / "annual.csv"

    Q_daily = get_camels_streamflow(stations, cache_dir, region_id, camels_qobs)

    if monthly_path.exists() and annual_path.exists():
        Q_monthly = pd.read_csv(monthly_path, index_col=0, parse_dates=True)
        Q_annual = pd.read_csv(annual_path, index_col=0, parse_dates=True)
    else:
        Q_daily, Q_monthly, Q_annual = prepare_frequencies(Q_daily)
        Q_monthly.to_csv(monthly_path)
        Q_annual.to_csv(annual_path)
        logger.info("Cached monthly/annual data to %s", cache_subdir)

    return Q_daily, Q_monthly, Q_annual

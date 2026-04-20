"""Spatial overview figures: CONUS map and per-region basin/flowline maps."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)


def fig_conus_overview(
    camels_attrs,
    regions_config: dict,
    figure_dir: Path,
    filename: str = "conus_basin_overview.png",
) -> None:
    """CONUS map showing all study basins colored by region.

    Parameters
    ----------
    camels_attrs : geopandas.GeoDataFrame
        Full CAMELS attributes (with geometry) from get_camels().
    regions_config : dict
        CAMELS_REGIONS dict from basins.py.
    figure_dir : Path
        Output directory for the figure.
    """
    import geopandas as gpd

    region_palette = plt.cm.Set2.colors
    sorted_regions = sorted(regions_config.keys())

    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot all CAMELS basins in light gray as background
    camels_attrs.plot(ax=ax, color="whitesmoke", edgecolor="lightgray", linewidth=0.2)

    # Overlay study basins colored by region
    legend_handles = []
    for i, region_id in enumerate(sorted_regions):
        color = region_palette[i % len(region_palette)]
        station_ids = regions_config[region_id]["stations"]
        region_basins = camels_attrs.loc[camels_attrs.index.isin(station_ids)]

        if region_basins.empty:
            continue

        region_basins.plot(
            ax=ax,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.7,
        )

        # Plot gauge points
        for sid in station_ids:
            if sid in camels_attrs.index:
                row = camels_attrs.loc[sid]
                lon = row.get("gauge_lon", None)
                lat = row.get("gauge_lat", None)
                if lon is not None and lat is not None:
                    ax.plot(
                        lon,
                        lat,
                        "k^",
                        markersize=6,
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                        zorder=5,
                    )

        label = f"{region_id} ({regions_config[region_id]['climate_type']})"
        legend_handles.append(mpatches.Patch(color=color, alpha=0.7, label=label))

    # Add gauge symbol to legend
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="black",
            markersize=6,
            label="USGS gauge",
        )
    )

    ax.legend(handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"CAMELS Study Basins: {len(sorted_regions)} Regions, "
        f"{sum(len(r['stations']) for r in regions_config.values())} Stations",
        fontsize=13,
        fontweight="bold",
    )

    # Set CONUS extent
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)

    fig.tight_layout()
    fig.savefig(Path(figure_dir) / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def fig_region_flowlines(
    camels_attrs,
    region_id: str,
    region_cfg: dict,
    figure_dir: Path,
    max_distance_km: int = 200,
) -> None:
    """Per-region map showing basin boundaries and NHD flowlines.

    Parameters
    ----------
    camels_attrs : geopandas.GeoDataFrame
        Full CAMELS attributes (with geometry).
    region_id : str
        Region identifier.
    region_cfg : dict
        Region config from CAMELS_REGIONS.
    figure_dir : Path
        Output directory.
    max_distance_km : int
        Maximum upstream distance (km) for flowline retrieval.
    """
    import geopandas as gpd
    from pynhd import NLDI

    station_ids = region_cfg["stations"]
    region_palette = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 10))

    nldi = NLDI()

    for i, sid in enumerate(station_ids):
        color = region_palette[i % len(region_palette)]

        # Plot basin polygon from CAMELS
        if sid in camels_attrs.index:
            basin_geom = camels_attrs.loc[[sid]]
            basin_geom.plot(
                ax=ax,
                color=color,
                alpha=0.15,
                edgecolor=color,
                linewidth=1.5,
            )

        # Retrieve upstream flowlines via NLDI
        try:
            flowlines = nldi.navigate_byid(
                "nwissite",
                f"USGS-{sid}",
                "upstreamTributaries",
                source="flowlines",
                distance=max_distance_km,
            )
            if flowlines is not None and not flowlines.empty:
                flowlines.plot(
                    ax=ax,
                    color=color,
                    linewidth=0.6,
                    alpha=0.7,
                )
        except Exception as e:
            logger.warning("Could not retrieve flowlines for %s: %s", sid, e)

        # Plot gauge point
        if sid in camels_attrs.index:
            row = camels_attrs.loc[sid]
            lon = row.get("gauge_lon", None)
            lat = row.get("gauge_lat", None)
            if lon is not None and lat is not None:
                ax.plot(
                    lon,
                    lat,
                    "^",
                    color=color,
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    zorder=10,
                    label=sid,
                )

    ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{region_id}: {region_cfg['description']}\n"
        f"({region_cfg['climate_type']}, {len(station_ids)} basins)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_aspect("equal")

    fig.tight_layout()
    filename = f"region_map_{region_id}.png"
    fig.savefig(Path(figure_dir) / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def create_spatial_figures(
    camels_attrs,
    regions_config: dict,
    figure_dir: Path,
) -> None:
    """Create all spatial overview figures.

    Parameters
    ----------
    camels_attrs : geopandas.GeoDataFrame
        Full CAMELS attributes from get_camels().
    regions_config : dict
        CAMELS_REGIONS dict.
    figure_dir : Path
        Output directory (e.g. figures/basin_data_verification/).
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # CONUS overview
    fig_conus_overview(camels_attrs, regions_config, figure_dir)

    # Per-region flowline maps
    for region_id, region_cfg in sorted(regions_config.items()):
        try:
            fig_region_flowlines(
                camels_attrs,
                region_id,
                region_cfg,
                figure_dir,
            )
        except Exception as e:
            logger.error("Failed to create map for %s: %s", region_id, e)

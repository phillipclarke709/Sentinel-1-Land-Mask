from pathlib import Path
import math
import rasterio
from rasterio.warp import transform_bounds


def snap_to_worldcover_grid(value: float) -> int:
    """
    Snap latitude or longitude to the ESA WorldCover 3° grid.
    Handles negative coordinates correctly.
    """
    return int(math.floor(value / 3.0) * 3)


def worldcover_tile_name(lat: int, lon: int, suffix: str = "_Map.tif") -> str:
    """
    Construct an ESA WorldCover tile filename from
    the southwest corner coordinates.
    """
    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_str = f"W{abs(lon):03d}" if lon < 0 else f"E{lon:03d}"
    return f"ESA_WorldCover_10m_2021_V200_{lat_str}{lon_str}{suffix}"


from typing import Union, List, Optional, Tuple

def find_required_worldcover_tiles(
    s1_path: Union[Path, str],
    worldcover_dir: Union[Path, str],
    bounds_wgs84: Optional[Tuple[float, float, float, float]] = None,
    filename_suffix: str = "_Map.tif",
) -> List[Path]:

    """
    Determine which ESA WorldCover 3° tiles intersect
    the Sentinel-1 raster footprint.

    Returns a list of existing tile paths.
    """

    if bounds_wgs84 is None:
        # -------------------------------------------------
        # Read Sentinel-1 bounds
        # -------------------------------------------------
        with rasterio.open(s1_path) as src:
            bounds_utm = src.bounds
            crs_utm = src.crs

        west, south, east, north = transform_bounds(
            crs_utm,
            "EPSG:4326",
            bounds_utm.left,
            bounds_utm.bottom,
            bounds_utm.right,
            bounds_utm.top,
            densify_pts=21,
        )
    else:
        west, south, east, north = bounds_wgs84

    # -------------------------------------------------
    # Select intersecting WorldCover tiles (3° grid)
    # -------------------------------------------------
    tiles = set()

    lat = snap_to_worldcover_grid(south)
    while lat < north:
        lon = snap_to_worldcover_grid(west)
        while lon < east:
            tile_s = lat
            tile_n = lat + 3
            tile_w = lon
            tile_e = lon + 3

            intersects = not (
                tile_e <= west or
                tile_w >= east or
                tile_n <= south or
                tile_s >= north
            )

            if intersects:
                tiles.add(worldcover_tile_name(lat, lon, suffix=filename_suffix))

            lon += 3
        lat += 3

    # -------------------------------------------------
    # Resolve existing paths
    # -------------------------------------------------
    paths = []
    worldcover_dir = Path(worldcover_dir)

    for t in sorted(tiles):
        p = worldcover_dir / t
        if p.exists():
            paths.append(p)

    if not paths:
        raise RuntimeError("No matching WorldCover tiles found.")

    return paths

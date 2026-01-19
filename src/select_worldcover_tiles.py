from pathlib import Path
import math
import rasterio
from rasterio.warp import transform_bounds

def snap_to_worldcover_grid(value):
    """
    Snap latitude or longitude to the WorldCover 3° grid
    with correct handling for negative values.
    """
    return int(math.floor(value / 3.0) * 3)

def worldcover_tile_name(lat, lon):
    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_str = f"W{abs(lon):03d}" if lon < 0 else f"E{lon:03d}"
    return f"ESA_WorldCover_10m_2021_V200_{lat_str}{lon_str}_Map.tif"

def find_required_worldcover_tiles(s1_path, worldcover_dir):

    # -------------------------------------------------
    # Read Sentinel-1 bounds
    # -------------------------------------------------
    with rasterio.open(s1_path) as src:
        bounds_utm = src.bounds
        crs_utm = src.crs

    west, south, east, north = transform_bounds(
        crs_utm, "EPSG:4326",
        bounds_utm.left, bounds_utm.bottom,
        bounds_utm.right, bounds_utm.top,
        densify_pts=21
    )

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
                tiles.add(worldcover_tile_name(lat, lon))

            lon += 3
        lat += 3

    # -------------------------------------------------
    # Resolve paths
    # -------------------------------------------------
    paths = []
    for t in sorted(tiles):
        p = Path(worldcover_dir) / t
        if p.exists():
            paths.append(p)

    if not paths:
        raise RuntimeError("No matching WorldCover tiles found.")

    print("Selected WorldCover tiles:")
    for p in paths:
        print(" ", p.name)

    return paths

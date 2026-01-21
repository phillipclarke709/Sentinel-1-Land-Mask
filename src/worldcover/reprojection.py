from typing import Tuple, List, Optional, Union
import os

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import array_bounds


def reproject_worldcover_to_s1(
    wc_mosaic: np.ndarray,
    wc_transform: rasterio.Affine,
    dst_transform: rasterio.Affine,
    dst_crs,
    dst_shape: Tuple[int, int],
    dst_nodata: int = -1,
) -> np.ndarray:
    """
    Reproject WorldCover mosaic (EPSG:4326) onto the Sentinel-1 grid.

    Returns:
        wc_reproj : int16 array aligned exactly to Sentinel-1
    """

    wc_reproj = np.full(dst_shape, dst_nodata, dtype="int16")

    reproject(
        source=wc_mosaic,
        destination=wc_reproj,
        src_transform=wc_transform,
        src_crs="EPSG:4326",
        src_nodata=0,              # WorldCover nodata
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=dst_nodata,
        resampling=Resampling.nearest,
    )

    return wc_reproj


def reproject_worldcover_tiles_to_s1(
    tile_paths: List,
    dst_transform: rasterio.Affine,
    dst_crs,
    dst_shape: Tuple[int, int],
    dst_nodata: int = -1,
    num_threads: Optional[Union[int, str]] = None,
) -> np.ndarray:
    """
    Reproject individual WorldCover tiles (EPSG:4326) onto the Sentinel-1 grid.

    Returns:
        wc_reproj : int16 array aligned exactly to Sentinel-1
    """

    wc_reproj = np.full(dst_shape, dst_nodata, dtype="int16")

    height, width = dst_shape
    dst_left, dst_bottom, dst_right, dst_top = array_bounds(
        height, width, dst_transform
    )

    first_write = True
    if num_threads is None or num_threads == "ALL_CPUS":
        num_threads = os.cpu_count() or 1

    for path in tile_paths:
        with rasterio.open(path) as src:
            src_left, src_bottom, src_right, src_top = transform_bounds(
                src.crs,
                dst_crs,
                *src.bounds,
                densify_pts=21,
            )
            if (
                src_right <= dst_left
                or src_left >= dst_right
                or src_top <= dst_bottom
                or src_bottom >= dst_top
            ):
                continue

            reproject(
                source=rasterio.band(src, 1),
                destination=wc_reproj,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=0,              # WorldCover nodata
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=dst_nodata,
                resampling=Resampling.nearest,
                init_dest_nodata=first_write,
                num_threads=num_threads,
            )
            first_write = False

    return wc_reproj

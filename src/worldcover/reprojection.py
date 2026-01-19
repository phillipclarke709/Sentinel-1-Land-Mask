from typing import Tuple

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


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

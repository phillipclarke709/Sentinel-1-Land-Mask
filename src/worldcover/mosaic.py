from pathlib import Path
from typing import List, Tuple

import rasterio
import numpy as np
from rasterio.merge import merge
from rasterio.warp import Resampling


def mosaic_worldcover_tiles(
    tile_paths: List[Path],
) -> Tuple[np.ndarray, rasterio.Affine]:
    """
    Load and mosaic ESA WorldCover tiles in geographic coordinates (EPSG:4326).

    Returns:
        wc_mosaic  : uint8 numpy array (2D)
        wc_transform : affine transform of the mosaic
    """
    datasets = [rasterio.open(p) for p in tile_paths]

    try:
        wc_mosaic, wc_transform = merge(
            datasets,
            resampling=Resampling.nearest,
        )
    finally:
        for ds in datasets:
            ds.close()

    # merge() returns (bands, rows, cols)
    wc_mosaic = wc_mosaic[0].astype("uint8")

    return wc_mosaic, wc_transform

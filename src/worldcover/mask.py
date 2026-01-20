from typing import Tuple
import numpy as np
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes


def build_land_mask(
    worldcover_array: np.ndarray,
    sar_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a boolean land mask from reprojected WorldCover data and
    apply it to a Sentinel-1 SAR array.
    """

    # Land = any non-zero class except open water (80); nodata is 0 or -1.
    land_mask = (
        (worldcover_array != 0)
        & (worldcover_array != 80)
        & (worldcover_array != -1)
    )

    # Morphological cleanup on land only to preserve coastal water.
    land_mask = binary_opening(land_mask, iterations=1)
    land_mask = binary_closing(land_mask, iterations=2)
    land_mask = binary_fill_holes(land_mask)

    sar_land_removed = sar_array.copy()
    sar_land_removed[land_mask] = np.nan

    return land_mask.astype(bool), sar_land_removed

from typing import List
import numpy as np
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes


def build_land_mask(
    wc_reproj: np.ndarray,
    land_classes: List[int],
    dst_nodata: int,
) -> np.ndarray:
    """
    Build a boolean land mask from reprojected WorldCover data
    and apply morphological cleanup.
    """

    land_mask = np.zeros(wc_reproj.shape, dtype=bool)

    valid_wc = wc_reproj != dst_nodata
    land_mask[valid_wc] = np.isin(wc_reproj[valid_wc], land_classes)

    # Morphological cleanup
    land_mask = binary_opening(land_mask, iterations=1)
    land_mask = binary_closing(land_mask, iterations=2)
    land_mask = binary_fill_holes(land_mask)

    return land_mask.astype("uint8")

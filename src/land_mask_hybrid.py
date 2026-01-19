print("Beginning script...")

# Standard Libraries
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.merge import merge
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes

# Other Self Made Modules
from worldcover.tiles import find_required_worldcover_tiles
from worldcover.mosaic import mosaic_worldcover_tiles
from worldcover.reprojection import reproject_worldcover_to_s1
DST_NODATA = -1

# =====================================================
# PATHS
# =====================================================
HH_PATH = Path(
    "data/input/Browser_images/"
    "2025-03-31-00_00_2025-03-31-23_59_"
    "Sentinel-1_IW_HH+HV_HH_(Raw).tiff"
)

WORLDCOVER_DIR = Path("data/worldcover/ESA_Worldcover")

OUT_MASK = Path("data/output/land_mask_worldcover_extended.tif")
OUT_IMG  = Path("data/output/S1_HH_landmasked_worldcover_extended.tif")

# WorldCover land classes (everything except open water = 80)
LAND_CLASSES = [10, 20, 30, 40, 50, 60, 70, 90, 95, 100]

# =====================================================
# LOAD SENTINEL-1 IMAGE
# =====================================================
print("Loading Sentinel-1 image...")
with rasterio.open(HH_PATH) as src:
    hh = src.read(1).astype("float32")
    profile = src.profile
    dst_crs = src.crs
    dst_transform = src.transform
    dst_shape = hh.shape

# =====================================================
# DERIVE AOI FROM VALID SAR DATA (CRITICAL FIX)
# =====================================================
print("Computing valid-data bounds from Sentinel-1...")

valid = np.isfinite(hh)
rows, cols = np.where(valid)

row_min, row_max = rows.min(), rows.max()
col_min, col_max = cols.min(), cols.max()

# Convert pixel indices → map coordinates
left,  top    = rasterio.transform.xy(dst_transform, row_min, col_min, offset="ul")
right, bottom = rasterio.transform.xy(dst_transform, row_max, col_max, offset="lr")

# Convert to lat/lon
west, south, east, north = transform_bounds(
    dst_crs, "EPSG:4326",
    left, bottom, right, top,
    densify_pts=21
)

print(f"AOI (WGS84): W={west:.2f}, S={south:.2f}, E={east:.2f}, N={north:.2f}")

# =====================================================
# WORLDCOVER TILE SELECTION (MODULE)
# =====================================================
print("Selecting required WorldCover tiles...")
WC_PATHS = find_required_worldcover_tiles(HH_PATH, WORLDCOVER_DIR)

print("Selected WorldCover tiles:")
for p in WC_PATHS:
    print(" ", p.name)

# =====================================================
# LOAD & MOSAIC WORLDCOVER (GEOGRAPHIC)
# =====================================================
print("Loading and mosaicking WorldCover tiles...")
wc_mosaic, wc_transform = mosaic_worldcover_tiles(WC_PATHS)

# =====================================================
# REPROJECT WORLDCOVER TO SENTINEL-1 GRID (SAFE)
# =====================================================
print("Reprojecting WorldCover mosaic to Sentinel-1 grid...")

wc_reproj = reproject_worldcover_to_s1(
    wc_mosaic,
    wc_transform,
    dst_transform,
    dst_crs,
    dst_shape,
)

valid_wc = wc_reproj != DST_NODATA

# =====================================================
# BUILD LAND MASK
# =====================================================
print("Building land mask...")
land_mask = np.zeros(dst_shape, dtype=bool)
valid_wc = wc_reproj != DST_NODATA
land_mask[valid_wc] = np.isin(wc_reproj[valid_wc], LAND_CLASSES)

# =====================================================
# MORPHOLOGICAL CLEANUP
# =====================================================
print("Applying morphological operations...")
land_mask = binary_opening(land_mask, iterations=1)
land_mask = binary_closing(land_mask, iterations=2)
land_mask = binary_fill_holes(land_mask)
land_mask = land_mask.astype("uint8")

# =====================================================
# SAVE LAND MASK
# =====================================================
print("Saving land mask...")
mask_profile = profile.copy()
mask_profile.update(dtype="uint8", nodata=0)

with rasterio.open(OUT_MASK, "w", **mask_profile) as dst:
    dst.write(land_mask, 1)

# =====================================================
# APPLY MASK TO SENTINEL-1
# =====================================================
print("Applying land mask to Sentinel-1 HH...")
hh_masked = np.where(land_mask == 1, np.nan, hh)

img_profile = profile.copy()
img_profile.update(dtype="float32", nodata=np.nan)

with rasterio.open(OUT_IMG, "w", **img_profile) as dst:
    dst.write(hh_masked, 1)

print("Extended WorldCover land mask complete.")

# =====================================================
# OPTIONAL VISUAL CHECK
# =====================================================
if __name__ == "__main__":
    print("Loading images for visual check...")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.title("WorldCover (reprojected)")
    plt.imshow(wc_reproj, cmap="tab20")

    plt.subplot(132)
    plt.title("Final Land Mask")
    plt.imshow(land_mask, cmap="gray")

    plt.subplot(133)
    plt.title("Masked Sentinel-1 HH")
    plt.imshow(hh_masked, cmap="gray")

    plt.tight_layout()
    plt.show()

print("Please close the plot window to end the script.")

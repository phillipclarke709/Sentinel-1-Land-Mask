import time

start_time = time.time()
print("Beginning script...")

# Standard Libraries
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from scipy.ndimage import binary_closing, binary_fill_holes, binary_dilation
 
# Other Self Made Modules
from worldcover.tiles import find_required_worldcover_tiles
from worldcover.reprojection import reproject_preprocessed_landmask_tiles_to_s1

# =====================================================
# PATHS
# =====================================================
HH_PATH = Path(
    "data/input/Browser_images(4)/2025-02-24-00_00_2025-02-24-23_59_Sentinel-1_EW_HH+HV_HH_(Raw).tiff"  #Path to your input HH image
)
HV_PATH = Path(
    "data/input/Browser_images(4)/2025-02-24-00_00_2025-02-24-23_59_Sentinel-1_EW_HH+HV_HV_(Raw).tiff"  #Path to your input HV image
)

WORLDCOVER_DIR = Path("data/worldcover/preprocessed") #Path to preprocessed WorldCover tiles directory

# =====================================================
# LOAD SENTINEL-1 IMAGE
# =====================================================
print("Loading Sentinel-1 HH image...")
with rasterio.open(HH_PATH) as src:
    hh = src.read(1).astype("float32")
    profile = src.profile
    dst_crs = src.crs
    dst_transform = src.transform
    dst_shape = hh.shape

# =====================================================
# DERIVE AREA OF INTEREST FROM VALID SAR DATA 
# =====================================================
print("Computing valid-data bounds from Sentinel-1...")

valid = np.isfinite(hh)
row_any = valid.any(axis=1)
col_any = valid.any(axis=0)

if not row_any.any() or not col_any.any():
    raise RuntimeError("No finite pixels found in the HH image.")

row_min = row_any.argmax()
row_max = len(row_any) - row_any[::-1].argmax() - 1
col_min = col_any.argmax()
col_max = len(col_any) - col_any[::-1].argmax() - 1

# Convert pixel indices to map coordinates
left,  top    = rasterio.transform.xy(dst_transform, row_min, col_min, offset="ul")
right, bottom = rasterio.transform.xy(dst_transform, row_max, col_max, offset="lr")

# Convert to lat/lon
west, south, east, north = transform_bounds(
    dst_crs, "EPSG:4326",
    left, bottom, right, top,
    densify_pts=21
)

print(f"AOI (WGS84): W={west:.2f}, S={south:.2f}, E={east:.2f}, N={north:.2f}")

_bounds_tag = f"W{west:.2f}_S{south:.2f}_E{east:.2f}_N{north:.2f}"
OUT_HH_IMG = Path(f"data/output/hh_masked_{_bounds_tag}.tif")
OUT_HV_IMG = Path(f"data/output/hv_masked_{_bounds_tag}.tif")

# =====================================================
# WORLDCOVER TILE SELECTION CALLS
# =====================================================
print("Selecting required WorldCover tiles...")
WC_PATHS = find_required_worldcover_tiles(
    HH_PATH,
    WORLDCOVER_DIR,
    bounds_wgs84=(west, south, east, north),
    filename_suffix="_preprocessed.tif",
)

print("Selected WorldCover tiles:")
for p in WC_PATHS:
    print(" ", p.name)

# =====================================================
# REPROJECT WORLDCOVER TILES TO SENTINEL-1 GRID
# =====================================================
print("Reprojecting WorldCover tiles to Sentinel-1 grid...")
land_mask = reproject_preprocessed_landmask_tiles_to_s1(
    WC_PATHS,
    dst_transform,
    dst_crs,
    dst_shape,
    num_threads=None,
)

nodata_count = np.sum(land_mask == 255)
if nodata_count == land_mask.size:
    raise RuntimeError(
        "WorldCover land mask is all nodata within the AOI. "
        "Check that the required tiles cover the scene bounds."
    )

# =====================================================
# CLEANING & BUFFING LAND MASK COASTLINE
# =====================================================

land_mask = land_mask == 1
land_mask = binary_closing(land_mask, iterations=1)
land_mask = binary_fill_holes(land_mask)
land_mask = binary_dilation(land_mask, iterations=2)

# =====================================================
# BUILD LAND MASK
# =====================================================
print("Applying land mask to Sentinel-1 HH...")
hh_masked = hh
hh_masked[land_mask] = np.nan

print("Loading Sentinel-1 HV image...")
with rasterio.open(HV_PATH) as src:
    if src.crs != dst_crs or src.transform != dst_transform or src.shape != dst_shape:
        raise RuntimeError("HV grid does not match HH (CRS/transform/shape).")
    hv = src.read(1).astype("float32")

hv_masked = hv
hv_masked[land_mask] = np.nan

# =====================================================
# APPLY MASK TO SENTINEL-1
# =====================================================
img_profile = profile.copy()
img_profile.update(dtype="float32", nodata=np.nan)

with rasterio.open(OUT_HH_IMG, "w", **img_profile) as dst:
    dst.write(hh_masked, 1)

with rasterio.open(OUT_HV_IMG, "w", **img_profile) as dst:
    dst.write(hv_masked, 1)

print("Extended WorldCover land mask complete.")
print(f"Script ran in {time.time() - start_time:.2f} seconds.")

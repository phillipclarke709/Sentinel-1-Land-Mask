print("Beginning script...")

# Standard Libraries
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform_bounds

# Other Self Made Modules
from worldcover.tiles import find_required_worldcover_tiles
from worldcover.mosaic import mosaic_worldcover_tiles
from worldcover.reprojection import reproject_worldcover_to_s1
from worldcover.mask import build_land_mask
DST_NODATA = -1

# =====================================================
# PATHS
# =====================================================
HH_PATH = Path(
    "data/input/Browser_images(1)/2025-03-28-00_00_2025-03-28-23_59_Sentinel-1_IW_HH+HV_HH_(Raw).tiff"  #Path to your input HH image
)
HV_PATH = Path(
    "data/input/Browser_images(1)/2025-03-28-00_00_2025-03-28-23_59_Sentinel-1_IW_HH+HV_HV_(Raw).tiff"  #Path to your input HV image
)

WORLDCOVER_DIR = Path("data/worldcover/ESA_Worldcover") #Path to your WorldCover tiles directory

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

print("Loading Sentinel-1 HV image...")
with rasterio.open(HV_PATH) as src:
    hv = src.read(1).astype("float32")

# =====================================================
# DERIVE AREA OF INTEREST FROM VALID SAR DATA 
# =====================================================
print("Computing valid-data bounds from Sentinel-1...")

valid = np.isfinite(hh)
rows, cols = np.where(valid)

row_min, row_max = rows.min(), rows.max()
col_min, col_max = cols.min(), cols.max()

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
WC_PATHS = find_required_worldcover_tiles(HH_PATH, WORLDCOVER_DIR)

print("Selected WorldCover tiles:")
for p in WC_PATHS:
    print(" ", p.name)

# =====================================================
# LOAD & MOSAIC WORLDCOVER
# =====================================================
print("Loading and mosaicking WorldCover tiles...")
wc_mosaic, wc_transform = mosaic_worldcover_tiles(WC_PATHS)

# =====================================================
# REPROJECT WORLDCOVER TO SENTINEL-1 GRID
# =====================================================
print("Reprojecting WorldCover mosaic to Sentinel-1 grid...")

wc_reproj = reproject_worldcover_to_s1(
    wc_mosaic,
    wc_transform,
    dst_transform,
    dst_crs,
    dst_shape,
)

print("wc_reproj min/max:", wc_reproj.min(), wc_reproj.max())
print("wc_reproj nodata count:", np.sum(wc_reproj == -1))
print("wc_reproj water count:", np.sum(wc_reproj == 80))

# =====================================================
# BUILD LAND MASK
# =====================================================
print("Building land mask and applying to Sentinel-1 HH...")
land_mask, hh_masked = build_land_mask(wc_reproj, hh)
print("land_mask true count:", np.sum(land_mask))
hv_masked = hv.copy()
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

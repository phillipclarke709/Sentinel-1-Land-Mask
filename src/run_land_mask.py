import time

from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from scipy.ndimage import binary_closing, binary_fill_holes, binary_dilation

from worldcover.tiles import find_required_worldcover_tiles
from worldcover.reprojection import reproject_preprocessed_landmask_tiles_to_s1

print("Beginning script...")

# =====================================================
# PATHS
# =====================================================
SMALL_SCENE_MPX = 10.0
MEDIUM_SCENE_MPX = 50.0

SCENES = [
    (
        Path("data/input/Browser_images/2025-03-31-00_00_2025-03-31-23_59_Sentinel-1_IW_HH+HV_HH_(Raw).tiff"),
        Path("data/input/Browser_images/2025-03-31-00_00_2025-03-31-23_59_Sentinel-1_IW_HH+HV_HV_(Raw).tiff"),
    ),
    (
        Path("data/input/Browser_images(1)/2025-03-28-00_00_2025-03-28-23_59_Sentinel-1_IW_HH+HV_HH_(Raw).tiff"),
        Path("data/input/Browser_images(1)/2025-03-28-00_00_2025-03-28-23_59_Sentinel-1_IW_HH+HV_HV_(Raw).tiff"),
    ),
    (
        Path("data/input/Browser_images(2)/2025-04-30-00_00_2025-04-30-23_59_Sentinel-1_IW_HH+HV_HH_(Raw).tiff"),
        Path("data/input/Browser_images(2)/2025-04-30-00_00_2025-04-30-23_59_Sentinel-1_IW_HH+HV_HV_(Raw).tiff"),
    ),
    (
        Path("data/input/Browser_images(3)/2025-03-29-00_00_2025-03-29-23_59_Sentinel-1_EW_HH+HV_HH_(Raw).tiff"),
        Path("data/input/Browser_images(3)/2025-03-29-00_00_2025-03-29-23_59_Sentinel-1_EW_HH+HV_HV_(Raw).tiff"),
     ),
    (
        Path("data/input/Browser_images(4)/2025-02-24-00_00_2025-02-24-23_59_Sentinel-1_EW_HH+HV_HH_(Raw).tiff"),
        Path("data/input/Browser_images(4)/2025-02-24-00_00_2025-02-24-23_59_Sentinel-1_EW_HH+HV_HV_(Raw).tiff"),
    ),
    (
        Path("data/input/rtc/ASF H/test_HH.tif"),
        Path("data/input/rtc/ASF H/test_HV.tif"),
    ),
]

WORLDCOVER_DIR = Path("data/worldcover/preprocessed") #Path to preprocessed WorldCover tiles directory

# =====================================================
# PROCESS SCENES
# =====================================================
total_start = time.time()
for idx, (HH_PATH, HV_PATH) in enumerate(SCENES, start=1):
    scene_start = time.time()
    print(f"Processing scene {idx}/{len(SCENES)}...")

    # -------------------------------------------------
    # LOAD SENTINEL-1 HH
    # -------------------------------------------------
    print("Loading Sentinel-1 HH image...")
    with rasterio.open(HH_PATH) as src:
        hh = src.read(1).astype("float32")
        profile = src.profile
        dst_crs = src.crs
        dst_transform = src.transform
        dst_shape = hh.shape
    print(f"HH shape: {dst_shape[0]} x {dst_shape[1]}")
    current_mpx = hh.size / 1_000_000
    res_x, res_y = src.res
    if current_mpx < SMALL_SCENE_MPX:
        size_label = "small"
    elif current_mpx < MEDIUM_SCENE_MPX:
        size_label = "medium"
    else:
        size_label = "large"

    # -------------------------------------------------
    # DERIVE AOI FROM VALID SAR DATA
    # -------------------------------------------------
    print("Computing valid-data bounds from Sentinel-1...")

    valid = np.isfinite(hh)
    row_any = valid.any(axis=1)
    col_any = valid.any(axis=0)

    if not row_any.any() or not col_any.any():
        print("No finite pixels found in the HH image. Skipping scene.")
        continue

    row_min = row_any.argmax()
    row_max = len(row_any) - row_any[::-1].argmax() - 1
    col_min = col_any.argmax()
    col_max = len(col_any) - col_any[::-1].argmax() - 1

    left,  top    = rasterio.transform.xy(dst_transform, row_min, col_min, offset="ul")
    right, bottom = rasterio.transform.xy(dst_transform, row_max, col_max, offset="lr")

    west, south, east, north = transform_bounds(
        dst_crs, "EPSG:4326",
        left, bottom, right, top,
        densify_pts=21
    )

    print(f"AOI (WGS84): W={west:.2f}, S={south:.2f}, E={east:.2f}, N={north:.2f}")
    if dst_crs.is_geographic:
        lat_center = (south + north) / 2.0
        res_x_m = abs(res_x) * 111320.0 * np.cos(np.deg2rad(lat_center))
        res_y_m = abs(res_y) * 110574.0
        res_m = max(res_x_m, res_y_m)
    else:
        res_m = max(abs(res_x), abs(res_y))
    res_m = round(res_m / 10.0) * 10.0
    if size_label == "small":
        runtime_note = "<20s"
    elif size_label == "medium":
        runtime_note = "<60s"
    else:
        runtime_note = "<3min"
    print(f"Resolution is ~{res_m:.0f}m; expect {runtime_note} runtime.")

    _bounds_tag = f"W{west:.2f}_S{south:.2f}_E{east:.2f}_N{north:.2f}"
    OUT_HH_IMG = Path(f"data/output/hh_masked_{_bounds_tag}.tif")
    OUT_HV_IMG = Path(f"data/output/hv_masked_{_bounds_tag}.tif")

    # -------------------------------------------------
    # WORLDCOVER TILE SELECTION
    # -------------------------------------------------
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

    # -------------------------------------------------
    # REPROJECT WORLDCOVER TO SENTINEL-1 GRID
    # -------------------------------------------------
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
        print(
            "WorldCover land mask is all nodata within the AOI. "
            "Check that the required tiles cover the scene bounds."
        )
        continue

    # -------------------------------------------------
    # CLEANING & BUFFING LAND MASK COASTLINE
    # -------------------------------------------------
    land_mask = land_mask == 1
    land_mask = binary_closing(land_mask, iterations=1)
    land_mask = binary_fill_holes(land_mask)
    land_mask = binary_dilation(land_mask, iterations=2)

    # -------------------------------------------------
    # APPLY MASK TO SENTINEL-1 HH
    # -------------------------------------------------
    print("Applying land mask to Sentinel-1 HH...")
    hh_masked = hh
    hh_masked[land_mask] = np.nan

    # -------------------------------------------------
    # LOAD & APPLY MASK TO SENTINEL-1 HV
    # -------------------------------------------------
    print("Loading Sentinel-1 HV image...")
    with rasterio.open(HV_PATH) as src:
        if src.crs != dst_crs or src.transform != dst_transform or src.shape != dst_shape:
            print("HV grid does not match HH (CRS/transform/shape). Skipping scene.")
            continue
        hv = src.read(1).astype("float32")

    hv_masked = hv
    hv_masked[land_mask] = np.nan

    # -------------------------------------------------
    # WRITE OUTPUTS
    # -------------------------------------------------
    img_profile = profile.copy()
    img_profile.update(dtype="float32", nodata=np.nan)

    with rasterio.open(OUT_HH_IMG, "w", **img_profile) as dst:
        dst.write(hh_masked, 1)

    with rasterio.open(OUT_HV_IMG, "w", **img_profile) as dst:
        dst.write(hv_masked, 1)

    print("Extended WorldCover land mask complete.")
    scene_seconds = time.time() - scene_start
    print(f"Scene ran in {scene_seconds:.2f} seconds.")
print(f"Script ran in {time.time() - total_start:.2f} seconds.")

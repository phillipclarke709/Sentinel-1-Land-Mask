import time

from pathlib import Path
import tkinter as tk
from tkinter import filedialog
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

def prompt_for_scene_file() -> Path:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected_file = filedialog.askopenfilename(
        title="Select Input Scene",
        filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")],
    )
    root.destroy()

    if not selected_file:
        raise RuntimeError("No input file selected.")

    return Path(selected_file)

WORLDCOVER_DIR = Path("data/worldcover/preprocessed") #Path to preprocessed WorldCover tiles directory

# =====================================================
# PROCESS SCENES
# =====================================================
try:
    scene_path = prompt_for_scene_file()
except RuntimeError as exc:
    print(str(exc))
    raise SystemExit(0)

run_output_dir = Path("data/output") / time.strftime("%Y%m%d_%Hh%Mm")
run_output_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving outputs to: {run_output_dir}")
SCENES = [scene_path]
for idx, SCENE_PATH in enumerate(SCENES, start=1):
    scene_start = time.time()
    print(f"Processing scene {idx}/{len(SCENES)}...")

    # -------------------------------------------------
    # LOAD SCENE
    # -------------------------------------------------
    print("Loading scene image...")
    with rasterio.open(SCENE_PATH) as src:
        scene_data = src.read(1).astype("float32")
        profile = src.profile
        dst_crs = src.crs
        dst_transform = src.transform
        dst_shape = scene_data.shape
        res_x, res_y = src.res
    print(f"Scene shape: {dst_shape[0]} x {dst_shape[1]}")
    current_mpx = scene_data.size / 1_000_000
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

    valid = np.isfinite(scene_data)
    row_any = valid.any(axis=1)
    col_any = valid.any(axis=0)

    if not row_any.any() or not col_any.any():
        print("No finite pixels found in the scene image. Skipping scene.")
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

    scene_stem = SCENE_PATH.stem
    out_scene_img = run_output_dir / f"{scene_stem}_masked{SCENE_PATH.suffix}"

    # -------------------------------------------------
    # WORLDCOVER TILE SELECTION
    # -------------------------------------------------
    print("Selecting required WorldCover tiles...")
    WC_PATHS = find_required_worldcover_tiles(
        SCENE_PATH,
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
    # APPLY MASK TO SCENE
    # -------------------------------------------------
    print("Applying land mask to scene image...")
    scene_masked = scene_data.copy()
    scene_masked[land_mask] = np.nan

    # -------------------------------------------------
    # WRITE OUTPUT
    # -------------------------------------------------
    img_profile = profile.copy()
    img_profile.update(dtype="float32", nodata=np.nan)

    with rasterio.open(out_scene_img, "w", **img_profile) as dst:
        dst.write(scene_masked, 1)
    print(f"Saved output to: {out_scene_img}")

    print("Extended WorldCover land mask complete.")
    scene_seconds = time.time() - scene_start
    print(f"Scene ran in {scene_seconds:.2f} seconds.")

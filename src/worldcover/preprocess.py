from pathlib import Path

import numpy as np
import rasterio


def preprocess_worldcover_tiles(
    input_dir: Path,
    output_dir: Path,
) -> None:
    """
    Convert WorldCover class tiles to land masks at native resolution.

    Land = 1, water = 0, nodata = 255.
    Output filenames preserve bounds and add a '_preprocessed' suffix.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = sorted(input_dir.glob("ESA_WorldCover_10m_2021_V200_*_Map.tif"))
    total = len(tile_paths)

    for idx, tile_path in enumerate(tile_paths, start=1):
        out_name = tile_path.name.replace("_Map.tif", "_preprocessed.tif")
        out_path = output_dir / out_name
        print(f"[{idx}/{total}] Preprocessing {tile_path.name} -> {out_name}")

        with rasterio.open(tile_path) as src:
            data = src.read(1)
            profile = src.profile

        land_mask = np.zeros(data.shape, dtype="uint8")
        land_mask[data == 0] = 255
        land_mask[(data != 0) & (data != 80)] = 1

        profile.update(dtype="uint8", nodata=255, count=1)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(land_mask, 1)
        print(f"[{idx}/{total}] Saved {out_name}")


if __name__ == "__main__":
    INPUT_DIR = Path("data/worldcover/ESA_Worldcover")
    OUTPUT_DIR = Path("data/worldcover/preprocessed")

    preprocess_worldcover_tiles(INPUT_DIR, OUTPUT_DIR)

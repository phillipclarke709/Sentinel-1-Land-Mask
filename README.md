# Sentinel-1 Land Masking with ESA WorldCover

This project provides a Python-based workflow to generate accurate land masks for
Sentinel-1 SAR imagery using ESA WorldCover as a static land prior, with correct
handling of SAR geometry and geocoding.

## Features
- Works with geocoded Sentinel-1 GRD / RTC imagery
- Automatic selection of required ESA WorldCover tiles
- Correct reprojection and nodata handling
- Preserves true Sentinel-1 acquisition geometry
- Suitable for Arctic and coastal environments

## Requirements
- Python 3.9+
- rasterio
- numpy
- scipy
- matplotlib

## Data (not included)
This repository does not include:
- Sentinel-1 imagery
- ESA WorldCover tiles

These must be downloaded separately.

## Usage
```bash
python src/land_mask_hyrbid.py

#!/usr/bin/env python
# coding: utf-8
"""
Step 1 — Preprocessing: Sentinel-2 reprojection, ortho clipping, band correlation.

For each study area this script:
  1. Reprojects Sentinel-2 (S2) bands to SWEREF99 TM (EPSG:3006)
  2. Clips S2 bands and orthophoto bands to the study area polygon
  3. Creates a 10 m fishnet grid aligned to S2 pixels
  4. Aggregates orthophoto bands to 10 m via zonal statistics (to match S2 resolution)
  5. Creates a mask excluding snow, water, shadow and bright pixels
  6. Runs Ordinary Least Squares (OLS) regression between S2 and ortho bands
     (NIR, Red, Green) — coefficients are used in Calibration_NDVI.py

Requires: arcpy (ArcGIS Pro with Spatial Analyst and Image Analyst extensions)
"""

import os

os.chdir(r"C:\TEMP\Vanessa_Henriksson")
print(os.getcwd())

import pandas as pd
import geopandas as gpd
from pathlib import Path
from arcgis_functions import *

# Allow overwriting existing output files
arcpy.env.overwriteOutput = True

shp_path = "Data/proglacial_outlines.shp"

# Load study area polygons (one row per glacier proglacial area)
study_areas = gpd.read_file(shp_path)

# Drop areas not used in this analysis (Stuorra, Laevas, Helags)
study_areas = study_areas.drop(index=[8, 12, 16])

# Load lookup table linking glacier names to input data file paths
# (orthophoto path, S2 band paths, sun azimuth/altitude for hillshade)
lookup = pd.read_csv("Data/Python/lookup_table.csv")
lookup.head()

# Join study area geometries with input file paths on glacier name
studarea_merge = study_areas.merge(
    lookup,
    left_on="Glacier_na",   # column in study areas shapefile
    right_on="Glacier_name",  # matching column in lookup table
    how="left"
)

studarea_merge.head()

# S2 band identifiers: Green (B3), Red (B4), NIR (B8), SWIR (B11), Scene Classification (SCL)
s2_bands = ["s2_b3", "s2_b4", "s2_b8", "s2_b11", "s2_SCL"]
# Orthophoto bands: Band_1=Red, Band_2=Green, Band_3=Blue, Band_4=NIR
ortho_bands = ["Band_1", "Band_2", "Band_3", "Band_4"]

# Dict to cache reprojected S2 raster paths (avoids reprojecting the same tile twice)
s2_band_reprojected_dict = {}
all_coefficients = []  # accumulates OLS results across all areas and bands

# Output folder for reprojected S2 rasters (shared across all study areas)
reprojections_dir = Path(f"Data/Python/Outputs/s2_reprojections")
reprojections_dir.mkdir(parents=True, exist_ok=True)

# Reproject all unique S2 tiles to SWEREF99 TM once before the main loop
# (multiple study areas may share the same S2 tile)
for band in s2_bands:
    unique_band_list = studarea_merge[band].unique()
    print(f"Found {unique_band_list.size} unique bands for {band}")
    for idx, band_path in enumerate(unique_band_list):
        print(band_path)
        s2_band_reprojected_dict[band_path] = reproject_s2(band_path, idx, band)
        print(f"Reprojecting {band} number {idx}")

# ── Main loop: process each study area ────────────────────────────────────────
for area in studarea_merge.itertuples():
    area_name = area.Glacier_na

    # Reset per-area path variables
    s2_NIR_clipped_path = None
    s2_Green_clipped_path = None
    s2_Red_clipped_path = None
    s2_SWIR_clipped_path = None
    s2_SCL_clipped_path = None

    s2_NIR_reproj_path = None
    s2_Green_reproj_path = None
    s2_Red_reproj_path = None
    s2_SWIR_reproj_path = None
    s2_SCL_reproj_path = None

    s2_fishnet_path = None  # will be set once from first S2 band clip

    ortho_Red_clipped_path = None
    ortho_Green_clipped_path = None
    ortho_Blue_clipped_path = None
    ortho_NIR_clipped_path = None

    geom = area.geometry
    out_dir = Path(f"Data/Python/Outputs/{area_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract single-area polygon as a shapefile for masking/clipping operations
    select_area_shp_path = select_area_shp(area_name, shp_path)

    # ── Clip the full orthophoto (all 4 bands) to this study area ─────────────
    ortho_src_path = getattr(area, "ortho")
    clipped_ortho_path = clip_ortho(area_name, select_area_shp_path, ortho_src_path)

    # ── Clip each S2 band and create a 10 m fishnet grid ──────────────────────
    for band in s2_bands:
        src_path = getattr(area, band)
        s2_reproj_path = s2_band_reprojected_dict[src_path]
        # clip_s2 returns the clipped raster path and a fishnet shapefile path
        s2_clipped_path, s2_fishnet_shp_path = clip_s2(area_name, s2_reproj_path, band, select_area_shp_path)

        # Keep only the first fishnet (they are all identical in extent/cell size)
        if s2_fishnet_path is None:
            s2_fishnet_path = s2_fishnet_shp_path

        # Store clipped + reprojected paths by band name for later use
        if band == "s2_b8":
            s2_NIR_clipped_path = s2_clipped_path
            s2_NIR_reproj_path = s2_reproj_path
        elif band == "s2_b3":
            s2_Green_clipped_path = s2_clipped_path
            s2_Green_reproj_path = s2_reproj_path
        elif band == "s2_b11":
            s2_SWIR_clipped_path = s2_clipped_path
            s2_SWIR_reproj_path = s2_reproj_path
        elif band == "s2_b4":
            s2_Red_clipped_path = s2_clipped_path
            s2_Red_reproj_path = s2_reproj_path
        elif band == "s2_SCL":
            s2_SCL_clipped_path = s2_clipped_path
            s2_SCL_reproj_path = s2_reproj_path

        print(f"{band} clipped & reprojected for {area_name}")

    # ── Aggregate each ortho band to 10 m using S2 fishnet as zonal zones ─────
    # The fishnet cells align to S2 pixels; zonal mean gives the average ortho
    # reflectance within each S2-sized cell, enabling a like-for-like comparison.
    for ortho_band in ortho_bands:
        ortho_band_path = f"{ortho_src_path}\\{ortho_band}"
        if ortho_band == "Band_1":   # Red
            ortho_Red_clipped_path = clip_ortho_band(area_name, select_area_shp_path, s2_Red_reproj_path, s2_fishnet_path, s2_Red_clipped_path, ortho_band, ortho_band_path)
        elif ortho_band == "Band_2":  # Green
            ortho_Green_clipped_path = clip_ortho_band(area_name, select_area_shp_path, s2_Green_reproj_path, s2_fishnet_path, s2_Green_clipped_path, ortho_band, ortho_band_path)
        elif ortho_band == "Band_3":  # Blue
            ortho_Blue_clipped_path = clip_ortho_band(area_name, select_area_shp_path, s2_NIR_reproj_path, s2_fishnet_path, s2_NIR_clipped_path, ortho_band, ortho_band_path)
        elif ortho_band == "Band_4":  # NIR
            ortho_NIR_clipped_path = clip_ortho_band(area_name, select_area_shp_path, s2_NIR_reproj_path, s2_fishnet_path, s2_NIR_clipped_path, ortho_band, ortho_band_path)

        print(f"Orthophoto clipped for {ortho_band} in {area_name} area")

    # ── Create a quality mask to exclude unreliable pixels ────────────────────
    # Masks out: snow (high NDSI), water (positive NDWI), shadow (low brightness
    # or SCL class 2), and very bright surfaces.
    # Returns an inverted mask where valid pixels = 1, masked pixels = NoData.
    area_inverted_mask_path = create_mask(area_name, s2_NIR_clipped_path, s2_Green_clipped_path, s2_SWIR_clipped_path, s2_SCL_clipped_path, ortho_Red_clipped_path, ortho_Green_clipped_path, ortho_Blue_clipped_path)

    print(f"Mask created for {area_name} area")

    # ── OLS band correlation: ortho band ~ S2 band ────────────────────────────
    # For each band pair, samples pixel values at S2 fishnet centroids (masked),
    # normalises both (ortho /255, S2 (DN-1000)/10000 → reflectance), then fits
    # an OLS line: S2_reflectance = k * ortho_normalised + m
    # The slope (k) and intercept (m) are used in Calibration_NDVI.py to convert
    # ortho DN values into pseudo-reflectance comparable to S2.

    all_coefficients.append(band_correlation(area_name,
                     select_area_shp_path,
                     s2_fishnet_path,
                     area_inverted_mask_path,
                     s2_NIR_clipped_path,
                     "s2_b8",
                     ortho_NIR_clipped_path,
                     "Band_4"))

    all_coefficients.append(band_correlation(area_name,
                     select_area_shp_path,
                     s2_fishnet_path,
                     area_inverted_mask_path,
                     s2_Red_clipped_path,
                     "s2_b4",
                     ortho_Red_clipped_path,
                     "Band_1"))

    all_coefficients.append(band_correlation(area_name,
                     select_area_shp_path,
                     s2_fishnet_path,
                     area_inverted_mask_path,
                     s2_Green_clipped_path,
                     "s2_b3",
                     ortho_Green_clipped_path,
                     "Band_2"))

    print(f"Band correlations done in {area_name}!")

# ── Export all OLS coefficients to CSV ────────────────────────────────────────
coefficients_csv_path = "Data/Python/Outputs/OLS_coefficients_summary.csv"
pd.DataFrame(all_coefficients).to_csv(coefficients_csv_path, index=False)
print(f"Coefficient summary saved to {coefficients_csv_path}")

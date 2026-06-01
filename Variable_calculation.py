"""
Step 4b — Main script for 1 m terrain variable calculation and sampling.

This script orchestrates the terrain variable derivation pipeline for each
study area at 1 m resolution. It calls the helper functions from High_res_script.py:

  1. mosaic_dem         — merge LiDAR DEM tiles, clip to study area, fill sinks
  2. calculate_variables — derive slope, aspect (sin/cos), hillshade, landforms,
                           distance from glacier, and profile curvature
  3. sample_areas       — resample vegetation to 1 m, create stratified sample
                          points, and extract all variable values at those points

The resulting per-area sample shapefile ({area_name}_samples.shp) contains
one row per sample point with columns for each terrain variable plus the
vegetation class label. This file is consumed by RF_block_test.py.

Lookup table (lookup_table_gd.csv) provides:
  - sun_azim, sun_alt: sun position at image acquisition (for hillshade)

Requires: arcpy (ArcGIS Pro with Spatial Analyst and Image Analyst extensions)
"""

import os

os.chdir(r"C:\TEMP\Vanessa_Henriksson")
print(os.getcwd())

import pandas as pd
import geopandas as gpd
from pathlib import Path
from arcgis_functions import *
from High_res_script import *

# Allow overwriting existing output files
arcpy.env.overwriteOutput = True

shp_path = "Data/proglacial_outlines.shp"
glacier_shp = "Data/glacier_polygon.shp"  # export from .gdb to .shp before running

# Load study area polygons
study_areas = gpd.read_file(shp_path)

# Drop areas not included in this analysis
study_areas = study_areas.drop(index=[12, 14])

# Load lookup table which provides sun azimuth and altitude per area
# (needed for area-specific hillshade calculation)
lookup = pd.read_csv("Data/Python/lookup_table_gd.csv")

# Join study areas with sun parameters on glacier name
studarea_merge = study_areas.merge(
    lookup,
    left_on="Glacier_na",
    right_on="Glacier_name",
    how="left"
)

# ── Main loop: process terrain variables for each study area ──────────────────
for area in studarea_merge.itertuples():
    area_name = area.Glacier_na
    # Sun position at the time the orthophoto was acquired — from lookup table
    sun_azim = area.sun_azim  # compass bearing of sun (degrees)
    sun_alt  = area.sun_alt   # elevation angle of sun above horizon (degrees)

    out_dir = Path(f"Data/Python/Outputs/{area_name}")
    geodiv_out_dir = Path(f"Data/Python/Outputs/{area_name}/Geodiversity")
    geodiv_out_dir.mkdir(parents=True, exist_ok=True)

    # Isolate this area's polygon as a standalone shapefile for ArcPy masking
    select_area_shp_path = select_area_shp(area_name, shp_path)

    print(f"Processing: {area_name}")

    # Step 1: Merge LiDAR tiles → clip → fill sinks → returns merged DEM path
    merged_dem_path = mosaic_dem(select_area_shp_path, area_name, str(geodiv_out_dir))
    print("Merged, clipped and filled DEM.")

    # Step 2: Derive all terrain variables from the merged DEM
    calculate_variables(area_name, str(geodiv_out_dir), merged_dem_path, sun_azim, sun_alt, select_area_shp_path, glacier_shp)
    print("Variables calculated.")

    # Step 3: Create stratified sample points and extract variable values
    sample_areas(str(geodiv_out_dir), area_name, merged_dem_path)
    print("Study area sampled.")

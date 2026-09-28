"""
Step 4b — Terrain variable calculation at 1 m from the DSM/DEM.

This script orchestrates terrain variable derivation for each study area at
1 m resolution (configurable via terrain_variables.dem_resample_target_m). It
calls the helper functions from High_res_script.py:

  1. mosaic_dem         — merge DSM/DEM tiles, resample to the target resolution
                          (e.g. from 0.5 m DSM tiles to 1 m), clip to study area,
                          fill sinks
  2. calculate_variables — derive slope, aspect (sin/cos), hillshade, landforms,
                           distance from glacier, and profile curvature

Sampling (extracting variable + TRI/SWI + predicted-vegetation values at
sample points, producing {area_name}_samples.shp) has been split out into
Sample_areas_1m.py, so terrain variables can be computed — and inspected —
independently of Veg_RF_polygon_cv.py's predicted vegetation raster. Run
Sample_areas_1m.py after both this script and Veg_RF_polygon_cv.py have been
run for the areas you want sampled.

Lookup table (paths.lookup_table_gd) provides:
  - sun_azim, sun_alt: sun position at image acquisition (for hillshade)

All paths, the study-area exclusion list, and the geomorphon search radius
are read from config.yaml.

Requires: arcpy (ArcGIS Pro with Spatial Analyst and Image Analyst extensions), PyYAML
"""

import os

import pandas as pd
import geopandas as gpd
from pathlib import Path
from arcgis_functions import *
from High_res_script import *

from config_utils import load_config, resolve_path

cfg = load_config()
os.chdir(cfg.paths.base_dir)
print(os.getcwd())

# Allow overwriting existing output files
arcpy.env.overwriteOutput = True

shp_path = cfg.paths.outlines_shp
glacier_shp = cfg.paths.glacier_shp  # export from .gdb to .shp before running
dem_tiles_dir = resolve_path(cfg, cfg.paths.dem_tiles_dir)
glacier_polygon_scratch_path = resolve_path(cfg, cfg.paths.glacier_polygon_scratch_gdb)
outputs_1m_dir = cfg.paths.outputs_1m

dem_resample_target_m = cfg.terrain_variables.dem_resample_target_m  # target resolution (m) for the mosaicked DSM/DEM
geomorphon_search_radius = dem_resample_target_m * cfg.terrain_variables.geomorphon_search_radius_multiplier  # metres

# Load study area polygons
study_areas = gpd.read_file(shp_path)

# Drop areas not included in this analysis
study_areas = study_areas.drop(index=cfg.study_areas.exclude_indices)
if cfg.study_areas.only:
    study_areas = study_areas[study_areas["Glacier_na"].isin(cfg.study_areas.only)]

# Load lookup table which provides sun azimuth and altitude per area
# (needed for area-specific hillshade calculation)
lookup = pd.read_csv(cfg.paths.lookup_table_gd)

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

    out_dir = Path(f"{outputs_1m_dir}/{area_name}")
    geodiv_out_dir = Path(f"{outputs_1m_dir}/{area_name}/Geodiversity")
    geodiv_out_dir.mkdir(parents=True, exist_ok=True)

    # Isolate this area's polygon as a standalone shapefile for ArcPy masking
    select_area_shp_path = select_area_shp(area_name, shp_path, outputs_1m_dir)

    print(f"Processing: {area_name}")

    # Step 1: Merge DSM/DEM tiles → resample to target resolution → clip → fill sinks
    merged_dem_path = mosaic_dem(select_area_shp_path, area_name, str(geodiv_out_dir), dem_tiles_dir,
                                 target_resolution_m=dem_resample_target_m)
    print("Merged, resampled, clipped and filled DEM.")

    # Step 2: Derive all terrain variables from the merged DEM
    calculate_variables(area_name, str(geodiv_out_dir), merged_dem_path, sun_azim, sun_alt,
                        select_area_shp_path, glacier_shp, glacier_polygon_scratch_path,
                        geomorphon_search_radius)
    print("Variables calculated.")

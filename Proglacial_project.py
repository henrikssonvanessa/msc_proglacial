#!/usr/bin/env python
# coding: utf-8
"""
Step 0 — Orthophoto preprocessing: clip source orthophotos to study areas.

For each study area, clips the full 4-band source orthophoto (path from the
'ortho' column in paths.lookup_table_ortho) to the study area polygon,
producing {area_name}_ortho_clip.tif under paths.outputs_1m/{area_name}/ —
the file Veg_RF_polygon_cv.py trains and predicts from. Run this before
Veg_RF_polygon_cv.py whenever source orthophotos change or a new study area
is added.

This used to also handle Sentinel-2 reprojection/clipping and an OLS
ortho-vs-S2 band correlation step (for calibrating ortho NDVI against
Sentinel-2 NDVI). That preprocessing has been dropped from this script —
orthophoto clipping is the only output any current script in this repo
depends on.

Requires: arcpy (ArcGIS Pro with Spatial Analyst extension), PyYAML
"""

import os

import pandas as pd
import geopandas as gpd
from pathlib import Path
from arcgis_functions import *

from config_utils import load_config, resolve_path

cfg = load_config()
os.chdir(cfg.paths.base_dir)
print(os.getcwd())

# Allow overwriting existing output files
arcpy.env.overwriteOutput = True

shp_path = cfg.paths.outlines_shp
outputs_1m_dir = cfg.paths.outputs_1m

# Load study area polygons
study_areas = gpd.read_file(shp_path)
study_areas = study_areas.drop(index=cfg.study_areas.exclude_indices)
if cfg.study_areas.only:
    study_areas = study_areas[study_areas["Glacier_na"].isin(cfg.study_areas.only)]

# Load lookup table linking glacier names to orthophoto source paths
lookup = pd.read_csv(resolve_path(cfg, cfg.paths.lookup_table_ortho))

# Join study area geometries with orthophoto source paths on glacier name
studarea_merge = study_areas.merge(
    lookup,
    left_on="Glacier_na",     # column in study areas shapefile
    right_on="Glacier_name",  # matching column in lookup table
    how="left"
)

# ── Main loop: clip the orthophoto for each study area ─────────────────────────
for area in studarea_merge.itertuples():
    area_name = area.Glacier_na
    ortho_src_path = area.ortho

    out_dir = Path(f"{outputs_1m_dir}/{area_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract single-area polygon as a shapefile for masking/clipping
    select_area_shp_path = select_area_shp(area_name, shp_path, outputs_1m_dir)

    clip_ortho(area_name, select_area_shp_path, ortho_src_path, outputs_1m_dir)
    print(f"Orthophoto clipped for {area_name}")

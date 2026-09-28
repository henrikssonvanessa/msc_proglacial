"""
Step 4c — Sample terrain variables + TRI/SWI + predicted vegetation at 1 m.

Split out from Variable_calculation.py so terrain variables can be computed
(and inspected) independently of the predicted vegetation raster — this is
the only script in the 1 m pipeline that needs Veg_RF_polygon_cv.py's output.
Run this after both Variable_calculation.py (terrain variables) and
Veg_RF_polygon_cv.py (predicted vegetation) have been run for the areas you
want sampled.

For each study area, calls High_res_script.py's sample_areas():
  - Resamples the predicted vegetation raster to the target analysis
    resolution and adds it, along with TRI/SWI (each skippable via config),
    to the terrain variable rasters Variable_calculation.py already produced
  - Creates stratified sample points and extracts all variable values,
    producing {area_name}_samples.shp — the input RF_block_test.py reads

All paths, TRI/SWI toggles, and sampling settings are read from config.yaml.

Requires: arcpy (ArcGIS Pro with Spatial Analyst and Image Analyst extensions), PyYAML
"""

import os

import arcpy
import geopandas as gpd
from pathlib import Path
from High_res_script import sample_areas

from config_utils import load_config, resolve_path

cfg = load_config()
os.chdir(cfg.paths.base_dir)
print(os.getcwd())

# Allow overwriting existing output files
arcpy.env.overwriteOutput = True

shp_path = cfg.paths.outlines_shp
arcgis_toolbox_path = cfg.paths.arcgis_toolbox_data_management
predicted_vegetation_dir = resolve_path(cfg, cfg.paths.predicted_vegetation_dir)
tri_swi_1m_dir = resolve_path(cfg, cfg.paths.tri_swi_1m_dir)
outputs_1m_dir = cfg.paths.outputs_1m
dem_resample_target_m = cfg.terrain_variables.dem_resample_target_m

# Load study area polygons
study_areas = gpd.read_file(shp_path)
study_areas = study_areas.drop(index=cfg.study_areas.exclude_indices)
if cfg.study_areas.only:
    study_areas = study_areas[study_areas["Glacier_na"].isin(cfg.study_areas.only)]

# ── Main loop: sample each study area ──────────────────────────────────────────
for area in study_areas.itertuples():
    area_name = area.Glacier_na

    geodiv_out_dir = Path(f"{outputs_1m_dir}/{area_name}/Geodiversity")
    # The 1 m DEM produced by Variable_calculation.py's mosaic_dem() — must
    # already exist; run Variable_calculation.py first if it doesn't.
    merged_dem_path = str(geodiv_out_dir / f"{area_name}_DEM.tif")

    print(f"Processing: {area_name}")

    # TRI/SWI are pre-computed elsewhere (e.g. QGIS/SAGA) — not derived by this script
    tri_file_path = f"{tri_swi_1m_dir}/{cfg.geodiversity.tri_filename_1m.format(area=area_name)}"
    swi_file_path = f"{tri_swi_1m_dir}/{cfg.geodiversity.swi_filename_1m.format(area=area_name)}"
    sample_areas(str(geodiv_out_dir), area_name, merged_dem_path, arcgis_toolbox_path,
                predicted_vegetation_dir, tri_file_path, swi_file_path,
                cfg.high_res_sampling.sample_fraction, cfg.high_res_sampling.sample_max_per_class,
                cfg.high_res_sampling.min_sample_distance_m,
                use_tri=cfg.features.use_tri, use_swi=cfg.features.use_swi,
                target_resolution_m=dem_resample_target_m)
    print("Study area sampled.")

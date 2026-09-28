"""
Step 5b — Sample terrain variables + TRI/SWI + snow + predicted vegetation at 20 m.

Split out from Low_res_script.py so terrain variables can be computed (and
inspected) independently of the predicted vegetation raster — this is the
only script in the 20 m pipeline that needs Veg_RF_polygon_cv.py's output.
Run this after both Low_res_script.py (terrain variables) and
Veg_RF_polygon_cv.py (predicted vegetation) have been run for the areas you
want sampled.

For each study area:
  1. Aggregates the 0.4 m binary vegetation raster to the target resolution
     (vegetation fraction), applies a threshold to create a binary
     classification (vegetated cell = fraction >= low_res.veg_fraction_threshold)
  2. Creates stratified sample points and extracts all terrain variables
     (produced by Low_res_script.py) plus TRI, SWI, and snow cover — each
     skippable via config — at those locations, producing
     {area_name}_samples.shp — the input rf_vegetation_20m.py reads

TRI and SWI are loaded from pre-computed rasters rather than derived from the
DEM, because they were computed by SAGA GIS using algorithms not available in
ArcPy. Snow cover is a Sentinel-2-derived fraction raster.

All paths, the vegetation threshold, the snow-file lookup, the TRI/SWI/snow
toggles, and the sampling settings are read from config.yaml.

Requires: arcpy (ArcGIS Pro with Spatial Analyst and Image Analyst extensions), PyYAML
"""

import os

import arcpy
from arcpy.sa import *
import numpy as np
import geopandas as gpd
from pathlib import Path

from config_utils import load_config, resolve_path

cfg = load_config()
os.chdir(cfg.paths.base_dir)
print(os.getcwd())

# Allow overwriting existing output files
arcpy.env.overwriteOutput = True

# ── CONFIG ────────────────────────────────────────────────────────────────────
USE_TRI  = cfg.features.use_tri
USE_SWI  = cfg.features.use_swi
USE_SNOW = cfg.features.use_snow
TARGET_RESOLUTION = cfg.low_res.target_resolution    # output pixel size in metres
VEG_THRESHOLD     = cfg.low_res.veg_fraction_threshold  # min vegetation fraction to classify a cell as vegetated
SNOW_DIR          = resolve_path(cfg, cfg.paths.snow_cover_dir)
TRI_SWI_DIR       = resolve_path(cfg, cfg.paths.tri_swi_20m_dir)
ARCGIS_TOOLBOX_PATH = cfg.paths.arcgis_toolbox_data_management
PREDICTED_VEGETATION_DIR = resolve_path(cfg, cfg.paths.predicted_vegetation_dir)
OUTPUTS_20M_DIR   = cfg.paths.outputs_20m
# ─────────────────────────────────────────────────────────────────────────────

# Lookup: which Sentinel-2 snow cover tile covers each study area
SNOW_FILE_LOOKUP = vars(cfg.snow_file_lookup)

shp_path = cfg.paths.outlines_shp
study_areas = gpd.read_file(shp_path)
study_areas = study_areas.drop(index=cfg.study_areas.exclude_indices)
if cfg.study_areas.only:
    study_areas = study_areas[study_areas["Glacier_na"].isin(cfg.study_areas.only)]


def sample_areas_lowres(geodiv_out_dir, area_name, target_resolution, resampled_dem_path, snow_file_path,
                        tri_file_path, swi_file_path, predicted_vegetation_dir, arcgis_toolbox_path,
                        veg_threshold=0.3, sample_fraction=0.50, sample_min_per_class=50,
                        sample_max_per_class=500, use_tri=True, use_swi=True, use_snow=True):
    """
    Create a binary low-resolution vegetation raster, then extract all predictor values.

    This function handles the special aggregation logic required to convert the
    high-resolution (0.4 m) binary vegetation raster into a coarser classification:

    Step 1: Recode the binary veg raster (1=veg, 2=non-veg) to (1=veg, 0=non-veg)
    Step 2: Aggregate (MEAN) to target_resolution → each cell value = vegetation
            fraction (0–1) (cell_factor = target_resolution / 0.4)
    Step 3: Apply threshold: fraction >= veg_threshold → classified as vegetation (1),
            else non-vegetation (2)

    TRI, SWI, and snow cover are loaded directly from pre-computed rasters
    rather than derived from the DEM, because they were computed by SAGA GIS
    on the target-resolution DEM using algorithms not available in ArcPy.

    Sampling:
      - Stratified by vegetation class, sample_fraction of the smaller class,
        clipped between sample_min_per_class and sample_max_per_class
      - Minimum distance = 1 × target_resolution to reduce spatial autocorrelation

    Parameters
    ----------
    geodiv_out_dir     : str   Output directory (must already contain the terrain
                               variable rasters produced by Low_res_script.py)
    area_name          : str   Glacier name
    target_resolution  : int   Pixel size in metres
    resampled_dem_path : str   Path to the resampled DEM (snap raster reference;
                               produced by Low_res_script.py)
    snow_file_path     : str   Path to the snow cover raster for this area
    tri_file_path      : str   Path to the pre-computed TRI raster
    swi_file_path      : str   Path to the pre-computed SWI raster
    predicted_vegetation_dir : str  Directory holding the 0.4 m predicted vegetation
                               rasters (config: paths.predicted_vegetation_dir)
    arcgis_toolbox_path: str   Path to ArcGIS Pro's Data Management Tools.tbx
                               (config: paths.arcgis_toolbox_data_management)
    veg_threshold      : float Minimum vegetation fraction for "vegetated" class
    sample_fraction    : float Fraction of the smaller class to sample
    sample_min_per_class : int Minimum samples per class
    sample_max_per_class : int Maximum samples per class
    use_tri            : bool Include the TRI raster as a predictor (config: features.use_tri).
                              Set False to sanity-check the pipeline without a real TRI raster —
                              must match the toggle used when rf_vegetation_20m.py reads the result.
    use_swi            : bool Include the SWI raster as a predictor (config: features.use_swi).
                              Same caveat as use_tri.
    use_snow           : bool Include the snow cover raster as a predictor (config: features.use_snow).
                              Same caveat as use_tri. snow_file_path may be None when False.
    """
    target_resolution = int(target_resolution)

    arcpy.ImportToolbox(arcgis_toolbox_path)
    arcpy.CheckOutExtension("ImageExt")
    arcpy.CheckOutExtension("ImageAnalyst")
    arcpy.env.overwriteOutput = True

    variables_folder = os.path.abspath(str(geodiv_out_dir))

    # Step 1: Recode 0.4 m vegetation raster: 1=veg → 1, 2=non-veg → 0
    veg_raster_hires = os.path.join(predicted_vegetation_dir, f"{area_name}_predicted_vegetation.tif")
    veg_binary = os.path.join(variables_folder, f"{area_name}_veg_binary.tif")
    with arcpy.EnvManager(snapRaster=resampled_dem_path):
        binary = arcpy.sa.Con(arcpy.Raster(veg_raster_hires) == 1, 1, 0)
        binary.save(veg_binary)

    # Step 2: Aggregate from 0.4 m to target resolution using mean
    # cell_factor = target_res / native_res (native ortho resolution is 0.4 m)
    # The result is vegetation fraction per target-resolution cell (0.0 – 1.0)
    cell_factor = int(target_resolution / 0.4)
    veg_fraction = os.path.join(variables_folder, f"{area_name}_veg_fraction.tif")
    with arcpy.EnvManager(snapRaster=resampled_dem_path):
        aggregated = arcpy.sa.Aggregate(arcpy.Raster(veg_binary), cell_factor, "MEAN")
        aggregated.save(veg_fraction)

    print(f"  Vegetation fraction raster created at {target_resolution}m.")

    # Step 3: Apply threshold — fraction >= veg_threshold → 1 (veg), else → 2 (non-veg)
    veg_raster_lowres = os.path.join(variables_folder, f"{area_name}_predicted_vegetation_{target_resolution}m.tif")
    with arcpy.EnvManager(snapRaster=resampled_dem_path):
        binary_lowres = arcpy.sa.Con(arcpy.Raster(veg_fraction) >= veg_threshold, 1, 2)
        binary_lowres.save(veg_raster_lowres)

    print(f"  Vegetation threshold ({veg_threshold}) applied — binary raster created.")

    # Count pixels per class to determine sample size dynamically
    raster_array = arcpy.RasterToNumPyArray(arcpy.Raster(veg_raster_lowres), nodata_to_value=0)
    n_veg    = int(np.sum(raster_array == 1))
    n_nonveg = int(np.sum(raster_array == 2))
    n_min    = min(n_veg, n_nonveg)

    # Clip sample size between sample_min_per_class and sample_max_per_class
    n_samples = int(np.clip(n_min * sample_fraction, sample_min_per_class, sample_max_per_class))

    print(f"  Veg pixels: {n_veg}, Non-veg pixels: {n_nonveg}")
    print(f"  Sampling {n_samples} per class ({sample_fraction*100:.0f}% of smaller class, "
          f"min={sample_min_per_class}, max={sample_max_per_class})")

    # Collect terrain variable rasters from the output folder (produced earlier
    # by Low_res_script.py). Exclude intermediate rasters that are not predictors
    variables_files_list = []
    exclude = {
        f"{area_name}_aspect.tif",
        f"{area_name}_aspect_rad.tif",
        f"{area_name}_DEM_clip.tif",
        f"{area_name}_DEM_fill.tif",
        f"{area_name}_geomorph.tif",
        f"{area_name}_veg_binary.tif",    # intermediate only
        f"{area_name}_veg_fraction.tif",  # intermediate only
    }

    for file in os.listdir(variables_folder):
        if file.endswith(".tif") and file not in exclude:
            variables_files_list.append(Raster(os.path.join(variables_folder, file)))

    # Add TRI, SWI and snow cover — loaded directly from source. Each is
    # skippable via config (features.use_tri / use_swi / use_snow) when that
    # raster isn't available yet; rf_vegetation_20m.py must use the same
    # toggles to match the resulting column layout.
    if use_tri:
        variables_files_list.append(Raster(tri_file_path))
    if use_swi:
        variables_files_list.append(Raster(swi_file_path))
    if use_snow:
        variables_files_list.append(Raster(snow_file_path))

    # Minimum spacing between sample points = 1 pixel at target resolution
    min_distance_m = target_resolution * 1

    # Create stratified random sample locations
    sample_points_shp = os.path.join(variables_folder, f"{area_name}_sample_points.shp")
    arcpy.management.CreateSpatialSamplingLocations(
        in_study_area=arcpy.Raster(veg_raster_lowres),
        out_features=sample_points_shp,
        sampling_method="STRAT_ID",
        strata_id_field="Value",
        num_samples=100,
        num_samples_per_strata=n_samples,
        min_distance=f"{min_distance_m} Meters"
    )
    print("  Sampling locations created.")

    # Extract all variable values at sample locations
    samples_shp = os.path.join(variables_folder, f"{area_name}_samples.shp")
    arcpy.sa.Sample(
        variables_files_list, sample_points_shp,
        samples_shp, "NEAREST", "FID",
        "CURRENT_SLICE", [], "", None, "", "ROW_WISE", "FEATURE_CLASS"
    )
    print("  Sampling complete.")


# ── Main loop: sample each study area at the target resolution ────────────────
for area in study_areas.itertuples():
    area_name = area.Glacier_na

    geodiv_out_dir = Path(OUTPUTS_20M_DIR) / f"{area_name}/Geodiversity"
    # The 20 m DEM produced by Low_res_script.py's resample_clip_fill_dem() —
    # must already exist; run Low_res_script.py first if it doesn't.
    resampled_dem_path = str(geodiv_out_dir / f"{area_name}_DEM_{TARGET_RESOLUTION}m.tif")

    snow_file_path = None
    if USE_SNOW:
        if area_name not in SNOW_FILE_LOOKUP:
            print(f"  ⚠ No snow file defined for {area_name} — skipping.")
            continue
        snow_file_path = os.path.join(SNOW_DIR, SNOW_FILE_LOOKUP[area_name])

    tri_file_path  = os.path.join(TRI_SWI_DIR, cfg.geodiversity.tri_filename_20m.format(area=area_name))
    swi_file_path  = os.path.join(TRI_SWI_DIR, cfg.geodiversity.swi_filename_20m.format(area=area_name))

    print(f"\nProcessing: {area_name}")
    print(f"  Snow file: {SNOW_FILE_LOOKUP[area_name] if USE_SNOW else '(skipped)'}")

    # Aggregate vegetation, apply threshold, sample all variables
    sample_areas_lowres(str(geodiv_out_dir), area_name,
                        TARGET_RESOLUTION, resampled_dem_path,
                        snow_file_path, tri_file_path, swi_file_path,
                        PREDICTED_VEGETATION_DIR, ARCGIS_TOOLBOX_PATH,
                        veg_threshold=VEG_THRESHOLD,
                        sample_fraction=cfg.low_res.sample_fraction,
                        sample_min_per_class=cfg.low_res.sample_min_per_class,
                        sample_max_per_class=cfg.low_res.sample_max_per_class,
                        use_tri=USE_TRI, use_swi=USE_SWI, use_snow=USE_SNOW)
    print("  Study area sampled.")

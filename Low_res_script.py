"""
Step 5 — Low-resolution (20 m) terrain variable calculation.

This script produces the 20 m terrain variable rasters used by
Sample_areas_20m.py and the 20 m geodiversity indices. It mirrors the 1 m
pipeline (Variable_calculation.py) but operates at coarser resolution.

For each study area:
  1. resample_clip_fill_dem — resamples the native-resolution DSM/DEM mosaic
     (kept on disk by High_res_script.py's mosaic_dem() as {area}_DEM_native.tif)
     directly to the 20 m target resolution using bilinear interpolation, then
     clips and fills sinks. Deliberately bypasses the 1 m analysis grid
     ({area}_DEM.tif) to avoid cascading through an intermediate resolution.
  2. calculate_variables    — derives slope, aspect (sin/cos), hillshade,
     landforms (search radius scaled to the target resolution), distance, and
     curvature

Vegetation aggregation, snow cover integration, TRI/SWI, and sample
extraction have been split out into Sample_areas_20m.py, so terrain variables
can be computed — and inspected — independently of the predicted vegetation
raster. Run Sample_areas_20m.py after both this script and
Veg_RF_polygon_cv.py have been run for the areas you want sampled.

Outputs are written to: {paths.outputs_20m}/{area_name}/Geodiversity/

All paths, the target resolution, and the geomorphon search radius are read
from config.yaml.

Requires: arcpy (ArcGIS Pro with Spatial Analyst and Image Analyst extensions), PyYAML
"""

import os

import math
import arcpy
from arcpy.sa import *
import pandas as pd
import geopandas as gpd
from pathlib import Path
from arcgis_functions import *

from config_utils import load_config, resolve_path

cfg = load_config()
os.chdir(cfg.paths.base_dir)
print(os.getcwd())

# Allow overwriting existing output files (e.g. the per-area {area}_select.shp
# already created by the 1 m pipeline for the same area)
arcpy.env.overwriteOutput = True

# ── CONFIG ────────────────────────────────────────────────────────────────────
TARGET_RESOLUTION = cfg.low_res.target_resolution    # output pixel size in metres
GLACIER_POLYGON_SCRATCH_PATH = resolve_path(cfg, cfg.paths.glacier_polygon_scratch_gdb)
GEOMORPHON_SEARCH_RADIUS_MULTIPLIER = cfg.terrain_variables.geomorphon_search_radius_multiplier
OUTPUTS_1M_DIR    = cfg.paths.outputs_1m
# ─────────────────────────────────────────────────────────────────────────────

shp_path    = cfg.paths.outlines_shp
glacier_shp = cfg.paths.glacier_shp

study_areas = gpd.read_file(shp_path)
study_areas = study_areas.drop(index=cfg.study_areas.exclude_indices)
if cfg.study_areas.only:
    study_areas = study_areas[study_areas["Glacier_na"].isin(cfg.study_areas.only)]

# Merge study areas with sun position from lookup table (needed for hillshade)
lookup = pd.read_csv(cfg.paths.lookup_table_gd)
studarea_merge = study_areas.merge(
    lookup,
    left_on="Glacier_na",
    right_on="Glacier_name",
    how="left"
)


def calculate_variables(area_name, geodiv_out_dir, merged_dem_path, sun_azim, sun_alt, area_shp, glacier_shp,
                        glacier_polygon_scratch_path, target_resolution=20, geomorphon_search_radius_multiplier=3):
    """
    Derive terrain variables from the resampled DEM.

    Identical to the 1 m version in High_res_script.py except:
      - Geomorphon landform search radius scales with resolution
        (target_resolution * geomorphon_search_radius_multiplier) so the
        neighbourhood used for landform classification covers a comparable
        physical extent at coarser pixel sizes.

    Variables derived: slope, aspect (sin/cos), hillshade, landforms, distance
    from glacier, and profile curvature. Saved to geodiv_out_dir.

    Parameters
    ----------
    area_name        : str   Glacier name
    geodiv_out_dir   : str   Output directory
    merged_dem_path  : str   Path to the resampled DEM
    sun_azim         : float Sun azimuth at image acquisition (degrees)
    sun_alt          : float Sun altitude at image acquisition (degrees)
    area_shp         : str   Single-area shapefile path
    glacier_shp      : str   Glacier polygon shapefile path
    glacier_polygon_scratch_path : str  Scratch feature class path for the
                             per-area glacier polygon selection
    target_resolution: int   Pixel size in metres
    geomorphon_search_radius_multiplier : int  Search radius (m) = target_resolution * multiplier
    """
    merged_dem = Raster(merged_dem_path)

    # Slope
    Output_raster = f"{geodiv_out_dir}/{area_name}_slope.tif"
    Slope = Output_raster
    with arcpy.EnvManager(mask=area_shp, snapRaster=merged_dem_path):
        Output_raster = arcpy.sa.Slope(merged_dem, "DEGREE", 1, "PLANAR", "METER", "GPU_THEN_CPU")
        Output_raster.save(Slope)
    print("  Slope")

    # Aspect (raw degrees)
    Output_raster_2_ = f"{geodiv_out_dir}/{area_name}_aspect.tif"
    Aspect = Output_raster_2_
    with arcpy.EnvManager(mask=area_shp, snapRaster=merged_dem_path):
        Output_raster_2_ = arcpy.sa.Aspect(merged_dem, "PLANAR", "METER", "GEODESIC_AZIMUTHS", "GPU_THEN_CPU")
        Output_raster_2_.save(Aspect)

    # Convert aspect to radians then decompose into sin and cos
    # to encode the circular aspect variable as two continuous linear predictors
    aspect_rad = f"{geodiv_out_dir}/{area_name}_aspect_rad.tif"
    with arcpy.EnvManager(snapRaster=merged_dem_path):
        Calculation = (Raster(Aspect)) * (math.pi / 180)
        Calculation.save(aspect_rad)

    aspect_sin = f"{geodiv_out_dir}/{area_name}_aspect_sin.tif"
    with arcpy.EnvManager(snapRaster=merged_dem_path):
        Calculation = Sin(aspect_rad)
        Calculation.save(aspect_sin)

    aspect_cos = f"{geodiv_out_dir}/{area_name}_aspect_cos.tif"
    with arcpy.EnvManager(snapRaster=merged_dem_path):
        Calculation = Cos(aspect_rad)
        Calculation.save(aspect_cos)
    print("  Aspect")

    # Hillshade: solar illumination based on sun position
    HillSha_MHM_1 = f"{geodiv_out_dir}/{area_name}_hillshade.tif"
    Hillshade = HillSha_MHM_1
    with arcpy.EnvManager(mask=area_shp, snapRaster=merged_dem_path):
        HillSha_MHM_1 = arcpy.sa.Hillshade(merged_dem, sun_azim, sun_alt, "SHADOWS", 1)
        HillSha_MHM_1.save(Hillshade)
    print("  Hillshade")

    # Geomorphon landforms: search radius scales with resolution
    # (target_resolution × multiplier) so the neighbourhood is physically
    # comparable to the 1 m analysis
    Geomorp_MHM_1 = f"{geodiv_out_dir}/{area_name}_landforms.tif"
    Geomorphon_Landforms = Geomorp_MHM_1
    Output_geomorphons_raster = f"{geodiv_out_dir}/{area_name}_geomorph.tif"
    geomorphon_search_radius = target_resolution * geomorphon_search_radius_multiplier
    with arcpy.EnvManager(mask=area_shp, snapRaster=merged_dem_path):
        Geomorp_MHM_1 = arcpy.sa.GeomorphonLandforms(merged_dem, Output_geomorphons_raster, 1, "METERS", geomorphon_search_radius, None, "METER")
        Geomorp_MHM_1.save(Geomorphon_Landforms)
    print("  Landforms")

    # Select glacier polygon for this area (for distance calculation)
    glacier_polygon_Select = glacier_polygon_scratch_path
    arcpy.analysis.Select(in_features=glacier_shp, out_feature_class=glacier_polygon_Select, where_clause=f"Glacier_na = '{area_name}'")

    # Distance from glacier: chronosequence proxy
    Suottas_distance_tif = f"{geodiv_out_dir}/{area_name}_distance.tif"
    Distance_Accumulation = Suottas_distance_tif
    Output_Back_Direction_Raster = ""
    Output_Source_Direction_Raster = ""
    Output_Source_Location_Raster = ""
    with arcpy.EnvManager(snapRaster=merged_dem_path):
        Suottas_distance_tif = arcpy.sa.DistanceAccumulation(glacier_polygon_Select, "", merged_dem, "", "", "BINARY 1 -30 30", "", "BINARY 1 45", Output_Back_Direction_Raster, Output_Source_Direction_Raster, Output_Source_Location_Raster, "", "", "", "FROM_SOURCE", "PLANAR")
        Suottas_distance_tif.save(Distance_Accumulation)
    print("  Distance")

    # Profile curvature: concavity/convexity of slope
    Surface_MHM_1 = f"{geodiv_out_dir}/{area_name}_curvature.tif"
    Surface_Parameters = Surface_MHM_1
    with arcpy.EnvManager(mask=area_shp, snapRaster=merged_dem_path):
        Surface_MHM_1 = arcpy.sa.SurfaceParameters(merged_dem, "PROFILE_CURVATURE", "QUADRATIC", "",
                                                    "FIXED_NEIGHBORHOOD", "METER", "DEGREE", "GEODESIC_AZIMUTHS",
                                                    "NORTH_POLE_ASPECT", area_shp)
        Surface_MHM_1.save(Surface_Parameters)
    print("  Curvature")


def resample_clip_fill_dem(area_shp, area_name, geodiv_out_dir, target_resolution, outputs_1m_dir):
    """
    Resample the native-resolution DSM/DEM mosaic to target_resolution metres,
    then clip and fill.

    Instead of re-mosaicking DSM tiles, this function reuses the native-
    resolution mosaic produced (and kept on disk) by High_res_script.py's
    mosaic_dem() — {area_name}_DEM_native.tif — and downsamples it directly to
    target_resolution with bilinear interpolation (appropriate for continuous
    elevation data). It deliberately does NOT use {area_name}_DEM.tif (the
    version already resampled to the 1 m analysis grid), to avoid cascading
    through an intermediate resolution — resampling native DSM (e.g. 0.5 m)
    straight to 20 m is more accurate than resampling native -> 1 m -> 20 m.

    Parameters
    ----------
    area_shp         : str  Single-area shapefile path
    area_name        : str  Glacier name
    geodiv_out_dir   : str  Output directory for the low-resolution geodiversity products
    target_resolution: int  Target pixel size in metres
    outputs_1m_dir   : str  Path to the 1 m outputs directory (config: paths.outputs_1m),
                            used to locate the native-resolution DEM mosaic

    Returns
    -------
    str  Path to the resampled DEM raster
    """
    # Path to the native-resolution DSM/DEM mosaic created in the 1 m pipeline,
    # before its own resample to the 1 m analysis grid
    merged_dem_native_path = os.path.join(outputs_1m_dir, f"{area_name}/Geodiversity/{area_name}_DEM_native.tif")

    # Resample directly from native resolution to target_resolution using bilinear interpolation
    resampled_dem_path = os.path.join(geodiv_out_dir, f"{area_name}_DEM_{target_resolution}m.tif")
    arcpy.management.Resample(
        in_raster=merged_dem_native_path,
        out_raster=resampled_dem_path,
        cell_size=target_resolution,
        resampling_type="BILINEAR"
    )
    print(f"  DEM resampled from native resolution to {target_resolution}m.")

    # Clip to study area boundary
    dem_clip_path = os.path.join(geodiv_out_dir, f"{area_name}_DEM_clip.tif")
    with arcpy.EnvManager(snapRaster=resampled_dem_path):
        dem_clip = arcpy.sa.ExtractByMask(resampled_dem_path, area_shp, "INSIDE", "DEFAULT")
        dem_clip.save(dem_clip_path)

    # Fill sinks (required for SWI calculation)
    dem_fill_path = os.path.join(geodiv_out_dir, f"{area_name}_DEM_fill.tif")
    with arcpy.EnvManager(snapRaster=resampled_dem_path):
        dem_fill = arcpy.sa.Fill(dem_clip, None)
        dem_fill.save(dem_fill_path)

    return resampled_dem_path


# ── Main loop: process all study areas at the target resolution ───────────────
for area in studarea_merge.itertuples():
    area_name = area.Glacier_na
    sun_azim  = area.sun_azim
    sun_alt   = area.sun_alt

    # Output directory for low-resolution products (separate from 1 m outputs)
    geodiv_out_dir = Path(cfg.paths.outputs_20m) / f"{area_name}/Geodiversity"
    geodiv_out_dir.mkdir(parents=True, exist_ok=True)

    select_area_shp_path = select_area_shp(area_name, shp_path, OUTPUTS_1M_DIR)

    print(f"\nProcessing: {area_name}")

    # Step 1: Resample native-resolution DSM/DEM mosaic to the target resolution, clip and fill sinks
    resampled_dem_path = resample_clip_fill_dem(
        select_area_shp_path, area_name,
        str(geodiv_out_dir), TARGET_RESOLUTION, OUTPUTS_1M_DIR
    )
    print(f"  DEM resampled to {TARGET_RESOLUTION}m, clipped and filled.")

    # Step 2: Derive terrain variables at the target resolution
    calculate_variables(area_name, str(geodiv_out_dir), resampled_dem_path,
                        sun_azim, sun_alt, select_area_shp_path, glacier_shp,
                        GLACIER_POLYGON_SCRATCH_PATH, TARGET_RESOLUTION,
                        GEOMORPHON_SEARCH_RADIUS_MULTIPLIER)
    print("  Variables calculated.")

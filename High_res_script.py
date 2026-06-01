"""
ArcPy helper functions for 1 m terrain variable derivation (Step 4a).

These functions are imported and called by Variable_calculation.py.

Functions
---------
mosaic_dem         -- Mosaic LiDAR DEM tiles, clip to study area, and fill sinks
calculate_variables -- Derive terrain variables from the DEM using ArcPy Spatial Analyst
sample_areas       -- Create stratified sample points and extract variable values

All terrain variables are derived from the Lantmäteriet Markhöjdmodell (LiDAR DEM)
at ~1 m resolution and saved to:
  Data/Python/Outputs/{area_name}/Geodiversity/

Requires: arcpy (ArcGIS Pro with Spatial Analyst, Image Analyst and Image Ext extensions)
"""

import math
import os
import numpy as np

import arcpy
from arcpy.sa import *


def mosaic_dem(area_shp, area_name, geodiv_out_dir):
    """
    Merge LiDAR DEM tiles for one study area, clip to the study area, and fill sinks.

    The LiDAR DEM is delivered as multiple tiles; they must be mosaicked into a
    single raster before terrain analysis. Sink filling ensures hydrological
    consistency (no artificial closed depressions) and is required for correct
    derivation of the SWI (Saga Wetness Index).

    Parameters
    ----------
    area_shp       : str  Path to the single-area shapefile (used as mask)
    area_name      : str  Glacier name
    geodiv_out_dir : str  Output directory for geodiversity products

    Returns
    -------
    str  Path to the merged (non-clipped) DEM raster
    """
    dem_folder = f"C:/TEMP/Vanessa_Henriksson/Data/DEM/Markhöjdmodell/{area_name}"

    # Collect all .tif DEM tiles in the area-specific folder
    files_list = os.listdir(dem_folder)
    dem_files_list = []
    for file in files_list:
        if file.endswith(".tif"):
            dem_files_list.append(Raster(f"{dem_folder}/{file}"))

    # Merge all tiles into a single raster using the study area polygon as mask
    merged_dem_path = f"{geodiv_out_dir}/{area_name}_DEM.tif"
    with arcpy.EnvManager(mask=area_shp):
        Output_mosaic = arcpy.management.MosaicToNewRaster(
            input_rasters=dem_files_list,
            output_location=geodiv_out_dir,
            pixel_type="32_BIT_FLOAT",
            raster_dataset_name_with_extension=f"{area_name}_DEM.tif",
            number_of_bands=1)[0]

    # Clip the merged DEM to the exact study area boundary
    dem_clip_tif = f"{geodiv_out_dir}/{area_name}_DEM_clip.tif"
    Extract_by_Mask = dem_clip_tif
    with arcpy.EnvManager(snapRaster=merged_dem_path):
        dem_clip_tif = arcpy.sa.ExtractByMask(merged_dem_path, area_shp, "INSIDE", "DEFAULT")
        dem_clip_tif.save(Extract_by_Mask)

    # Fill sinks in the clipped DEM (removes spurious low points that would
    # otherwise interrupt flow accumulation and SWI calculation)
    output_surface_raster = f"{geodiv_out_dir}/{area_name}_DEM_fill.tif"
    Fill = output_surface_raster
    with arcpy.EnvManager(snapRaster=merged_dem_path):
        output_surface_raster = arcpy.sa.Fill(dem_clip_tif, None)
        output_surface_raster.save(Fill)

    return merged_dem_path


def calculate_variables(area_name, geodiv_out_dir, merged_dem_path, sun_azim, sun_alt, area_shp, glacier_shp):
    """
    Derive terrain variables from the LiDAR DEM using ArcPy Spatial Analyst.

    Variables derived
    -----------------
    slope        -- Steepness in degrees (controls water movement and soil stability)
    aspect       -- Compass direction the slope faces (0–360°)
    aspect_sin   -- Sine of aspect (in radians) — captures north-south gradient
    aspect_cos   -- Cosine of aspect (in radians) — captures east-west gradient
                    (aspect is circular, so sin/cos encode it as linear variables)
    hillshade    -- Solar illumination (0–255) based on sun azimuth/altitude at
                    the time the orthophoto was acquired — proxy for solar radiation
    landforms    -- Geomorphon landform classes (1–10: flat, summit, ridge, shoulder,
                    spur, slope, hollow, footslope, valley, depression) derived from
                    a 3 m search radius analysis of the local relief pattern
    distance     -- Distance accumulation from the glacier polygon — a proxy for
                    time since deglaciation (closer = more recently deglaciated)
    curvature    -- Profile curvature (concave vs convex slope, in degrees/100 m) —
                    affects water concentration and soil development

    All variables are snapped to the merged DEM grid and masked to the study area.

    Parameters
    ----------
    area_name      : str   Glacier name
    geodiv_out_dir : str   Output directory
    merged_dem_path: str   Path to the merged (full-extent) DEM
    sun_azim       : float Sun azimuth angle in degrees (from lookup table)
    sun_alt        : float Sun altitude angle in degrees (from lookup table)
    area_shp       : str   Path to single-area shapefile (spatial mask)
    glacier_shp    : str   Path to glacier polygon shapefile (for distance calculation)
    """
    merged_dem = Raster(merged_dem_path)

    # Slope: rate of change in elevation, in degrees (0 = flat, 90 = vertical)
    Output_raster = f"{geodiv_out_dir}/{area_name}_slope.tif"
    Slope = Output_raster
    with arcpy.EnvManager(mask=area_shp, snapRaster=merged_dem_path):
        Output_raster = arcpy.sa.Slope(merged_dem, "DEGREE", 1, "PLANAR", "METER", "GPU_THEN_CPU")
        Output_raster.save(Slope)
    print("Slope")

    # Aspect: compass direction the slope faces (0–360°, -1 = flat)
    Output_raster_2_ = f"{geodiv_out_dir}/{area_name}_aspect.tif"
    Aspect = Output_raster_2_
    with arcpy.EnvManager(mask=area_shp, snapRaster=merged_dem_path):
        Output_raster_2_ = arcpy.sa.Aspect(merged_dem, "PLANAR", "METER", "GEODESIC_AZIMUTHS", "GPU_THEN_CPU")
        Output_raster_2_.save(Aspect)

    # Convert aspect from degrees to radians for trigonometric transformation
    aspect_rad = f"{geodiv_out_dir}/{area_name}_aspect_rad.tif"
    with arcpy.EnvManager(snapRaster=merged_dem_path):
        Calculation = (Raster(Aspect)) * (math.pi / 180)
        Calculation.save(aspect_rad)

    # Aspect sine: positive = south-facing, negative = north-facing
    aspect_sin = f"{geodiv_out_dir}/{area_name}_aspect_sin.tif"
    with arcpy.EnvManager(snapRaster=merged_dem_path):
        Calculation = Sin(aspect_rad)
        Calculation.save(aspect_sin)

    # Aspect cosine: positive = east-facing, negative = west-facing
    aspect_cos = f"{geodiv_out_dir}/{area_name}_aspect_cos.tif"
    with arcpy.EnvManager(snapRaster=merged_dem_path):
        Calculation = Cos(aspect_rad)
        Calculation.save(aspect_cos)
    print("Aspect")

    # Hillshade: solar illumination index computed from sun position at image acquisition time
    # Shadows parameter = "SHADOWS" includes both illuminated and shadowed areas
    HillSha_MHM_1 = f"{geodiv_out_dir}/{area_name}_hillshade.tif"
    Hillshade = HillSha_MHM_1
    with arcpy.EnvManager(mask=area_shp, snapRaster=merged_dem_path):
        HillSha_MHM_1 = arcpy.sa.Hillshade(merged_dem, sun_azim, sun_alt, "SHADOWS", 1)
        HillSha_MHM_1.save(Hillshade)
    print("Hillshade")

    # Geomorphon landforms: classifies each cell by comparing it to neighbours
    # within a 3 m search radius. The 10 classes capture topographic position
    # (e.g. ridge, valley, slope) and are important for vegetation niche differentiation.
    Geomorp_MHM_1 = f"{geodiv_out_dir}/{area_name}_landforms.tif"
    Geomorphon_Landforms = Geomorp_MHM_1
    Output_geomorphons_raster = f"{geodiv_out_dir}/{area_name}_geomorph.tif"
    with arcpy.EnvManager(mask=area_shp, snapRaster=merged_dem_path):
        Geomorp_MHM_1 = arcpy.sa.GeomorphonLandforms(merged_dem, Output_geomorphons_raster, 1, "METERS", 3, None, "METER")
        Geomorp_MHM_1.save(Geomorphon_Landforms)
    print("Landforms")

    # Select glacier polygon for this specific area from the master shapefile
    glacier_polygon_Select = "C:\\TEMP\\Vanessa_Henriksson\\RD_GIS\\RD_GIS.gdb\\glacier_polygon_Select"
    arcpy.analysis.Select(in_features=glacier_shp, out_feature_class=glacier_polygon_Select, where_clause=f"Glacier_na = '{area_name}'")

    # Distance accumulation: horizontal distance from glacier boundary for each pixel
    # This represents a chronosequence proxy — pixels close to the glacier were
    # deglaciated more recently and have had less time for soil and vegetation development.
    # BINARY 1 -30 30 = passable slopes between -30° and 30°; BINARY 1 45 = max turn angle
    Suottas_distance_tif = f"{geodiv_out_dir}/{area_name}_distance.tif"
    Distance_Accumulation = Suottas_distance_tif
    Output_Back_Direction_Raster = ""
    Output_Source_Direction_Raster = ""
    Output_Source_Location_Raster = ""
    with arcpy.EnvManager(snapRaster=merged_dem_path):
        Suottas_distance_tif = arcpy.sa.DistanceAccumulation(glacier_polygon_Select, "", merged_dem, "", "", "BINARY 1 -30 30", "", "BINARY 1 45", Output_Back_Direction_Raster, Output_Source_Direction_Raster, Output_Source_Location_Raster, "", "", "", "FROM_SOURCE", "PLANAR")
        Suottas_distance_tif.save(Distance_Accumulation)
    print("Distance")

    # Profile curvature: rate of change of slope along the downslope direction
    # Negative = concave (water accumulates); Positive = convex (water disperses)
    Surface_MHM_1 = f"{geodiv_out_dir}/{area_name}_curvature.tif"
    Surface_Parameters = Surface_MHM_1
    with arcpy.EnvManager(mask=area_shp, snapRaster=merged_dem_path):
        Surface_MHM_1 = arcpy.sa.SurfaceParameters(merged_dem, "PROFILE_CURVATURE", "QUADRATIC", "",
                                                    "FIXED_NEIGHBORHOOD", "METER", "DEGREE", "GEODESIC_AZIMUTHS",
                                                    "NORTH_POLE_ASPECT", area_shp)
        Surface_MHM_1.save(Surface_Parameters)
    print("Curvature")


def sample_areas(geodiv_out_dir, area_name, merged_dem_path):
    """
    Create stratified sample points and extract terrain variable values at each point.

    Sampling strategy:
      - The predicted vegetation raster (~0.4 m) is resampled to 1 m using NEAREST
        neighbour (to match DEM resolution) so that vegetation and terrain variables
        are spatially aligned.
      - Stratified sampling by vegetation class (1=veg, 2=non-veg) ensures both
        classes are represented. Sample size = 10% of the smaller class, max 2000.
      - Minimum distance of 2 m between sample points reduces spatial autocorrelation.
      - All terrain variable rasters are then sampled at the selected point locations
        using NEAREST neighbour extraction.

    The resulting shapefile ({area_name}_samples.shp) has one row per sample point
    with columns v_raster_1 ... v_raste_11 containing the terrain variable values.
    This file is the primary input for RF_block_test.py.

    Parameters
    ----------
    geodiv_out_dir : str  Output directory containing terrain variable rasters
    area_name      : str  Glacier name
    merged_dem_path: str  Path to merged DEM (used as snap raster for resampling)
    """
    arcpy.ImportToolbox(r"c:\program files\arcgis\pro\Resources\ArcToolbox\toolboxes\Data Management Tools.tbx")
    arcpy.CheckOutExtension("ImageExt")
    arcpy.CheckOutExtension("ImageAnalyst")

    # Use absolute path to avoid ArcPy relative path issues
    variables_folder = os.path.abspath(str(geodiv_out_dir))
    variables_files = os.listdir(variables_folder)
    variables_files_list = []

    # Exclude intermediate/non-predictor rasters from the variable list
    exclude = {
        f"{area_name}_aspect.tif",          # raw aspect in degrees — not used directly
        f"{area_name}_aspect_rad.tif",       # intermediate (aspect in radians)
        f"{area_name}_DEM_clip.tif",         # intermediate DEM
        f"{area_name}_DEM_fill.tif",         # intermediate DEM
        f"{area_name}_geomorph.tif",         # raw geomorphon output (landforms is the classified version)
        f"{area_name}_predicted_vegetation.tif",
        f"{area_name}_predicted_vegetation_1m.tif"  # appended manually below
    }

    for file in variables_files:
        if file.endswith(".tif") and file not in exclude:
            variables_files_list.append(Raster(os.path.join(variables_folder, file)))

    # Resample vegetation from 0.4 m to 1 m using NEAREST (categorical data)
    # so it aligns exactly with the 1 m DEM-derived variables
    predicted_vegetation_04m = f"C:\\TEMP\\Vanessa_Henriksson\\Data\\Python\\Outputs\\Predicted_vegetation\\{area_name}_predicted_vegetation.tif"
    predicted_vegetation_1m = os.path.join(variables_folder, f"{area_name}_predicted_vegetation_1m.tif")
    with arcpy.EnvManager(snapRaster=merged_dem_path):
        arcpy.management.Resample(predicted_vegetation_04m, predicted_vegetation_1m, "1", "NEAREST")

    # Add the 1 m vegetation raster as the last variable to sample
    variables_files_list.append(arcpy.Raster(predicted_vegetation_1m))

    print("Predicted vegetation resampled to 1m.")

    # Count pixels per class to set a proportional sample size
    arr = arcpy.RasterToNumPyArray(predicted_vegetation_1m, nodata_to_value=0)
    veg_count = int(np.sum(arr == 1))
    non_veg_count = int(np.sum(arr == 2))
    min_class_count = min(veg_count, non_veg_count)

    print(f"Vegetation pixels: {veg_count}, Non-vegetation pixels: {non_veg_count}")

    # 10% of smallest class, capped at 2000 per class
    num_samples_per_strata = min(int(min_class_count * 0.10), 2000)
    num_samples = num_samples_per_strata * 2  # total across both classes

    print(f"Dynamic sampling: {num_samples_per_strata} samples per strata ({num_samples} total)")

    # Create stratified random sample locations; STRAT_ID uses the raster value
    # (1 or 2) as the stratification field so both classes are equally represented
    sample_points_shp = os.path.join(variables_folder, f"{area_name}_sample_points.shp")
    arcpy.management.CreateSpatialSamplingLocations(
        in_study_area=arcpy.Raster(predicted_vegetation_1m),
        out_features=sample_points_shp,
        sampling_method="STRAT_ID",
        strata_id_field="Value",
        num_samples=num_samples,
        num_samples_per_strata=num_samples_per_strata,
        min_distance="2 Meters"  # minimum 2 m spacing to reduce spatial autocorrelation
    )
    print("Sampling locations created.")

    # Extract all terrain variable values at the sample locations
    # ROW_WISE output means one row per sample point, one column per variable
    samples_shp = os.path.join(variables_folder, f"{area_name}_samples.shp")
    arcpy.sa.Sample(variables_files_list, sample_points_shp, samples_shp, "NEAREST", "FID", "CURRENT_SLICE", [], "", None, "", "ROW_WISE", "FEATURE_CLASS")
    print("Sampling complete.")

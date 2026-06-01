"""
Step 5 — Low-resolution (20 m) processing: terrain variables, vegetation aggregation,
snow cover integration, and sample extraction.

This script produces the 20 m dataset used by rf_vegetation_20m.py and the
20 m geodiversity indices. It mirrors the 1 m pipeline (Variable_calculation.py)
but operates at coarser resolution and adds snow cover as an additional variable.

For each study area:
  1. resample_clip_fill_dem — resamples the existing 1 m DEM to 20 m using
     bilinear interpolation, clips and fills sinks
  2. calculate_variables    — derives slope, aspect (sin/cos), hillshade,
     landforms (search radius scaled to 20 m), distance, and curvature at 20 m
  3. sample_areas_lowres    — aggregates the 0.4 m binary vegetation raster to
     20 m (vegetation fraction), applies a 0.3 threshold to create a 20 m binary
     classification, then creates stratified sample points and extracts all
     terrain variables + TRI + SWI + snow cover at those locations

Key difference from 1 m pipeline:
  - Snow cover: a Sentinel-2-derived snow cover fraction raster is added as an
    extra predictor. High snow cover in the growing season limits vegetation
    establishment. TRI and SWI are loaded from pre-computed 20 m rasters rather
    than re-derived from the DEM.
  - Vegetation fraction threshold: a cell is classified as "vegetated" if ≥ 30%
    of its sub-pixels (at 0.4 m) were classified as vegetation.
  - Sample size is dynamic: 50% of the smaller class, capped at 500 per class.

Outputs are written to: Data/Python/Outputs_20m/{area_name}/Geodiversity/

Requires: arcpy (ArcGIS Pro with Spatial Analyst and Image Analyst extensions)
"""

import os
os.chdir(r"C:\TEMP\Vanessa_Henriksson")
print(os.getcwd())

import math
import arcpy
from arcpy.sa import *
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from arcgis_functions import *
from NDVI_functions import *
from variable_functions import *

# ── CONFIG ────────────────────────────────────────────────────────────────────
TARGET_RESOLUTION  = 20    # output pixel size in metres
VEG_THRESHOLD      = 0.3   # minimum vegetation fraction to classify a 20 m cell as vegetated
SNOW_DIR           = r"C:\TEMP\Vanessa_Henriksson\Data\Snow_data"
TRI_SWI_DIR        = r"C:\TEMP\Vanessa_Henriksson\Data\DEM\TRI_SWI_20m"
BASE_DIR           = r"C:\TEMP\Vanessa_Henriksson"
# ─────────────────────────────────────────────────────────────────────────────

# Lookup: which Sentinel-2 snow cover tile covers each study area
# Snow cover was derived from multi-temporal S2 composites at 20 m resolution
SNOW_FILE_LOOKUP = {
    # T33WXQ tile (southern areas)
    "Suottas":              "T33WXQ_snow_cover.tif",
    "Vartas":               "T33WXQ_snow_cover.tif",
    "Mikka":                "T33WXQ_snow_cover.tif",
    "Ruotes":               "T33WXQ_snow_cover.tif",
    "Pårte":                "T33WXQ_snow_cover.tif",
    # T33WXR tile (northern areas)
    "Kårsa":                "T33WXR_snow_cover.tif",
    "Riuko":                "T33WXR_snow_cover.tif",
    "Gallan":               "T33WXR_snow_cover.tif",
    "Unna_Räita":           "T33WXR_snow_cover.tif",
    "Vaktpost":             "T33WXR_snow_cover.tif",
    "Storglaciären":        "T33WXR_snow_cover.tif",
    "Isfall":               "T33WXR_snow_cover.tif",
    "Ballinriehppe_II_III": "T33WXR_snow_cover.tif",
    "Rabots":               "T33WXR_snow_cover.tif",
    # Areas with individual snow cover files
    "Stuorra":              "Stuorra_snow_cover.tif",
    "Helags":               "Helags_snow_cover.tif",
}

arcpy.env.overwriteOutput = True

shp_path    = "Data/proglacial_outlines.shp"
glacier_shp = "Data/glacier_polygon.shp"

study_areas = gpd.read_file(shp_path)
study_areas = study_areas.drop(index=[12, 14])

# Merge study areas with sun position from lookup table (needed for hillshade)
lookup = pd.read_csv("Data/Python/lookup_table_gd.csv")
studarea_merge = study_areas.merge(
    lookup,
    left_on="Glacier_na",
    right_on="Glacier_name",
    how="left"
)


def calculate_variables(area_name, geodiv_out_dir, merged_dem_path, sun_azim, sun_alt, area_shp, glacier_shp, target_resolution=20):
    """
    Derive terrain variables from the 20 m resampled DEM.

    Identical to the 1 m version in High_res_script.py except:
      - Geomorphon landform search radius scales with resolution (target_resolution * 3)
        so the neighbourhood used for landform classification covers a comparable
        physical extent at coarser pixel sizes.

    Variables derived: slope, aspect (sin/cos), hillshade, landforms, distance
    from glacier, and profile curvature. Saved to geodiv_out_dir.

    Parameters
    ----------
    area_name        : str   Glacier name
    geodiv_out_dir   : str   Output directory
    merged_dem_path  : str   Path to the resampled 20 m DEM
    sun_azim         : float Sun azimuth at image acquisition (degrees)
    sun_alt          : float Sun altitude at image acquisition (degrees)
    area_shp         : str   Single-area shapefile path
    glacier_shp      : str   Glacier polygon shapefile path
    target_resolution: int   Pixel size in metres (default 20)
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
    # (3× the pixel size) so the neighbourhood is physically comparable
    # to the 1 m analysis which used a 3 m search radius
    Geomorp_MHM_1 = f"{geodiv_out_dir}/{area_name}_landforms.tif"
    Geomorphon_Landforms = Geomorp_MHM_1
    Output_geomorphons_raster = f"{geodiv_out_dir}/{area_name}_geomorph.tif"
    with arcpy.EnvManager(mask=area_shp, snapRaster=merged_dem_path):
        Geomorp_MHM_1 = arcpy.sa.GeomorphonLandforms(merged_dem, Output_geomorphons_raster, 1, "METERS", target_resolution * 3, None, "METER")
        Geomorp_MHM_1.save(Geomorphon_Landforms)
    print("  Landforms")

    # Select glacier polygon for this area (for distance calculation)
    glacier_polygon_Select = r"C:\TEMP\Vanessa_Henriksson\RD_GIS\RD_GIS.gdb\glacier_polygon_Select"
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


def resample_clip_fill_dem(area_shp, area_name, geodiv_out_dir, target_resolution):
    """
    Resample the existing 1 m DEM to target_resolution metres, then clip and fill.

    Instead of re-mosaicking LiDAR tiles, this function reuses the merged 1 m DEM
    produced in Variable_calculation.py and downsamples it with bilinear
    interpolation (appropriate for continuous elevation data).

    Parameters
    ----------
    area_shp         : str  Single-area shapefile path
    area_name        : str  Glacier name
    geodiv_out_dir   : str  Output directory for 20 m geodiversity products
    target_resolution: int  Target pixel size in metres

    Returns
    -------
    str  Path to the resampled DEM raster
    """
    # Path to the merged 1 m DEM created in the 1 m pipeline
    merged_dem_path = os.path.join(BASE_DIR, f"Data/Python/Outputs/{area_name}/Geodiversity/{area_name}_DEM.tif")

    # Resample to target resolution using bilinear interpolation
    resampled_dem_path = os.path.join(geodiv_out_dir, f"{area_name}_DEM_{target_resolution}m.tif")
    arcpy.management.Resample(
        in_raster=merged_dem_path,
        out_raster=resampled_dem_path,
        cell_size=target_resolution,
        resampling_type="BILINEAR"
    )
    print(f"  DEM resampled to {target_resolution}m.")

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


def sample_areas_lowres(geodiv_out_dir, area_name, target_resolution, resampled_dem_path, snow_file_path, tri_file_path, swi_file_path, veg_threshold=0.3):
    """
    Create a binary 20 m vegetation raster, then extract all predictor values.

    This function handles the special aggregation logic required to convert the
    high-resolution (0.4 m) binary vegetation raster into a 20 m classification:

    Step 1: Recode the binary veg raster (1=veg, 2=non-veg) to (1=veg, 0=non-veg)
    Step 2: Aggregate (MEAN) to 20 m → each cell value = vegetation fraction (0–1)
            (cell_factor = 20 / 0.4 = 50: each 20 m cell contains 50×50 = 2500
            sub-pixels that are averaged)
    Step 3: Apply threshold: fraction ≥ 0.3 → classified as vegetation (1),
            else non-vegetation (2)

    TRI, SWI, and snow cover are loaded directly from pre-computed 20 m rasters
    rather than derived from the DEM, because they were computed by SAGA GIS
    on the 20 m DEM using algorithms not available in ArcPy.

    Sampling:
      - Stratified by vegetation class, 50% of the smaller class, max 500
      - Minimum distance = 1 × target_resolution to reduce spatial autocorrelation

    Parameters
    ----------
    geodiv_out_dir     : str   Output directory
    area_name          : str   Glacier name
    target_resolution  : int   Pixel size in metres (20)
    resampled_dem_path : str   Path to the 20 m DEM (snap raster reference)
    snow_file_path     : str   Path to the snow cover raster for this area
    tri_file_path      : str   Path to the pre-computed 20 m TRI raster
    swi_file_path      : str   Path to the pre-computed 20 m SWI raster
    veg_threshold      : float Minimum vegetation fraction for "vegetated" class (default 0.3)
    """
    target_resolution = int(target_resolution)

    arcpy.ImportToolbox(r"c:\program files\arcgis\pro\Resources\ArcToolbox\toolboxes\Data Management Tools.tbx")
    arcpy.CheckOutExtension("ImageExt")
    arcpy.CheckOutExtension("ImageAnalyst")
    arcpy.env.overwriteOutput = True

    variables_folder = os.path.abspath(str(geodiv_out_dir))

    # Step 1: Recode 0.4 m vegetation raster: 1=veg → 1, 2=non-veg → 0
    veg_raster_hires = os.path.join(BASE_DIR, "Data", "Python", "Outputs", "Predicted_vegetation",
                                    f"{area_name}_predicted_vegetation.tif")
    veg_binary = os.path.join(variables_folder, f"{area_name}_veg_binary.tif")
    with arcpy.EnvManager(snapRaster=resampled_dem_path):
        binary = arcpy.sa.Con(arcpy.Raster(veg_raster_hires) == 1, 1, 0)
        binary.save(veg_binary)

    # Step 2: Aggregate from 0.4 m to target resolution using mean
    # cell_factor = target_res / native_res = 20 / 0.4 = 50
    # The result is vegetation fraction per 20 m cell (0.0 – 1.0)
    cell_factor = int(target_resolution / 0.4)
    veg_fraction = os.path.join(variables_folder, f"{area_name}_veg_fraction.tif")
    with arcpy.EnvManager(snapRaster=resampled_dem_path):
        aggregated = arcpy.sa.Aggregate(arcpy.Raster(veg_binary), cell_factor, "MEAN")
        aggregated.save(veg_fraction)

    print(f"  Vegetation fraction raster created at {target_resolution}m.")

    # Step 3: Apply threshold — fraction ≥ VEG_THRESHOLD → 1 (veg), else → 2 (non-veg)
    veg_raster_lowres = os.path.join(variables_folder, f"{area_name}_predicted_vegetation_{target_resolution}m.tif")
    with arcpy.EnvManager(snapRaster=resampled_dem_path):
        binary_20m = arcpy.sa.Con(arcpy.Raster(veg_fraction) >= veg_threshold, 1, 2)
        binary_20m.save(veg_raster_lowres)

    print(f"  Vegetation threshold ({veg_threshold}) applied — binary raster created.")

    # Count pixels per class to determine sample size dynamically
    raster_array = arcpy.RasterToNumPyArray(arcpy.Raster(veg_raster_lowres), nodata_to_value=0)
    n_veg    = int(np.sum(raster_array == 1))
    n_nonveg = int(np.sum(raster_array == 2))
    n_min    = min(n_veg, n_nonveg)

    SAMPLE_FRACTION = 0.50
    # Clip sample size between 50 and 500 per class
    n_samples = int(np.clip(n_min * SAMPLE_FRACTION, 50, 500))

    print(f"  Veg pixels: {n_veg}, Non-veg pixels: {n_nonveg}")
    print(f"  Sampling {n_samples} per class ({SAMPLE_FRACTION*100:.0f}% of smaller class, min=50, max=500)")

    # Collect terrain variable rasters from the output folder
    # Exclude intermediate rasters that are not predictors
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

    # Add TRI, SWI and snow cover — already at 20 m, loaded directly from source
    variables_files_list.append(Raster(tri_file_path))
    variables_files_list.append(Raster(swi_file_path))
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


# ── Main loop: process all study areas at 20 m ────────────────────────────────
for area in studarea_merge.itertuples():
    area_name = area.Glacier_na
    sun_azim  = area.sun_azim
    sun_alt   = area.sun_alt

    # Output directory for 20 m products (separate from 1 m outputs)
    geodiv_out_dir = Path(BASE_DIR) / f"Data/Python/Outputs_{TARGET_RESOLUTION}m/{area_name}/Geodiversity"
    geodiv_out_dir.mkdir(parents=True, exist_ok=True)

    select_area_shp_path = select_area_shp(area_name, shp_path)

    if area_name not in SNOW_FILE_LOOKUP:
        print(f"  ⚠ No snow file defined for {area_name} — skipping.")
        continue

    snow_file_path = os.path.join(SNOW_DIR, SNOW_FILE_LOOKUP[area_name])
    tri_file_path  = os.path.join(TRI_SWI_DIR, f"TIF_TRI_{area_name}_DEM_clip.tif")
    swi_file_path  = os.path.join(TRI_SWI_DIR, f"TIF_SWI_{area_name}_DEM_fill.tif")

    print(f"\nProcessing: {area_name}")
    print(f"  Snow file: {SNOW_FILE_LOOKUP[area_name]}")

    # Step 1: Resample existing 1 m DEM to 20 m, clip and fill sinks
    resampled_dem_path = resample_clip_fill_dem(
        select_area_shp_path, area_name,
        str(geodiv_out_dir), TARGET_RESOLUTION
    )
    print(f"  DEM resampled to {TARGET_RESOLUTION}m, clipped and filled.")

    # Step 2: Derive terrain variables at 20 m
    calculate_variables(area_name, str(geodiv_out_dir), resampled_dem_path,
                        sun_azim, sun_alt, select_area_shp_path, glacier_shp, TARGET_RESOLUTION)
    print("  Variables calculated.")

    # Step 3: Aggregate vegetation, apply threshold, sample all variables
    sample_areas_lowres(str(geodiv_out_dir), area_name,
                        TARGET_RESOLUTION, resampled_dem_path,
                        snow_file_path, tri_file_path, swi_file_path,
                        veg_threshold=VEG_THRESHOLD)
    print("  Study area sampled.")

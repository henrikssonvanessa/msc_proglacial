"""
ArcPy helper functions for band calibration and NDVI computation (Step 2).

Functions
---------
calibrate_ortho    -- Apply OLS linear calibration to ortho Red and NIR bands
calculate_NDVI     -- Compute NDVI from raw ortho, calibrated ortho, and S2 bands
resample_ortho     -- Aggregate ortho NDVI to 10 m via zonal statistics
sample_NDVI        -- Sample all three NDVI products at S2 fishnet centroids
mask_samples       -- Clip NDVI samples to exclude snow/water/shadow pixels
NDVI_regression    -- Run OLS comparing calibrated and non-calibrated NDVI to S2 NDVI

Requires: arcpy (ArcGIS Pro with Spatial Analyst and Image Analyst extensions)
"""

import arcpy
from arcpy.ia import *
from arcpy.sa import *


def calibrate_ortho(area_name,
                    NDVI_out_dir,
                    clipped_ortho_band_1,
                    clipped_ortho_band_4,
                    red_k,
                    red_m,
                    NIR_k,
                    NIR_m):
    """
    Apply per-area OLS calibration to convert ortho DN values to pseudo-reflectance.

    The linear calibration equation is:
        calibrated_reflectance = k * (DN / 255) + m

    where k (slope) and m (intercept) were derived by regressing normalised
    ortho values against Sentinel-2 surface reflectance in Proglacial_project.py.
    Dividing by 255 first normalises ortho DN (0–255) to the 0–1 range before
    applying the per-area scale factor and offset.

    Parameters
    ----------
    area_name          : str    Glacier name
    NDVI_out_dir       : Path   Output directory for NDVI products
    clipped_ortho_band_1 : str  Path to clipped ortho Band_1 (Red, DN 0–255)
    clipped_ortho_band_4 : str  Path to clipped ortho Band_4 (NIR, DN 0–255)
    red_k, red_m       : float  OLS slope and intercept for Red band
    NIR_k, NIR_m       : float  OLS slope and intercept for NIR band

    Returns
    -------
    tuple  (output_path_band1, output_path_band4) — calibrated raster paths
    """
    # Calibrate Red band (Band_1)
    output_path_1 = f"{NDVI_out_dir}/{area_name}_ortho_band_1_cal.tif"
    with arcpy.EnvManager(snapRaster=clipped_ortho_band_1):
        Calibration = red_k * (Raster(clipped_ortho_band_1)/255) + red_m
        Calibration.save(output_path_1)

    # Calibrate NIR band (Band_4)
    output_path_4 = f"{NDVI_out_dir}/{area_name}_ortho_band_4_cal.tif"
    with arcpy.EnvManager(snapRaster=clipped_ortho_band_4):
        arcpy.Raster(clipped_ortho_band_4)
        Calibration = NIR_k * (Raster(clipped_ortho_band_4)/255) + NIR_m
        Calibration.save(output_path_4)

    return output_path_1, output_path_4


def calculate_NDVI(area_name,
                   NDVI_out_dir,
                   clipped_ortho_band_4_path,
                   clipped_ortho_band_1_path,
                   output_path_1,
                   output_path_4,
                   clipped_s2_b4_path,
                   clipped_s2_b8_path):
    """
    Compute NDVI for three spectral scenarios.

    NDVI = (NIR - Red) / (NIR + Red)
    Values range from -1 (bare soil/rock) to +1 (dense vegetation).
    Three versions are computed so that the effect of calibration can be assessed:

    1. Non-calibrated ortho: raw DN / 255 bands — sensor-specific, not reflectance
    2. Calibrated ortho: OLS-adjusted pseudo-reflectance bands
    3. Sentinel-2: true surface reflectance from L2A product

    Parameters
    ----------
    area_name              : str  Glacier name
    NDVI_out_dir           : Path Output directory
    clipped_ortho_band_4_path : str  Raw ortho NIR band path
    clipped_ortho_band_1_path : str  Raw ortho Red band path
    output_path_1          : str  Calibrated ortho Red band path
    output_path_4          : str  Calibrated ortho NIR band path
    clipped_s2_b4_path     : str  Clipped S2 Red (B4) band path
    clipped_s2_b8_path     : str  Clipped S2 NIR (B8) band path

    Returns
    -------
    tuple  (NDVI_non_cal_path, NDVI_cal_path, NDVI_s2_path)
    """
    clipped_ortho_band_1 = Raster(clipped_ortho_band_1_path)
    clipped_ortho_band_4 = Raster(clipped_ortho_band_4_path)
    output_1 = Raster(output_path_1)
    output_4 = Raster(output_path_4)
    clipped_s2_b4 = Raster(clipped_s2_b4_path)
    clipped_s2_b8 = Raster(clipped_s2_b8_path)

    # NDVI from raw (non-calibrated) ortho — DN / 255 used as a simple normalisation
    NDVI_output_path_non_cal = f"{NDVI_out_dir}/{area_name}_NDVI_ortho_non-cal.tif"
    with arcpy.EnvManager(snapRaster=clipped_ortho_band_1_path):
        NDVI = (clipped_ortho_band_4 - clipped_ortho_band_1) / (clipped_ortho_band_4 + clipped_ortho_band_1)
        NDVI.save(NDVI_output_path_non_cal)

    # NDVI from calibrated ortho (pseudo-reflectance)
    NDVI_output_path_cal = f"{NDVI_out_dir}/{area_name}_NDVI_ortho_cal.tif"
    with arcpy.EnvManager(snapRaster=clipped_ortho_band_1_path):
        NDVI = (output_4 - output_1) / (output_4 + output_1)
        NDVI.save(NDVI_output_path_cal)

    # NDVI from Sentinel-2 L2A surface reflectance
    NDVI_output_path_s2 = f"{NDVI_out_dir}/{area_name}_NDVI_S2.tif"
    with arcpy.EnvManager(snapRaster=clipped_s2_b8):
        NDVI = (clipped_s2_b8 - clipped_s2_b4) / (clipped_s2_b8 + clipped_s2_b4)
        NDVI.save(NDVI_output_path_s2)

    return NDVI_output_path_non_cal, NDVI_output_path_cal, NDVI_output_path_s2


def resample_ortho(area_name,
                   s2_fishnet_shp_path,
                   NDVI_out_dir,
                   NDVI_output_path_s2,
                   NDVI_output_path_cal,
                   NDVI_output_path_non_cal):
    """
    Resample ortho NDVI rasters from ~0.4 m to 10 m using zonal mean statistics.

    The S2 fishnet (10 m cells aligned to S2 pixels) is used as the zonal zone
    layer. For each cell, the mean NDVI of all ortho pixels within it is
    calculated. This produces a 10 m ortho NDVI product that can be directly
    compared to the 10 m S2 NDVI.

    Parameters
    ----------
    area_name             : str   Glacier name
    s2_fishnet_shp_path   : str   10 m fishnet shapefile (zonal zones)
    NDVI_out_dir          : Path  Output directory
    NDVI_output_path_s2   : str   S2 NDVI raster path (not resampled — already 10 m)
    NDVI_output_path_cal  : str   Calibrated ortho NDVI raster path (~0.4 m)
    NDVI_output_path_non_cal: str Non-calibrated ortho NDVI raster path (~0.4 m)

    Returns
    -------
    tuple  (ortho_NDVI_10m, ortho_NDVI_non_cal_10m) — resampled raster objects
    """
    # Resample calibrated NDVI to 10 m
    ortho_NDVI_10m = f"{NDVI_out_dir}/{area_name}_ortho_NDVI_10m.tif"
    Zonal_Statistics = ortho_NDVI_10m
    ortho_NDVI_10m = arcpy.sa.ZonalStatistics(s2_fishnet_shp_path, "FID", Raster(NDVI_output_path_cal), "MEAN", "DATA", "CURRENT_SLICE", 90, "AUTO_DETECT", "ARITHMETIC", 360)
    ortho_NDVI_10m.save(Zonal_Statistics)

    # Resample non-calibrated NDVI to 10 m (for comparison)
    ortho_NDVI_non_cal_10m = f"{NDVI_out_dir}/{area_name}_ortho_NDVI_non-cal_10m.tif"
    Zonal_Statistics = ortho_NDVI_non_cal_10m
    ortho_NDVI_non_cal_10m = arcpy.sa.ZonalStatistics(s2_fishnet_shp_path, "FID", Raster(NDVI_output_path_non_cal), "MEAN", "DATA", "CURRENT_SLICE", 90, "AUTO_DETECT", "ARITHMETIC", 360)
    ortho_NDVI_non_cal_10m.save(Zonal_Statistics)

    return ortho_NDVI_10m, ortho_NDVI_non_cal_10m


def sample_NDVI(ortho_NDVI_10m,
                ortho_NDVI_non_cal_10m,
                NDVI_output_path_s2,
                s2_fishnet_shp_path,
                NDVI_out_dir,
                area_name):
    """
    Sample all three NDVI rasters at S2 fishnet cell centroids and remove NoData.

    Sampling at fishnet centroids gives one row per 10 m cell with values from:
      v_raster_1 — calibrated ortho NDVI (10 m)
      v_raster_2 — non-calibrated ortho NDVI (10 m)
      v_raster_3 — S2 NDVI (10 m)

    Points where any of the three values is NoData (-9999) are identified and
    removed using Select + Erase so that the OLS regression only uses complete
    pixel triplets.

    Parameters
    ----------
    ortho_NDVI_10m       : raster  Calibrated ortho NDVI at 10 m
    ortho_NDVI_non_cal_10m: raster Non-calibrated ortho NDVI at 10 m
    NDVI_output_path_s2  : str     S2 NDVI raster path
    s2_fishnet_shp_path  : str     10 m fishnet shapefile (sampling locations)
    NDVI_out_dir         : Path    Output directory
    area_name            : str     Glacier name

    Returns
    -------
    tuple  (NDVI_samples_path, nodata_points_path, clean_samples_path)
    """
    # Sample all three NDVI rasters at fishnet centroids
    NDVI_samples = f"{NDVI_out_dir}/{area_name}_NDVI_samples.shp"
    arcpy.sa.Sample(
        [ortho_NDVI_10m, ortho_NDVI_non_cal_10m, NDVI_output_path_s2],
        s2_fishnet_shp_path,
        NDVI_samples,
        "NEAREST", "FID", "CURRENT_SLICE", [], "", None, "",
        "ROW_WISE", "FEATURE_CLASS")

    # Select points where any band returned NoData
    sample_nodata = f"{NDVI_out_dir}/{area_name}_NDVI_nodata_samples.shp"
    arcpy.analysis.Select(
        in_features=NDVI_samples,
        out_feature_class=sample_nodata,
        where_clause="v_raster_1 = -9999 Or v_raster_2 = -9999 Or v_raster_3 = -9999")

    # Erase (remove) the NoData points from the full sample set
    sample_nodata_erase = f"{NDVI_out_dir}/{area_name}_NDVI_sample_clean.shp"
    arcpy.analysis.Erase(
        in_features=NDVI_samples,
        erase_features=sample_nodata,
        out_feature_class=sample_nodata_erase)

    return NDVI_samples, sample_nodata, sample_nodata_erase


def mask_samples(out_dir,
                 NDVI_out_dir,
                 area_name,
                 sample_nodata_erase):
    """
    Clip NDVI samples to only keep pixels outside the snow/water/shadow mask.

    The inverted quality mask from Step 1 is converted to polygons, and the
    sample points are clipped to those polygons. This ensures that the OLS
    comparison between ortho NDVI and S2 NDVI only uses pixels from
    vegetated or bare ground areas (not snow, water, or shadows).

    Parameters
    ----------
    out_dir            : Path  Main output directory for this area
    NDVI_out_dir       : Path  NDVI sub-directory for this area
    area_name          : str   Glacier name
    sample_nodata_erase: str   Path to clean NDVI sample shapefile (NoData removed)

    Returns
    -------
    str  Path to the masked and clipped NDVI sample shapefile
    """
    raster_mask = Raster(f"{out_dir}/{area_name}_mask_inv.tif")

    # Convert the inverted mask raster to polygons for use as a clip boundary
    polygon_mask = f"{out_dir}/{area_name}_polygon_mask.shp"
    arcpy.conversion.RasterToPolygon(in_raster=raster_mask, out_polygon_features=polygon_mask, simplify="NO_SIMPLIFY")

    # Clip samples to the valid-pixel polygons
    NDVI_sample_clip = f"{NDVI_out_dir}/{area_name}_NDVI_sample_clip.shp"
    arcpy.analysis.Clip(in_features=sample_nodata_erase, clip_features=polygon_mask, out_feature_class=NDVI_sample_clip)

    return NDVI_sample_clip


def NDVI_regression(NDVI_out_dir,
                    area_name,
                    NDVI_sample_clip):
    """
    Run OLS regressions comparing ortho NDVI (calibrated and non-calibrated) to S2 NDVI.

    Two models are fitted:
      Model 1: S2_NDVI (v_raster_3) ~ calibrated_ortho_NDVI (v_raster_1)
      Model 2: S2_NDVI (v_raster_3) ~ non_calibrated_ortho_NDVI (v_raster_2)

    A slope near 1 and intercept near 0 in Model 1 (calibrated) compared to
    Model 2 (non-calibrated) indicates that the calibration has improved the
    spectral agreement between the two sensors. Output PDF reports provide
    OLS diagnostics and scatter plots.

    Parameters
    ----------
    NDVI_out_dir    : Path  Output directory for NDVI products
    area_name       : str   Glacier name
    NDVI_sample_clip: str   Path to the clean, masked NDVI sample shapefile
    """
    # OLS: S2 NDVI ~ calibrated ortho NDVI
    NDVI_OLS_shp = f"{NDVI_out_dir}/{area_name}_NDVI_OLS.shp"
    NDVI_OLS_pdf = f"{NDVI_out_dir}/{area_name}_NDVI_OLS.pdf"
    arcpy.stats.OrdinaryLeastSquares(
        Input_Feature_Class=NDVI_sample_clip,
        Unique_ID_Field="FID_",
        Output_Feature_Class=NDVI_OLS_shp,
        Dependent_Variable="v_raster_3",     # S2 NDVI
        Explanatory_Variables=["v_raster_1"],  # calibrated ortho NDVI
        Output_Report_File=NDVI_OLS_pdf)

    # OLS: S2 NDVI ~ non-calibrated ortho NDVI (baseline comparison)
    NDVI_OLS_non_cal_shp = f"{NDVI_out_dir}/{area_name}_NDVI_OLS_non-cal.shp"
    NDVI_OLS_non_cal_pdf = f"{NDVI_out_dir}/{area_name}_NDVI_OLS_non-cal.pdf"
    arcpy.stats.OrdinaryLeastSquares(
        Input_Feature_Class=NDVI_sample_clip,
        Unique_ID_Field="FID_",
        Output_Feature_Class=NDVI_OLS_non_cal_shp,
        Dependent_Variable="v_raster_3",      # S2 NDVI
        Explanatory_Variables=["v_raster_2"],  # non-calibrated ortho NDVI
        Output_Report_File=NDVI_OLS_non_cal_pdf)

"""
Step 2 — Band calibration and NDVI computation.

The orthophoto sensor captures reflectance in arbitrary DN units (0–255) that
are not directly comparable to Sentinel-2 surface reflectance values.
This script applies per-area OLS calibration equations derived in Step 1
(Proglacial_project.py) to convert the ortho Red and NIR bands into
pseudo-reflectance values that are spectrally consistent with S2.

For each study area the script:
  1. Reads the per-area OLS coefficients (slope k, intercept m) for the
     Red (Band_1) and NIR (Band_4) orthophoto bands
  2. Applies the linear calibration:  calibrated = k * (DN / 255) + m
  3. Computes NDVI from:
       - Raw (non-calibrated) ortho bands
       - Calibrated ortho bands
       - Sentinel-2 Red (B4) and NIR (B8) bands
  4. Resamples both ortho NDVI layers to 10 m using zonal statistics
     (matching S2 pixel size) for comparison
  5. Samples NDVI values at S2 fishnet centroids, removes NoData,
     applies the quality mask, and runs OLS comparing calibrated
     ortho NDVI to S2 NDVI

Requires: arcpy (ArcGIS Pro with Spatial Analyst and Image Analyst extensions)
"""

import os

os.chdir(r"C:\TEMP\Vanessa_Henriksson")
print(os.getcwd())

import pandas as pd
import geopandas as gpd
from pathlib import Path
from arcgis_functions import *
from NDVI_functions import *

# Allow overwriting existing output files
arcpy.env.overwriteOutput = True

shp_path = "Data/proglacial_outlines.shp"

# Load study area polygons
study_areas = gpd.read_file(shp_path)

# Drop areas not included in this step
study_areas = study_areas.drop(index=[8, 12, 16])

# Load the calibration lookup table which contains per-area OLS coefficients:
#   NIR_coeff, NIR_intercept  → k and m for ortho Band_4 (NIR)
#   red_coeff, red_intercept  → k and m for ortho Band_1 (Red)
lookup = pd.read_csv("Data/Python/lookup_table_coeff.csv")
lookup.head()

# Join study areas with coefficients on glacier name
studarea_merge = study_areas.merge(
    lookup,
    left_on="Glacier_na",
    right_on="Glacier_name",
    how="left"
)

print(studarea_merge.head())

# ── Main loop: calibrate bands and compute NDVI for each study area ────────────
for area in studarea_merge.itertuples():
    area_name = area.Glacier_na

    # Read OLS calibration coefficients for this area
    NIR_k = area.NIR_coeff       # slope for NIR band
    NIR_m = area.NIR_intercept   # intercept for NIR band
    red_k = area.red_coeff       # slope for Red band
    red_m = area.red_intercept   # intercept for Red band

    out_dir = Path(f"Data/Python/Outputs/{area_name}")
    NDVI_out_dir = Path(f"Data/Python/Outputs/{area_name}/NDVI")
    NDVI_out_dir.mkdir(parents=True, exist_ok=True)

    # Extract this area's polygon for masking
    select_area_shp_path = select_area_shp(area_name, shp_path)

    # Paths to pre-processed inputs from Step 1
    # Ortho bands are accessed as sub-layers of the clipped multi-band raster
    clipped_ortho_band_1_path = f"{out_dir}/{area_name}_ortho_clip.tif/Band_1"  # Red
    clipped_ortho_band_4_path = f"{out_dir}/{area_name}_ortho_clip.tif/Band_4"  # NIR
    clipped_s2_b4_path = f"{out_dir}/{area_name}_s2_b4_clip.tif"  # S2 Red
    clipped_s2_b8_path = f"{out_dir}/{area_name}_s2_b8_clip.tif"  # S2 NIR

    # Step 2a: Apply OLS linear calibration to convert ortho DN → pseudo-reflectance
    # calibrated = k * (DN / 255) + m  (same linear form used in band_correlation)
    output_path_1, output_path_4 = calibrate_ortho(area_name,
                                                   NDVI_out_dir,
                                                   clipped_ortho_band_1_path,
                                                   clipped_ortho_band_4_path,
                                                   red_k,
                                                   red_m,
                                                   NIR_k,
                                                   NIR_m)

    print(f"Orthophotos calibrated for {area_name}.")

    # Step 2b: Compute NDVI for three scenarios:
    #   (i)  non-calibrated ortho: raw DN values divided by 255
    #   (ii) calibrated ortho: pseudo-reflectance from step 2a
    #   (iii) Sentinel-2: L2A surface reflectance
    # NDVI = (NIR - Red) / (NIR + Red); ranges from -1 to +1
    NDVI_output_path_non_cal, NDVI_output_path_cal, NDVI_output_path_s2 = calculate_NDVI(area_name,
                                                                                         NDVI_out_dir,
                                                                                         clipped_ortho_band_4_path,
                                                                                         clipped_ortho_band_1_path,
                                                                                         output_path_1,
                                                                                         output_path_4,
                                                                                         clipped_s2_b4_path,
                                                                                         clipped_s2_b8_path)

    print(f"NDVI calculated for {area_name}.")

    # The fishnet was created in Step 1; reuse it here as zonal zones
    s2_fishnet_shp_path = f"{out_dir}/{area_name}_s2_b8_fishnet.shp"

    # Step 2c: Resample ortho NDVI from ~0.4 m to 10 m using zonal mean statistics
    # (one mean NDVI value per 10 m S2 cell) so the two NDVI products are
    # spatially comparable at the same resolution
    ortho_NDVI_10m, ortho_NDVI_non_cal_10m = resample_ortho(area_name,
                                                            s2_fishnet_shp_path,
                                                            NDVI_out_dir,
                                                            NDVI_output_path_s2,
                                                            NDVI_output_path_cal,
                                                            NDVI_output_path_non_cal)

    print(f"Orthophotos resampled for {area_name} using Zonal Statistics.")

    # Step 2d: Sample NDVI values at fishnet centroids and remove NoData points
    NDVI_samples, sample_nodata, sample_nodata_erase = sample_NDVI(ortho_NDVI_10m,
                                                                   ortho_NDVI_non_cal_10m,
                                                                   NDVI_output_path_s2,
                                                                   s2_fishnet_shp_path,
                                                                   NDVI_out_dir,
                                                                   area_name)

    print(f"NDVI sampled for {area_name}")

    # Step 2e: Apply the quality mask from Step 1 to remove snow/water/shadow
    # pixels from the NDVI comparison samples
    NDVI_sample_clip = mask_samples(out_dir,
                                    NDVI_out_dir,
                                    area_name,
                                    sample_nodata_erase)

    print(f"Mask applied on samples for {area_name}.")

    # Step 2f: OLS regression — calibrated ortho NDVI ~ S2 NDVI
    # and non-calibrated ortho NDVI ~ S2 NDVI (for comparison)
    # A slope near 1 and intercept near 0 indicates good spectral agreement.
    NDVI_regression(NDVI_out_dir,
                    area_name,
                    NDVI_sample_clip)

    print(f"OLS complete for {area_name}.")

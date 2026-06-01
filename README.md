# MSc Thesis: Vegetation Mapping and Geodiversity in Swedish Proglacial Areas

This repository contains the Python code developed for Vanessa Henriksson's MSc thesis, which investigates the distribution of vegetation and geodiversity across selected proglacial areas in Sweden.

## Overview

The project combines high-resolution orthophotos and LiDAR-derived terrain data to:

1. Map binary vegetation cover (vegetation / non-vegetation) using Random Forest classification
2. Analyse the relationship between geodiversity and vegetation distribution using Random Forest classification
3. Compute terrain-based geodiversity indices

Scripts for the initial attempt to correct orthophotos against Sentinel-2 data is also included.

Analyses are carried out at two spatial resolutions: **1 m** (orthophoto-based) and **20 m** (Sentinel-2-based).

## Study Areas

Approximately 16 proglacial areas in northern Sweden including Kårsa, Suottas, Vartas, Mikka, Ruotes, Pårte, Storglaciären, Rabots, Isfall, Riuko, Gallan, Unna Räita, Vaktpost, Ballinriehppe II/III, Stuorra, and Helags. All spatial data are projected in **SWEREF99 TM (EPSG:3006)**.

## Data Sources

| Data  | Resolution | Description                                           |
|-------|---------|-------------------------------------------------------|
| Orthophotos | 0.4 m | 4-band (R, G, B, NIR) aerial imagery                  |
| Sentinel-2 | 10–20 m | Bands B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR), SCL |
| DTM   | 1 m | Lantmäteriet Markhöjdmodell                           |
| Snow cover | 20 m | Sentinel-2-derived snow cover fraction                |

## Repository Structure

```
.
├── arcgis_functions.py        # ArcPy helper functions (reprojection, clipping, zonal stats)
├── Proglacial_project.py      # Step 1: Preprocess S2 and orthophoto bands; OLS band correlation
├── NDVI_functions.py          # NDVI calculation helpers (corrected and non-corrected)
├── Calibration_NDVI.py        # Step 2: Correct ortho bands, compute NDVI, OLS correlation
├── Veg_RF_polygon_cv.py       # Step 3: RF vegetation classification with polygon-based cross-validation
├── High_res_script.py         # DTM helpers: mosaic, clip, fill, terrain variable calculation
├── Variable_calculation.py    # Step 4: Calculate terrain variables at 1 m from LiDAR DEM
├── Low_res_script.py          # Step 5: Calculate terrain variables at 20 m; integrate snow cover
├── RF_block_test.py           # Step 6a: RF vegetation prediction at 1 m with block cross-validation
├── rf_vegetation_20m.py       # Step 6b: RF vegetation prediction at 20 m with block cross-validation
├── ndvi_validation.py         # Validate predicted vegetation rasters against S2 NDVI
├── Selection_ratio.py         # Selection ratio and partial dependence plots (both resolutions)
├── geodiversity_index_HL.py   # Geodiversity index: sliding-window diversity (Landforms x TRI method)
├── geodiversity_index_PCA.py  # Geodiversity index: PCA composite of terrain variables
└── geodiv_veg_all_areas.py    # Step 8: Geodiversity–vegetation correlation across all study areas
```

## Workflow

### 1. Preprocessing (`Proglacial_project.py`, `arcgis_functions.py`)
- Reproject Sentinel-2 bands to SWEREF99 TM
- Clip S2 bands and orthophotos to each study area polygon
- Create 10 m fishnet grids aligned to S2 pixels
- Aggregate orthophoto bands to 10 m via zonal statistics
- Compute OLS correlation coefficients between S2 and aggregated ortho bands

### 2. NDVI Correction (`Calibration_NDVI.py`, `NDVI_functions.py`)
- Apply per-area OLS coefficients to correct orthophoto Red and NIR bands
- Compute NDVI for corrected ortho, non-corrected ortho and Sentinel-2 independently
- Compute OLS correlation

### 3. Vegetation Mapping (`Veg_RF_polygon_cv.py`)
- Train a Random Forest classifier (300 trees, max depth 20) on 4-band orthophoto pixels sampled within training polygons
- Binary target: vegetation (1) vs. non-vegetation (2)
- Validation strategies:
  - Random 70/30 train–test split
  - Polygon-based GroupKFold cross-validation (5 folds)
  - "External" validation with additional polygons (Suottas, Vartas)
- Produce predicted vegetation rasters at 0.4 m resolution

### 4. Terrain Variable Calculation (`Variable_calculation.py`, `High_res_script.py`)
- Mosaic and clip LiDAR DEM tiles per study area
- Derive terrain variables via ArcPy Spatial Analyst: slope, aspect (sin/cos), curvature, hillshade, TRI, SWI, landforms, distance from glacier

### 5. Low-Resolution Processing (`Low_res_script.py`)
- Aggregate vegetation from 1 m to 20 m
- Re-calculate terrain rasters at 20 m resolution
- Add snow cover as an additional feature for 20 m models

### 6. RF Prediction with Block Cross-Validation (`RF_block_test.py`, `rf_vegetation_20m.py`, `Selection_ratio.py`)
- Spatial block cross-validation to account for autocorrelation (200 m blocks at 1 m resolution; 400 m at 20 m resolution)
- Features: Landforms, Distance, Aspect (sin/cos), Elevation, Curvature, Hillshade, TRI, SWI (+ Snow cover at 20 m)
- Permutation importance and partial dependence plots
- Selection ratio analysis: compares observed vs. expected vegetation frequency across terrain variables
- Combined Partial dependence plots for all RF models

### 7. Geodiversity Indices (`geodiversity_index_HL.py`, `geodiversity_index_PCA.py`)
- **Landforms x TRI index**: local diversity computed over a 10x10 pixel sliding window
- **PCA index**: composite of Curvature, Landforms, TRI, and SWI; computed at both 1 m and 20 m
- Both indices classified into five levels (Very low to Very high)

### 8. Vegetation–Geodiversity Relationship (`geodiv_veg_all_areas.py`)
- Point-biserial correlation between predicted vegetation and geodiversity indices across all study areas
- Calculation of dominant geodiversity class per study area at both 1 m and 20 m resolution

## Dependencies

### Python packages
- `arcpy` (ArcGIS Pro) — required for preprocessing and terrain variable calculation
- `geopandas`, `fiona`, `shapely`
- `rasterio`
- `scikit-learn`
- `numpy`, `pandas`, `scipy`
- `matplotlib`, `seaborn`

### Software
- ArcGIS Pro with Spatial Analyst and Image Analyst extensions

## Notes

- Local paths (e.g. `C:\TEMP\Vanessa_Henriksson\`) are hardcoded throughout the scripts. Update these to match your environment before running.
- Study area outlines are read from `Data/proglacial_outlines.shp`.
- Lookup tables mapping glacier names to input data paths are stored in `Data/Python/`.
- Outputs are written to `Data/Python/Outputs/` (1 m) and `Data/Python/Outputs_20m/` (20 m).

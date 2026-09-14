# MSc Thesis: Vegetation Mapping and Geodiversity in Swedish Proglacial Areas

This repository contains the Python code developed for Vanessa Henriksson's MSc thesis, which investigates the distribution of vegetation and geodiversity across selected proglacial areas in Sweden.

## Overview

The project combines high-resolution orthophotos and LiDAR-derived terrain data to:

1. Map binary vegetation cover (vegetation / non-vegetation) using Random Forest classification
2. Analyse the relationship between geodiversity and vegetation distribution using Random Forest classification
3. Compute terrain-based geodiversity indices

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
| TRI / SWI | 1 m & 20 m | Terrain Ruggedness Index / SAGA Wetness Index, generated manually (QGIS/SAGA) — not computed by any script in this repo |

## Configuration

All tunable settings — base paths, the study-area exclusion list, Random Forest
hyperparameters, cross-validation settings, and thresholds (vegetation
fraction, sampling fractions/caps, geomorphon search radius, geodiversity
window size, etc.) live in **`config.yaml`** at the repository root, instead
of being hardcoded in each script.

To run the pipeline on your machine:

1. Open `config.yaml` and set `paths.base_dir` to your working directory
   (the equivalent of the old hardcoded `C:\TEMP\Vanessa_Henriksson`), and
   adjust any other paths that differ in your setup.
2. Every script loads this file automatically via `config_utils.load_config()`.
   To use a config file at a different location (e.g. per-collaborator or
   per-study configs), set the `PROGLACIAL_CONFIG` environment variable to
   its path instead of editing `config.yaml` directly.

See the comments in `config.yaml` for what each setting controls and which
script(s) read it.

## Repository Structure

```
.
├── config.yaml                # Central configuration: paths, study areas, model/CV
│                               # settings, thresholds — edit this to adapt the pipeline
├── config_utils.py             # Loads config.yaml into an attribute-accessible object
├── arcgis_functions.py         # Shared ArcPy helper (per-area shapefile selection)
├── Veg_RF_polygon_cv.py        # Step 1: RF vegetation classification with polygon-based cross-validation
├── High_res_script.py          # DTM helpers: mosaic, clip, fill, terrain variable calculation
├── Variable_calculation.py     # Step 2: Calculate terrain variables at 1 m from LiDAR DEM
├── Low_res_script.py           # Step 3: Calculate terrain variables at 20 m; integrate snow cover
├── RF_block_test.py            # Step 4a: RF vegetation prediction at 1 m with block cross-validation
├── rf_vegetation_20m.py        # Step 4b: RF vegetation prediction at 20 m with block cross-validation
├── ndvi_validation.py          # Validate predicted vegetation rasters against S2 NDVI
├── Selection_ratio.py          # Selection ratio and partial dependence plots (both resolutions)
├── geodiversity_index_HL.py    # Geodiversity index: sliding-window diversity (Landforms x TRI method)
├── geodiversity_index_PCA.py   # Geodiversity index: PCA composite of terrain variables
└── geodiv_veg_all_areas.py     # Step 6: Geodiversity–vegetation correlation across all study areas
```

> **Note:** Earlier preprocessing steps that reprojected/clipped Sentinel-2 and
> orthophoto imagery and calibrated ortho NDVI against Sentinel-2 (formerly
> `Proglacial_project.py`, `Calibration_NDVI.py`, `NDVI_functions.py`) have
> been removed — that preprocessing has already been run and its outputs
> exist in `Data/Python/Outputs/`. `ndvi_validation.py` still expects the
> `{area}_NDVI_S2.tif` rasters those scripts produced; if you need to
> regenerate them from scratch for a new area, that preprocessing will need
> to be reimplemented.
>
> **TRI/SWI:** neither the 1 m nor 20 m pipeline computes TRI or SWI —
> both are generated externally (QGIS/SAGA) and read from
> `paths.tri_swi_1m_dir` / `paths.tri_swi_20m_dir` in `config.yaml`, using
> the filename patterns in `geodiversity.tri_filename_1m` /
> `swi_filename_1m` / `tri_filename_20m` / `swi_filename_20m`. Generate
> these before running `Variable_calculation.py`, `RF_block_test.py`,
> `Selection_ratio.py`, or either geodiversity index script.
>
> **Predicted vegetation:** `Veg_RF_polygon_cv.py` is the only vegetation-
> classification script in the repo, so its output at
> `paths.predicted_vegetation_1m` is the single canonical vegetation raster
> that every downstream step (terrain sampling, `geodiv_veg_all_areas.py`,
> `ndvi_validation.py`) reads from.

## Workflow

### 1. Vegetation Mapping (`Veg_RF_polygon_cv.py`)
- Train a Random Forest classifier (300 trees, max depth 20 by default — configurable) on 4-band orthophoto pixels sampled within training polygons
- Binary target: vegetation (1) vs. non-vegetation (2)
- Validation strategies:
  - Random 70/30 train–test split
  - Polygon-based GroupKFold cross-validation (5 folds)
  - "External" validation with additional polygons (Suottas, Vartas)
- Exports a `classification_results_comparison.csv` with the delta between polygon CV and random split, per area and class
- Produce predicted vegetation rasters at 0.4 m resolution

### 2. Terrain Variable Calculation (`Variable_calculation.py`, `High_res_script.py`)
- Mosaic and clip LiDAR DEM tiles per study area
- Derive terrain variables via ArcPy Spatial Analyst: slope, aspect (sin/cos), curvature, hillshade, landforms, distance from glacier
- TRI and SWI are *not* derived here — they are loaded from the externally-generated rasters described above

### 3. Low-Resolution Processing (`Low_res_script.py`)
- Aggregate vegetation from 1 m to 20 m
- Re-calculate terrain rasters at 20 m resolution
- Add snow cover as an additional feature for 20 m models

### 4. RF Prediction with Block Cross-Validation (`RF_block_test.py`, `rf_vegetation_20m.py`, `Selection_ratio.py`)
- Spatial block cross-validation to account for autocorrelation (200 m blocks at 1 m resolution; 400 m at 20 m resolution — configurable)
- Features: Landforms, Distance, Aspect (sin/cos), Elevation, Curvature, Hillshade, TRI, SWI (+ Snow cover at 20 m)
- Also evaluates a random 70/30 split for comparison, and exports a `metrics_comparison.csv` with the delta between block CV and random split per area — a large negative delta flags optimistic bias in the random split
- Permutation importance and partial dependence plots
- Selection ratio analysis: compares observed vs. expected vegetation frequency across terrain variables
- Combined Partial dependence plots for all RF models

### 5. Geodiversity Indices (`geodiversity_index_HL.py`, `geodiversity_index_PCA.py`)
- **Landforms x TRI index**: local diversity computed over a 10x10 pixel sliding window (configurable)
- **PCA index**: composite of Curvature, Landforms, TRI, and SWI; computed at both 1 m and 20 m
- Both indices classified into five levels (Very low to Very high)

### 6. Vegetation–Geodiversity Relationship (`geodiv_veg_all_areas.py`)
- Point-biserial correlation between predicted vegetation and geodiversity indices across all study areas
- Calculation of dominant geodiversity class per study area at both 1 m and 20 m resolution

`ndvi_validation.py` independently validates predicted vegetation rasters against Sentinel-2 NDVI and can be run once the vegetation rasters and (pre-existing) NDVI_S2 rasters are available.

## Dependencies

### Python packages
- `arcpy` (ArcGIS Pro) — required for terrain variable calculation
- `geopandas`, `fiona`, `shapely`
- `rasterio`
- `scikit-learn`
- `numpy`, `pandas`, `scipy`
- `matplotlib`, `seaborn`
- `PyYAML` — for loading `config.yaml`

### Software
- ArcGIS Pro with Spatial Analyst and Image Analyst extensions

## Notes

- Base paths are centralised in `config.yaml` (`paths.base_dir`, plus per-input overrides) — edit that file to match your environment rather than the scripts.
- Study area outlines are read from the path in `config.yaml`'s `paths.outlines_shp` (default: `Data/proglacial_outlines.shp`).
- Lookup tables mapping glacier names to input data paths are stored in `Data/Python/`.
- Outputs are written to `Data/Python/Outputs/` (1 m) and `Data/Python/Outputs_20m/` (20 m).

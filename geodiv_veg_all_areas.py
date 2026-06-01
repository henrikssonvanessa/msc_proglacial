#!/usr/bin/env python
# coding: utf-8
"""
Geodiversity–vegetation relationship across all study areas.

This script quantifies the statistical relationship between each geodiversity
index (H&L and dos Santos PCA) and predicted binary vegetation cover for every
study area, at both 1 m and 20 m resolution.

Point-biserial correlation
--------------------------
The point-biserial correlation (r_pb) is the appropriate measure when one
variable is continuous and the other is binary. Here:
  - Binary variable:     predicted vegetation (1 = vegetated, other = non-veg)
  - Continuous variable: geodiversity index value (continuous float)

r_pb is mathematically equivalent to the Pearson correlation coefficient when
one variable is dichotomous. A positive r_pb indicates that higher geodiversity
values tend to co-occur with vegetated pixels; negative indicates the opposite.

For each area and resolution the script also reports:
  - Which geodiversity class (Very low → Very high) has the highest vegetation
    cover percentage — the 'dominant class' for vegetation.

All rasters are clipped to the study area polygon before analysis to exclude
any cells that overlap with adjacent areas or nodata zones.

Outputs
-------
  Data/Python/Outputs/Geo_veg_relation/all_areas_geodiv_veg_summary.csv
  Data/Python/Outputs/Geo_veg_relation/all_areas_geodiv_veg_correlation.png
"""

import os
os.chdir(r"C:\TEMP\Vanessa_Henriksson")
print(os.getcwd())

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fix random seed so subsampling is reproducible across runs
np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
RESOLUTIONS = {
    '1m':  'Data/Python/Outputs',
    '20m': 'Data/Python/Outputs_20m',
}

NODATA        = -9999
SAMPLE_PIXELS = 200_000   # cap for point-biserial correlation (speed + memory)
GEODIV_LABELS = {1: 'Very low', 2: 'Low', 3: 'Medium', 4: 'High', 5: 'Very high'}
# ─────────────────────────────────────────────────────────────────────────────

study_areas = gpd.read_file("Data/proglacial_outlines.shp")
study_areas = study_areas.drop(index=[12, 14])
if study_areas.crs is None:
    study_areas = study_areas.set_crs('EPSG:3006')  # SWEREF99 TM
area_names  = [area.Glacier_na for area in study_areas.itertuples()]

out_folder = Path("Data/Python/Outputs/Geo_veg_relation")
out_folder.mkdir(parents=True, exist_ok=True)

FALLBACK_CRS = 'EPSG:3006'  # used when a raster file has no embedded CRS metadata


def _apply_polygon_mask(arr, transform, shape, geometry):
    """
    Set pixels outside the study area polygon to NaN.

    Uses rasterio's geometry_mask to create a boolean array that is True
    outside the polygon (invert=False). All data is in SWEREF99 TM so no
    reprojection is needed before masking.

    Parameters
    ----------
    arr      : ndarray  2D float array (modified in place)
    transform: affine   Raster transform (pixel → coordinate mapping)
    shape    : tuple    (height, width) of the raster
    geometry : shapely  Polygon geometry defining the study area boundary

    Returns
    -------
    ndarray  Same array with out-of-polygon pixels set to NaN
    """
    outside = geometry_mask(
        [geometry.__geo_interface__],
        transform=transform, invert=False, out_shape=shape,
    )
    arr[outside] = np.nan
    return arr


def load_raster_array(path, clip_geometry=None):
    """
    Load a raster band as a float array with NoData replaced by NaN.

    Optionally clips the result to a polygon geometry so that only pixels
    within the study area boundary are retained.

    Parameters
    ----------
    path          : str            Path to the raster file
    clip_geometry : shapely or None  Study area polygon for clipping (optional)

    Returns
    -------
    ndarray or None  Float array, or None if the file could not be opened
    """
    try:
        with rasterio.open(path) as src:
            arr    = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            arr[arr == NODATA] = np.nan
            if clip_geometry is not None:
                arr = _apply_polygon_mask(
                    arr, src.transform, (src.height, src.width), clip_geometry,
                )
        return arr
    except Exception as e:
        print(f"  ⚠ Could not load {path}: {e}")
        return None


def align_to_reference(path_ref, path_other, resampling=Resampling.nearest,
                        clip_geometry=None):
    """
    Reproject path_other to match the pixel grid of path_ref.

    Necessary when two rasters cover the same area but have different
    resolutions or slight alignment differences (e.g. 1 m vegetation vs
    1 m geodiversity index from a different origin). Nearest-neighbour is
    the default because the vegetation raster is categorical (1/2); bilinear
    is used for continuous geodiversity arrays.

    Parameters
    ----------
    path_ref      : str           Path to the reference raster (defines target grid)
    path_other    : str           Path to the raster to be reprojected
    resampling    : Resampling    Resampling method (default: nearest)
    clip_geometry : shapely/None  Optional polygon for masking after alignment

    Returns
    -------
    ndarray  Float array aligned to the reference grid
    """
    with rasterio.open(path_ref) as ref_src:
        ref_shape     = (ref_src.height, ref_src.width)
        ref_transform = ref_src.transform
        ref_crs       = ref_src.crs if ref_src.crs is not None else FALLBACK_CRS

    with rasterio.open(path_other) as src:
        arr    = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[arr == NODATA] = np.nan

        # Only reproject if the grid does not already match the reference
        if arr.shape != ref_shape or src.transform != ref_transform:
            dst = np.full(ref_shape, np.nan, dtype=float)
            reproject(
                source=arr, destination=dst,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                resampling=resampling
            )
            arr = dst

    if clip_geometry is not None:
        arr = _apply_polygon_mask(arr, ref_transform, ref_shape, clip_geometry)
    return arr


def point_biserial_corr(veg_arr, geodiv_arr):
    """
    Compute the point-biserial correlation between binary vegetation and a
    continuous geodiversity index.

    The point-biserial r is the standard Pearson correlation applied when
    one variable is dichotomous (0/1 or 1/2). It tests whether pixels with
    higher geodiversity values are more likely to be vegetated.

    If the valid pixel count exceeds SAMPLE_PIXELS, a random subsample is
    drawn (with the seed fixed globally) to keep computation time manageable
    while preserving statistical representativeness.

    Parameters
    ----------
    veg_arr    : ndarray  Binary vegetation raster (1 = vegetated)
    geodiv_arr : ndarray  Continuous geodiversity index (same grid as veg_arr)

    Returns
    -------
    tuple  (r, p, n) — correlation coefficient, p-value, sample size
           Returns (nan, nan, 0) if fewer than 10 valid pixels are available.
    """
    # Keep only pixels where both arrays have valid (non-NaN) values
    valid_mask = ~np.isnan(veg_arr) & ~np.isnan(geodiv_arr)
    veg_flat   = veg_arr[valid_mask]
    geo_flat   = geodiv_arr[valid_mask]

    if len(veg_flat) < 10:
        return np.nan, np.nan, 0

    # Subsample if too many pixels (reproducible via np.random.seed at top)
    if len(veg_flat) > SAMPLE_PIXELS:
        idx      = np.random.choice(len(veg_flat), SAMPLE_PIXELS, replace=False)
        veg_flat = veg_flat[idx]
        geo_flat = geo_flat[idx]

    r, p = stats.pointbiserialr(veg_flat, geo_flat)
    return round(r, 4), round(p, 6), len(veg_flat)


def dominant_geodiv_class(veg_arr, geodiv_classified):
    """
    Find which geodiversity class has the highest vegetation cover percentage.

    For each of the 5 geodiversity classes, counts the proportion of pixels
    within that class that are classified as vegetated (value == 1).
    Returns the label and cover percentage of the class with the highest
    vegetation cover — i.e. the terrain condition most associated with
    vegetation presence.

    Parameters
    ----------
    veg_arr           : ndarray  Binary vegetation raster (1 = vegetated)
    geodiv_classified : ndarray  Integer geodiversity class raster (1–5)

    Returns
    -------
    tuple  (class_label, cover_pct) — e.g. ('High', 72.3)
           Returns ('-', nan) if no valid data is found.
    """
    best_class  = None
    best_cover  = -1

    rows = []
    for c in sorted(GEODIV_LABELS.keys()):
        mask  = (geodiv_classified == c) & ~np.isnan(veg_arr)
        valid = veg_arr[mask]
        if len(valid) == 0:
            continue
        cover = np.sum(valid == 1) / len(valid) * 100
        rows.append((c, cover, int(np.sum(mask))))
        if cover > best_cover:
            best_cover = cover
            best_class = c

    if best_class is None:
        return '-', np.nan
    return GEODIV_LABELS[best_class], round(best_cover, 1)


def bin_hl_to_classes(hl_arr):
    """
    Classify the continuous H&L geodiversity index into 5 levels.

    Uses the same mean ± 0.5/1.5 SD scheme applied in geodiversity_index_HL.py,
    but computed from the per-area distribution rather than the global one.
    This is used here only to identify the dominant geodiversity class for
    vegetation cover; the continuous index is used for the correlation.

    Parameters
    ----------
    hl_arr : ndarray  Continuous H&L geodiversity index array

    Returns
    -------
    ndarray or None  Integer class array (1–5), or None if no valid data
    """
    valid  = hl_arr[~np.isnan(hl_arr)]
    if len(valid) == 0:
        return None
    gm, gs = valid.mean(), valid.std()
    bounds = [-np.inf, gm - 1.5*gs, gm - 0.5*gs,
               gm + 0.5*gs, gm + 1.5*gs, np.inf]
    classified = np.full(hl_arr.shape, np.nan)
    for k in range(5):
        mask = (hl_arr >= bounds[k]) & (hl_arr < bounds[k+1])
        classified[mask] = k + 1
    return classified


# ── Main loop: compute correlations for all study areas ───────────────────────
print("Processing all areas...")
summary_rows = []

for area_row in study_areas.itertuples():
    area_name = area_row.Glacier_na
    area_geom = area_row.geometry  # polygon used to clip rasters

    print(f"\n  {area_name}")
    row = {'Area': area_name}

    for res_label, res_folder in RESOLUTIONS.items():

        # Resolve paths for this area and resolution
        # The 1 m vegetation raster lives in a shared folder (not per-area)
        veg_path = f"{res_folder}/{area_name}/Geodiversity/{area_name}_predicted_vegetation_{res_label}.tif"
        if res_label == '1m':
            veg_path = f"Data/Python/Outputs/Predicted_vegetation/{area_name}_predicted_vegetation.tif"

        hl_path            = f"{res_folder}/{area_name}/Geodiversity/{area_name}_geoindex_hl_{res_label}.tif"
        ds_path            = f"{res_folder}/{area_name}/Geodiversity/{area_name}_geoindex_{res_label}.tif"
        ds_classified_path = f"{res_folder}/{area_name}/Geodiversity/{area_name}_geoindex_classified_{res_label}.tif"

        # Load vegetation raster clipped to this area's polygon
        veg_arr = load_raster_array(veg_path, clip_geometry=area_geom)
        if veg_arr is None:
            # Fill with NaN if the file is missing
            for col in [f'H&L r ({res_label})', f'DS r ({res_label})',
                        f'H&L dominant class ({res_label})', f'H&L veg in dominant % ({res_label})',
                        f'DS dominant class ({res_label})', f'DS veg in dominant % ({res_label})']:
                row[col] = np.nan
            continue

        # ── H&L method ────────────────────────────────────────────────────────
        hl_arr = load_raster_array(hl_path, clip_geometry=area_geom)
        if hl_arr is not None:
            # Align H&L grid to vegetation grid if they differ (e.g. due to
            # different origins from ArcPy vs rasterio outputs)
            if hl_arr.shape != veg_arr.shape:
                hl_arr = align_to_reference(veg_path, hl_path, Resampling.bilinear,
                                            clip_geometry=area_geom)

            # Point-biserial r: vegetation (binary) ~ H&L index (continuous)
            r, p, n = point_biserial_corr(veg_arr, hl_arr)
            row[f'H&L r ({res_label})'] = r

            # Identify which geodiversity class has the most vegetation
            hl_classified = bin_hl_to_classes(hl_arr)
            if hl_classified is not None:
                dom_class, dom_cover = dominant_geodiv_class(veg_arr, hl_classified)
                row[f'H&L dominant class ({res_label})']    = dom_class
                row[f'H&L veg in dominant % ({res_label})'] = dom_cover
        else:
            row[f'H&L r ({res_label})']                 = np.nan
            row[f'H&L dominant class ({res_label})']    = '-'
            row[f'H&L veg in dominant % ({res_label})'] = np.nan

        # ── dos Santos PCA method ─────────────────────────────────────────────
        ds_arr = load_raster_array(ds_path, clip_geometry=area_geom)
        if ds_arr is not None:
            if ds_arr.shape != veg_arr.shape:
                ds_arr = align_to_reference(veg_path, ds_path, Resampling.bilinear,
                                            clip_geometry=area_geom)

            # Point-biserial r: vegetation (binary) ~ dos Santos index (continuous)
            r, p, n = point_biserial_corr(veg_arr, ds_arr)
            row[f'DS r ({res_label})'] = r

            # Load pre-classified dos Santos raster (avoids re-classifying here)
            ds_classified = load_raster_array(ds_classified_path, clip_geometry=area_geom)
            if ds_classified is not None:
                if ds_classified.shape != veg_arr.shape:
                    ds_classified = align_to_reference(veg_path, ds_classified_path,
                                                       clip_geometry=area_geom)
                dom_class, dom_cover = dominant_geodiv_class(veg_arr, ds_classified)
                row[f'DS dominant class ({res_label})']    = dom_class
                row[f'DS veg in dominant % ({res_label})'] = dom_cover
        else:
            row[f'DS r ({res_label})']                 = np.nan
            row[f'DS dominant class ({res_label})']    = '-'
            row[f'DS veg in dominant % ({res_label})'] = np.nan

    summary_rows.append(row)
    print(f"    H&L r: 1m={row.get('H&L r (1m)', 'n/a')}, 20m={row.get('H&L r (20m)', 'n/a')} | "
          f"DS r: 1m={row.get('DS r (1m)', 'n/a')}, 20m={row.get('DS r (20m)', 'n/a')}")

# ── Build and save summary table ──────────────────────────────────────────────
col_order = [
    'Area',
    'H&L r (1m)', 'H&L r (20m)',
    'DS r (1m)',  'DS r (20m)',
    'H&L dominant class (1m)',    'H&L veg in dominant % (1m)',
    'H&L dominant class (20m)',   'H&L veg in dominant % (20m)',
    'DS dominant class (1m)',     'DS veg in dominant % (1m)',
    'DS dominant class (20m)',    'DS veg in dominant % (20m)',
]

df_summary = pd.DataFrame(summary_rows)[col_order]
df_summary.to_csv(out_folder / "all_areas_geodiv_veg_summary.csv",
                  index=False, encoding='utf-8-sig')

print("\nSummary table:")
print(df_summary.to_string(index=False))
print("\nSaved to all_areas_geodiv_veg_summary.csv")

# ── Bar chart: point-biserial r per area, both methods, both resolutions ──────
# Two subplots (top = 1 m, bottom = 20 m); each subplot shows paired bars for
# the H&L (orange) and dos Santos PCA (blue) methods.
# r values are annotated above/below bars; the zero line is drawn for reference.
matplotlib.rcParams['font.family'] = 'Calibri'

df_plot = df_summary.sort_values('Area').reset_index(drop=True)
plot_names = df_plot['Area'].tolist()

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

x     = np.arange(len(plot_names))
width = 0.35

for ax, res_label in zip(axes, ['1m', '20m']):
    hl_vals = df_plot[f'H&L r ({res_label})'].values
    ds_vals = df_plot[f'DS r ({res_label})'].values

    bars_hl = ax.bar(x - width/2, hl_vals, width=width,
                     color='darkorange', alpha=0.7, label='Landforms x TRI method')
    bars_ds = ax.bar(x + width/2, ds_vals, width=width,
                     color='steelblue',  alpha=0.7, label='PCA method')

    # Compute y-axis padding to make room for the legend without overlapping bars
    def _safe_max_abs(vals):
        v = vals[~np.isnan(vals)]
        return float(np.max(np.abs(v))) if len(v) > 0 else 0.0

    all_vals   = np.concatenate([hl_vals, ds_vals])
    valid_vals = all_vals[~np.isnan(all_vals)]
    y_min = valid_vals.min() if len(valid_vals) else -0.1
    y_max = valid_vals.max() if len(valid_vals) else 0.1
    span  = y_max - y_min

    if ax is axes[0]:
        # Place legend in the corner with the smallest bar values
        left_abs  = _safe_max_abs(np.concatenate([hl_vals[:2],  ds_vals[:2]]))
        right_abs = _safe_max_abs(np.concatenate([hl_vals[-2:], ds_vals[-2:]]))
        top_val   = float(np.nanmax(all_vals)) if len(valid_vals) > 0 else 0.0
        bot_val   = float(abs(np.nanmin(all_vals))) if len(valid_vals) > 0 else 0.0
        h_side    = 'left' if left_abs <= right_abs else 'right'
        v_side    = 'lower' if top_val >= bot_val else 'upper'
        # Give extra padding on the side where the legend will be placed
        pad_top = span * (0.55 if v_side == 'upper' else 0.3)
        pad_bot = span * (0.55 if v_side == 'lower' else 0.3)
    else:
        pad_top = span * 0.3
        pad_bot = span * 0.3

    ax.set_ylim(y_min - pad_bot, y_max + pad_top)

    # Annotate bar heights with r values
    for bar, val in zip(bars_hl, hl_vals):
        if not np.isnan(val):
            y_pos = val + 0.005 if val >= 0 else val - 0.005
            va    = 'bottom'    if val >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val:.2f}', ha='center', va=va, fontsize=16, rotation=90)
    for bar, val in zip(bars_ds, ds_vals):
        if not np.isnan(val):
            y_pos = val + 0.005 if val >= 0 else val - 0.005
            va    = 'bottom'    if val >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val:.2f}', ha='center', va=va, fontsize=16, rotation=90)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("Point-biserial r", fontsize=22)
    ax.set_title(f"Geodiversity–Vegetation correlation — {res_label}", fontsize=23,
                 fontweight='bold')
    if ax is axes[0]:
        ax.legend(fontsize=21, loc=f'{v_side} {h_side}', framealpha=1.0)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax.tick_params(axis='y', labelsize=18)

# Only the bottom subplot shows x-axis labels (shared x-axis)
axes[1].set_xticks(x)
axes[1].set_xticklabels(
    [a.replace('_', ' ').replace(' II III', '').replace('_II_III', '') for a in plot_names],
    rotation=45, ha='right', fontsize=20)

plt.tight_layout()
plt.savefig(out_folder / "all_areas_geodiv_veg_correlation.png",
            bbox_inches='tight', dpi=150)
plt.close()

print("Correlation figure saved.")
print("\nAll done!")

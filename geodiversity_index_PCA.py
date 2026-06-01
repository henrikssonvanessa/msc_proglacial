#!/usr/bin/env python
# coding: utf-8
"""
Step 7b — PCA-weighted composite geodiversity index (dos Santos method).

This script computes a geodiversity index following the method of
dos Santos et al., using Principal Component Analysis (PCA) to derive
data-driven weights for combining terrain variables.

Method
------
Variables used: Curvature, Landforms, TRI, SWI

These four variables capture different aspects of terrain geodiversity:
  - Curvature: shape of the terrain surface (concave/convex)
  - Landforms: topographic position class (valley, ridge, slope etc.)
  - TRI (Terrain Ruggedness Index): local elevation variability
  - SWI (SAGA Wetness Index): topographic wetness potential

The index is computed in three passes:

Pass 1 — Global normalisation
  Load all four variables for every study area. Track the global min and max
  of each variable across all areas. Normalise to 0–1 using these global values
  so that variables are on the same scale and comparable across areas.

Pass 2 — PCA weights
  Stack all normalised pixels from all areas into one matrix and fit PCA.
  The weight of each variable is derived from its contribution across all
  principal components, following dos Santos:
      raw_weight_i = Σ_j |eigenvector_ij| × eigenvalue_j
      weight_i = raw_weight_i / Σ raw_weights
  This gives data-driven weights that reflect each variable's overall
  explanatory power across the terrain dataset.

Pass 3 — Weighted composite index
  Index = Σ_i  weight_i × normalised_variable_i
  Higher values indicate greater geodiversity.

Classification
  The index is classified into 5 levels using global mean ± 0.5/1.5 SD
  (same scheme as the H&L method for comparability).

Outputs (per resolution, per area):
  {area}_geoindex_{res}.tif            — continuous index (float32)
  {area}_geoindex_classified_{res}.tif — 5-class index (int16)
  geodiversity_weights_{res}.csv       — PCA-derived variable weights
  Figures/Geodiversity/geodiversity_index_{res}.png — overview figure

Requires: rasterio, scikit-learn (PCA), numpy
"""

import os
os.chdir(r"C:\TEMP\Vanessa_Henriksson")
print(os.getcwd())

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── CONFIG ────────────────────────────────────────────────────────────────────
RESOLUTIONS = {
    '1m':  'Data/Python/Outputs',
    '20m': 'Data/Python/Outputs_20m',
}

# Variables included in the geodiversity index
FEATURE_COLS = ['Curvature', 'Landforms', 'TRI', 'SWI']

# Filename templates — {area} is replaced with the glacier name
RASTER_NAMES = {
    'Curvature':  '{area}_curvature.tif',
    'Landforms':  '{area}_landforms.tif',
    'TRI':        '{area}_TRI.tif',
    'SWI':        '{area}_SWI.tif'
}

NODATA_VAL = -9999
# ─────────────────────────────────────────────────────────────────────────────

study_areas = gpd.read_file("Data/proglacial_outlines.shp")
study_areas = study_areas.drop(index=[12, 14])
area_names  = [area.Glacier_na for area in study_areas.itertuples()]


def load_raster(path):
    """Load raster as a float array with NoData replaced by NaN."""
    with rasterio.open(path) as src:
        arr    = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[arr == NODATA_VAL] = np.nan

        meta      = src.meta.copy()
        shape     = arr.shape
        transform = src.transform
        crs       = src.crs

    return arr, meta, shape, transform, crs


def align_to_reference(src_arr, src_transform, src_crs,
                       ref_shape, ref_transform, ref_crs):
    """
    Reproject src_arr to match the reference grid using bilinear resampling.

    Used to ensure all four variables are pixel-aligned before combining them
    into the composite index.
    """
    dst = np.full(ref_shape, np.nan, dtype=float)

    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear
    )

    return dst


def get_raster_path(res_folder, area_name, feature):
    """
    Return the path to the variable raster for a given area and resolution.

    TRI and SWI at 20 m were pre-computed by SAGA GIS and use a different
    naming convention and directory from the ArcPy-derived 1 m outputs.
    """
    if res_folder.endswith("Outputs_20m"):
        if feature == 'TRI':
            return rf"C:\TEMP\Vanessa_Henriksson\Data\DEM\TRI_SWI_20m\TIF_TRI_{area_name}_DEM_clip.tif"
        elif feature == 'SWI':
            return rf"C:\TEMP\Vanessa_Henriksson\Data\DEM\TRI_SWI_20m\TIF_SWI_{area_name}_DEM_fill.tif"

    filename = RASTER_NAMES[feature].replace('{area}', area_name)
    return f"{res_folder}/{area_name}/Geodiversity/{filename}"


# ── Process each resolution ───────────────────────────────────────────────────
for res_label, res_folder in RESOLUTIONS.items():
    print(f"\n{'='*60}")
    print(f"Processing {res_label} resolution")
    print(f"{'='*60}")

    fig_folder = Path(f"{res_folder}/Figures/Geodiversity")
    fig_folder.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: load all variable rasters and track global min/max ─────────────
    # Global min/max ensures the same normalisation scale is used across all areas,
    # so a normalised value of 0.5 is physically comparable regardless of area.
    print("\nPass 1 — collecting global min/max per variable...")
    global_min = {f:  np.inf for f in FEATURE_COLS}
    global_max = {f: -np.inf for f in FEATURE_COLS}

    # Cache arrays and metadata for use in passes 2 and 3
    area_arrays     = {area: {} for area in area_names}
    area_metas      = {}
    area_shapes     = {}
    area_transforms = {area: {} for area in area_names}
    area_crs        = {}

    for area_name in area_names:
        for feature in FEATURE_COLS:
            path = get_raster_path(res_folder, area_name, feature)
            try:
                arr, meta, shape, transform, crs = load_raster(path)

                area_arrays[area_name][feature]     = arr
                area_transforms[area_name][feature] = transform
                area_crs[area_name]                 = crs
                area_metas[area_name]               = meta
                area_shapes[area_name]              = shape

                # Update running global min/max for this variable
                valid = arr[~np.isnan(arr)]
                if len(valid) > 0:
                    global_min[feature] = min(global_min[feature], valid.min())
                    global_max[feature] = max(global_max[feature], valid.max())
            except Exception as e:
                print(f"  ⚠ Could not load {feature} for {area_name}: {e}")

    print("\nGlobal variable ranges:")
    for f in FEATURE_COLS:
        print(f"  {f:15s}: {global_min[f]:.3f} – {global_max[f]:.3f}")

    # ── Pass 2: normalise all pixels and fit PCA to derive weights ─────────────
    # Collect normalised pixels from all areas into a single matrix.
    # SWI is used as the reference grid — all other variables are aligned to it
    # before stacking, because SWI at 20 m may have slightly different alignment.
    print("\nPass 2 — normalizing and computing PCA weights...")

    all_pixels = []  # will hold normalised pixel rows from all areas

    for area_name in area_names:
        if 'SWI' not in area_arrays[area_name]:
            continue

        ref_feature   = 'SWI'
        ref_arr       = area_arrays[area_name][ref_feature]
        ref_shape     = ref_arr.shape
        ref_transform = area_transforms[area_name][ref_feature]
        ref_crs       = area_crs[area_name]

        aligned = []  # normalised arrays aligned to the SWI grid

        for feature in FEATURE_COLS:
            arr = area_arrays[area_name].get(feature)
            if arr is None:
                break

            transform = area_transforms[area_name][feature]

            # Reproject to SWI grid if needed
            if arr.shape != ref_shape or transform != ref_transform:
                arr = align_to_reference(arr, transform, ref_crs,
                                         ref_shape, ref_transform, ref_crs)

            # Global min-max normalisation: (x - min) / (max - min) → [0, 1]
            rng  = global_max[feature] - global_min[feature]
            norm = (arr - global_min[feature]) / rng if rng != 0 else np.zeros_like(arr)

            aligned.append(norm)

        if len(aligned) < len(FEATURE_COLS):
            continue

        # Stack variables along the last axis: shape = (rows, cols, n_vars)
        stacked    = np.stack(aligned, axis=-1)
        stacked_2d = stacked.reshape(-1, len(FEATURE_COLS))
        # Keep only rows where all variables are valid
        valid      = ~np.any(np.isnan(stacked_2d), axis=1)
        all_pixels.append(stacked_2d[valid])

    all_pixels = np.vstack(all_pixels)
    print(f"  Total valid pixels for PCA: {len(all_pixels):,}")

    # Fit PCA on the combined pixel matrix from all areas
    pca = PCA(n_components=len(FEATURE_COLS))
    pca.fit(all_pixels)

    eigenvalues  = pca.explained_variance_      # variance explained by each PC
    eigenvectors = pca.components_.T            # shape: (n_vars, n_components)

    # dos Santos weight formula:
    # For each variable i, sum |loading_ij| × eigenvalue_j across all PCs j
    # Then normalise so weights sum to 1.
    # This weights variables by how strongly they contribute to PCs with high
    # explanatory power.
    raw_weights = np.sum(np.abs(eigenvectors * eigenvalues), axis=1)
    weights     = raw_weights / raw_weights.sum()

    print("\nPCA variable weights (dos Santos method):")
    weight_df = pd.DataFrame({
        'Variable': FEATURE_COLS,
        'Weight':   np.round(weights, 4),
        'Weight_%': np.round(weights * 100, 2)
    }).sort_values('Weight_%', ascending=False)
    print(weight_df.to_string(index=False))

    weight_df.to_csv(
        Path(res_folder) / f"geodiversity_weights_{res_label}.csv",
        index=False, encoding='utf-8-sig'
    )

    print(f"\n  PCA explained variance per component:")
    for i, ev in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {ev*100:.1f}% "
              f"(cumulative: {pca.explained_variance_ratio_[:i+1].sum()*100:.1f}%)")

    # ── Pass 3: compute weighted composite index per area ──────────────────────
    print("\nPass 3 — computing geodiversity index per area...")

    all_geoindex_values = []
    geoindex_maps       = {}

    for area_name in area_names:
        if 'SWI' not in area_arrays[area_name]:
            continue

        ref_feature   = 'SWI'
        ref_arr       = area_arrays[area_name][ref_feature]
        ref_shape     = ref_arr.shape
        ref_transform = area_transforms[area_name][ref_feature]
        ref_crs       = area_crs[area_name]

        normalized = {}  # normalised variable arrays aligned to SWI grid

        for feature in FEATURE_COLS:
            arr = area_arrays[area_name].get(feature)
            if arr is None:
                continue

            transform = area_transforms[area_name][feature]

            if arr.shape != ref_shape or transform != ref_transform:
                arr = align_to_reference(arr, transform, ref_crs,
                                         ref_shape, ref_transform, ref_crs)

            rng = global_max[feature] - global_min[feature]
            normalized[feature] = (arr - global_min[feature]) / rng if rng != 0 else np.zeros_like(arr)

        if len(normalized) < len(FEATURE_COLS):
            print(f"  ⚠ Skipping {area_name} — missing variables")
            continue

        # Weighted sum: index = Σ weight_i × normalised_variable_i
        geoindex = np.zeros(ref_shape, dtype=float)
        nan_mask = np.zeros(ref_shape, dtype=bool)

        for j, feature in enumerate(FEATURE_COLS):
            arr = normalized[feature]
            nan_mask |= np.isnan(arr)
            # Replace NaN with 0 for summation; will be masked after loop
            geoindex += weights[j] * np.where(np.isnan(arr), 0, arr)

        geoindex[nan_mask] = np.nan

        geoindex_maps[area_name] = geoindex
        valid_vals = geoindex[~nan_mask]

        if len(valid_vals) == 0:
            print(f"  ⚠ {area_name}: no valid pixels after masking — skipping")
            del geoindex_maps[area_name]
            continue

        all_geoindex_values.append(valid_vals)
        print(f"  {area_name}: Geoindex range {valid_vals.min():.3f} – {valid_vals.max():.3f}")

    # ── Classify into 5 geodiversity levels ───────────────────────────────────
    all_vals    = np.concatenate(all_geoindex_values)
    global_mean = all_vals.mean()
    global_std  = all_vals.std()

    bounds = [
        -np.inf,
        global_mean - 1.5 * global_std,
        global_mean - 0.5 * global_std,
        global_mean + 0.5 * global_std,
        global_mean + 1.5 * global_std,
        np.inf
    ]
    class_labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']

    print(f"\nGlobal Geoindex — mean: {global_mean:.3f}, std: {global_std:.3f}")
    print("Classification boundaries:")
    for k, label in enumerate(class_labels):
        print(f"  {label:10s}: {bounds[k]:.3f} – {bounds[k+1]:.3f}")

    # ── Save rasters ──────────────────────────────────────────────────────────
    cmap = plt.cm.RdYlGn

    for area_name, geoindex in geoindex_maps.items():
        meta = area_metas[area_name].copy()
        meta.update({
            'height':    geoindex.shape[0],
            'width':     geoindex.shape[1],
            'transform': area_transforms[area_name]['SWI'],  # aligned to SWI grid
            'dtype':     'float32',
            'count':     1,
            'nodata':    -9999,
            'compress':  'lzw',
            'crs':       rasterio.crs.CRS.from_epsg(3006),
        })

        # Create 5-class integer raster
        classified = np.zeros(geoindex.shape, dtype=np.int16)
        for k in range(5):
            classified[(geoindex >= bounds[k]) & (geoindex < bounds[k+1])] = k + 1
        classified[np.isnan(geoindex)] = -9999

        # Save continuous index
        out_cont = Path(res_folder) / area_name / "Geodiversity" / \
                   f"{area_name}_geoindex_{res_label}.tif"
        with rasterio.open(out_cont, 'w', **meta) as dst:
            dst.write(geoindex.astype('float32'), 1)

        # Save classified index
        meta_class = meta.copy()
        meta_class['dtype'] = 'int16'
        out_class = Path(res_folder) / area_name / "Geodiversity" / \
                    f"{area_name}_geoindex_classified_{res_label}.tif"
        with rasterio.open(out_class, 'w', **meta_class) as dst:
            dst.write(classified.astype('int16'), 1)

    # ── Combined overview figure ──────────────────────────────────────────────
    n_cols_fig = 4
    n_rows_fig = int(np.ceil(len(geoindex_maps) / n_cols_fig))
    fig, axes  = plt.subplots(n_rows_fig, n_cols_fig,
                               figsize=(5 * n_cols_fig, 4 * n_rows_fig),
                               constrained_layout=True)
    axes_flat  = axes.flatten()

    # Shared colour scale: global mean ± 2 SD
    vmin = global_mean - 2 * global_std
    vmax = global_mean + 2 * global_std
    im   = None

    for i, (area_name, geoindex) in enumerate(geoindex_maps.items()):
        ax  = axes_flat[i]
        im  = ax.imshow(geoindex, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(area_name.replace("_", " "), fontsize=10, fontweight='bold')
        ax.axis('off')

    for j in range(len(geoindex_maps), len(axes_flat)):
        axes_flat[j].set_visible(False)

    if im is not None:
        fig.colorbar(im, ax=axes_flat[:len(geoindex_maps)],
                     orientation='vertical', fraction=0.02, pad=0.02,
                     label='Geoindex', shrink=0.6)

    fig.suptitle(f"Geodiversity Index — {res_label} resolution (dos Santos method)",
                 fontsize=14, fontweight='bold')
    plt.savefig(fig_folder / f"geodiversity_index_{res_label}.png",
                bbox_inches='tight', dpi=150)
    plt.close()

    print(f"\nFigures and rasters saved for {res_label} resolution.")

print("\nAll done!")

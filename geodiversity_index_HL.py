#!/usr/bin/env python
# coding: utf-8
"""
Step 7a — Hjort & Luoto (H&L) geodiversity index.

This script computes a geodiversity index following the approach of
Hjort & Luoto (2006), adapted for proglacial terrain.

Method
------
Geodiversity = normalised landform diversity × normalised terrain roughness (TRI)

1. Landform diversity: for each pixel, count the number of unique geomorphon
   landform classes (1–10) within a 10 × 10 pixel moving window (10 m × 10 m
   at 1 m resolution; 200 m × 200 m at 20 m resolution). Higher diversity
   indicates more varied terrain — ridges, valleys, slopes etc. present nearby.

2. TRI (Terrain Ruggedness Index): measures the mean absolute difference in
   elevation between a pixel and its 8 neighbours. High TRI = rough, highly
   variable terrain; low TRI = smooth terrain.

3. Both variables are globally normalised to 0–1 across all study areas using
   their global min/max values (so values are comparable across areas).

4. Index = normalised_landform_diversity × normalised_TRI
   The multiplicative form rewards locations that are both morphologically
   diverse AND topographically rough — if either component is near zero, the
   index is low.

5. The index is classified into 5 levels using global mean ± 0.5/1.5 SD:
     1 = Very low, 2 = Low, 3 = Medium, 4 = High, 5 = Very high

Outputs (per resolution, per area):
  {area}_geoindex_hl_{res}.tif            — continuous index (float32)
  {area}_geoindex_hl_classified_{res}.tif — 5-class index (int16)
  Figures/Geodiversity/geodiversity_index_hl_{res}.png — overview figure

Requires: rasterio, scipy, numpy
"""

import os
os.chdir(r"C:\TEMP\Vanessa_Henriksson")
print(os.getcwd())

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import generic_filter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
RESOLUTIONS = {
    '1m':  'Data/Python/Outputs',    # 1 m: window = 10 × 10 m physical extent
    '20m': 'Data/Python/Outputs_20m', # 20 m: window = 200 × 200 m physical extent
}

WINDOW_SIZE = 10   # pixels; at 1 m res = 10 m, at 20 m res = 200 m

NODATA_VAL  = -9999
# ─────────────────────────────────────────────────────────────────────────────

study_areas = gpd.read_file("Data/proglacial_outlines.shp")
study_areas = study_areas.drop(index=[12, 14])
area_names  = [area.Glacier_na for area in study_areas.itertuples()]


def load_raster(path):
    """Load a raster as a float array with NoData replaced by NaN."""
    with rasterio.open(path) as src:
        arr    = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[arr == NODATA_VAL] = np.nan
        meta      = src.meta.copy()
        transform = src.transform
        crs       = src.crs
    return arr, meta, transform, crs


def align_to_reference(src_arr, src_transform, src_crs,
                       ref_shape, ref_transform, ref_crs):
    """
    Reproject src_arr to match the reference grid (shape, transform, CRS).

    Used to ensure TRI and landform diversity rasters are on exactly the same
    pixel grid before multiplying them together.
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


def get_tri_path(res_folder, area_name):
    """
    Return the path to the TRI raster for a given area and resolution.

    At 20 m resolution, TRI was pre-computed by SAGA GIS and stored in a
    separate directory with a different naming convention.
    """
    if res_folder.endswith('Outputs_20m'):
        return rf"C:\TEMP\Vanessa_Henriksson\Data\DEM\TRI_SWI_20m\TIF_TRI_{area_name}_DEM_clip.tif"
    return f"{res_folder}/{area_name}/Geodiversity/{area_name}_TRI.tif"


def count_unique_landforms(values, nodata_val=NODATA_VAL):
    """
    Count the number of unique valid landform classes in a moving window.

    Used as the function passed to scipy.ndimage.generic_filter.
    NaN values (masked pixels) are excluded before counting.
    Returns 0 if no valid values are present in the window.
    """
    valid = values[~np.isnan(values)]
    return len(np.unique(valid)) if len(valid) > 0 else 0


def minmax_normalize(arr, global_min, global_max):
    """
    Normalise array to 0–1 using the global min/max across all study areas.

    Global normalisation ensures values are comparable across areas — a value
    of 0.5 means the same thing regardless of which area it comes from.
    Returns an array of zeros if global_min == global_max (no variation).
    """
    rng = global_max - global_min
    if rng == 0:
        return np.zeros_like(arr)
    return (arr - global_min) / rng


# ── Process each resolution ───────────────────────────────────────────────────
for res_label, res_folder in RESOLUTIONS.items():
    print(f"\n{'='*60}")
    print(f"Processing {res_label} resolution — Hjort & Luoto method")
    print(f"{'='*60}")

    fig_folder = Path(f"{res_folder}/Figures/Geodiversity")
    fig_folder.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: compute landform diversity and load TRI ────────────────────────
    # Also track global min/max for both variables across all areas
    # (needed for normalisation in Pass 2)
    print("\nPass 1 — computing landform diversity and loading TRI...")

    landform_div_maps  = {}  # {area: 2D diversity array}
    tri_maps           = {}  # {area: 2D TRI array}
    tri_transforms     = {}
    tri_crs_store      = {}
    area_metas         = {}
    area_transforms    = {}
    area_crs_store     = {}

    global_tri_min =  np.inf
    global_tri_max = -np.inf
    global_div_min =  np.inf
    global_div_max = -np.inf

    for area_name in area_names:
        # Load geomorphon landform raster (integer classes 1–10)
        lf_path = f"{res_folder}/{area_name}/Geodiversity/{area_name}_landforms.tif"
        try:
            lf_arr, meta, transform, crs = load_raster(lf_path)
        except Exception as e:
            print(f"  ⚠ Could not load Landforms for {area_name}: {e}")
            continue

        area_metas[area_name]      = meta
        area_transforms[area_name] = transform
        area_crs_store[area_name]  = crs

        # Apply moving-window diversity: count unique landform classes per window
        # generic_filter applies count_unique_landforms to each WINDOW_SIZE × WINDOW_SIZE
        # neighbourhood and assigns the result to the central pixel.
        print(f"  {area_name}: computing landform diversity ({WINDOW_SIZE}x{WINDOW_SIZE} window)...")
        lf_div = generic_filter(
            lf_arr,
            count_unique_landforms,
            size=WINDOW_SIZE,
            mode='constant',
            cval=np.nan  # treat pixels outside the raster as NaN
        )
        # Propagate NoData mask from the original landform raster
        lf_div[np.isnan(lf_arr)] = np.nan
        landform_div_maps[area_name] = lf_div

        valid_div = lf_div[~np.isnan(lf_div)]
        if len(valid_div) > 0:
            global_div_min = min(global_div_min, valid_div.min())
            global_div_max = max(global_div_max, valid_div.max())

        # Load TRI (pre-computed separately)
        tri_path = get_tri_path(res_folder, area_name)
        try:
            tri_arr, _, tri_transform, tri_crs = load_raster(tri_path)
            tri_maps[area_name]       = tri_arr
            tri_transforms[area_name] = tri_transform
            tri_crs_store[area_name]  = tri_crs

            valid_tri = tri_arr[~np.isnan(tri_arr)]
            if len(valid_tri) > 0:
                global_tri_min = min(global_tri_min, valid_tri.min())
                global_tri_max = max(global_tri_max, valid_tri.max())
        except Exception as e:
            print(f"  ⚠ Could not load TRI for {area_name}: {e}")

    print(f"\n  Landform diversity range: {global_div_min:.1f} – {global_div_max:.1f}")
    print(f"  TRI range:                {global_tri_min:.3f} – {global_tri_max:.3f}")

    # ── Pass 2: compute H&L geodiversity index per area ───────────────────────
    print("\nPass 2 — computing Hjort & Luoto geodiversity index...")

    geoindex_maps       = {}
    all_geoindex_values = []

    for area_name in area_names:
        if area_name not in landform_div_maps or area_name not in tri_maps:
            print(f"  ⚠ Skipping {area_name} — missing data")
            continue

        lf_div  = landform_div_maps[area_name]
        tri_arr = tri_maps[area_name]

        # Align TRI to the landform grid if they differ in extent or cell alignment
        ref_shape     = lf_div.shape
        ref_transform = area_transforms[area_name]
        ref_crs       = area_crs_store[area_name]

        if tri_arr.shape != ref_shape or tri_transforms[area_name] != ref_transform:
            tri_arr = align_to_reference(
                tri_arr, tri_transforms[area_name], tri_crs_store[area_name],
                ref_shape, ref_transform, ref_crs
            )

        # Normalise both components to 0–1 using global ranges
        lf_norm  = minmax_normalize(lf_div,  global_div_min, global_div_max)
        tri_norm = minmax_normalize(tri_arr, global_tri_min, global_tri_max)

        # H&L index: product of diversity and roughness
        # High index requires BOTH diverse landforms AND rough terrain
        geoindex = lf_norm * tri_norm

        # Apply NoData mask
        nan_mask = np.isnan(lf_div) | np.isnan(tri_arr)
        geoindex[nan_mask] = np.nan

        valid_vals = geoindex[~nan_mask]
        if len(valid_vals) == 0:
            print(f"  ⚠ {area_name}: no valid pixels — skipping")
            continue

        geoindex_maps[area_name] = geoindex
        all_geoindex_values.append(valid_vals)
        print(f"  {area_name}: Geoindex range {valid_vals.min():.3f} – {valid_vals.max():.3f}")

    # ── Classify into 5 geodiversity levels using global statistics ───────────
    # Mean ± 0.5/1.5 SD gives 5 equal-probability classes under a normal distribution
    all_vals    = np.concatenate(all_geoindex_values)
    global_mean = all_vals.mean()
    global_std  = all_vals.std()

    bounds = [
        -np.inf,
        global_mean - 1.5 * global_std,   # upper bound of class 1 (Very low)
        global_mean - 0.5 * global_std,   # upper bound of class 2 (Low)
        global_mean + 0.5 * global_std,   # upper bound of class 3 (Medium)
        global_mean + 1.5 * global_std,   # upper bound of class 4 (High)
        np.inf                             # class 5 (Very high)
    ]
    class_labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']

    print(f"\nGlobal Geoindex — mean: {global_mean:.3f}, std: {global_std:.3f}")
    print("Classification boundaries:")
    for k, label in enumerate(class_labels):
        print(f"  {label:10s}: {bounds[k]:.3f} – {bounds[k+1]:.3f}")

    # ── Save rasters ──────────────────────────────────────────────────────────
    cmap = plt.cm.RdYlGn  # red (low) → yellow (medium) → green (high)

    for area_name, geoindex in geoindex_maps.items():
        meta = area_metas[area_name].copy()
        meta.update({
            'height':    geoindex.shape[0],
            'width':     geoindex.shape[1],
            'transform': area_transforms[area_name],
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
                   f"{area_name}_geoindex_hl_{res_label}.tif"
        with rasterio.open(out_cont, 'w', **meta) as dst:
            dst.write(geoindex.astype('float32'), 1)

        # Save classified index
        meta_class = meta.copy()
        meta_class['dtype'] = 'int16'
        out_class = Path(res_folder) / area_name / "Geodiversity" / \
                    f"{area_name}_geoindex_hl_classified_{res_label}.tif"
        with rasterio.open(out_class, 'w', **meta_class) as dst:
            dst.write(classified.astype('int16'), 1)

    # ── Combined overview figure: one subplot per study area ──────────────────
    n_cols_fig = 4
    n_rows_fig = int(np.ceil(len(geoindex_maps) / n_cols_fig))
    fig, axes  = plt.subplots(n_rows_fig, n_cols_fig,
                               figsize=(5 * n_cols_fig, 4 * n_rows_fig),
                               constrained_layout=True)
    axes_flat  = axes.flatten()

    # Colour scale centred on global mean ± 2 SD
    vmin = global_mean - 2 * global_std
    vmax = global_mean + 2 * global_std
    im   = None

    for i, (area_name, geoindex) in enumerate(geoindex_maps.items()):
        ax = axes_flat[i]
        im = ax.imshow(geoindex, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(area_name.replace("_", " "), fontsize=10, fontweight='bold')
        ax.axis('off')
        # Zoom each subplot to the non-NaN bounding box to remove empty space
        valid_rows = np.any(~np.isnan(geoindex), axis=1)
        valid_cols = np.any(~np.isnan(geoindex), axis=0)
        if valid_rows.any() and valid_cols.any():
            rmin, rmax = np.where(valid_rows)[0][[0, -1]]
            cmin, cmax = np.where(valid_cols)[0][[0, -1]]
            pad = max((rmax - rmin) * 0.05, (cmax - cmin) * 0.05, 5)
            ax.set_xlim(cmin - pad, cmax + pad)
            ax.set_ylim(rmax + pad, rmin - pad)

    for j in range(len(geoindex_maps), len(axes_flat)):
        axes_flat[j].set_visible(False)

    if im is not None:
        fig.colorbar(im, ax=axes_flat[:len(geoindex_maps)],
                     orientation='vertical', fraction=0.02, pad=0.02,
                     label='Geoindex (H&L)', shrink=0.6)

    fig.suptitle(
        f"Geodiversity Index — {res_label} resolution\n"
        f"Hjort & Luoto method (landform diversity × TRI, {WINDOW_SIZE}x{WINDOW_SIZE} window)",
        fontsize=14, fontweight='bold'
    )
    plt.savefig(fig_folder / f"geodiversity_index_hl_{res_label}.png",
                bbox_inches='tight', dpi=150)
    plt.close()

    print(f"\nFigures and rasters saved for {res_label} resolution.")

print("\nAll done!!")

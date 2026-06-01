#!/usr/bin/env python
# coding: utf-8
"""
NDVI validation — testing whether the predicted vegetation map is consistent
with the spectral signal captured by Sentinel-2.

Rationale
---------
The vegetation map was produced from orthophoto spectral bands, not from S2
directly. If the classification is ecologically valid, pixels classified as
vegetation should have higher S2 NDVI values than non-vegetation pixels.
This script tests that assumption statistically.

Approach
--------
1. The binary predicted vegetation raster (~0.4 m, 1=veg, 2=non-veg) is
   reprojected to 10 m using average resampling → proportion of sub-pixels
   classified as vegetation within each 10 m S2 cell (vegetation fraction).
2. S2 NDVI at 10 m is loaded for the same area.
3. Statistical tests:
     - Spearman correlation: vegetation fraction vs S2 NDVI (continuous)
     - Mann-Whitney U test: NDVI in vegetated pixels (fraction > 0.5) vs
       non-vegetated pixels. One-sided: tests whether vegetation pixels
       have significantly higher NDVI.
     - Rank-biserial correlation (r_rb): effect size for the U test.
4. Results are saved to a CSV and visualised as side-by-side box plots.

Outputs
-------
  Data/Python/Outputs/Geo_veg_relation/all_areas_ndvi_validation.csv
  Data/Python/Outputs/Geo_veg_relation/all_areas_ndvi_validation_boxplot.png
"""

import os
os.chdir(r"C:\TEMP\Vanessa_Henriksson")
print(os.getcwd())

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures without a display
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
NODATA        = -9999
SAMPLE_PIXELS = 200_000  # cap on pixels used for Spearman correlation (speed)
# ─────────────────────────────────────────────────────────────────────────────

study_areas = gpd.read_file("Data/proglacial_outlines.shp")
study_areas = study_areas.drop(index=[12, 14])
area_names  = [area.Glacier_na for area in study_areas.itertuples()]

out_folder = Path("Data/Python/Outputs/Geo_veg_relation")
out_folder.mkdir(parents=True, exist_ok=True)

print("Running NDVI validation...")
ndvi_rows = []  # accumulates one results dict per study area

for area_name in area_names:
    ndvi_path = f"Data/Python/Outputs/{area_name}/NDVI/{area_name}_NDVI_S2.tif"
    veg_path  = f"Data/Python/Outputs/Predicted_vegetation/{area_name}_predicted_vegetation.tif"

    if not Path(ndvi_path).exists():
        print(f"  ⚠ No NDVI found for {area_name} (tried: {ndvi_path})")
        continue
    if not Path(veg_path).exists():
        print(f"  ⚠ No predicted vegetation found for {area_name}")
        continue

    print(f"  {area_name}")

    # Load S2 NDVI at its native 10 m resolution; this is used as the reference
    # grid — the vegetation raster will be reprojected to match it.
    with rasterio.open(ndvi_path) as src:
        ndvi_arr       = src.read(1).astype(float)
        ndvi_nodata    = src.nodata
        ndvi_shape     = (src.height, src.width)
        ndvi_transform = src.transform
        ndvi_crs       = src.crs
    if ndvi_nodata is not None:
        ndvi_arr[ndvi_arr == ndvi_nodata] = np.nan
    ndvi_arr[ndvi_arr == NODATA] = np.nan

    # Reproject the binary vegetation raster (~0.4 m) to 10 m using
    # average resampling. The result at each 10 m cell is the mean of the
    # ~0/1 sub-pixel values, i.e. the fraction of sub-pixels classified as
    # vegetation. Vegetation pixels are recoded to 1 and non-vegetation to 0
    # before resampling so the average equals the vegetation fraction.
    with rasterio.open(veg_path) as veg_src:
        veg_raw = veg_src.read(1).astype(float)
        veg_nd  = veg_src.nodata
        if veg_nd is not None:
            veg_raw[veg_raw == veg_nd] = np.nan
        veg_raw[veg_raw == NODATA] = np.nan
        # Recode: 1 (vegetation) → 1.0, 2 (non-vegetation) → 0.0
        veg_raw = np.where(veg_raw == 1, 1.0, np.where(veg_raw == 2, 0.0, np.nan))

        veg_frac = np.full(ndvi_shape, np.nan, dtype=float)
        reproject(
            source=veg_raw,
            destination=veg_frac,
            src_transform=veg_src.transform,
            src_crs=veg_src.crs,
            dst_transform=ndvi_transform,
            dst_crs=ndvi_crs,
            resampling=Resampling.average,  # average sub-pixel values → fraction
        )

    # Keep only pixels where both NDVI and vegetation fraction are valid
    valid      = ~np.isnan(ndvi_arr) & ~np.isnan(veg_frac)
    ndvi_valid = ndvi_arr[valid]
    veg_valid  = veg_frac[valid]

    if len(ndvi_valid) < 10:
        print(f"    Too few valid pixels, skipping.")
        continue

    # Spearman correlation — non-parametric rank correlation between
    # vegetation fraction and NDVI. Subsample if array is very large
    # to keep computation time manageable.
    if len(ndvi_valid) > SAMPLE_PIXELS:
        idx       = np.random.choice(len(ndvi_valid), SAMPLE_PIXELS, replace=False)
        ndvi_samp = ndvi_valid[idx]
        veg_samp  = veg_valid[idx]
    else:
        ndvi_samp, veg_samp = ndvi_valid, veg_valid

    rho, p_rho = stats.spearmanr(veg_samp, ndvi_samp)

    # Split pixels into vegetated (fraction > 0.5) and non-vegetated groups
    # for the Mann-Whitney U test
    veg_mask   = veg_valid > 0.5
    noveg_mask = ~veg_mask
    ndvi_veg   = ndvi_valid[veg_mask]
    ndvi_noveg = ndvi_valid[noveg_mask]

    if len(ndvi_veg) > 10 and len(ndvi_noveg) > 10:
        # One-sided test: alternative='greater' tests whether vegetation NDVI
        # is significantly higher than non-vegetation NDVI
        u_stat, p_mw = stats.mannwhitneyu(ndvi_veg, ndvi_noveg, alternative='greater')
        n1, n2 = len(ndvi_veg), len(ndvi_noveg)
        # Rank-biserial r: effect size; ranges -1 to +1 (0 = no effect)
        r_rb   = (2 * u_stat) / (n1 * n2) - 1
    else:
        u_stat, p_mw, r_rb = np.nan, np.nan, np.nan

    ndvi_rows.append({
        'Area':                         area_name,
        'N veg pixels (10 m)':          int(veg_mask.sum()),
        'N non-veg pixels (10 m)':      int(noveg_mask.sum()),
        'Mean NDVI veg':                round(float(np.nanmean(ndvi_veg)),   4) if len(ndvi_veg)   > 0 else np.nan,
        'Mean NDVI non-veg':            round(float(np.nanmean(ndvi_noveg)), 4) if len(ndvi_noveg) > 0 else np.nan,
        'Spearman rho':                 round(rho,    4),
        'Spearman p':                   round(p_rho,  6),
        'Mann-Whitney U':               round(u_stat, 1) if not np.isnan(u_stat) else np.nan,
        'Mann-Whitney p (veg>non-veg)': round(p_mw,   6) if not np.isnan(p_mw)  else np.nan,
        'Rank-biserial r':              round(r_rb,   4) if not np.isnan(r_rb)   else np.nan,
    })
    print(f"    Spearman rho={rho:.3f} (p={p_rho:.4f}) | "
          f"MW p={p_mw:.4f} | "
          f"mean NDVI veg={np.nanmean(ndvi_veg):.3f}  non-veg={np.nanmean(ndvi_noveg):.3f}")

if not ndvi_rows:
    print("No NDVI data found for any area. Check file paths.")
else:
    df_ndvi = pd.DataFrame(ndvi_rows)
    df_ndvi.to_csv(out_folder / "all_areas_ndvi_validation.csv", index=False, encoding='utf-8-sig')
    print("\nNDVI validation summary:")
    print(df_ndvi.to_string(index=False))
    print("\nSaved to all_areas_ndvi_validation.csv")

    # ── Box plot: NDVI distributions veg vs non-veg per study area ───────────
    # One subplot per area; boxes show the distribution of S2 NDVI values
    # in vegetated (orange) and non-vegetated (blue) pixels. The Mann-Whitney
    # p-value is annotated above each subplot.
    matplotlib.rcParams['font.family'] = 'Calibri'

    n_areas = len(ndvi_rows)
    fig, axes = plt.subplots(1, n_areas, figsize=(max(10, n_areas * 2.2), 6),
                             sharey=True)  # shared y-axis for easy comparison across areas
    if n_areas == 1:
        axes = [axes]

    for ax, row in zip(axes, ndvi_rows):
        area_name = row['Area']
        ndvi_path = f"Data/Python/Outputs/{area_name}/NDVI/{area_name}_NDVI_S2.tif"
        veg_path  = f"Data/Python/Outputs/Predicted_vegetation/{area_name}_predicted_vegetation.tif"

        # Re-load rasters to get pixel arrays for plotting
        with rasterio.open(ndvi_path) as src:
            ndvi_arr       = src.read(1).astype(float)
            ndvi_nodata    = src.nodata
            ndvi_shape     = (src.height, src.width)
            ndvi_transform = src.transform
            ndvi_crs       = src.crs
        if ndvi_nodata is not None:
            ndvi_arr[ndvi_arr == ndvi_nodata] = np.nan
        ndvi_arr[ndvi_arr == NODATA] = np.nan

        with rasterio.open(veg_path) as veg_src:
            veg_raw = veg_src.read(1).astype(float)
            veg_nd  = veg_src.nodata
            if veg_nd is not None:
                veg_raw[veg_raw == veg_nd] = np.nan
            veg_raw[veg_raw == NODATA] = np.nan
            veg_frac = np.full(ndvi_shape, np.nan, dtype=float)
            reproject(
                source=veg_raw,          destination=veg_frac,
                src_transform=veg_src.transform, src_crs=veg_src.crs,
                dst_transform=ndvi_transform,    dst_crs=ndvi_crs,
                resampling=Resampling.average,
            )

        valid      = ~np.isnan(ndvi_arr) & ~np.isnan(veg_frac)
        ndvi_valid = ndvi_arr[valid]
        veg_valid  = veg_frac[valid]
        ndvi_veg   = ndvi_valid[veg_valid > 0.5]
        ndvi_noveg = ndvi_valid[veg_valid <= 0.5]

        bp = ax.boxplot([ndvi_noveg, ndvi_veg],
                        tick_labels=['Non-veg', 'Veg'],
                        patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5))
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('darkorange')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_alpha(0.7)

        label = area_name.replace('_', ' ').replace(' II III', '').replace('_II_III', '')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4, axis='y')

        # Annotate with Mann-Whitney p-value
        p_mw = row['Mann-Whitney p (veg>non-veg)']
        if not np.isnan(p_mw):
            p_str = f"p={p_mw:.3f}" if p_mw >= 0.001 else "p<0.001"
            ax.text(0.5, 0.97, p_str, transform=ax.transAxes,
                    ha='center', va='top', fontsize=9)

    axes[0].set_ylabel("NDVI", fontsize=12)
    plt.suptitle("S2 NDVI — predicted vegetation vs non-vegetation pixels",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_folder / "all_areas_ndvi_validation_boxplot.png",
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Box plot saved to all_areas_ndvi_validation_boxplot.png")

print("\nAll done!")

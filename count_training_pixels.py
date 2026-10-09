#!/usr/bin/env python
# coding: utf-8
"""
Count training pixels per class for the vegetation classifier — read-only.

Uses exactly the same inputs and pixel extraction as Veg_RF_polygon_cv.py
(same orthophoto, training polygons, study-area filtering and NaN handling),
so the counts match what the classifier is trained on. Nothing is trained or
predicted, and no rasters are written.

Why: the classifier is trained on every pixel inside every training polygon,
so class balance depends on pixel counts, not polygon counts. If vegetation
polygons are larger, vegetation dominates the training data and the model
leans towards predicting vegetation.

For each study area, prints and saves:
  - vegetation / non-vegetation pixels and the vegetation share (%)
  - number of polygons per class
  - polygon size per class (median and largest, in pixels), and the share of
    each class's pixels that comes from its largest polygon

Outputs (in the predicted-vegetation folder, next to the classifier metrics):
  training_pixel_counts.csv          one row per study area
  training_pixel_counts_polygons.csv one row per training polygon

Requires: rasterio, geopandas, fiona, PyYAML
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pathlib import Path

from config_utils import load_config, resolve_path

cfg = load_config()

shp_path = resolve_path(cfg, cfg.paths.outlines_shp)
gdb = resolve_path(cfg, cfg.paths.training_gdb)
outputs_1m_dir = resolve_path(cfg, cfg.paths.outputs_1m)
predicted_vegetation_dir = Path(resolve_path(cfg, cfg.paths.predicted_vegetation_dir))

study_areas = gpd.read_file(shp_path)
study_areas = study_areas.drop(index=cfg.study_areas.exclude_indices)
if cfg.study_areas.only:
    study_areas = study_areas[study_areas["Glacier_na"].isin(cfg.study_areas.only)]

CLASS_NAMES = {1: "veg", 2: "nonveg"}

area_rows = []
polygon_rows = []

for area in study_areas.itertuples():
    area_name = area.Glacier_na
    ortho_fp = f"{outputs_1m_dir}/{area_name}/{area_name}_ortho_clip.tif"
    print(f"\n{'=' * 60}\n{area_name}\n{'=' * 60}")

    if not Path(ortho_fp).exists():
        print(f"  Skipping — {ortho_fp} not found.")
        continue

    # ── Training polygons (same filtering as Veg_RF_polygon_cv.py) ────────────
    df_full = gpd.read_file(filename=gdb, layer=f"{area_name}_training")
    if df_full.crs != study_areas.crs:
        df_full = df_full.to_crs(study_areas.crs)
    n_before_extent = len(df_full)
    df_full = df_full[df_full.intersects(area.geometry)]
    n_dropped_extent = n_before_extent - len(df_full)
    if n_dropped_extent:
        print(f"  Dropped {n_dropped_extent} training polygon(s) outside the study area boundary.")
    df = df_full.dropna().reset_index(drop=True)
    if df.empty:
        print("  No training data found, skipping.")
        continue

    # ── Count pixels per polygon (same extraction as Veg_RF_polygon_cv.py) ────
    with rasterio.open(ortho_fp) as ortho_raster:
        bands = ortho_raster.count
        for poly_idx, (_, row) in enumerate(df.iterrows()):
            label = row["Category"]
            out_img, _ = mask(ortho_raster, [row.geometry], crop=True, filled=False)
            pixels = out_img.reshape(bands, -1).T
            pixels = pixels[~np.any(np.isnan(pixels), axis=1)]
            polygon_rows.append({
                "area": area_name,
                "polygon": poly_idx,
                "class": CLASS_NAMES.get(label, str(label)),
                "n_pixels": int(pixels.shape[0]),
            })

    # ── Per-area summary ──────────────────────────────────────────────────────
    polys = pd.DataFrame([p for p in polygon_rows if p["area"] == area_name])
    polys = polys[polys["n_pixels"] > 0]   # empty polygons are skipped in training too
    row = {"area": area_name}
    for cls in ["veg", "nonveg"]:
        sizes = polys.loc[polys["class"] == cls, "n_pixels"]
        total = int(sizes.sum())
        row[f"{cls}_pixels"] = total
        row[f"{cls}_polygons"] = len(sizes)
        row[f"{cls}_median_polygon_px"] = int(sizes.median()) if len(sizes) else 0
        row[f"{cls}_largest_polygon_px"] = int(sizes.max()) if len(sizes) else 0
        row[f"{cls}_largest_polygon_share_pct"] = round(100 * sizes.max() / total, 1) if total else np.nan
    total_px = row["veg_pixels"] + row["nonveg_pixels"]
    row["veg_share_pct"] = round(100 * row["veg_pixels"] / total_px, 1) if total_px else np.nan
    area_rows.append(row)

    print(f"  Vegetation:     {row['veg_pixels']:>10,} px in {row['veg_polygons']:>3} polygons "
          f"(median {row['veg_median_polygon_px']:,} px, largest {row['veg_largest_polygon_px']:,} px "
          f"= {row['veg_largest_polygon_share_pct']}% of class)")
    print(f"  Non-vegetation: {row['nonveg_pixels']:>10,} px in {row['nonveg_polygons']:>3} polygons "
          f"(median {row['nonveg_median_polygon_px']:,} px, largest {row['nonveg_largest_polygon_px']:,} px "
          f"= {row['nonveg_largest_polygon_share_pct']}% of class)")
    print(f"  Vegetation share of training pixels: {row['veg_share_pct']}%")

# ── Save ──────────────────────────────────────────────────────────────────────
if area_rows:
    predicted_vegetation_dir.mkdir(parents=True, exist_ok=True)
    df_areas = pd.DataFrame(area_rows)
    df_areas.to_csv(predicted_vegetation_dir / "training_pixel_counts.csv", index=False)
    pd.DataFrame(polygon_rows).to_csv(
        predicted_vegetation_dir / "training_pixel_counts_polygons.csv", index=False)
    print(f"\nSaved training_pixel_counts.csv and training_pixel_counts_polygons.csv "
          f"to {predicted_vegetation_dir}")
    print("\n" + df_areas[["area", "veg_pixels", "nonveg_pixels", "veg_share_pct",
                           "veg_polygons", "nonveg_polygons",
                           "veg_median_polygon_px", "nonveg_median_polygon_px"]].to_string(index=False))
else:
    print("\nNo areas counted.")

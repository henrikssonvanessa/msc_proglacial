#!/usr/bin/env python
# coding: utf-8
"""
Step 6b — Random Forest vegetation prediction at 20 m with spatial block cross-validation.

This script mirrors RF_block_test.py but operates on the 20 m dataset produced
in Low_res_script.py. The key differences from the 1 m model are:

  - Block size: 400 m (doubled from 200 m) because at 20 m resolution each
    block covers the same number of pixels, but the physical extent needs to
    be larger to capture the broader-scale spatial autocorrelation structure.
  - Additional predictor: Snow_cover — a Sentinel-2-derived fraction of the
    growing season during which a pixel is snow-covered. Persistent snow
    cover prevents or delays vegetation establishment.
  - Features (10 variables): Landforms, Distance, Aspect_sin, Elevation,
    Curvature, Aspect_cos, Hillshade, TRI, SWI, Snow_cover

The outputs are written to Data/Python/Outputs_20m/ so they do not overwrite
the 1 m results.

See RF_block_test.py for a full explanation of the spatial block CV approach.
"""

import os
os.chdir(r"C:\TEMP\Vanessa_Henriksson")
print(os.getcwd())

import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate, train_test_split
from sklearn.metrics import (accuracy_score, recall_score, make_scorer,
                             precision_score, f1_score)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
BLOCK_SIZE = 400   # spatial block size in metres (doubled vs 1 m model)
N_FOLDS    = 5
RESOLUTION = 20
# ─────────────────────────────────────────────────────────────────────────────

study_areas = gpd.read_file("Data/proglacial_outlines.shp")
study_areas = study_areas.drop(index=[12, 14])

fig_folder = Path(f"Data/Python/Outputs_{RESOLUTION}m/Figures")
fig_folder.mkdir(parents=True, exist_ok=True)

# Column rename mapping for the 20 m sample shapefile
# The column order from ArcPy Sample differs slightly from the 1 m version
# because snow cover (v_raste_12) was appended last in Low_res_script.py
new_names = {
    "v_raster_1": "Aspect_cos",
    "v_raster_2": "Aspect_sin",
    "v_raster_3": "Curvature",
    "v_raster_4": "Elevation",
    "v_raster_5": "Distance",
    "v_raster_6": "Hillshade",
    "v_raster_7": "Landforms",
    "v_raster_8": "Vegetation",   # target variable
    "v_raster_9": "Slope",
    "v_raste_10": "TRI",
    "v_raste_11": "SWI",
    "v_raste_12": "Snow_cover",   # additional predictor vs 1 m model
}

# Snow_cover added as the 10th predictor at 20 m
feature_cols = ['Landforms', 'Distance', 'Aspect_sin', 'Elevation',
                'Curvature', 'Aspect_cos', 'Hillshade', 'TRI', 'SWI', 'Snow_cover']

random_split_rows = []
block_cv_rows     = []
perm_veg_rows     = []

for area in study_areas.itertuples():
    area_name      = area.Glacier_na
    geodiv_out_dir = f"Data/Python/Outputs_{RESOLUTION}m/{area_name}/Geodiversity"
    print(f"\n{'='*60}\nProcessing {area_name}\n{'='*60}")

    # ── Load and clean sample data ─────────────────────────────────────────────
    gdf_full = gpd.read_file(f"{geodiv_out_dir}/{area_name}_samples.shp")
    gdf_full = gdf_full.replace(-9999, np.nan)  # replace ArcPy NoData sentinel
    gdf      = gdf_full.dropna()
    print(f"Samples before NaN removal: {len(gdf_full)}")
    print(f"Samples after NaN removal:  {len(gdf)}")

    gdf = gdf.rename(columns=new_names)

    # ── Assign spatial block IDs ───────────────────────────────────────────────
    # Larger blocks (400 m) than the 1 m model to account for the coarser pixel
    # size and broader autocorrelation structure at 20 m resolution
    coords    = np.array([(geom.x, geom.y) for geom in gdf.geometry])
    block_col = (np.floor(coords[:, 0] / BLOCK_SIZE).astype(int),
                 np.floor(coords[:, 1] / BLOCK_SIZE).astype(int))
    block_ids = block_col[0] * 10_000_000 + block_col[1]
    unique_blocks = {v: i for i, v in enumerate(np.unique(block_ids))}
    block_ids = np.array([unique_blocks[b] for b in block_ids])

    n_blocks = len(np.unique(block_ids))
    print(f"Number of spatial blocks ({BLOCK_SIZE}m): {n_blocks}")
    n_folds_area = min(n_blocks, N_FOLDS)
    if n_blocks < N_FOLDS:
        print(f"  ⚠ Fewer blocks than folds — reducing folds to {n_blocks}")

    # ── Feature matrix and target ─────────────────────────────────────────────
    x = gdf[feature_cols]
    # Recode: 1 (vegetation) → 1, 2 (non-vegetation) → 0
    y = gdf['Vegetation'].replace({1: 1, 2: 0})

    # ── Spatial block cross-validation ────────────────────────────────────────
    gkf     = GroupKFold(n_splits=n_folds_area)
    scoring = {
        'accuracy':  'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall':    make_scorer(recall_score,    zero_division=0),
        'f1':        make_scorer(f1_score,        zero_division=0),
    }

    cv_results = cross_validate(
        RandomForestClassifier(n_estimators=100, random_state=42),
        x, y,
        groups=block_ids,
        cv=gkf,
        scoring=scoring,
        return_estimator=True,
        n_jobs=-1
    )

    print("\n── Spatial Block CV Results ──")
    block_row = {'area': area_name, 'n_samples': len(gdf),
                 'n_blocks': n_blocks, 'n_folds': n_folds_area}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        scores = cv_results[f'test_{metric}']
        block_row[f'{metric}_mean'] = round(scores.mean(), 4)
        block_row[f'{metric}_std']  = round(scores.std(),  4)
        print(f"  {metric.capitalize():10s}: {scores.mean():.3f} ± {scores.std():.3f}  "
              f"(folds: {np.round(scores, 3)})")
    block_cv_rows.append(block_row)

    # ── Random split (for comparison) ─────────────────────────────────────────
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)
    rf_rs = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_rs.fit(x_train, y_train)
    preds_rs = rf_rs.predict(x_test)

    rs_row = {
        'area':      area_name,
        'n_samples': len(gdf),
        'accuracy':  round(accuracy_score(y_test,  preds_rs), 4),
        'precision': round(precision_score(y_test, preds_rs, zero_division=0), 4),
        'recall':    round(recall_score(y_test,    preds_rs, zero_division=0), 4),
        'f1':        round(f1_score(y_test,        preds_rs, zero_division=0), 4),
    }
    random_split_rows.append(rs_row)
    print(f"\n── Random Split Results ──")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print(f"  {metric.capitalize():10s}: {rs_row[metric]:.3f}")

    # ── Refit on ALL data for importance and PDP plots ─────────────────────────
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x, y)

    veg_samples = x[y == 1]  # vegetation-only subset for PDP histogram overlays

    # ── Feature importance (MDI) ──────────────────────────────────────────────
    plt.figure(figsize=(12, 10))
    plt.barh(x.columns, rf.feature_importances_)
    plt.xlabel("Feature Importance (MDI)")
    plt.title(f"Random Forest Feature Importance — {area_name} ({RESOLUTION}m)")
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/{area_name}_feat_imp.png")
    plt.close()
    print("Feature importance figure saved.")

    # ── Permutation importance (accuracy) ────────────────────────────────────
    perm_acc = permutation_importance(rf, x, y, n_repeats=30,
                                      random_state=42, n_jobs=-1)
    indices = np.argsort(perm_acc.importances_mean)
    plt.figure(figsize=(12, 10))
    plt.barh(np.array(x.columns)[indices],
             perm_acc.importances_mean[indices],
             xerr=perm_acc.importances_std[indices])
    plt.xlabel("Permutation Importance (mean decrease in accuracy)")
    plt.title(f"Permutation Importance — {area_name} ({RESOLUTION}m)")
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/{area_name}_perm_imp.png")
    plt.close()
    print("Permutation importance figure saved.")

    # ── Permutation importance (vegetation recall) ────────────────────────────
    veg_recall = make_scorer(recall_score, pos_label=1)
    perm_veg   = permutation_importance(rf, x, y, scoring=veg_recall,
                                        n_repeats=30, random_state=42, n_jobs=-1)
    indices = np.argsort(perm_veg.importances_mean)
    plt.figure(figsize=(12, 10))
    plt.barh(np.array(x.columns)[indices],
             perm_veg.importances_mean[indices],
             xerr=perm_veg.importances_std[indices])
    plt.xlabel("Permutation Importance (mean decrease in vegetation recall)")
    plt.title(f"Permutation Importance — Vegetation Recall — {area_name} ({RESOLUTION}m)")
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/{area_name}_perm_veg.png")
    plt.close()
    print("Permutation importance (vegetation recall) figure saved.")

    perm_row = {'area': area_name}
    for feat, mean_val, std_val in zip(feature_cols,
                                       perm_veg.importances_mean,
                                       perm_veg.importances_std):
        perm_row[f'{feat}_mean'] = round(mean_val, 4)
        perm_row[f'{feat}_std']  = round(std_val,  4)
    perm_veg_rows.append(perm_row)

    # ── Partial dependence plots with histogram overlay ───────────────────────
    # Same approach as RF_block_test.py; 10 features arranged in 4 rows × 3 cols
    n_features = len(feature_cols)
    n_cols_fig = 3
    n_rows_fig = int(np.ceil(n_features / n_cols_fig))

    fig, axes = plt.subplots(n_rows_fig, n_cols_fig,
                             figsize=(6 * n_cols_fig, 5 * n_rows_fig))
    axes_flat = axes.flatten()

    disp = PartialDependenceDisplay.from_estimator(
        rf, x, feature_cols,
        response_method='predict_proba',
        grid_resolution=50,
        kind="average",
        ax=axes_flat[:n_features]
    )

    for i, feature in enumerate(feature_cols):
        ax_pdp  = axes_flat[i]
        ax_hist = ax_pdp.twinx()
        ax_hist.hist(x[feature],           bins=30, alpha=0.15, color='grey',  density=True)
        ax_hist.hist(veg_samples[feature], bins=30, alpha=0.25, color='green', density=True)
        ax_hist.set_ylabel("Density", fontsize=8, color='grey')
        ax_hist.tick_params(axis='y', labelsize=7, labelcolor='grey')

    for j in range(n_features, len(axes_flat)):
        axes_flat[j].set_visible(False)

    legend_elements = [
        Patch(facecolor='green', alpha=0.4, label='Vegetation samples'),
        Patch(facecolor='grey',  alpha=0.3, label='All samples'),
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.8)
    plt.suptitle(
        f"Partial Dependence of Vegetation Probability — {area_name} ({RESOLUTION}m)\n"
        f"(green = vegetation samples, grey = all samples)",
        y=1.01, fontsize=13
    )
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/{area_name}_pdd_veg.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("PDD figure saved.")

    # ── Spearman correlation matrix ───────────────────────────────────────────
    corr = gdf[feature_cols].corr(method='spearman')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'Spearman Correlation Matrix — {area_name} ({RESOLUTION}m)')
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/{area_name}_corr_matrix_spearman.png")
    plt.close()
    print("Spearman correlation matrix saved.")

# ── Export summary CSVs ───────────────────────────────────────────────────────
csv_folder = Path(f"Data/Python/Outputs_{RESOLUTION}m")

df_random = pd.DataFrame(random_split_rows)
df_random.to_csv(csv_folder / "metrics_random_split.csv", index=False, encoding='utf-8-sig')
print("\nRandom split metrics saved.")

df_block = pd.DataFrame(block_cv_rows)
df_block.to_csv(csv_folder / "metrics_block_cv.csv", index=False, encoding='utf-8-sig')
print("Block CV metrics saved.")

df_perm = pd.DataFrame(perm_veg_rows)
df_perm.to_csv(csv_folder / "perm_importance_veg_recall.csv", index=False, encoding='utf-8-sig')
print("Permutation importance (vegetation recall) saved.")

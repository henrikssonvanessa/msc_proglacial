#!/usr/bin/env python
# coding: utf-8
"""
Step 6a — Random Forest vegetation prediction at 1 m with spatial block cross-validation.

Why spatial block cross-validation?
------------------------------------
Terrain data has strong spatial autocorrelation: pixels close together tend to
have similar values. A random train/test split can assign spatially adjacent pixels
to both sets, meaning the model is evaluated on data that is almost identical to
some of its training data. This inflates performance metrics.

Spatial block CV addresses this by dividing the study area into rectangular blocks
(200 m × 200 m here) and holding out entire blocks as test sets. This tests how
well the model generalises across space — a more realistic scenario when predicting
vegetation for unmapped areas.

For each study area:
  1. Loads the sample shapefile ({area_name}_samples.shp) produced in Step 4
  2. Assigns each sample point to a spatial block based on its coordinates
  3. Runs GroupKFold CV (5 folds) with block membership as the group variable
  4. Also runs a random 70/30 split for comparison (to quantify the optimism bias)
  5. Refits on ALL data to compute feature importances and partial dependence plots
  6. Saves feature importance, permutation importance, PDP, and Spearman correlation
     figures, and exports summary metrics to CSV

Features (9 variables at 1 m):
  Landforms, Distance, Aspect_sin, Elevation, Curvature, Aspect_cos,
  Hillshade, TRI, SWI

Target: Vegetation (1=vegetated, recoded to 1; 2=non-vegetated, recoded to 0)

Outputs:
  Data/Python/Outputs/Figures/{area}_feat_imp.png
  Data/Python/Outputs/Figures/{area}_perm_imp.png
  Data/Python/Outputs/Figures/{area}_perm_veg.png
  Data/Python/Outputs/Figures/{area}_pdd_veg.png
  Data/Python/Outputs/Figures/{area}_corr_matrix_spearman.png
  Data/Python/Outputs/metrics_random_split.csv
  Data/Python/Outputs/metrics_block_cv.csv
  Data/Python/Outputs/perm_importance_veg_recall.csv
"""

import os
os.chdir(r"C:\TEMP\Vanessa_Henriksson")
print(os.getcwd())

import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, recall_score, make_scorer,
                             precision_score, f1_score)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures without a display

# ── CONFIG ────────────────────────────────────────────────────────────────────
BLOCK_SIZE = 200   # spatial block size in metres; should be larger than the
                   # range of spatial autocorrelation in the terrain data
N_FOLDS    = 5     # number of spatial CV folds
# ─────────────────────────────────────────────────────────────────────────────

study_areas = gpd.read_file("Data/proglacial_outlines.shp")
study_areas = study_areas.drop(index=[12, 14])

fig_folder = Path("Data/Python/Outputs/Figures")
fig_folder.mkdir(parents=True, exist_ok=True)

# Storage for summary CSVs (one row per study area)
random_split_rows = []
block_cv_rows     = []
perm_veg_rows     = []

for area in study_areas.itertuples():
    area_name = area.Glacier_na
    geodiv_out_dir = f"Data/Python/Outputs/{area_name}/Geodiversity"
    print(f"\n{'='*60}\nProcessing {area_name}\n{'='*60}")

    # ── Load sample data ───────────────────────────────────────────────────────
    # The samples shapefile has one row per sample point; columns v_raster_1…
    # v_raste_11 contain terrain variable values extracted by ArcPy in Step 4.
    gdf_full = gpd.read_file(f"{geodiv_out_dir}/{area_name}_samples.shp")
    gdf_full = gdf_full.replace(-9999, np.nan)  # replace ArcPy NoData sentinel
    gdf = gdf_full.dropna()  # remove rows with any missing variable
    print(f"Samples before NaN removal: {len(gdf_full)}")
    print(f"Samples after NaN removal:  {len(gdf)}")

    # Rename generic ArcPy column names to meaningful variable names
    # Note: ArcPy truncates long column names — "v_raste_10" is not "v_raster_10"
    new_names = {
        "v_raster_1": "Aspect_cos", "v_raster_2": "Aspect_sin",
        "v_raster_3": "Curvature",  "v_raster_4": "Elevation",
        "v_raster_5": "Distance",   "v_raster_6": "Hillshade",
        "v_raster_7": "Landforms",  "v_raster_8": "Slope",
        "v_raster_9": "SWI", "v_raste_10": "TRI", "v_raste_11": "Vegetation"
    }
    gdf = gdf.rename(columns=new_names)

    # ── Assign spatial block IDs ───────────────────────────────────────────────
    # Divide the coordinate space into a regular grid of BLOCK_SIZE × BLOCK_SIZE m.
    # Each point is assigned a block ID based on which grid cell it falls in.
    # Points in the same block are always kept together in CV splits.
    coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])
    block_col = (np.floor(coords[:, 0] / BLOCK_SIZE).astype(int),
                 np.floor(coords[:, 1] / BLOCK_SIZE).astype(int))
    # Create a unique integer ID per block using a large multiplier to avoid collisions
    block_ids = block_col[0] * 10_000_000 + block_col[1]
    # Remap to consecutive integers starting at 0 (required by GroupKFold)
    unique_blocks = {v: i for i, v in enumerate(np.unique(block_ids))}
    block_ids = np.array([unique_blocks[b] for b in block_ids])

    n_blocks = len(np.unique(block_ids))
    print(f"Number of spatial blocks ({BLOCK_SIZE}m): {n_blocks}")
    if n_blocks < N_FOLDS:
        print(f"  ⚠ Fewer blocks than folds — reducing folds to {n_blocks}")
        n_folds_area = n_blocks
    else:
        n_folds_area = N_FOLDS

    # ── Feature matrix and target vector ──────────────────────────────────────
    feature_cols = ['Landforms', 'Distance', 'Aspect_sin', 'Elevation',
                    'Curvature', 'Aspect_cos', 'Hillshade', 'TRI', 'SWI']
    x = gdf[feature_cols]
    # Recode: 1 (vegetation) → 1, 2 (non-vegetation) → 0
    y = gdf['Vegetation'].replace({1: 1, 2: 0}).astype(int)

    # ── Spatial block cross-validation ────────────────────────────────────────
    # GroupKFold respects block membership: all points in a block are in the
    # same fold (either all training or all test), never split across folds.
    gkf = GroupKFold(n_splits=n_folds_area)

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

    # ── Random split (for comparison) ────────────────────────────────────────
    # This ignores spatial structure and is expected to give inflated metrics
    # relative to block CV. The delta between the two indicates bias magnitude.
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)
    rf_rs = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_rs.fit(x_train, y_train)
    preds_rs = rf_rs.predict(x_test)

    rs_row = {
        'area':       area_name,
        'n_samples':  len(gdf),
        'accuracy':   round(accuracy_score(y_test,  preds_rs), 4),
        'precision':  round(precision_score(y_test, preds_rs, zero_division=0), 4),
        'recall':     round(recall_score(y_test,    preds_rs, zero_division=0), 4),
        'f1':         round(f1_score(y_test,        preds_rs, zero_division=0), 4),
    }
    random_split_rows.append(rs_row)
    print(f"\n── Random Split Results ──")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print(f"  {metric.capitalize():10s}: {rs_row[metric]:.3f}")

    # ── Refit on ALL data for importance and PDP plots ─────────────────────────
    # Cross-validation estimates generalisation performance; importance and PDP
    # plots use a model trained on all available data to maximise stability.
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x, y)

    veg_samples = x[y == 1]   # vegetation-only subset for histogram overlays in PDPs

    # ── Feature importance (MDI — Mean Decrease in Impurity) ─────────────────
    # MDI measures how much each feature reduces node impurity on average across
    # all trees. Higher = more important for splitting. Can be biased towards
    # continuous or high-cardinality features.
    plt.figure(figsize=(12, 10))
    importances = rf.feature_importances_
    plt.barh(x.columns, importances)
    plt.xlabel("Feature Importance (MDI)")
    plt.title(f"Random Forest Feature Importance — {area_name}")
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/{area_name}_feat_imp.png")
    plt.close()
    print("Feature importance figure saved.")

    # ── Permutation importance (accuracy) ────────────────────────────────────
    # Measures the decrease in accuracy when each feature's values are randomly
    # shuffled (breaking its relationship with the target). More robust than MDI
    # because it is measured on the actual data distribution.
    perm_acc = permutation_importance(rf, x, y, n_repeats=30,
                                      random_state=42, n_jobs=-1)
    indices = np.argsort(perm_acc.importances_mean)
    plt.figure(figsize=(12, 10))
    plt.barh(np.array(x.columns)[indices],
             perm_acc.importances_mean[indices],
             xerr=perm_acc.importances_std[indices])
    plt.xlabel("Permutation Importance (mean decrease in accuracy)")
    plt.title(f"Permutation Importance — {area_name}")
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/{area_name}_perm_imp.png")
    plt.close()
    print("Permutation importance figure saved.")

    # ── Permutation importance (vegetation recall) ────────────────────────────
    # Same as above but scored on vegetation recall rather than overall accuracy.
    # This reveals which features are most important specifically for correctly
    # identifying vegetated pixels (more ecologically relevant metric here).
    veg_recall = make_scorer(recall_score, pos_label=1)
    perm_veg = permutation_importance(rf, x, y, scoring=veg_recall,
                                      n_repeats=30, random_state=42, n_jobs=-1)
    indices = np.argsort(perm_veg.importances_mean)
    plt.figure(figsize=(12, 10))
    plt.barh(np.array(x.columns)[indices],
             perm_veg.importances_mean[indices],
             xerr=perm_veg.importances_std[indices])
    plt.xlabel("Permutation Importance (mean decrease in vegetation recall)")
    plt.title(f"Permutation Importance — Vegetation Recall — {area_name}")
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/{area_name}_perm_veg.png")
    plt.close()
    print("Permutation importance (vegetation recall) figure saved.")

    row = {'area': area_name}
    for feat, imp, std in zip(feature_cols, perm_veg.importances_mean, perm_veg.importances_std):
        row[f'{feat}_mean'] = round(imp, 4)
        row[f'{feat}_std']  = round(std, 4)
    perm_veg_rows.append(row)

    # ── Partial dependence plots (PDPs) with histogram overlay ────────────────
    # A PDP shows the marginal effect of one feature on predicted vegetation
    # probability, averaged over all other feature values. The orange line
    # is the PDP (probability of vegetation); grey histogram = all sample
    # distribution; green histogram = vegetation-only samples. This reveals
    # whether vegetation probability increases or decreases with each variable
    # and at what threshold.
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 5 * n_rows))
    axes_flat = axes.flatten()

    disp = PartialDependenceDisplay.from_estimator(
        rf, x, feature_cols,
        response_method='predict_proba',
        grid_resolution=50,
        kind="average",
        ax=axes_flat[:n_features]
    )

    # Overlay histograms on each subplot to show data support
    for i, feature in enumerate(feature_cols):
        ax_pdp = axes_flat[i]
        ax_hist = ax_pdp.twinx()  # second y-axis on the right for density

        ax_hist.hist(x[feature], bins=30, alpha=0.15,
                     color='grey', density=True)        # all samples
        ax_hist.hist(veg_samples[feature], bins=30, alpha=0.25,
                     color='green', density=True)       # vegetation samples only

        ax_hist.set_ylabel("Density", fontsize=8, color='grey')
        ax_hist.tick_params(axis='y', labelsize=7, labelcolor='grey')

    for j in range(n_features, len(axes_flat)):
        axes_flat[j].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.4, label='Vegetation samples'),
        Patch(facecolor='grey',  alpha=0.3, label='All samples'),
    ]
    fig.legend(handles=legend_elements, loc='lower right',
               fontsize=10, framealpha=0.8)

    plt.suptitle(
        f"Partial Dependence of Vegetation Probability — {area_name}\n"
        f"(green = vegetation samples, grey = all samples)",
        y=1.01, fontsize=13
    )
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/{area_name}_pdd_veg.png",
                bbox_inches='tight', dpi=150)
    plt.close()
    print("PDD for vegetation (all predictors) figure saved.")

    # ── Spearman correlation matrix ───────────────────────────────────────────
    # Checks for multicollinearity among predictors using Spearman rank
    # correlation (non-parametric; appropriate for non-normally distributed data).
    corr = gdf[feature_cols].corr(method='spearman')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'Spearman Correlation Matrix — {area_name}')
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/{area_name}_corr_matrix_spearman.png")
    plt.close()
    print("Spearman correlation matrix saved.")

# ── Export summary CSVs ───────────────────────────────────────────────────────
csv_folder = Path("Data/Python/Outputs")

df_random = pd.DataFrame(random_split_rows)
df_random.to_csv(csv_folder / "metrics_random_split.csv", index=False)
print("\nRandom split metrics saved to metrics_random_split.csv")

df_block = pd.DataFrame(block_cv_rows)
df_block.to_csv(csv_folder / "metrics_block_cv.csv", index=False)
print("Block CV metrics saved to metrics_block_cv.csv")

df_perm_veg = pd.DataFrame(perm_veg_rows)
df_perm_veg.to_csv(csv_folder / "perm_importance_veg_recall.csv", index=False)
print("Permutation importance (vegetation recall) saved to perm_importance_veg_recall.csv")

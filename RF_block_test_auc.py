#!/usr/bin/env python
# coding: utf-8
"""
Step 6a — Random Forest vegetation prediction at 1 m with spatial block cross-validation.

Copy of RF_block_test.py with permutation importance also scored by ROC AUC,
both on the refit model and on the held-out blocks of each block-CV fold (see
"Permutation importance (ROC AUC)" below). Everything else is unchanged.

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

Features (10 variables at 1 m by default):
  Landforms, Distance, Aspect_sin, Elevation, Curvature, Aspect_cos,
  Hillshade, Slope, TRI, SWI
  TRI/SWI can be excluded via config.yaml's features.use_tri / features.use_swi
  (e.g. to sanity-check the pipeline before those rasters are available) —
  Sample_areas_1m.py must be re-run with the same toggles first, since the
  sample shapefile's columns depend on which layers were included at sampling
  time.

Target: Vegetation (1=vegetated, recoded to 1; 2=non-vegetated, recoded to 0)

Outputs:
  Data/Python/Outputs/Figures/{area}_feat_imp.png
  Data/Python/Outputs/Figures/{area}_perm_imp.png
  Data/Python/Outputs/Figures/{area}_perm_veg.png
  Data/Python/Outputs/Figures/{area}_perm_auc.png
  Data/Python/Outputs/Figures/{area}_perm_auc_heldout.png
  Data/Python/Outputs/Figures/{area}_perm_auc_heldout_groups.png
  Data/Python/Outputs/Figures/{area}_pdd_veg.png
  Data/Python/Outputs/Figures/{area}_corr_matrix_spearman.png
  Data/Python/Outputs/metrics_random_split.csv
  Data/Python/Outputs/metrics_block_cv.csv
  Data/Python/Outputs/metrics_comparison.csv     (random split vs block CV, with deltas)
  Data/Python/Outputs/perm_importance_veg_recall.csv
  Data/Python/Outputs/perm_importance_auc.csv
  Data/Python/Outputs/perm_importance_auc_heldout.csv
  Data/Python/Outputs/perm_importance_auc_heldout_groups.csv
"""

import os

import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, recall_score, make_scorer,
                             precision_score, f1_score, roc_auc_score)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures without a display

from config_utils import load_config

cfg = load_config()
os.chdir(cfg.paths.base_dir)
print(os.getcwd())

# ── CONFIG ────────────────────────────────────────────────────────────────────
BLOCK_SIZE = cfg.cross_validation.block_size_1m   # spatial block size in metres; should be larger than the
                                                   # range of spatial autocorrelation in the terrain data
N_FOLDS    = cfg.cross_validation.block_cv_folds  # number of spatial CV folds
RF_PARAMS  = dict(n_estimators=cfg.random_forest.terrain_classifier.n_estimators, random_state=cfg.random_state)
# Variables shuffled together for grouped held-out importance (correlated
# variables share importance, so shuffling one alone can understate them).
# A group is skipped if any of its members is not in the model.
IMPORTANCE_GROUPS = {
    'Distance+Elevation': ['Distance', 'Elevation'],
    'Slope+SWI':          ['Slope', 'SWI'],
}
# ─────────────────────────────────────────────────────────────────────────────


def grouped_auc_drop(est, x_test, y_test, cols, n_repeats, rng):
    """Mean drop in ROC AUC when the columns in `cols` are shuffled together
    (same row order for all of them, so their relationship to each other is
    kept but their link to vegetation is broken)."""
    pos = list(est.classes_).index(1)
    base = roc_auc_score(y_test, est.predict_proba(x_test)[:, pos])
    drops = []
    for _ in range(n_repeats):
        x_perm = x_test.copy()
        order = rng.permutation(len(x_test))
        x_perm[cols] = x_test[cols].to_numpy()[order]
        drops.append(base - roc_auc_score(y_test, est.predict_proba(x_perm)[:, pos]))
    return np.mean(drops)


# ── Sample shapefile column mapping ─────────────────────────────────────────────
# High_res_script.py's sample_areas() (called from Sample_areas_1m.py) always
# samples 8 fixed terrain variables first (in this order), then appends TRI
# and SWI if enabled (in that order — see sample_areas()'s "Add TRI and SWI"
# block), then the vegetation target last. ArcPy's Sample tool names columns
# positionally (v_raster_1, v_raster_2, ...), truncated to "v_raste_N" once N
# reaches two digits — so the mapping below must move in lockstep with
# features.use_tri / features.use_swi in config.yaml and with which rasters
# Sample_areas_1m.py was actually run with (re-run it if you change these
# after the fact).
BASE_FEATURE_COLS = ['Aspect_cos', 'Aspect_sin', 'Curvature', 'Elevation',
                     'Distance', 'Hillshade', 'Landforms', 'Slope']
OPTIONAL_FEATURE_COLS = []
if cfg.features.use_tri:
    OPTIONAL_FEATURE_COLS.append('TRI')
if cfg.features.use_swi:
    OPTIONAL_FEATURE_COLS.append('SWI')

ALL_SAMPLE_COLS = BASE_FEATURE_COLS + OPTIONAL_FEATURE_COLS + ['Vegetation']
new_names = {
    (f"v_raster_{i}" if i < 10 else f"v_raste_{i}"): name
    for i, name in enumerate(ALL_SAMPLE_COLS, start=1)
}

study_areas = gpd.read_file(cfg.paths.outlines_shp)
study_areas = study_areas.drop(index=cfg.study_areas.exclude_indices)
if cfg.study_areas.only:
    study_areas = study_areas[study_areas["Glacier_na"].isin(cfg.study_areas.only)]

fig_folder = Path(cfg.paths.outputs_1m) / "Figures"
fig_folder.mkdir(parents=True, exist_ok=True)

# Storage for summary CSVs (one row per study area)
random_split_rows = []
block_cv_rows     = []
perm_veg_rows     = []
perm_auc_rows     = []
perm_auc_heldout_rows = []
perm_auc_groups_rows  = []

for area in study_areas.itertuples():
    area_name = area.Glacier_na
    geodiv_out_dir = f"{cfg.paths.outputs_1m}/{area_name}/Geodiversity"
    print(f"\n{'='*60}\nProcessing {area_name}\n{'='*60}")

    # ── Load sample data ───────────────────────────────────────────────────────
    # The samples shapefile has one row per sample point; columns v_raster_1…
    # v_raste_11 contain terrain variable values extracted by ArcPy in Step 4.
    gdf_full = gpd.read_file(f"{geodiv_out_dir}/{area_name}_samples.shp")
    gdf_full = gdf_full.replace(cfg.nodata_value, np.nan)  # replace ArcPy NoData sentinel
    gdf = gdf_full.dropna()  # remove rows with any missing variable
    print(f"Samples before NaN removal: {len(gdf_full)}")
    print(f"Samples after NaN removal:  {len(gdf)}")

    # Rename generic ArcPy column names to meaningful variable names
    # (mapping built above from config.yaml's features.use_tri / use_swi)
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
    # Excludes 'Vegetation' (the target). Landforms/Hillshade/Slope/TRI/SWI are
    # included only if enabled in config.yaml — Landforms, Hillshade and Slope
    # are always sampled regardless (they're base terrain variables), so toggling
    # features.use_landforms / use_hillshade / use_slope doesn't require re-sampling, unlike
    # features.use_tri/use_swi.
    feature_cols = ['Distance', 'Aspect_sin', 'Elevation',
                    'Curvature', 'Aspect_cos']
    if cfg.features.use_landforms:
        feature_cols.insert(0, 'Landforms')
    if cfg.features.use_hillshade:
        feature_cols.append('Hillshade')
    if cfg.features.use_slope:
        feature_cols.append('Slope')
    feature_cols += OPTIONAL_FEATURE_COLS
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
        'auc':       'roc_auc',   # NaN for a fold whose test blocks hold only one class
    }

    cv_results = cross_validate(
        RandomForestClassifier(**RF_PARAMS),
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
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        scores = cv_results[f'test_{metric}']
        # nanmean/nanstd: AUC is NaN for single-class test folds; other metrics never are
        block_row[f'{metric}_mean'] = round(np.nanmean(scores), 4)
        block_row[f'{metric}_std']  = round(np.nanstd(scores),  4)
        label = 'AUC' if metric == 'auc' else metric.capitalize()
        print(f"  {label:10s}: {np.nanmean(scores):.3f} ± {np.nanstd(scores):.3f}  "
              f"(folds: {np.round(scores, 3)})")
    block_cv_rows.append(block_row)

    # ── Random split (for comparison) ────────────────────────────────────────
    # This ignores spatial structure and is expected to give inflated metrics
    # relative to block CV. The delta between the two indicates bias magnitude.
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.cross_validation.random_split_test_size, random_state=cfg.random_state)
    rf_rs = RandomForestClassifier(**RF_PARAMS)
    rf_rs.fit(x_train, y_train)
    preds_rs = rf_rs.predict(x_test)
    proba_rs = rf_rs.predict_proba(x_test)[:, list(rf_rs.classes_).index(1)]

    rs_row = {
        'area':       area_name,
        'n_samples':  len(gdf),
        'accuracy':   round(accuracy_score(y_test,  preds_rs), 4),
        'precision':  round(precision_score(y_test, preds_rs, zero_division=0), 4),
        'recall':     round(recall_score(y_test,    preds_rs, zero_division=0), 4),
        'f1':         round(f1_score(y_test,        preds_rs, zero_division=0), 4),
        'auc':        round(roc_auc_score(y_test,   proba_rs), 4),
    }
    random_split_rows.append(rs_row)
    print(f"\n── Random Split Results ──")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        label = 'AUC' if metric == 'auc' else metric.capitalize()
        print(f"  {label:10s}: {rs_row[metric]:.3f}")

    # ── Refit on ALL data for importance and PDP plots ─────────────────────────
    # Cross-validation estimates generalisation performance; importance and PDP
    # plots use a model trained on all available data to maximise stability.
    rf = RandomForestClassifier(**RF_PARAMS)
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
                                      random_state=cfg.random_state, n_jobs=-1)
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
                                      n_repeats=30, random_state=cfg.random_state, n_jobs=-1)
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

    # ── Permutation importance (ROC AUC) ──────────────────────────────────────
    # Same as above but scored on ROC AUC: the probability that a randomly chosen
    # vegetated sample gets a higher predicted vegetation probability than a
    # randomly chosen non-vegetated one (1 = perfect separation, 0.5 = random).
    # Unlike accuracy it does not depend on the 0.5 threshold and is not
    # dominated by the majority (non-vegetation) class.
    perm_auc = permutation_importance(rf, x, y, scoring='roc_auc',
                                      n_repeats=30, random_state=cfg.random_state, n_jobs=-1)
    indices = np.argsort(perm_auc.importances_mean)
    plt.figure(figsize=(12, 10))
    plt.barh(np.array(x.columns)[indices],
             perm_auc.importances_mean[indices],
             xerr=perm_auc.importances_std[indices])
    plt.xlabel("Permutation Importance (mean decrease in ROC AUC)")
    plt.title(f"Permutation Importance — ROC AUC — {area_name}")
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/{area_name}_perm_auc.png")
    plt.close()
    print("Permutation importance (ROC AUC) figure saved.")

    row = {'area': area_name}
    for feat, imp, std in zip(feature_cols, perm_auc.importances_mean, perm_auc.importances_std):
        row[f'{feat}_mean'] = round(imp, 4)
        row[f'{feat}_std']  = round(std, 4)
    perm_auc_rows.append(row)

    # ── Permutation importance (ROC AUC) on held-out blocks ───────────────────
    # The importances above use the refit model on its own training data, which
    # can overstate variables that mainly help the model recognise location.
    # Here each block-CV fold model is instead scored on its test blocks (never
    # seen in training), and the results are averaged across folds. The error
    # bars are the SD across folds, i.e. how much importance varies between
    # parts of the study area. Folds whose test blocks contain only one class
    # are skipped, since AUC is undefined there.
    # GroupKFold is deterministic, so gkf.split() reproduces the same folds, in
    # the same order, as cross_validate() used for cv_results['estimator'].
    #
    # Grouped importance: each IMPORTANCE_GROUPS group is also shuffled as a
    # whole on the same held-out folds. Compared with the sum of its members'
    # individual importances: group >> sum means the members overlap (the model
    # falls back on one when the other is shuffled); group ≈ sum means they
    # contribute independently and the individual results can be read alone.
    active_groups = {name: members for name, members in IMPORTANCE_GROUPS.items()
                     if all(m in feature_cols for m in members)}
    group_rng = np.random.default_rng(cfg.random_state)
    fold_imps = []
    fold_group_imps = []
    n_folds_skipped = 0
    for est, (_, test_idx) in zip(cv_results['estimator'],
                                  gkf.split(x, y, groups=block_ids)):
        x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]
        if y_test.nunique() < 2 or len(est.classes_) < 2:
            n_folds_skipped += 1
            continue
        res = permutation_importance(est, x_test, y_test, scoring='roc_auc',
                                     n_repeats=10, random_state=cfg.random_state, n_jobs=-1)
        fold_imps.append(res.importances_mean)
        fold_group_imps.append([grouped_auc_drop(est, x_test, y_test, members, 10, group_rng)
                                for members in active_groups.values()])

    if fold_imps:
        fold_imps = np.array(fold_imps)            # shape (n_folds_used, n_features)
        heldout_mean = fold_imps.mean(axis=0)
        heldout_std  = fold_imps.std(axis=0)
        n_folds_used = len(fold_imps)
        print(f"Held-out permutation importance: {n_folds_used} folds used, "
              f"{n_folds_skipped} skipped (single-class test fold).")

        indices = np.argsort(heldout_mean)
        plt.figure(figsize=(12, 10))
        plt.barh(np.array(x.columns)[indices],
                 heldout_mean[indices],
                 xerr=heldout_std[indices])
        plt.xlabel("Permutation Importance (mean decrease in ROC AUC on held-out blocks)")
        plt.title(f"Permutation Importance — ROC AUC, held-out blocks — {area_name}\n"
                  f"(mean ± SD across {n_folds_used} folds)")
        plt.tight_layout()
        plt.savefig(f"{fig_folder}/{area_name}_perm_auc_heldout.png")
        plt.close()
        print("Permutation importance (ROC AUC, held-out blocks) figure saved.")

        row = {'area': area_name, 'n_folds_used': n_folds_used,
               'n_folds_skipped': n_folds_skipped}
        for feat, imp, std in zip(feature_cols, heldout_mean, heldout_std):
            row[f'{feat}_mean'] = round(imp, 4)
            row[f'{feat}_std']  = round(std, 4)
        perm_auc_heldout_rows.append(row)

        if active_groups:
            fold_group_imps = np.array(fold_group_imps)   # shape (n_folds_used, n_groups)
            group_names, group_vals, sum_vals = [], [], []
            for g, (name, members) in enumerate(active_groups.items()):
                member_idx = [feature_cols.index(m) for m in members]
                group_per_fold = fold_group_imps[:, g]
                sum_per_fold = fold_imps[:, member_idx].sum(axis=1)
                rho = (gdf[members].corr(method='spearman').iloc[0, 1]
                       if len(members) == 2 else np.nan)
                print(f"  Group {name}: shuffled together {group_per_fold.mean():.4f}, "
                      f"sum of individual {sum_per_fold.mean():.4f}, Spearman rho {rho:.2f}")
                perm_auc_groups_rows.append({
                    'area': area_name, 'group': name, 'members': ', '.join(members),
                    'spearman_rho': round(rho, 3),
                    'group_mean': round(group_per_fold.mean(), 4),
                    'group_std': round(group_per_fold.std(), 4),
                    'sum_individual_mean': round(sum_per_fold.mean(), 4),
                    'sum_individual_std': round(sum_per_fold.std(), 4),
                    'n_folds_used': n_folds_used,
                })
                group_names.append(f"{name}\n(rho = {rho:.2f})" if np.isfinite(rho) else name)
                group_vals.append((group_per_fold.mean(), group_per_fold.std()))
                sum_vals.append((sum_per_fold.mean(), sum_per_fold.std()))

            # Two bars per group: shuffled together vs. sum of individual importances
            pos = np.arange(len(group_names))
            h = 0.38
            plt.figure(figsize=(10, 2 + 1.5 * len(group_names)))
            plt.barh(pos + h / 2, [v[0] for v in group_vals], height=h,
                     xerr=[v[1] for v in group_vals], label='Shuffled together')
            plt.barh(pos - h / 2, [v[0] for v in sum_vals], height=h,
                     xerr=[v[1] for v in sum_vals], label='Sum of individual importances')
            plt.yticks(pos, group_names)
            plt.xlabel("Permutation Importance (mean decrease in ROC AUC on held-out blocks)")
            plt.title(f"Grouped permutation importance — held-out blocks — {area_name}\n"
                      f"(mean ± SD across {n_folds_used} folds)")
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(f"{fig_folder}/{area_name}_perm_auc_heldout_groups.png")
            plt.close()
            print("Grouped permutation importance (ROC AUC, held-out blocks) figure saved.")
    else:
        print("Held-out permutation importance skipped — no fold had both classes in its test blocks.")

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
csv_folder = Path(cfg.paths.outputs_1m)

df_random = pd.DataFrame(random_split_rows)
df_random.to_csv(csv_folder / "metrics_random_split.csv", index=False)
print("\nRandom split metrics saved to metrics_random_split.csv")

df_block = pd.DataFrame(block_cv_rows)
df_block.to_csv(csv_folder / "metrics_block_cv.csv", index=False)
print("Block CV metrics saved to metrics_block_cv.csv")

df_perm_veg = pd.DataFrame(perm_veg_rows)
df_perm_veg.to_csv(csv_folder / "perm_importance_veg_recall.csv", index=False)
print("Permutation importance (vegetation recall) saved to perm_importance_veg_recall.csv")

df_perm_auc = pd.DataFrame(perm_auc_rows)
df_perm_auc.to_csv(csv_folder / "perm_importance_auc.csv", index=False)
print("Permutation importance (ROC AUC) saved to perm_importance_auc.csv")

df_perm_auc_heldout = pd.DataFrame(perm_auc_heldout_rows)
df_perm_auc_heldout.to_csv(csv_folder / "perm_importance_auc_heldout.csv", index=False)
print("Permutation importance (ROC AUC, held-out blocks) saved to perm_importance_auc_heldout.csv")

df_perm_auc_groups = pd.DataFrame(perm_auc_groups_rows)
df_perm_auc_groups.to_csv(csv_folder / "perm_importance_auc_heldout_groups.csv", index=False)
print("Grouped permutation importance (ROC AUC, held-out blocks) saved to "
      "perm_importance_auc_heldout_groups.csv")

# ── Comparison table: random split vs block CV side by side ───────────────────
# Negative delta values indicate that the random split was optimistically biased
# (inflated metrics due to spatial autocorrelation between train and test pixels).
df_comp = df_random.drop(columns=["n_samples"]).rename(columns={
    "accuracy":  "rs_accuracy",
    "precision": "rs_precision",
    "recall":    "rs_recall",
    "f1":        "rs_f1",
    "auc":       "rs_auc",
}).merge(
    df_block.rename(columns={
        "accuracy_mean":  "cv_accuracy_mean",  "accuracy_std":  "cv_accuracy_std",
        "precision_mean": "cv_precision_mean", "precision_std": "cv_precision_std",
        "recall_mean":    "cv_recall_mean",    "recall_std":    "cv_recall_std",
        "f1_mean":        "cv_f1_mean",        "f1_std":        "cv_f1_std",
        "auc_mean":       "cv_auc_mean",       "auc_std":       "cv_auc_std",
    }),
    on="area",
    how="outer"
)

# Delta = block CV mean − random split (positive = block CV is higher than random split)
df_comp["delta_accuracy"]  = (df_comp["cv_accuracy_mean"]  - df_comp["rs_accuracy"]).round(4)
df_comp["delta_precision"] = (df_comp["cv_precision_mean"] - df_comp["rs_precision"]).round(4)
df_comp["delta_recall"]    = (df_comp["cv_recall_mean"]    - df_comp["rs_recall"]).round(4)
df_comp["delta_f1"]        = (df_comp["cv_f1_mean"]        - df_comp["rs_f1"]).round(4)
df_comp["delta_auc"]       = (df_comp["cv_auc_mean"]       - df_comp["rs_auc"]).round(4)

df_comp.to_csv(csv_folder / "metrics_comparison.csv", index=False)
print("Comparison table saved to metrics_comparison.csv")
print("\nDelta columns = block_cv_mean − random_split  (negative means block CV is lower, "
      "indicating the random split was optimistically biased)")

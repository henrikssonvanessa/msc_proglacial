#!/usr/bin/env python
# coding: utf-8
"""
Spatial block CV for the terrain RF with pooled out-of-fold metrics and
block-bootstrap confidence intervals.

Standalone alternative to the CV part of RF_block_test.py / rf_vegetation_20m.py
— it does not modify them or their outputs. It reads the same
{area}_samples.shp files and uses the same features (including the
features.use_* toggles), spatial blocks, and RF settings.

Why a separate evaluation:
  - With GroupKFold, a test fold can contain (almost) no vegetation samples
    when vegetation is patchy. Precision/recall/F1 are then undefined, and
    sklearn's scorers report 0, which drags down the fold means.
  - A standard deviation over 5 folds is a weak measure of uncertainty.

What this script does instead:
  1. Runs spatial block CV (blocks are kept whole, as before) and pools the
     out-of-fold predictions from all folds, so every sample is predicted
     exactly once by a model that never saw its block.
  2. Computes accuracy, balanced accuracy, precision, recall, F1 and ROC AUC
     once on the pooled predictions (threshold 0.5, like RF predict()).
  3. Gets 95% confidence intervals by bootstrapping BLOCKS (not samples),
     which respects spatial dependence. A wide interval means the estimate is
     genuinely uncertain, e.g. because vegetation occurs in very few blocks.

Model comparison (categorical Landforms):
  sklearn's RandomForestClassifier treats Landforms (geomorphon classes 1–10)
  as an ordered number, so it can only split it as "class <= k", even though
  the classes have no natural order. HistGradientBoostingClassifier ("hgb")
  is run alongside the RF with Landforms declared categorical, so it can group
  any subset of classes at a split. Both models use the same folds, so their
  metrics and importances are directly comparable.

  Importance is out-of-fold permutation importance: in each test fold a
  feature is shuffled and the drop in pooled ROC AUC is recorded (averaged
  over N_PERM_REPEATS). Unlike impurity importance it is not biased towards
  continuous variables. FEATURE_GROUPS are also shuffled jointly, because
  correlated terrain variables (Landforms, Curvature, TRI) share information
  and each one alone can look unimportant.

Settings (constants below):
  RESOLUTION : 1 or 20 — which sample set to evaluate
  CV_SCHEME  : "group"              GroupKFold, same fold scheme as the existing scripts
               "stratified_group"   StratifiedGroupKFold (shuffled, seeded): blocks stay
                                    whole but vegetation is spread across folds
               "leave_one_block_out" each block is held out in turn (best use of data
                                    for areas with few blocks; slower for many blocks)
  N_BOOT     : number of block-bootstrap resamples
  MODELS     : "rf" (RandomForest) and/or "hgb" (HistGradientBoosting, categorical Landforms)
  N_PERM_REPEATS : permutation repeats per feature (0 = skip importance)

Outputs:
  {outputs_dir}/metrics_block_cv_pooled_{CV_SCHEME}.csv — one row per area and
  model, with each metric plus _ci_low / _ci_high columns, and n_blocks,
  n_blocks_with_veg, and n_test_folds_without_veg for diagnosing sparse areas.
  {outputs_dir}/importance_block_cv_pooled_{CV_SCHEME}.csv — one row per area,
  model and feature (or feature group), with the mean/std AUC drop and rank.

Requires: geopandas, scikit-learn, PyYAML
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, LeaveOneGroupOut

from config_utils import load_config

cfg = load_config()
os.chdir(cfg.paths.base_dir)

# ── SETTINGS ──────────────────────────────────────────────────────────────────
RESOLUTION = 1             # 1 or 20
CV_SCHEME  = "group"       # "group", "stratified_group", or "leave_one_block_out"
N_BOOT     = 1000          # block-bootstrap resamples (lower this for a quicker run)
MODELS     = ["rf", "hgb"] # "rf" = RandomForest, "hgb" = HistGradientBoosting (categorical Landforms)
N_PERM_REPEATS = 5         # permutation-importance repeats per feature (0 = skip)
CATEGORICAL_FEATURES = ['Landforms']
# Features shuffled together for grouped importance (members not in use are ignored)
FEATURE_GROUPS = {"terrain_form": ['Landforms', 'Curvature', 'TRI']}
# ─────────────────────────────────────────────────────────────────────────────

N_FOLDS   = cfg.cross_validation.block_cv_folds
RF_PARAMS = dict(n_estimators=cfg.random_forest.terrain_classifier.n_estimators,
                 random_state=cfg.random_state, n_jobs=-1)
# early_stopping=False: the automatic early stopping holds out a random (not
# spatial) validation split, which would let spatial autocorrelation leak in.
HGB_PARAMS = dict(max_iter=200, early_stopping=False, random_state=cfg.random_state)
METRICS   = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"]


def resolution_settings():
    """
    Return (outputs_dir, block_size, new_names, feature_cols) for RESOLUTION.

    The column mapping mirrors RF_block_test.py (1 m) / rf_vegetation_20m.py
    (20 m): ArcPy's Sample tool names columns positionally, so the mapping must
    match the order the rasters were sampled in (and the features.use_* toggles
    the sampling scripts were run with).
    """
    optional = []
    if RESOLUTION == 1:
        base = ['Aspect_cos', 'Aspect_sin', 'Curvature', 'Elevation',
                'Distance', 'Hillshade', 'Landforms', 'Slope']
        if cfg.features.use_tri:
            optional.append('TRI')
        if cfg.features.use_swi:
            optional.append('SWI')
        all_cols = base + optional + ['Vegetation']
        outputs_dir = cfg.paths.outputs_1m
        block_size = cfg.cross_validation.block_size_1m
    elif RESOLUTION == 20:
        base = ['Aspect_cos', 'Aspect_sin', 'Curvature', 'Elevation',
                'Distance', 'Hillshade', 'Landforms', 'Vegetation', 'Slope']
        if cfg.features.use_tri:
            optional.append('TRI')
        if cfg.features.use_swi:
            optional.append('SWI')
        if cfg.features.use_snow:
            optional.append('Snow_cover')
        all_cols = base + optional
        outputs_dir = cfg.paths.outputs_20m
        block_size = cfg.cross_validation.block_size_20m
    else:
        raise ValueError("RESOLUTION must be 1 or 20")

    new_names = {
        (f"v_raster_{i}" if i < 10 else f"v_raste_{i}"): name
        for i, name in enumerate(all_cols, start=1)
    }

    feature_cols = ['Landforms', 'Distance', 'Aspect_sin', 'Elevation',
                    'Curvature', 'Aspect_cos', 'Hillshade']
    if cfg.features.use_slope:
        feature_cols.append('Slope')
    feature_cols += optional
    return outputs_dir, block_size, new_names, feature_cols


def assign_blocks(gdf, block_size):
    """Integer block ID per sample from a regular block_size x block_size m grid."""
    xs = np.array([g.x for g in gdf.geometry])
    ys = np.array([g.y for g in gdf.geometry])
    raw = (np.floor(xs / block_size).astype(np.int64) * 10_000_000
           + np.floor(ys / block_size).astype(np.int64))
    _, ids = np.unique(raw, return_inverse=True)
    return ids


def make_cv(n_blocks):
    """Build the cross-validator for CV_SCHEME (folds reduced if blocks are fewer)."""
    if CV_SCHEME == "leave_one_block_out":
        return LeaveOneGroupOut()
    n_folds = min(N_FOLDS, n_blocks)
    if CV_SCHEME == "group":
        return GroupKFold(n_splits=n_folds)
    if CV_SCHEME == "stratified_group":
        return StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                    random_state=cfg.random_state)
    raise ValueError('CV_SCHEME must be "group", "stratified_group" or "leave_one_block_out"')


def make_model(model_name, feature_cols):
    """RandomForest ("rf") or HistGradientBoosting with categorical Landforms ("hgb")."""
    if model_name == "rf":
        return RandomForestClassifier(**RF_PARAMS)
    if model_name == "hgb":
        cat_idx = [feature_cols.index(c) for c in CATEGORICAL_FEATURES if c in feature_cols]
        return HistGradientBoostingClassifier(categorical_features=cat_idx or None, **HGB_PARAMS)
    raise ValueError('MODELS entries must be "rf" or "hgb"')


def veg_proba(model, X):
    """P(vegetation); 0 if the training fold contained no vegetation."""
    classes = list(model.classes_)
    if 1 in classes:
        return model.predict_proba(X)[:, classes.index(1)]
    return np.zeros(len(X))


def permutation_sets(feature_cols):
    """{name: column indices} — each feature alone, plus FEATURE_GROUPS with >= 2 members in use."""
    sets = {f: [i] for i, f in enumerate(feature_cols)}
    for group, members in FEATURE_GROUPS.items():
        idx = [feature_cols.index(m) for m in members if m in feature_cols]
        if len(idx) >= 2:
            sets[f"group:{group}"] = idx
    return sets


def pooled_oof(X, y, blocks, cv, model_name, feature_cols, perm_sets, rng):
    """
    Out-of-fold P(vegetation) for every sample, plus the number of vegetation
    samples in each test fold (0 = a fold where per-fold recall is undefined).

    Also returns out-of-fold predictions with each entry of perm_sets shuffled
    within the test fold ({name: array (N_PERM_REPEATS, n_samples)}), so
    permutation importance is measured on data the model never saw. Group
    members are shuffled with the same row order, keeping their relationship
    to each other but breaking their link to vegetation.
    """
    proba = np.full(len(y), np.nan)
    perm_proba = {name: np.full((N_PERM_REPEATS, len(y)), np.nan) for name in perm_sets}
    veg_in_test_fold = []
    for train_idx, test_idx in cv.split(X, y, groups=blocks):
        model = make_model(model_name, feature_cols)
        model.fit(X[train_idx], y[train_idx])
        X_test = X[test_idx]
        proba[test_idx] = veg_proba(model, X_test)
        for name, cols in perm_sets.items():
            for r in range(N_PERM_REPEATS):
                X_perm = X_test.copy()
                order = rng.permutation(len(test_idx))
                X_perm[:, cols] = X_test[order][:, cols]
                perm_proba[name][r, test_idx] = veg_proba(model, X_perm)
        veg_in_test_fold.append(int(y[test_idx].sum()))
    return proba, perm_proba, veg_in_test_fold


def permutation_importance_rows(y, proba, perm_proba):
    """Mean/std drop in pooled ROC AUC per feature (or group), ranked within the model."""
    if N_PERM_REPEATS == 0:
        return []
    base_auc = roc_auc_score(y, proba)
    rows = []
    for name, preds in perm_proba.items():
        drops = np.array([base_auc - roc_auc_score(y, p) for p in preds])
        rows.append({"feature": name, "auc_drop_mean": drops.mean(), "auc_drop_std": drops.std()})
    # rank single features only, so groups don't push individual ranks down
    singles = sorted((r for r in rows if not r["feature"].startswith("group:")),
                     key=lambda r: r["auc_drop_mean"], reverse=True)
    for rank, r in enumerate(singles, start=1):
        r["rank"] = rank
    return rows


def compute_metrics(y, proba):
    """Metrics at threshold 0.5; NaN where undefined (e.g. recall with no vegetation)."""
    pred = proba > 0.5
    pos = y == 1
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    tn = int(np.sum(~pred & ~pos))
    fn = int(np.sum(~pred & pos))
    n = tp + fp + tn + fn

    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
          else np.nan)
    balanced = ((recall + specificity) / 2
                if np.isfinite(recall) and np.isfinite(specificity) else np.nan)
    auc = roc_auc_score(y, proba) if 0 < pos.sum() < len(y) else np.nan
    return {"accuracy": (tp + tn) / n, "balanced_accuracy": balanced,
            "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def block_bootstrap_ci(y, proba, blocks, n_boot, rng):
    """95% percentile CIs from resampling whole blocks with replacement."""
    unique_blocks = np.unique(blocks)
    idx_by_block = [np.where(blocks == b)[0] for b in unique_blocks]
    draws = {m: [] for m in METRICS}
    for _ in range(n_boot):
        chosen = rng.integers(0, len(unique_blocks), size=len(unique_blocks))
        idx = np.concatenate([idx_by_block[i] for i in chosen])
        res = compute_metrics(y[idx], proba[idx])
        for m in METRICS:
            draws[m].append(res[m])

    ci = {}
    for m in METRICS:
        vals = np.asarray(draws[m], dtype=float)
        vals = vals[~np.isnan(vals)]
        # too few valid resamples (metric mostly undefined) -> no interval
        ci[m] = tuple(np.percentile(vals, [2.5, 97.5])) if len(vals) >= 10 else (np.nan, np.nan)
    return ci


def main():
    outputs_dir, block_size, new_names, feature_cols = resolution_settings()
    print(f"Resolution: {RESOLUTION} m | CV scheme: {CV_SCHEME} | block size: {block_size} m | "
          f"bootstrap resamples: {N_BOOT}")
    print(f"Features: {feature_cols} | models: {MODELS} | permutation repeats: {N_PERM_REPEATS}")
    perm_sets = permutation_sets(feature_cols) if N_PERM_REPEATS > 0 else {}

    study_areas = gpd.read_file(cfg.paths.outlines_shp)
    study_areas = study_areas.drop(index=cfg.study_areas.exclude_indices)
    if cfg.study_areas.only:
        study_areas = study_areas[study_areas["Glacier_na"].isin(cfg.study_areas.only)]

    rng = np.random.default_rng(cfg.random_state)
    rows = []
    imp_rows = []

    for area in study_areas.itertuples():
        area_name = area.Glacier_na
        samples_fp = f"{outputs_dir}/{area_name}/Geodiversity/{area_name}_samples.shp"
        print(f"\n{'=' * 60}\nProcessing {area_name}\n{'=' * 60}")
        if not Path(samples_fp).exists():
            print(f"  Skipping — {samples_fp} not found.")
            continue

        gdf_full = gpd.read_file(samples_fp)
        gdf_full = gdf_full.replace(cfg.nodata_value, np.nan)
        gdf = gdf_full.dropna().rename(columns=new_names)

        X = gdf[feature_cols].to_numpy()
        y = gdf['Vegetation'].replace({1: 1, 2: 0}).astype(int).to_numpy()
        blocks = assign_blocks(gdf, block_size)
        n_blocks = len(np.unique(blocks))
        n_blocks_with_veg = len(np.unique(blocks[y == 1]))

        if n_blocks < 2 or y.sum() == 0 or y.sum() == len(y):
            print(f"  Skipping — need at least 2 blocks and both classes "
                  f"(blocks: {n_blocks}, vegetation samples: {int(y.sum())}).")
            continue

        cv = make_cv(n_blocks)
        print(f"  Samples: {len(y)} | blocks: {n_blocks} (with vegetation: {n_blocks_with_veg})")

        for model_name in MODELS:
            proba, perm_proba, veg_in_test_fold = pooled_oof(
                X, y, blocks, cv, model_name, feature_cols, perm_sets, rng)
            n_folds_without_veg = int(sum(v == 0 for v in veg_in_test_fold))

            metrics = compute_metrics(y, proba)
            ci = block_bootstrap_ci(y, proba, blocks, N_BOOT, rng)

            print(f"\n  [{model_name}] test folds without vegetation: "
                  f"{n_folds_without_veg}/{len(veg_in_test_fold)}")
            for m in METRICS:
                print(f"  {m:18s}: {metrics[m]:.3f}  [95% CI {ci[m][0]:.3f} – {ci[m][1]:.3f}]")

            row = {"area": area_name, "model": model_name, "n_samples": len(y),
                   "n_blocks": n_blocks, "n_blocks_with_veg": n_blocks_with_veg,
                   "n_test_folds": len(veg_in_test_fold),
                   "n_test_folds_without_veg": n_folds_without_veg, "cv_scheme": CV_SCHEME}
            for m in METRICS:
                row[m] = round(metrics[m], 4)
                row[f"{m}_ci_low"] = round(ci[m][0], 4)
                row[f"{m}_ci_high"] = round(ci[m][1], 4)
            rows.append(row)

            imp = permutation_importance_rows(y, proba, perm_proba)
            if imp:
                print(f"  Permutation importance (AUC drop):")
                for r in sorted(imp, key=lambda r: r["auc_drop_mean"], reverse=True):
                    print(f"    {r['feature']:24s}: {r['auc_drop_mean']:.4f} ± {r['auc_drop_std']:.4f}")
            for r in imp:
                imp_rows.append({"area": area_name, "model": model_name, "feature": r["feature"],
                                 "auc_drop_mean": round(r["auc_drop_mean"], 5),
                                 "auc_drop_std": round(r["auc_drop_std"], 5),
                                 "rank": r.get("rank", np.nan), "cv_scheme": CV_SCHEME})

    if not rows:
        print("\nNo areas evaluated.")
        return

    df = pd.DataFrame(rows)
    out_fp = Path(outputs_dir) / f"metrics_block_cv_pooled_{CV_SCHEME}.csv"
    df.to_csv(out_fp, index=False)
    print(f"\nSaved {out_fp}")
    print(df[["area", "model", "n_blocks", "n_blocks_with_veg", "n_test_folds_without_veg",
              "accuracy", "accuracy_ci_low", "accuracy_ci_high",
              "balanced_accuracy", "auc", "recall", "precision"]].to_string(index=False))

    if imp_rows:
        imp_df = pd.DataFrame(imp_rows)
        imp_fp = Path(outputs_dir) / f"importance_block_cv_pooled_{CV_SCHEME}.csv"
        imp_df.to_csv(imp_fp, index=False)
        print(f"\nSaved {imp_fp}")
        # Landforms rank per area and model: does categorical handling (hgb) lift it?
        lf = imp_df[imp_df["feature"] == "Landforms"]
        if not lf.empty:
            print("\nLandforms rank (1 = most important) per area:")
            print(lf.pivot(index="area", columns="model", values="rank").to_string())


if __name__ == "__main__":
    main()

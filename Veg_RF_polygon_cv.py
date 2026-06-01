"""
Step 3b — Random Forest vegetation classification with polygon-based cross-validation.

This script is an extension of Veg_RF.py that adds a spatially-aware validation
strategy: polygon-based GroupKFold cross-validation (CV).

Problem with random pixel splitting:
    Training polygons contain thousands of pixels, and nearby pixels are spatially
    autocorrelated (similar spectral values). A random 70/30 split can assign pixels
    from the same polygon to both training and test sets, inflating performance metrics
    because the model has effectively "seen" the neighbourhood of every test pixel.

Polygon-based CV solution:
    Each training polygon is assigned a group ID. GroupKFold ensures that all pixels
    from a given polygon are kept in the same fold (either all training or all test).
    This means the model is always evaluated on entirely unseen polygons, giving a more
    realistic estimate of performance on new spatial locations.

For each study area the script:
  1. Loads training polygons from a file geodatabase
  2. Samples all ortho pixels (4 bands) inside each polygon, tracking polygon membership
  3. Runs both a random 70/30 split (for comparison) and polygon GroupKFold CV
  4. Trains a final model on ALL training data and predicts the full raster
  5. Runs external validation on Suottas and Vartas (held-out areas)
  6. Exports metrics CSVs and a comparison table

Vegetation encoding: 1 = vegetation, 2 = non-vegetation (in rasters and training data)

Requires: rasterio, scikit-learn, geopandas, fiona
"""

import pandas as pd
import geopandas as gpd
import fiona
import rasterio
from rasterio.mask import mask
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import classification_report
from pathlib import Path
import matplotlib.pyplot as plt

# Maximum number of CV folds; automatically reduced if fewer polygons exist
N_FOLDS = 5

shp_path = "C:\TEMP\Vanessa_Henriksson\Data\proglacial_outlines.shp"
gdb = r"C:\TEMP\Vanessa_Henriksson\Proglacial_outline\training_data.gdb"

study_areas = gpd.read_file(shp_path)
study_areas = study_areas.drop(index=[12, 14])  # remove areas not included in analysis

# Storage for metrics from the three validation approaches
results_random = []      # random 70/30 split metrics
results_polygon_cv = []  # polygon-based GroupKFold CV metrics
results_external = []    # external validation metrics (Suottas, Vartas)

for area in study_areas.itertuples():
    area_name = area.Glacier_na
    ortho_fp = f"C:\TEMP\Vanessa_Henriksson\Data\Python\Outputs\{area_name}\{area_name}_ortho_clip.tif"

    # Open the clipped 4-band orthophoto for this area
    ortho_raster = rasterio.open(ortho_fp)
    print(ortho_raster.meta)

    # Read all 4 bands into a numpy array: shape = (bands, rows, cols)
    img = ortho_raster.read()
    bands, rows, cols = img.shape

    # Load training polygons for this area from the file geodatabase
    # Each polygon has a 'Category' field: 1 = vegetation, 2 = non-vegetation
    df_full = gpd.read_file(filename=gdb, layer=f"{area_name}_training")
    df = df_full.dropna().reset_index(drop=True)

    if df.empty:
        print(f"No training data found for {area_name}, skipping.")
        continue

    # ── Sample pixels inside training polygons ────────────────────────────────
    # For each polygon, extract the spectral values of all ortho pixels that
    # fall within its boundary. Track which polygon each pixel came from using
    # poly_idx — this group ID is used later in GroupKFold.
    X = []            # feature matrix: rows = pixels, cols = 4 spectral bands
    y = []            # labels: 1 = vegetation, 2 = non-vegetation
    polygon_ids = []  # group ID per pixel — one integer per polygon

    for poly_idx, (_, row) in enumerate(df.iterrows()):
        geom = [row.geometry]
        label = row["Category"]

        # Mask the raster to only the pixels inside this polygon
        out_img, _ = mask(
            ortho_raster,
            geom,
            crop=True,
            filled=False  # masked arrays: NaN outside polygon boundary
        )

        # Reshape from (bands, rows, cols) to (pixels, bands)
        pixels = out_img.reshape(bands, -1).T
        # Remove any pixel with NaN in any band (masked/nodata pixels)
        pixels = pixels[~np.any(np.isnan(pixels), axis=1)]

        if pixels.shape[0] == 0:
            continue

        X.append(pixels)
        y.append(np.full(pixels.shape[0], label))
        polygon_ids.append(np.full(pixels.shape[0], poly_idx))

    if not X:
        print(f"No valid training pixels found for {area_name}, skipping.")
        continue

    X = np.vstack(X)
    y = np.concatenate(y)
    polygon_ids = np.concatenate(polygon_ids)  # same length as X and y

    n_polygons = len(np.unique(polygon_ids))
    print(f"\nTraining samples: {X.shape}")
    print(f"Vegetation pixels: {np.sum(y == 1)}")
    print(f"Non-vegetation pixels: {np.sum(y == 2)}")
    print(f"Number of polygons: {n_polygons}")

    # ── Random 70/30 split (for comparison with polygon CV) ───────────────────
    # This is equivalent to the approach in Veg_RF.py; included here so that
    # the two validation strategies can be directly compared.
    print("\n── Random Split ──")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    rf_random = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf_random.fit(X_train, y_train)
    y_pred_random = rf_random.predict(X_test)
    print(classification_report(y_test, y_pred_random))

    report_random = classification_report(y_test, y_pred_random, output_dict=True)
    for label, label_name in [(1, "vegetation"), (2, "non-vegetation")]:
        m = report_random[str(label)]
        results_random.append({
            "area": area_name,
            "class": label_name,
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1_score": round(m["f1-score"], 4),
            "support": m["support"],
            "accuracy": round(report_random["accuracy"], 4),
        })

    # ── Polygon-based GroupKFold cross-validation ─────────────────────────────
    # GroupKFold ensures every pixel from a given polygon lands in only one fold.
    # The model is therefore evaluated on polygons it has never seen, which is a
    # more honest estimate of generalisation to new spatial locations.
    print("\n── Polygon CV ──")
    n_folds_area = min(N_FOLDS, n_polygons)
    if n_polygons < N_FOLDS:
        print(f"  Note: only {n_polygons} polygons — using {n_folds_area} folds")

    gkf = GroupKFold(n_splits=n_folds_area)
    fold_reports = []

    for fold, (train_idx, test_idx) in enumerate(
            gkf.split(X, y, groups=polygon_ids)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        test_polygons = np.unique(polygon_ids[test_idx])
        print(f"  Fold {fold + 1}: {len(test_polygons)} test polygons, "
              f"{len(test_idx)} test pixels")

        rf_fold = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        rf_fold.fit(X_tr, y_tr)
        y_pred_fold = rf_fold.predict(X_te)
        fold_reports.append(
            classification_report(y_te, y_pred_fold, output_dict=True)
        )

    # Average metrics across all folds
    for label, label_name in [(1, "vegetation"), (2, "non-vegetation")]:
        key = str(label)
        precisions = [r[key]["precision"] for r in fold_reports if key in r]
        recalls    = [r[key]["recall"]    for r in fold_reports if key in r]
        f1s        = [r[key]["f1-score"]  for r in fold_reports if key in r]
        accs       = [r["accuracy"]       for r in fold_reports]

        results_polygon_cv.append({
            "area": area_name,
            "class": label_name,
            "precision_mean": round(float(np.mean(precisions)), 4),
            "precision_std":  round(float(np.std(precisions)),  4),
            "recall_mean":    round(float(np.mean(recalls)),    4),
            "recall_std":     round(float(np.std(recalls)),     4),
            "f1_mean":        round(float(np.mean(f1s)),        4),
            "f1_std":         round(float(np.std(f1s)),         4),
            "accuracy_mean":  round(float(np.mean(accs)),       4),
            "accuracy_std":   round(float(np.std(accs)),        4),
            "n_folds":        n_folds_area,
            "n_polygons":     n_polygons,
        })

        print(f"  {label_name}  precision={np.mean(precisions):.3f}±{np.std(precisions):.3f}  "
              f"recall={np.mean(recalls):.3f}±{np.std(recalls):.3f}  "
              f"f1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
              f"accuracy={np.mean(accs):.3f}±{np.std(accs):.3f}")

    # ── Final model trained on ALL training data ───────────────────────────────
    # Cross-validation is only for evaluation; the actual vegetation map is
    # produced by a model trained on the complete labelled dataset.
    rf_final = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf_final.fit(X, y)

    # ── Predict the full raster in memory-efficient chunks ────────────────────
    # Predicting all pixels at once can exhaust memory for large rasters.
    # Processing in chunks of 1 million pixels keeps memory usage manageable.
    raster_reshaped = img.reshape(bands, -1).T  # shape: (n_pixels, 4 bands)
    chunk_size = 1_000_000
    n_pixels = raster_reshaped.shape[0]
    ortho_predictions = np.empty(n_pixels, dtype='int32')
    for i in range(0, n_pixels, chunk_size):
        ortho_predictions[i:i + chunk_size] = rf_final.predict(
            raster_reshaped[i:i + chunk_size]
        )

    # Reshape predictions back to the original raster grid
    classified_ortho = ortho_predictions.reshape(rows, cols)

    meta = ortho_raster.meta.copy()
    meta.update({
        "count": 1,       # single-band output
        "dtype": "int32",
        "compress": "lzw",
        "nodata": -9999
    })

    # Save to a separate folder so it does not overwrite Predicted_vegetation/
    output_path = Path(
        f"C:\TEMP\Vanessa_Henriksson\Data\Python\Outputs"
        f"\Predicted_vegetation_polygon_cv\{area_name}_predicted_vegetation.tif"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(classified_ortho.astype("int32"), 1)

    print(f"{area_name} raster saved to {output_path}")

    # ── External validation for Suottas and Vartas ────────────────────────────
    # These two areas were withheld from training entirely.
    # External test polygons (with known Category labels) are used to assess
    # how well the model generalises to new, unseen proglacial areas.
    ext_gdb = r"C:\TEMP\Vanessa_Henriksson\Proglacial_outline\external_test_data.gdb"
    if area_name in ("Suottas", "Vartas"):
        df_ext = gpd.read_file(filename=ext_gdb, layer=area_name).dropna(subset=["Category"])
        y_true_ext, y_pred_ext = [], []
        with rasterio.open(output_path) as classified_raster:
            for _, row in df_ext.iterrows():
                geom = [row.geometry]
                # External test data uses 0 = non-vegetation; remap to 2 to
                # match the raster encoding (1 = veg, 2 = non-veg)
                true_label = 1 if row["Category"] == 1 else 2
                try:
                    out_img, _ = mask(classified_raster, geom, crop=True, filled=True, nodata=-9999)
                    pixels = out_img[0].flatten()
                    pixels = pixels[pixels != -9999]  # exclude nodata pixels
                    if pixels.size == 0:
                        continue
                    y_pred_ext.append(pixels)
                    y_true_ext.append(np.full(pixels.size, true_label))
                except Exception:
                    continue

        if y_true_ext:
            y_true_ext = np.concatenate(y_true_ext)
            y_pred_ext = np.concatenate(y_pred_ext)
            print(f"\n--- External raster validation for {area_name} ---")
            print(classification_report(y_true_ext, y_pred_ext,
                                        target_names=["vegetation", "non-vegetation"],
                                        labels=[1, 2]))
            ext_report = classification_report(y_true_ext, y_pred_ext, labels=[1, 2], output_dict=True)
            for label, label_name in [(1, "vegetation"), (2, "non-vegetation")]:
                if str(label) in ext_report:
                    m = ext_report[str(label)]
                    results_external.append({
                        "area": area_name,
                        "class": label_name,
                        "precision": round(m["precision"], 4),
                        "recall": round(m["recall"], 4),
                        "f1_score": round(m["f1-score"], 4),
                        "support": m["support"],
                        "accuracy": round(ext_report["accuracy"], 4),
                    })
        else:
            print(f"No valid external test pixels found for {area_name}.")

# ── Export metrics to CSV ──────────────────────────────────────────────────────
out_dir = Path(r"C:\TEMP\Vanessa_Henriksson\Data\Python\Outputs\Predicted_vegetation_polygon_cv")
out_dir.mkdir(parents=True, exist_ok=True)

df_random = pd.DataFrame(results_random)
df_random.to_csv(out_dir / "classification_results_random_split.csv", index=False)
print("\nRandom split metrics saved to classification_results_random_split.csv")

df_polygon_cv = pd.DataFrame(results_polygon_cv)
df_polygon_cv.to_csv(out_dir / "classification_results_polygon_cv.csv", index=False)
print("Polygon CV metrics saved to classification_results_polygon_cv.csv")

df_external = pd.DataFrame(results_external)
df_external.to_csv(out_dir / "classification_results_external.csv", index=False)
print("External validation metrics saved to classification_results_external.csv")

# ── Comparison table: random split vs polygon CV side by side ─────────────────
# Negative delta values indicate that the random split was optimistically biased
# (inflated accuracy due to spatial autocorrelation between train and test pixels).
df_comp = df_random.rename(columns={
    "precision": "rs_precision",
    "recall":    "rs_recall",
    "f1_score":  "rs_f1",
    "accuracy":  "rs_accuracy",
    "support":   "rs_support",
}).merge(
    df_polygon_cv.rename(columns={
        "precision_mean": "cv_precision_mean",
        "precision_std":  "cv_precision_std",
        "recall_mean":    "cv_recall_mean",
        "recall_std":     "cv_recall_std",
        "f1_mean":        "cv_f1_mean",
        "f1_std":         "cv_f1_std",
        "accuracy_mean":  "cv_accuracy_mean",
        "accuracy_std":   "cv_accuracy_std",
    }),
    on=["area", "class"],
    how="outer"
)

# Delta = polygon CV mean − random split (positive = CV is higher than random split)
df_comp["delta_precision"] = (df_comp["cv_precision_mean"] - df_comp["rs_precision"]).round(4)
df_comp["delta_recall"]    = (df_comp["cv_recall_mean"]    - df_comp["rs_recall"]).round(4)
df_comp["delta_f1"]        = (df_comp["cv_f1_mean"]        - df_comp["rs_f1"]).round(4)
df_comp["delta_accuracy"]  = (df_comp["cv_accuracy_mean"]  - df_comp["rs_accuracy"]).round(4)

df_comp.to_csv(out_dir / "classification_results_comparison.csv", index=False)
print("Comparison table saved to classification_results_comparison.csv")
print("\nDelta columns = polygon_cv_mean − random_split  (negative means polygon CV is lower, "
      "indicating the random split was optimistically biased)")

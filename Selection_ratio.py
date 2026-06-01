#!/usr/bin/env python
# coding: utf-8
"""
Step 8 — Selection ratio analysis and partial dependence plots.

This script analyses how vegetation presence relates to terrain conditions across
all study areas, using two complementary visualisations:

Selection ratio
---------------
The selection ratio for a variable's value range bin is:
    SR = (proportion of vegetation pixels in bin) / (proportion of all pixels in bin)

SR > 1: vegetation is found more often than expected by chance in that bin
         → that terrain condition is preferentially selected by vegetation
SR < 1: vegetation is less common than expected → vegetation avoids that condition
SR = 1 (red dashed line): no preference — vegetation uses the habitat in proportion
         to its availability

Bars are coloured by sample size: dark blue = n ≥ 10 (reliable), light = n < 10.

Partial dependence plots (PDPs)
--------------------------------
Each PDP shows the marginal effect of one terrain variable on the model's
predicted probability of vegetation, after averaging out the other variables.
The PDP is overlaid on the same axes as the selection ratio to show whether
the statistical model's predictions align with the raw habitat use patterns.

The script handles four resolution modes (set RESOLUTION at top):
  1  → 1 m model  (9 features, all study areas)
  20 → 20 m model (10 features including Snow_cover, all study areas)
  '5mm'  → drone model at 5 mm (Kårsa only, no Distance)
  '12cm' → drone model at 12 cm (Kårsa only)

Outputs (per variable):
  {data_folder}/Figures/{feature}_selection_ratio.png   one subplot per area
  {data_folder}/Figures/all_areas_selection_ratio.png   all areas overlaid
"""

import os
os.chdir(r"C:\TEMP\Vanessa_Henriksson")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
RESOLUTION      = 20   # set to 1, 20, '5mm', or '12cm'
N_BINS          = 10   # number of equal-width bins per variable for selection ratio
GRID_RESOLUTION = 50   # number of grid points for PDP curves
PDP_Y_PADDING   = 0.02 # extra padding above/below the global PDP y-axis range
# ─────────────────────────────────────────────────────────────────────────────

# ── Resolution-specific settings ─────────────────────────────────────────────
# Each resolution has its own output folder, feature list, column rename map,
# vegetation label encoding, and whether to restrict to Kårsa only.

if RESOLUTION == 1:
    data_folder   = "Data/Python/Outputs"
    feature_cols  = ['Landforms', 'Distance', 'Aspect_sin', 'Elevation',
                     'Curvature', 'Aspect_cos', 'Hillshade', 'TRI', 'SWI']
    new_names     = {
        "v_raster_1": "Aspect_cos", "v_raster_2": "Aspect_sin",
        "v_raster_3": "Curvature",  "v_raster_4": "Elevation",
        "v_raster_5": "Distance",   "v_raster_6": "Hillshade",
        "v_raster_7": "Landforms",  "v_raster_8": "Slope",
        "v_raster_9": "SWI",        "v_raste_10": "TRI",
        "v_raste_11": "Vegetation"
    }
    veg_values    = [1, 2]   # 1 = veg, 2 = non-veg
    veg_recode    = {1: 1, 2: 0}  # recode for RF (needs 0/1 target)
    karsa_only    = False

elif RESOLUTION == 20:
    data_folder   = "Data/Python/Outputs_20m"
    feature_cols  = ['Landforms', 'Distance', 'Aspect_sin', 'Elevation',
                     'Curvature', 'Aspect_cos', 'Hillshade', 'TRI', 'SWI',
                     'Snow_cover']
    new_names     = {
        "v_raster_1": "Aspect_cos", "v_raster_2": "Aspect_sin",
        "v_raster_3": "Curvature",  "v_raster_4": "Elevation",
        "v_raster_5": "Distance",   "v_raster_6": "Hillshade",
        "v_raster_7": "Landforms",  "v_raster_8": "Vegetation",
        "v_raster_9": "Slope",      "v_raste_10": "TRI",
        "v_raste_11": "SWI",        "v_raste_12": "Snow_cover",
    }
    veg_values    = [1, 2]
    veg_recode    = {1: 1, 2: 0}
    karsa_only    = False

elif RESOLUTION == '5mm':
    data_folder   = "Data/Python/Outputs_5mm"
    feature_cols  = ['Aspect_cos', 'Aspect_sin', 'Curvature', 'Elevation',
                     'Hillshade', 'Landforms', 'Slope', 'SWI', 'TRI']
    new_names     = {
        "v_raster_1": "Aspect_cos", "v_raster_2": "Aspect_sin",
        "v_raster_3": "Curvature",  "v_raster_4": "Hillshade",
        "v_raster_5": "Landforms",  "v_raster_6": "Slope",
        "v_raster_7": "SWI",        "v_raster_8": "TRI",
        "v_raster_9": "Elevation",  "v_raste_10": "Vegetation",
    }
    veg_values    = [0, 1]   # drone data uses 0 = non-veg, 1 = veg
    veg_recode    = None     # already 0/1
    karsa_only    = True     # drone data only available for Kårsa

elif RESOLUTION == '12cm':
    data_folder   = "Data/Python/Outputs_12cm"
    feature_cols  = ['Aspect_cos', 'Aspect_sin', 'Curvature', 'Distance',
                     'Elevation', 'Hillshade', 'Landforms', 'Slope', 'SWI', 'TRI']
    new_names     = {
        "v_raster_1": "Aspect_cos", "v_raster_2": "Aspect_sin",
        "v_raster_3": "Curvature",  "v_raster_4": "Elevation",
        "v_raster_5": "Distance",   "v_raster_6": "Hillshade",
        "v_raster_7": "Landforms",  "v_raster_8": "Slope",
        "v_raster_9": "SWI",        "v_raste_10": "TRI",
        "v_raste_11": "Vegetation",
    }
    veg_values    = [0, 1]
    veg_recode    = None
    karsa_only    = True

study_areas = gpd.read_file("Data/proglacial_outlines.shp")
study_areas = study_areas.drop(index=[12, 14])
if karsa_only:
    study_areas = study_areas[study_areas["Glacier_na"] == "Kårsa"]
area_names  = sorted([area.Glacier_na for area in study_areas.itertuples()])

fig_folder = Path(f"{data_folder}/Figures")
fig_folder.mkdir(parents=True, exist_ok=True)

n_cols = 4
n_rows = int(np.ceil(len(feature_cols) / n_cols))

# ── Pass 1: collect PDP values and selection ratios for all areas ─────────────
# We compute these first for all areas so we can set a consistent global y-axis
# range for the PDP curves across subplots (makes areas visually comparable).
pdp_store = {a: {} for a in area_names}   # {area: {feature: (grid_vals, avg_preds)}}
sr_store  = {f: {} for f in feature_cols}  # {feature: {area: (midpoints, sr, bins, counts)}}

pdp_global_min = {f:  np.inf for f in feature_cols}
pdp_global_max = {f: -np.inf for f in feature_cols}

for area in study_areas.itertuples():
    area_name      = area.Glacier_na
    geodiv_out_dir = f"{data_folder}/{area_name}/Geodiversity"
    print(f"Pass 1 — {area_name}")

    gdf = gpd.read_file(f"{geodiv_out_dir}/{area_name}_samples.shp")
    gdf = gdf.replace(-9999, np.nan).dropna()
    gdf = gdf.rename(columns=new_names)
    # Keep only rows with valid vegetation labels
    gdf = gdf[gdf['Vegetation'].isin(veg_values)]

    x         = gdf[feature_cols]
    # Apply recoding if needed (drone data already has 0/1 labels)
    y         = (gdf['Vegetation'].replace(veg_recode) if veg_recode else gdf['Vegetation']).astype(int)
    veg       = x[y == 1]   # feature values for vegetation pixels only
    total_veg = len(veg)
    total_all = len(x)

    # Fit RF on all data for this area (used for PDP computation)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x, y)

    for feature in feature_cols:
        feat_idx = feature_cols.index(feature)

        # PDP: marginal effect of this feature on predicted vegetation probability
        # grid_resolution=50 means 50 evenly-spaced values across the feature range
        pd_results = partial_dependence(
            rf, x,
            features=[feat_idx],
            response_method='predict_proba',
            grid_resolution=GRID_RESOLUTION,
            kind="average"
        )
        grid_vals = pd_results["grid_values"][0]
        avg_preds = pd_results["average"][0]
        pdp_store[area_name][feature] = (grid_vals, avg_preds)

        # Track global PDP y-axis range for consistent scaling across areas
        pdp_global_min[feature] = min(pdp_global_min[feature], avg_preds.min())
        pdp_global_max[feature] = max(pdp_global_max[feature], avg_preds.max())

        # Selection ratio:
        # 1. Bin the feature into N_BINS equal-width bins
        # 2. For each bin, count all samples and vegetation-only samples
        # 3. SR = (veg_count/total_veg) / (all_count/total_all)
        bins          = np.linspace(x[feature].min(), x[feature].max(), N_BINS + 1)
        bin_midpoints = (bins[:-1] + bins[1:]) / 2

        all_counts, _ = np.histogram(x[feature],   bins=bins)
        veg_counts, _ = np.histogram(veg[feature], bins=bins)

        # Proportions of available habitat and used habitat in each bin
        all_prop = all_counts / total_all
        veg_prop = veg_counts / total_veg

        # SR = used / available; undefined (NaN) if no pixels available in the bin
        with np.errstate(divide='ignore', invalid='ignore'):
            sr = np.where(all_prop > 0, veg_prop / all_prop, np.nan)

        sr_store[feature][area_name] = (bin_midpoints, sr, bins, all_counts)

# Set font for publication-quality figures
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 28})

# ── Pass 2: one figure per variable, one subplot per study area ────────────────
# Each subplot shows:
#   - Blue bars: selection ratio per bin (SR ≥ 1 → vegetation prefers that range)
#   - Red dashed line at SR = 1 (no selection)
#   - Orange line: PDP vegetation probability (right axis)
#   - Sample counts annotated in each bar (n < 10 = light blue = unreliable)
MIN_SAMPLES = 10  # minimum bin sample size for reliable SR estimation
n_areas     = len(area_names)
n_cols_fig  = min(3, n_areas)
n_rows_fig  = int(np.ceil(n_areas / n_cols_fig))

for feature in feature_cols:
    print(f"Pass 2 — plotting {feature}")

    fig, axes = plt.subplots(n_rows_fig, n_cols_fig, squeeze=False,
                             figsize=(6 * n_cols_fig, 5 * n_rows_fig))
    axes_flat = axes.flatten()

    for i, area_name in enumerate(area_names):
        ax_sr  = axes_flat[i]        # left axis: selection ratio (bars)
        ax_pdp = ax_sr.twinx()       # right axis: PDP probability (line)

        # Selection ratio bars
        bin_midpoints, sr, bins, all_counts = sr_store[feature][area_name]
        bin_width  = (bins[1] - bins[0]) * 0.85

        # Bars with fewer than MIN_SAMPLES are shown in light blue as unreliable
        bar_colors = ['steelblue' if c >= MIN_SAMPLES else 'lightsteelblue'
                      for c in all_counts]

        ax_sr.bar(bin_midpoints, sr,
                  width=bin_width,
                  color=bar_colors, alpha=0.5, edgecolor='white',
                  zorder=2)
        # Reference line: SR = 1 (neutral / no selection)
        ax_sr.axhline(1.0, color='red', linestyle='--',
                      linewidth=1.2, zorder=3)

        # Annotate each bar with sample count (rotated for readability)
        for mid, count in zip(bin_midpoints, all_counts):
            if count > 0:
                ax_sr.text(mid, 0.05, str(count),
                           ha='center', va='bottom',
                           fontsize=25, color='navy',
                           rotation=90, zorder=5)

        display_name = area_name.replace("_II_III", "").replace("_", " ")
        if i + n_cols_fig >= n_areas:
            ax_sr.set_xlabel(feature, fontsize=29)
        if i % n_cols_fig == 0:
            ax_sr.set_ylabel("Selection ratio", fontsize=28, color='steelblue')
        ax_sr.tick_params(axis='y', labelcolor='steelblue')
        ax_sr.set_title(display_name, fontsize=27, fontweight='bold')
        ax_sr.grid(True, linestyle='--', alpha=0.3, zorder=1)

        # PDP curve with fixed y-axis range across all areas for comparability
        grid_vals, avg_preds = pdp_store[area_name][feature]

        ax_pdp.plot(grid_vals, avg_preds,
                    color='darkorange', linewidth=2.8,
                    alpha=0.9, zorder=4)
        if i % n_cols_fig == n_cols_fig - 1:
            ax_pdp.set_ylabel("Vegetation probability (PDP)",
                              fontsize=28, color='black')
        ax_pdp.tick_params(axis='y', labelcolor='black')

        # Fixed y-axis range using global min/max ± padding
        y_min = pdp_global_min[feature] - PDP_Y_PADDING
        y_max = pdp_global_max[feature] + PDP_Y_PADDING
        ax_pdp.set_ylim(y_min, y_max)

    # Legend with colour/line type definitions
    legend_elements = [
        Patch(facecolor='steelblue',      alpha=0.6, label='Selection ratio (n >= 10)'),
        Patch(facecolor='lightsteelblue', alpha=0.6, label='Selection ratio (n < 10, unreliable)'),
        Line2D([0], [0], color='red',        linestyle='--', linewidth=1.5,
               label='Selection ratio = 1 (no selection)'),
        Line2D([0], [0], color='darkorange', linestyle='-',  linewidth=2.0,
               label='PDP — vegetation probability'),
    ]

    # Place legend in the first empty subplot (if any) on the last row
    if n_areas < len(axes_flat):
        ax_leg = axes_flat[n_areas]
        ax_leg.set_visible(True)
        ax_leg.axis('off')
        ax_leg.legend(handles=legend_elements, loc='center', fontsize=28, framealpha=0.9)
        for j in range(n_areas + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
    else:
        fig.legend(handles=legend_elements,
                   loc='upper center',
                   bbox_to_anchor=(0.5, -0.02),
                   fontsize=28, framealpha=0.9, ncol=2)

    plt.suptitle(
        f"Selection Ratio & Partial Dependence — {feature} ({RESOLUTION}m)",
        fontsize=32, y=1.0
    )
    plt.tight_layout()
    safe_feature = feature.replace('/', '_')  # make safe for filenames
    plt.savefig(f"{fig_folder}/{safe_feature}_selection_ratio.png",
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {feature}")

# ── Combined plot: selection ratio lines for all study areas on one figure ─────
# Shows each area as a coloured line, making it easy to compare habitat use
# patterns across areas for each terrain variable.
n_areas = len(study_areas)
cmap    = plt.get_cmap('tab20', n_areas)
colors  = [cmap(i) for i in range(n_areas)]

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(6 * n_cols, 5 * n_rows))
axes_flat = axes.flatten()

for j, feature in enumerate(feature_cols):
    ax = axes_flat[j]
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.2, zorder=0)

    for i, area_name in enumerate(area_names):
        if area_name not in sr_store[feature]:
            continue
        mids, sr, _, _ = sr_store[feature][area_name]
        display_name = area_name.replace("_II_III", "").replace("_", " ")
        ax.plot(mids, sr,
                color=colors[i], linewidth=1.5,
                alpha=0.85, label=display_name)

    if j + n_cols >= len(feature_cols):
        ax.set_xlabel(feature, fontsize=29)
    if j % n_cols == 0:
        ax.set_ylabel("Selection ratio", fontsize=28)
    ax.set_title(feature, fontsize=27, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

for j in range(len(feature_cols), len(axes_flat)):
    axes_flat[j].set_visible(False)

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(
    handles[1:], labels[1:],
    loc='upper center',
    bbox_to_anchor=(0.5, -0.02),
    fontsize=27,
    title="Study area",
    title_fontsize=28,
    framealpha=0.9,
    ncol=5
)

plt.suptitle(f"Selection Ratio — All Study Areas ({RESOLUTION}m)\n(red dashed = no selection)",
             fontsize=32, y=1.01)
plt.tight_layout()
plt.savefig(f"{fig_folder}/all_areas_selection_ratio.png",
            bbox_inches='tight', dpi=150)
plt.close()
print("Combined selection ratio figure saved.")

# Reset matplotlib to defaults after using custom fonts
plt.rcParams.update({'font.family': matplotlib.rcParamsDefault['font.family'],
                     'font.size':   matplotlib.rcParamsDefault['font.size']})

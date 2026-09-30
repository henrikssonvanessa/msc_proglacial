"""
Step 3c — Compute TRI (Terrain Ruggedness Index) and SWI (SAGA Wetness Index)
via SAGA GIS.

Automates what was previously a manual QGIS/SAGA step. For each study area,
at both 1 m and 20 m resolution, calls saga_cmd to derive:
  - TRI  from {area_name}_DEM_clip.tif (the clipped, unfilled DEM)
  - SWI  from {area_name}_DEM_fill.tif (the sink-filled DEM — required for
         hydrologically consistent flow accumulation)

Both DEMs are produced by Variable_calculation.py (1 m) / Low_res_script.py
(20 m); run those first. Outputs are written to paths.tri_swi_1m_dir /
tri_swi_20m_dir using the existing geodiversity.tri_filename_1m/20m and
swi_filename_1m/20m patterns — the same locations every other script in this
pipeline already reads TRI/SWI from (Sample_areas_1m.py, Sample_areas_20m.py,
RF_block_test.py, rf_vegetation_20m.py, Selection_ratio.py,
geodiversity_index_HL.py/PCA.py). Once this has been run for an area,
features.use_tri / features.use_swi can be set back to true with no other
changes needed.

SAGA tools used (library -- tool number):
  ta_morphometry 16  "Terrain Ruggedness Index (TRI)"
  ta_hydrology   15  "SAGA Wetness Index" (the wetness index itself is that
                     tool's TWI output parameter, despite the tool's name)

Verified against SAGA 9.11.1's saga_cmd (config: paths.saga_cmd). If you're on
a different SAGA version and a call fails, run e.g. `saga_cmd ta_hydrology 15`
with no other arguments from a terminal — it prints that build's exact
parameter names/choice indices, the fastest way to spot a mismatch. Tool
parameters (search radius, SWI suction, etc.) are configurable under `saga:`
in config.yaml.

Per-area/per-resolution failures are logged and skipped rather than stopping
the whole run, so one missing DEM or a SAGA error for one area doesn't block
the rest.

Requires: SAGA GIS (saga_cmd at paths.saga_cmd), geopandas, PyYAML
"""

import os
import subprocess
from pathlib import Path

import geopandas as gpd

from config_utils import load_config, resolve_path

cfg = load_config()
os.chdir(cfg.paths.base_dir)
print(os.getcwd())

SAGA_CMD = cfg.paths.saga_cmd
SAGA_ENV_BAT = cfg.paths.saga_env_bat

# Each study area is processed once per resolution; the only things that
# differ between 1 m and 20 m are the output directory/filenames and where
# the input DEMs live.
RESOLUTIONS = [
    {
        "label": "1m",
        "outputs_dir": cfg.paths.outputs_1m,
        "tri_swi_dir": resolve_path(cfg, cfg.paths.tri_swi_1m_dir),
        "tri_filename": cfg.geodiversity.tri_filename_1m,
        "swi_filename": cfg.geodiversity.swi_filename_1m,
    },
    {
        "label": "20m",
        "outputs_dir": cfg.paths.outputs_20m,
        "tri_swi_dir": resolve_path(cfg, cfg.paths.tri_swi_20m_dir),
        "tri_filename": cfg.geodiversity.tri_filename_20m,
        "swi_filename": cfg.geodiversity.swi_filename_20m,
    },
]

study_areas = gpd.read_file(cfg.paths.outlines_shp)
study_areas = study_areas.drop(index=cfg.study_areas.exclude_indices)
if cfg.study_areas.only:
    study_areas = study_areas[study_areas["Glacier_na"].isin(cfg.study_areas.only)]


def _quote(token):
    """Quote a command token for cmd.exe if it contains whitespace."""
    token = str(token)
    return f'"{token}"' if " " in token else token


def run_saga(args, description):
    """
    Run a saga_cmd command, raising with SAGA's own stdout/stderr on failure.

    SAGA is bundled via QGIS/OSGeo4W, which won't load (missing DLLs on PATH,
    GDAL_DATA/PROJ_LIB unset) unless its environment batch script is sourced
    first — the same way QGIS's own bin/saga_gui.bat calls o4w_env.bat before
    launching saga_gui.exe. So this runs everything through cmd.exe as
    `call "<saga_env_bat>" && "<saga_cmd>" <args...>` rather than invoking
    saga_cmd directly. If paths.saga_env_bat is empty (e.g. a standalone SAGA
    install that doesn't need it), the `call` step is skipped.

    Parameters
    ----------
    args        : list[str]  Arguments after saga_cmd itself, e.g.
                             ["ta_morphometry", "16", "-DEM=...", "-TRI=..."]
    description : str        Short label used in the raised error / log line
    """
    saga_call = " ".join(_quote(a) for a in [SAGA_CMD] + args)
    command = f'call "{SAGA_ENV_BAT}" && {saga_call}' if SAGA_ENV_BAT else saga_call
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    print(f"    {description} done.")


def compute_tri(dem_clip_path, out_path, params):
    """Derive TRI from the clipped DEM via SAGA's ta_morphometry tool 16."""
    run_saga([
        "ta_morphometry", "16",
        f"-DEM={dem_clip_path}",
        f"-TRI={out_path}",
        f"-MODE={params.mode}",
        f"-RADIUS={params.radius}",
        f"-DW_WEIGHTING={params.dw_weighting}",
    ], f"TRI -> {out_path.name}")


def compute_swi(dem_fill_path, out_path, params):
    """Derive SWI from the filled DEM via SAGA's ta_hydrology tool 15."""
    run_saga([
        "ta_hydrology", "15",
        f"-DEM={dem_fill_path}",
        f"-TWI={out_path}",
        f"-SUCTION={params.suction}",
        f"-AREA_TYPE={params.area_type}",
        f"-SLOPE_TYPE={params.slope_type}",
        f"-SLOPE_MIN={params.slope_min}",
        f"-SLOPE_OFF={params.slope_off}",
        f"-SLOPE_WEIGHT={params.slope_weight}",
    ], f"SWI -> {out_path.name}")


# ── Main loop: compute TRI/SWI for every area, at both resolutions ────────────
for res in RESOLUTIONS:
    print(f"\n{'='*60}\nResolution: {res['label']}\n{'='*60}")
    tri_swi_dir = Path(res["tri_swi_dir"])
    tri_swi_dir.mkdir(parents=True, exist_ok=True)

    for area in study_areas.itertuples():
        area_name = area.Glacier_na
        geodiv_out_dir = Path(res["outputs_dir"]) / area_name / "Geodiversity"
        dem_clip_path = geodiv_out_dir / f"{area_name}_DEM_clip.tif"
        dem_fill_path = geodiv_out_dir / f"{area_name}_DEM_fill.tif"

        print(f"\nProcessing: {area_name} ({res['label']})")

        if not dem_clip_path.exists():
            print(f"  ⚠ Skipping TRI — {dem_clip_path} not found "
                  f"(run Variable_calculation.py / Low_res_script.py first).")
        else:
            tri_out = tri_swi_dir / res["tri_filename"].format(area=area_name)
            try:
                compute_tri(dem_clip_path, tri_out, cfg.saga.tri)
            except RuntimeError as e:
                print(f"  ⚠ TRI failed for {area_name} ({res['label']}):\n{e}")

        if not dem_fill_path.exists():
            print(f"  ⚠ Skipping SWI — {dem_fill_path} not found "
                  f"(run Variable_calculation.py / Low_res_script.py first).")
        else:
            swi_out = tri_swi_dir / res["swi_filename"].format(area=area_name)
            try:
                compute_swi(dem_fill_path, swi_out, cfg.saga.swi)
            except RuntimeError as e:
                print(f"  ⚠ SWI failed for {area_name} ({res['label']}):\n{e}")

print("\nDone.")

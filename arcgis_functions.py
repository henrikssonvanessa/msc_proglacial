# -*- coding: utf-8 -*-
"""
ArcPy helper functions shared across the terrain/vegetation pipeline.

Functions
---------
select_area_shp   -- Extract a single study area polygon to its own shapefile
clip_ortho        -- Clip the full orthophoto (all bands) to the study area

Requires: arcpy (ArcGIS Pro with Spatial Analyst extension)
"""
import arcpy


def select_area_shp(area_name, shp_path, outputs_dir):
    """
    Extract a single glacier's polygon from the master shapefile.

    Uses an attribute query on 'Glacier_na' to isolate one study area,
    saving it as a standalone shapefile that can be used as a mask in
    subsequent ArcPy spatial operations.

    Parameters
    ----------
    area_name   : str  Name of the glacier (must match 'Glacier_na' field)
    shp_path    : str  Path to the master proglacial outlines shapefile
    outputs_dir : str  Base outputs directory (e.g. config paths.outputs_1m)

    Returns
    -------
    str  Path to the single-area output shapefile
    """
    area_name_select_shp = f"{outputs_dir}/{area_name}/{area_name}_select.shp"
    arcpy.analysis.Select(in_features=shp_path, out_feature_class=area_name_select_shp, where_clause=f"Glacier_na = '{area_name}'")
    return area_name_select_shp


def clip_ortho(area_name, area_shp, ortho_path, outputs_dir):
    """
    Clip the full 4-band orthophoto to the study area polygon.

    The clipped ortho (all bands intact) is kept as a single multi-band file
    and is used as input to Veg_RF_polygon_cv.py's Random Forest classifier.

    Parameters
    ----------
    area_name   : str  Glacier name
    area_shp    : str  Path to the single-area mask shapefile (from select_area_shp)
    ortho_path  : str  Path to the source orthophoto raster (config: per-area
                       'ortho' column in paths.lookup_table_ortho)
    outputs_dir : str  Base outputs directory (config: paths.outputs_1m)

    Returns
    -------
    str  Path to the clipped orthophoto raster ({area_name}_ortho_clip.tif)
    """
    ortho_raster = arcpy.Raster(ortho_path)
    out_path = f"{outputs_dir}/{area_name}/{area_name}_ortho_clip.tif"
    clipped = arcpy.sa.ExtractByMask(ortho_raster, area_shp, "INSIDE")
    clipped.save(out_path)
    return out_path

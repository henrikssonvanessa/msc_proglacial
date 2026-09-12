# -*- coding: utf-8 -*-
"""
ArcPy helper functions shared across the terrain/vegetation pipeline.

Functions
---------
select_area_shp   -- Extract a single study area polygon to its own shapefile

Requires: arcpy (ArcGIS Pro)
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

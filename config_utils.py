# -*- coding: utf-8 -*-
"""
Shared configuration loader for the proglacial vegetation/geodiversity pipeline.

Every script in this repository reads its tunable settings (paths, study-area
list, model hyperparameters, thresholds) from a single YAML file (config.yaml
at the repo root, by default) instead of hardcoding them. This makes it
possible to re-run the pipeline on a different machine or dataset by editing
one file rather than every script.

Usage
-----
    from config_utils import load_config, resolve_path

    cfg = load_config()
    os.chdir(cfg.paths.base_dir)
    shp_path = cfg.paths.outlines_shp          # relative to base_dir (cwd)
    gdb_path = resolve_path(cfg, cfg.paths.training_gdb)  # absolute path

Requires: PyYAML
"""
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

CONFIG_ENV_VAR = "PROGLACIAL_CONFIG"
DEFAULT_CONFIG_NAME = "config.yaml"


def _to_namespace(obj):
    """Recursively convert nested dicts (and the dicts/lists they contain) into
    attribute-accessible SimpleNamespace objects."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(config_path=None):
    """
    Load pipeline configuration from a YAML file.

    Resolution order for the config file location:
      1. Explicit `config_path` argument
      2. `PROGLACIAL_CONFIG` environment variable
      3. `config.yaml` next to this module (repo root)

    Parameters
    ----------
    config_path : str or Path, optional

    Returns
    -------
    SimpleNamespace  Nested, attribute-accessible configuration, e.g.
                     cfg.paths.base_dir,
                     cfg.random_forest.vegetation_classifier.n_estimators
    """
    if config_path is None:
        config_path = os.environ.get(CONFIG_ENV_VAR)
    if config_path is None:
        config_path = Path(__file__).resolve().parent / DEFAULT_CONFIG_NAME

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Edit config.yaml (in particular "
            f"paths.base_dir) for your environment, or set the {CONFIG_ENV_VAR} "
            f"environment variable to point at a different config file."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _to_namespace(raw)


def resolve_path(cfg, relative_path):
    """Join a config-relative path with the configured base_dir into an absolute path."""
    return str(Path(cfg.paths.base_dir) / relative_path)

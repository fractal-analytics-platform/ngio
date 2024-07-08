"""Functions for loading tables metadata from Zarr groups."""

from pathlib import Path

import zarr

from ngio.tables.v1 import (
    FeatureTable,
    MaskingRoiTable,
    RoiTable,
    load_table_meta_v1,
)

_loaders = {
    "1": load_table_meta_v1,
}


def _get_version(zarr_group: zarr.Group) -> str:
    """Get the version of a Fractal table."""
    version = zarr_group.attrs.get("fractal_table_version", None)
    if version is None:
        raise ValueError("Table version not found.")

    if not isinstance(version, str):
        raise ValueError("Table version must be a string.")
    return version


def load_table_meta(
    zarr_url: str | Path, table_name: str
) -> RoiTable | MaskingRoiTable | FeatureTable:
    """Load a table from a Zarr group."""
    table_path = Path(zarr_url) / "tables" / table_name
    table_group = zarr.open_group(table_path)
    version = _get_version(table_group)
    return _loaders[version](table_path)

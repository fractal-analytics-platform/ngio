"""Utility functions for working with tables in the v1 format."""

from pathlib import Path

import zarr

from ngio.tables.v1.specs import FeatureTable, MaskingRoiTable, RoiTable


def load_table_meta_v1(
    zarr_url: str | Path,
) -> RoiTable | MaskingRoiTable | FeatureTable:
    """Load a table from a Zarr group."""
    table_group = zarr.open_group(zarr_url)
    group_attrs = dict(table_group.attrs.asdict())

    # rename 'encoding-type' and 'encoding-version'
    # to 'encoding_type' and 'encoding_version'
    enc_type = group_attrs.pop("encoding-type")
    group_attrs["encoding_type"] = enc_type
    enc_version = group_attrs.pop("encoding-version")
    group_attrs["encoding_version"] = enc_version

    if group_attrs["type"] == "roi_table":
        return RoiTable(**group_attrs)

    elif group_attrs["type"] == "masking_roi_table":
        return MaskingRoiTable(**group_attrs)

    elif group_attrs["type"] == "feature_table":
        return FeatureTable(**group_attrs)

    else:
        raise ValueError(
            f"Unsupported table type: {group_attrs['type']},\
                supported types are: ['roi', 'masking_roi', 'feature']"
        )

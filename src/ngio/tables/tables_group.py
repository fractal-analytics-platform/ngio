"""Module for handling the /tables group in an OME-NGFF file.

The /tables group contains t
"""

from typing import Literal

import zarr

from ngio.io import StoreLike
from ngio.pydantic_utils import BaseWithExtraFields
from ngio.tables.v1 import FeatureTableV1, MaskingROITableV1, ROITableV1

ROITable = ROITableV1
IMPLEMENTED_ROI_TABLES = {"1": ROITableV1}

FeatureTable = FeatureTableV1
IMPLEMENTED_FEATURE_TABLES = {"1": FeatureTableV1}

MaskingROITable = MaskingROITableV1
IMPLEMENTED_MASKING_ROI_TABLES = {"1": MaskingROITableV1}

Table = ROITable | FeatureTable | MaskingROITable


class CommonMeta(BaseWithExtraFields):
    """Common metadata for all tables."""

    type: Literal["roi_table", "feature_table", "masking_roi_table"]
    fractal_table_version: str


def _find_table_impl(
    table_type: Literal["roi_table", "feature_table", "masking_roi_table"],
    version: str,
) -> Table:
    """Find the type of table in the group."""
    if table_type == "roi_table":
        if version not in IMPLEMENTED_ROI_TABLES:
            raise ValueError(f"ROI Table version {version} not implemented.")
        return IMPLEMENTED_ROI_TABLES[version]

    elif table_type == "feature_table":
        if version not in IMPLEMENTED_FEATURE_TABLES:
            raise ValueError(f"Feature Table version {version} not implemented.")
        return IMPLEMENTED_FEATURE_TABLES[version]

    elif table_type == "masking_roi_table":
        if version not in IMPLEMENTED_MASKING_ROI_TABLES:
            raise ValueError(f"Masking ROI Table version {version} not implemented.")
        return IMPLEMENTED_MASKING_ROI_TABLES[version]

    else:
        raise ValueError(f"Table type {table_type} not recognized.")


def _get_table_impl(group: zarr.Group) -> Table:
    common_meta = CommonMeta(**group.attrs)
    return _find_table_impl(
        table_type=common_meta.type, version=common_meta.fractal_table_version
    )(group=group)


class TableGroup:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(self, group: StoreLike | zarr.Group) -> None:
        """Initialize the LabelGroupHandler."""
        if not isinstance(group, zarr.Group):
            group = zarr.open_group(group, mode="a")

        if "tables" not in group:
            self._group = group.create_group("tables")
        else:
            self._group: zarr.Group = group["tables"]

    def _validate_list_of_tables(self, list_of_tables: list[str]) -> None:
        """Validate the list of tables."""
        list_of_groups = list(self._group.group_keys())

        for table_name in list_of_tables:
            if table_name not in list_of_groups:
                raise ValueError(f"Table {table_name} not found in the group.")

    def _get_list_of_tables(self) -> list[str]:
        """Return the list of tables."""
        list_of_tables = self._group.attrs.get("tables", [])
        self._validate_list_of_tables(list_of_tables)
        return list_of_tables

    def list(
        self,
        type: Literal["roi_table", "feature_table", "masking_roi_table"] | None = None,
    ) -> list[str]:
        """List all labels in the group.

        Args:
            type (str): The type of table to list.
                If None, all tables are listed.
                Allowed values are: 'roi_table', 'feature_table', 'masking_roi_table'.
        """
        list_of_tables = self._get_list_of_tables()
        self._validate_list_of_tables(list_of_tables=list_of_tables)
        if type is None:
            return list_of_tables

        else:
            if type not in ["roi_table", "feature_table", "masking_roi_table"]:
                raise ValueError(
                    f"Table type {type} not recognized. "
                    " Allowed values are: 'roi', 'feature', 'masking_roi'."
                )
            list_of_typed_tables = []
            for table_name in list_of_tables:
                table = self._group[table_name]
                common_meta = CommonMeta(**table.attrs)
                if common_meta.type == type:
                    list_of_typed_tables.append(table_name)
            return list_of_typed_tables

    def get_table(self, name: str) -> Table:
        """Get a label from the group."""
        list_of_tables = self._get_list_of_tables()
        if name not in list_of_tables:
            raise ValueError(f"Table {name} not found in the group.")

        return _get_table_impl(group=self._group[name])

    def new(
        self,
        name: str,
        table_type: str = "roi_table",
        overwrite: bool = False,
        version: str = "1",
        **type_specific_kwargs,
    ) -> Table:
        """Add a new table to the group."""
        list_of_tables = self._get_list_of_tables()
        if not overwrite and name in list_of_tables:
            raise ValueError(f"Table {name} already exists in the group.")

        if overwrite and name in list_of_tables:
            list_of_tables.remove(name)

        table_impl = _find_table_impl(table_type=table_type, version=version)
        new_table = table_impl._new(
            parent_group=self._group,
            name=name,
            overwrite=overwrite,
            **type_specific_kwargs,
        )

        self._group.attrs["tables"] = [*list_of_tables, name]

        return new_table

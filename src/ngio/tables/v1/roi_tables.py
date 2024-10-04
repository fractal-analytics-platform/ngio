"""Implementation of the ROI Table class.

This class follows the roi_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

from collections.abc import Iterable
from typing import Literal

import pandas as pd
import zarr
from pydantic import BaseModel

from ngio.core.roi import WorldCooROI
from ngio.tables.v1.generic_table import REQUIRED_COLUMNS, BaseTable, write_table_ad


class ROITableV1Meta(BaseModel):
    """Metadata for the ROI table."""

    fractal_table_version: Literal["1"] = "1"
    type: Literal["roi_table"] = "roi_table"


ORIGIN_COLUMNS = [
    "x_micrometer_original",
    "y_micrometer_original",
]
TRANSLATION_COLUMNS = ["translation_x", "translation_y", "translation_z"]
OPTIONAL_COLUMNS = ORIGIN_COLUMNS + TRANSLATION_COLUMNS


def create_empty_roi_table(
    include_origin: bool = False, include_translation: bool = False
) -> pd.DataFrame:
    """Create an empty ROI table."""
    origin_columns = ORIGIN_COLUMNS if include_origin else []
    translation_columns = TRANSLATION_COLUMNS if include_translation else []

    columns = [
        *REQUIRED_COLUMNS,
        *origin_columns,
        *translation_columns,
    ]
    table = pd.DataFrame(
        index=pd.Index([], name="FieldIndex", dtype="str"), columns=columns
    )

    return table


class ROITableV1:
    """Class to handle fractal ROI tables.

    To know more about the ROI table format, please refer to the
    specification at:
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(self, group: zarr.Group):
        """Initialize the class from an existing group.

        Args:
            group (zarr.Group): The group containing the
                ROI table.
        """
        self._meta = ROITableV1Meta(**group.attrs)
        self._table_handler = BaseTable(
            group=group, index_key="FieldIndex", index_type="str"
        )

    @classmethod
    def _new(
        cls,
        parent_group: zarr.Group,
        name: str,
        include_origin: bool = False,
        include_translation: bool = False,
        overwrite: bool = False,
    ):
        """Create a new ROI table.

        Note this method is not meant to be called directly.
        But should be called from the TablesHandler class.

        Args:
            parent_group (zarr.Group): The parent group where the ROI table
                will be created.
            name (str): The name of the ROI table.
            table (DataFrame): The ROI table as a DataFrame.
            include_origin (bool): Whether to include the origin columns in the table.
            include_translation (bool): Whether to include the translation columns
                in the table.
            overwrite (bool): Whether to overwrite the table if it already exists.
        """
        group = parent_group.create_group(name, overwrite=overwrite)

        table = create_empty_roi_table(
            include_origin=include_origin, include_translation=include_translation
        )

        meta = ROITableV1Meta()
        write_table_ad(
            group=group,
            table=table,
            index_key="FieldIndex",
            index_type="str",
            meta=meta,
        )
        return cls(group=group)

    @property
    def meta(self) -> ROITableV1Meta:
        """Return the metadata of the ROI table."""
        return self._meta

    @property
    def table_handler(self) -> BaseTable:
        """Return the table handler."""
        return self._table_handler

    @property
    def table(self) -> pd.DataFrame:
        """Return the ROI table as a DataFrame."""
        return self._table_handler.table

    @table.setter
    def table(self, table: pd.DataFrame):
        self._table_handler.table = table

    @property
    def list_field_indexes(self) -> list[str]:
        """Return a list of all field indexes in the table."""
        return self.table.index.tolist()

    def append_rois(self, rois: Iterable[WorldCooROI] | WorldCooROI) -> None:
        """Append ROIs to the current table.

        Args:
            rois (Iterable[WorldCooROI] | WorldCooROI): The ROIs to append.
        """
        if isinstance(rois, WorldCooROI):
            rois = [rois]

        rois_dict = {}
        for roi in rois:
            field_index = roi.infos.get("field_index", None)
            if field_index is None:
                raise ValueError("Field index is required in the ROI infos.")

            rois_dict[field_index] = {
                "x_micrometer": roi.x,
                "y_micrometer": roi.y,
                "z_micrometer": roi.z,
                "len_x_micrometer": roi.x_length,
                "len_y_micrometer": roi.y_length,
                "len_z_micrometer": roi.z_length,
            }

        table_df = self.table
        new_table_df = pd.DataFrame.from_dict(rois_dict, orient="index")

        if not table_df.empty:
            table_df = pd.concat([table_df, new_table_df], axis=0)
        else:
            table_df = new_table_df

        table_df.index.name = "FieldIndex"
        table_df.index = table_df.index.astype(str)
        self.table = table_df

    def get_roi(self, field_index) -> WorldCooROI:
        """Get an ROI from the table."""
        if field_index not in self.list_field_indexes:
            raise ValueError(f"Field index {field_index} is not in the table")

        table_df = self.table.loc[field_index]
        roi = WorldCooROI(
            x=table_df.loc["x_micrometer"],
            y=table_df.loc["y_micrometer"],
            z=table_df.loc["z_micrometer"],
            x_length=table_df.loc["len_x_micrometer"],
            y_length=table_df.loc["len_y_micrometer"],
            z_length=table_df.loc["len_z_micrometer"],
            unit="micrometer",
            infos={"field_index": field_index},
        )
        return roi

    @property
    def list_rois(self) -> list[WorldCooROI]:
        """List all ROIs in the table."""
        return [self.get_roi(field_index) for field_index in self.list_field_indexes]

    def write(self) -> None:
        """Write the crrent state of the table to the Zarr file."""
        self._table_handler.write(self.meta)

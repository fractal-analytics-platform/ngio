"""Implementation of the ROI Table class.

This class follows the roi_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

from pathlib import Path
from typing import Literal

import anndata as ad
import zarr
from pandas import DataFrame
from pydantic import BaseModel

from ngio.core.roi import WorldCooROI
from ngio.tables._utils import validate_roi_table


class ROITableFormattingError(Exception):
    """Error raised when an ROI table is not formatted correctly."""

    pass


class ROITableV1Meta(BaseModel):
    """Metadata for the ROI table."""

    fractal_table_version: Literal["1"] = "1"
    type: Literal["roi_table"] = "roi_table"


REQUIRED_COLUMNS = [
    "x_micrometer",
    "y_micrometer",
    "z_micrometer",
    "len_x_micrometer",
    "len_y_micrometer",
    "len_z_micrometer",
]
ORIGIN_COLUMNS = [
    "x_micrometer_original",
    "y_micrometer_original",
]
TRANSLATION_COLUMNS = ["translation_x", "translation_y", "translation_z"]
OPTIONAL_COLUMNS = ORIGIN_COLUMNS + TRANSLATION_COLUMNS


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
        self.table_group = group
        self._meta = ROITableV1Meta(**group.attrs)
        ad_table = ad.read_zarr(self.table_group)

        table = ad_table.to_df()
        table = validate_roi_table(
            table,
            required_columns=REQUIRED_COLUMNS,
            optional_columns=OPTIONAL_COLUMNS,
        )
        self._table = table

    @classmethod
    def _create_new(
        cls,
        parent_group: zarr.Group,
        name: str,
        table: DataFrame | None = None,
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
        table_group = parent_group.create_group(name, overwrite=overwrite)

        # setup empty dataframe with FieldIndex as index
        # and self._required_columns as columns
        origin_columns = ORIGIN_COLUMNS if include_origin else []
        translation_columns = TRANSLATION_COLUMNS if include_translation else []

        columns = [
            "FieldIndex",
            *REQUIRED_COLUMNS,
            *origin_columns,
            *translation_columns,
        ]

        if table is None:
            table = DataFrame(columns=columns)
        else:
            cls._validate_roi_table(table=table)
        # this should be possibe to do duing initialization
        # but for now this works
        table = table.set_index("FieldIndex")
        cls._write(group=table_group, table=table)
        return cls(table_group)

    @property
    def data_frame(self) -> DataFrame:
        """Return the ROI table as a DataFrame."""
        return self._table

    @data_frame.setter
    def data_frame(self, table: DataFrame):
        table = validate_roi_table(
            data_frame=table,
            required_columns=REQUIRED_COLUMNS,
            optional_columns=OPTIONAL_COLUMNS,
        )
        self._table = table

    @property
    def meta(self) -> ROITableV1Meta:
        """Return the metadata of the ROI table."""
        return self._meta

    @property
    def list_field_indexes(self) -> list[str]:
        """Return a list of all field indexes in the table."""
        return self.data_frame.index.tolist()

    def add_roi(
        self, field_index: str, roi: WorldCooROI, overwrite: bool = False
    ) -> None:
        """Add a new ROI to the table."""
        if field_index in self.list_field_indexes and not overwrite:
            raise ValueError(
                f"Field index {field_index} already exists in ROI table. "
                "Set overwrite=True to overwrite"
            )

        self.data_frame.loc[field_index] = {
            "x_micrometer": roi.x,
            "y_micrometer": roi.y,
            "z_micrometer": roi.z,
            "len_x_micrometer": roi.x_length,
            "len_y_micrometer": roi.y_length,
            "len_z_micrometer": roi.z_length,
        }

    def get_roi(self, field_index) -> WorldCooROI:
        """Get an ROI from the table."""
        if field_index not in self.list_field_indexes:
            raise ValueError(f"Field index {field_index} is not in the table")

        table_df = self.data_frame
        roi = WorldCooROI(
            field_index=field_index,
            x=table_df.loc[field_index, "x_micrometer"],
            y=table_df.loc[field_index, "y_micrometer"],
            z=table_df.loc[field_index, "z_micrometer"],
            x_length=table_df.loc[field_index, "len_x_micrometer"],
            y_length=table_df.loc[field_index, "len_y_micrometer"],
            z_length=table_df.loc[field_index, "len_z_micrometer"],
            unit="micrometer",
        )
        return roi

    @property
    def list_rois(self) -> list[WorldCooROI]:
        """List all ROIs in the table."""
        return [self.get_roi(field_index) for field_index in self.list_field_indexes]

    def write(self) -> None:
        """Write the crrent state of the table to the Zarr file."""
        data_frame = validate_roi_table(
            data_frame=self.data_frame,
            required_columns=REQUIRED_COLUMNS,
            optional_columns=OPTIONAL_COLUMNS,
        )
        self._write(group=self.table_group, table=data_frame)

    @staticmethod
    def _write(group: zarr.Group, table: DataFrame) -> None:
        ad_table = ad.AnnData(table)
        # anndata can only write from a store and not a group
        path = Path(group.store.path) / group.path
        ad_table.write_zarr(path)

        # Always make sure to write the metadata
        meta = ROITableV1Meta()
        group.attrs.update(meta.model_dump(exclude=None))

"""Implementation of the Masking ROI Table class.

This class follows the roi_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

from typing import Literal

import anndata as ad
import zarr
from pandas import DataFrame
from pydantic import BaseModel

from ngio.core.roi import WorldCooROI
from ngio.tables._utils import table_ad_to_df, table_df_to_ad


class ROITableFormattingError(Exception):
    """Error raised when an ROI table is not formatted correctly."""

    pass


class MaskingROITableV1Meta(BaseModel):
    """Metadata for the ROI table."""

    fractal_table_version: Literal["1"] = "1"
    type: Literal["roi_table"] = "masking_roi_table"
    region: dict[Literal["path"], str]
    instance_key: str = "label"


REQUIRED_COLUMNS = [
    "label",
    "x_micrometer",
    "y_micrometer",
    "z_micrometer",
    "len_x_micrometer",
    "len_y_micrometer",
    "len_z_micrometer",
]


class MaskingROITableV1:
    """Class to handle fractal ROI tables.

    To know more about the ROI table format, please refer to the
    specification at:
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    _index_type = "int"

    def __init__(self, group: zarr.Group):
        """Initialize the class from an existing group.

        Args:
            group (zarr.Group): The group containing the
                ROI table.
        """
        self.table_group = group
        self._meta = MaskingROITableV1Meta(**group.attrs)
        self._instance_key = self._meta.instance_key
        ad_table = ad.read_zarr(self.table_group)

        self._table = table_ad_to_df(
            ad_table, index_key=self._instance_key, index_type=self._index_type
        )

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
        raise NotImplementedError

    @property
    def table(self) -> DataFrame:
        """Return the ROI table as a DataFrame."""
        return self._table

    @table.setter
    def table(self, table: DataFrame):
        self._table = table

    def as_anndata(self):
        """Return the ROI table as an anndata object."""
        return table_df_to_ad(
            self.table, index_key=self._instance_key, index_type=self._index_type
        )

    def from_anndata(self, ad_table: ad.AnnData):
        """Set the table from an anndata object."""
        self.table = table_ad_to_df(
            ad_table, index_key=self._instance_key, index_type=self._index_type
        )

    @property
    def meta(self) -> MaskingROITableV1Meta:
        """Return the metadata of the ROI table."""
        return self._meta

    @property
    def instance_key(self) -> str:
        """Return the instance key of the ROI table."""
        return self._meta.instance_key

    @property
    def list_labels(self) -> list[int]:
        """Return a list of all labels in the table."""
        return self.table[self.instance_key].tolist()

    def add_roi(
        self, field_index: str, roi: WorldCooROI, overwrite: bool = False
    ) -> None:
        """Add a new ROI to the table."""
        raise NotImplementedError

    def get_roi(self, field_index) -> WorldCooROI:
        """Get an ROI from the table."""
        raise NotImplementedError

    @property
    def list_rois(self) -> list[WorldCooROI]:
        """List all ROIs in the table."""
        raise NotImplementedError

    def write(self) -> None:
        """Write the crrent state of the table to the Zarr file."""
        raise NotImplementedError

    @staticmethod
    def _write(group: zarr.Group, table: DataFrame) -> None:
        raise NotImplementedError

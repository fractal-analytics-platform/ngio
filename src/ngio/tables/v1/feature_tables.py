"""Implementation of the FeatureTableV1 class.

This class follows the feature_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

from pathlib import Path
from typing import Literal

import anndata as ad
import zarr
from pandas import DataFrame
from pydantic import BaseModel

from ngio.tables._utils import df_from_andata, df_to_andata


class FeatureTableFormattingError(Exception):
    """Error raised when an ROI table is not formatted correctly."""

    pass


class FeatureTableV1Meta(BaseModel):
    """Metadata for the ROI table."""

    region: dict[Literal["path"], str]
    instance_key: str = "label"
    fractal_table_version: Literal["1"] = "1"
    type: Literal["feature_table"] = "feature_table"


class FeatureTableV1:
    """Class to handle fractal Feature tables.

    To know more about the Feature table format, please refer to the
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(self, group: zarr.Group, implicit_conversion: bool = False):
        """Initialize the class from an existing group.

        Args:
            group (zarr.Group): The group containing the
                ROI table.
            implicit_conversion (bool): If True, the anndata object will be
                converted to a DataFrame without any checks. If False, the
                anndata object will be checked for compatibility and converted
                to a DataFrame. Default is False.
        """
        self.table_group = group
        self._meta = FeatureTableV1Meta(**group.attrs)
        self._instance_key = self._meta.instance_key

        ad_table = ad.read_zarr(self.table_group)
        table = df_from_andata(
            ad_table,
            index_key=self._instance_key,
        )
        self._table = table

    @classmethod
    def _create_new(
        cls,
        parent_group: zarr.Group,
        name: str,
        table: DataFrame | None,
        region: str,
        instance_key: str = "label",
        overwrite: bool = False,
    ) -> None:
        """Create a new Feature table.

        Args:
            parent_group (zarr.Group): The parent group where the ROI table
                will be created.
            name (str): The name of the ROI table.
            table (DataFrame): The ROI table as a DataFrame.
            region (str): The path to the region of interest.
            instance_key (str): The column name to use as the index of the DataFrame.
                Default is 'label'.
            overwrite (bool): If True, the table will be overwritten if it exists.
                Default is False.
        """
        if table is None:
            raise ValueError("Can not create an empty feature table.")

        group = parent_group.create_group(name, overwrite=overwrite)
        meta = FeatureTableV1Meta(region={"path": region}, instance_key=instance_key)
        # Always make sure to write the metadata (in case the write fails)
        group.attrs.update(meta.model_dump(exclude=None))

        cls._write(group=group, table=table, meta=meta)
        return cls(group=group)

    @property
    def data_frame(self) -> DataFrame:
        """Return the ROI table as a DataFrame."""
        return self._table

    @data_frame.setter
    def data_frame(self, table: DataFrame):
        self._table = table

    @property
    def meta(self) -> FeatureTableV1Meta:
        """Return the metadata of the feature table."""
        return self._meta

    @property
    def instance_key(self) -> str:
        """Return the instance key of the feature table."""
        return self._instance_key

    @property
    def list_labels(self) -> list[str]:
        """Return a list of all field indexes in the table."""
        return self.data_frame.index.tolist()

    def label_image_name(self, get_full_path: bool = False) -> str:
        """Return the name of the label image.

        The name is assumed to be after the last '/' in the path.
        Since this might fails, get_full_path is True, the full path is returned.

        Args:
            get_full_path (bool): If True, the full path is returned.
        """
        path = self.meta.region["path"]
        if get_full_path:
            return path

        return path.split("/")[-1]

    def write(self) -> None:
        """Write the crrent state of the table to the Zarr file."""
        self._write(group=self.table_group, table=self.data_frame, meta=self.meta)

    @staticmethod
    def _write(group: zarr.Group, table: DataFrame, meta: FeatureTableV1Meta) -> None:
        ad_table = df_to_andata(
            table, index_key=meta.instance_key, implicit_conversion=False
        )

        path = Path(group.store.path) / group.path
        ad_table.write_zarr(path)

        # Always make sure to write the metadata
        group.attrs.update(meta.model_dump(exclude=None))

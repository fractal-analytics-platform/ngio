"""Implementation of the FeatureTableV1 class.

This class follows the feature_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

from typing import Literal

import pandas as pd
import zarr
from pydantic import BaseModel

from ngio.tables.v1._generic_table import BaseTable, write_table_ad


class FeatureTableV1Meta(BaseModel):
    """Metadata for the ROI table."""

    region: dict[Literal["path"], str] | None = None
    instance_key: str = "label"
    fractal_table_version: Literal["1"] = "1"
    type: Literal["feature_table"] = "feature_table"


class FeatureTableV1:
    """Class to handle fractal Feature tables.

    To know more about the Feature table format, please refer to the
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(
        self,
        group: zarr.Group,
        validate_metadata: bool = True,
        validate_table: bool = True,
        index_key: str | None = None,
    ):
        """Initialize the class from an existing group.

        Args:
            group (zarr.Group): The group containing the
                ROI table.
            validate_metadata (bool): If True, the metadata is validated.
            validate_table (bool): If True, the table is validated.
            index_key (str): The column name to use as the index of the DataFrame.
        """
        if validate_metadata:
            self._meta = FeatureTableV1Meta(**group.attrs)
        else:
            self._meta = FeatureTableV1Meta.model_construct(**group.attrs)

        # Validate the table is not implemented for the Feature table
        validators = None
        validators = validators if validate_table else None

        index_key = index_key if index_key is not None else self._meta.instance_key
        self._table_handler = BaseTable(
            group=group,
            index_key=index_key,
            index_type="int",
            validators=validators,
        )

    @classmethod
    def _new(
        cls,
        parent_group: zarr.Group,
        name: str,
        label_image: str | None = None,
        instance_key: str = "label",
        overwrite: bool = False,
    ) -> None:
        """Create a new Feature table.

        Args:
            parent_group (zarr.Group): The parent group where the ROI table
                will be created.
            name (str): The name of the ROI table.
            label_image (str | None): The relative path to the label image used to
                create the feature table.
            instance_key (str): The column name to use as the index of the DataFrame.
                Default is 'label'.
            overwrite (bool): If True, the table will be overwritten if it exists.
                Default is False.
        """
        group = parent_group.create_group(name, overwrite=overwrite)
        table = pd.DataFrame(index=pd.Index([], name=instance_key, dtype="int"))

        label_image = label_image if label_image is not None else "None"
        meta = FeatureTableV1Meta(
            region={"path": label_image}, instance_key=instance_key
        )
        write_table_ad(
            group=group,
            table=table,
            index_key=instance_key,
            index_type="int",
            meta=meta,
        )
        return cls(group=group)

    @property
    def meta(self) -> FeatureTableV1Meta:
        """Return the metadata of the feature table."""
        return self._meta

    @property
    def table_handler(self) -> BaseTable:
        """Return the table handler."""
        return self._table_handler

    @property
    def table(self) -> pd.DataFrame:
        """Return the feature table as a DataFrame."""
        return self._table_handler.table

    @table.setter
    def table(self, table: pd.DataFrame):
        """Set the feature table."""
        raise NotImplementedError(
            "Setting the table is not implemented. Please use the 'set_table' method."
        )

    def set_table(self, table: pd.DataFrame):
        """Set the feature table."""
        self._table_handler.set_table(table)

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

    def write(self):
        """Write the table to the group."""
        self._table_handler.write(meta=self.meta)

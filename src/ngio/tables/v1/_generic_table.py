"""Implementation of a Generic Table class."""

from pathlib import Path
from typing import Literal

import anndata as ad
import pandas as pd
import zarr
from pydantic import BaseModel

from ngio.tables._utils import Validator, table_ad_to_df, table_df_to_ad, validate_table

REQUIRED_COLUMNS = [
    "x_micrometer",
    "y_micrometer",
    "z_micrometer",
    "len_x_micrometer",
    "len_y_micrometer",
    "len_z_micrometer",
]


def write_table_ad(
    group: zarr.Group,
    table: pd.DataFrame,
    index_key: str,
    index_type: Literal["int", "str"],
    meta: BaseModel,
    validators: list[Validator] | None = None,
) -> None:
    """Write a table to a Zarr group.

    Args:
        group (zarr.Group): The group to write the table to.
        table (pd.DataFrame): The table to write.
        index_key (str): The column name to use as the index of the DataFrame.
        index_type (str): The type of the index column in the DataFrame.
        meta (BaseModel): The metadata of the table.
        validators (list[Validator]): A list of functions to further validate the
            table.
    """
    ad_table = table_df_to_ad(
        table,
        index_key=index_key,
        index_type=index_type,
        validators=validators,
    )

    group_path = Path(group.store.path) / group.path
    ad_table.write_zarr(group_path)
    group.attrs.update(meta.model_dump(exclude=None))


class BaseTable:
    """A base table class to be used for table operations in all other tables."""

    def __init__(
        self,
        group: zarr.Group,
        index_key: str,
        index_type: Literal["int", "str"],
        validators: list[Validator] | None = None,
    ):
        """Initialize the class from an existing group.

        Args:
            group (zarr.Group): The group containing the
                ROI table.
            index_key (str): The column name to use as the index of the DataFrame.
            index_type (str): The type of the index column in the DataFrame.
            validators (list[Validator]): A list of functions to further validate the
                table.
        """
        self._table_group = group
        self._index_key = index_key
        self._index_type = index_type
        self._validators = validators

        table_ad = ad.read_zarr(self._table_group)

        self._table = table_ad_to_df(
            table_ad=table_ad,
            index_key=self._index_key,
            index_type=self._index_type,
            validators=self._validators,
        )

    @property
    def table(self) -> pd.DataFrame:
        """Return the ROI table as a DataFrame."""
        return self._table

    @table.setter
    def table(self, table: pd.DataFrame) -> None:
        raise NotImplementedError(
            "Setting the table directly is not supported. "
            "Please use the 'set_table' method."
        )

    def set_table(self, table: pd.DataFrame) -> None:
        table = validate_table(
            table_df=table,
            index_key=self.index_key,
            index_type=self.index_type,
            validators=self._validators,
        )
        self._table = table

    def as_anndata(self) -> ad.AnnData:
        """Return the ROI table as an AnnData object."""
        return table_df_to_ad(
            self.table, index_key=self.index_key, index_type=self.index_type
        )

    def from_anndata(self, table_ad: ad.AnnData) -> None:
        """Return the ROI table as an AnnData object."""
        table = table_ad_to_df(
            table_ad=table_ad,
            index_key=self.index_key,
            index_type=self.index_type,
            validators=self._validators,
        )
        # Don't use the setter to avoid re-validating the table
        self._table = table

    @property
    def index(self) -> list[int | str]:
        """Return a list of all the labels in the table."""
        return self.table.index.tolist()

    @property
    def group(self) -> zarr.Group:
        """Return the group of the table."""
        return self._table_group

    @property
    def group_path(self) -> Path:
        """Return the path of the group."""
        return Path(self._table_group.store.path) / self._table_group.path

    @property
    def index_key(self) -> str:
        """Return the index key of the table."""
        return self._index_key

    @property
    def index_type(self) -> Literal["int", "str"]:
        """Return the index type of the table."""
        return self._index_type

    @property
    def validators(self) -> list[Validator] | None:
        """Return the validators of the table."""
        return self._validators

    @validators.setter
    def validators(self, validators: list[Validator] | None) -> None:
        """Set the validators of the table."""
        self._validators = validators

    def add_validator(self, validator: Validator) -> None:
        """Add a validator to the table."""
        if self._validators is None:
            self._validators = []
        self._validators.append(validator)

    def write(self, meta: BaseModel) -> None:
        """Write the current state of the table to the Zarr file."""
        table = self.table
        table = validate_table(
            table_df=table,
            index_key=self.index_key,
            index_type=self.index_type,
            validators=self._validators,
        )
        write_table_ad(
            group=self._table_group,
            table=self.table,
            index_key=self.index_key,
            index_type=self.index_type,
            meta=meta,
            validators=self._validators,
        )

from collections.abc import Collection

import pandas as pd
from pandas import DataFrame

from ngio.tables.backends._abstract_backend import AbstractTableBackend
from ngio.utils import NgioFileNotFoundError


class JsonTableBackend(AbstractTableBackend):
    """A class to load and write small tables in the zarr group .attrs (json)."""

    @staticmethod
    def backend_name() -> str:
        """The name of the backend."""
        return "experimental_json_v1"

    @staticmethod
    def implements_anndata() -> bool:
        """Whether the handler implements the anndata protocol."""
        return False

    @staticmethod
    def implements_dataframe() -> bool:
        """Whether the handler implements the dataframe protocol."""
        return True

    def load_columns(self) -> list[str]:
        """List all labels in the group."""
        return list(self.load_as_dataframe().columns)

    def _get_table_group(self):
        try:
            table_group = self._group_handler.get_group(path="table")
        except NgioFileNotFoundError:
            table_group = self._group_handler.group.create_group("table")
        return table_group

    def load_as_dataframe(self, columns: Collection[str] | None = None) -> DataFrame:
        """List all labels in the group."""
        table_group = self._get_table_group()
        table_dict = dict(table_group.attrs)
        data_frame = pd.DataFrame.from_dict(table_dict)
        if columns is not None:
            data_frame = data_frame[columns]
        return data_frame

    def write_from_dataframe(
        self, table: DataFrame, metadata: dict | None = None
    ) -> None:
        """Consolidate the metadata in the store."""
        table_group = self._get_table_group()
        table_group.attrs.clear()
        table_group.attrs.update(table.to_dict())
        if metadata is not None:
            self._group_handler.write_attrs(metadata)

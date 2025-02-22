from collections.abc import Collection

import pandas as pd
from pandas import DataFrame

from ngio.tables.backends._abstract_backend import AbstractTableBackend


class JsonTableBackend(AbstractTableBackend):
    """A class to load and write small tables in the zarr group .attrs (json)."""

    @property
    def backend_name(self) -> str:
        """The name of the backend."""
        return "json"

    @property
    def implements_anndata(self) -> bool:
        """Whether the handler implements the anndata protocol."""
        return False

    @property
    def implements_dataframe(self) -> bool:
        """Whether the handler implements the dataframe protocol."""
        return True

    def load_columns(self) -> list[str]:
        """List all labels in the group."""
        return list(self.load_as_dataframe().columns)

    def load_as_dataframe(self, columns: Collection[str] | None = None) -> DataFrame:
        """List all labels in the group."""
        attrs = self._group_handler.load_attrs()
        table = attrs.get("table")
        if table is None:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(table)

    def write_from_dataframe(
        self, table: DataFrame, metadata: dict | None = None
    ) -> None:
        """Consolidate the metadata in the store."""
        attrs = self._group_handler.load_attrs()
        attrs["table"] = table.to_dict()
        if metadata is not None:
            attrs.update(metadata)
        self._group_handler.write_attrs(attrs)

from collections.abc import Collection
from pathlib import Path

from anndata import AnnData
from pandas import DataFrame

from ngio.tables.backends._abstract_backend import AbstractTableBackend
from ngio.tables.backends._anndata_utils import (
    anndata_to_dataframe,
    custom_read_zarr,
    dataframe_to_anndata,
)
from ngio.utils import NgioValueError


class AnnDataBackend(AbstractTableBackend):
    """A class to load and write tables from/to an AnnData object."""

    @staticmethod
    def backend_name() -> str:
        """The name of the backend."""
        return "anndata_v1"

    @staticmethod
    def implements_anndata() -> bool:
        """Whether the handler implements the anndata protocol."""
        return True

    @staticmethod
    def implements_dataframe() -> bool:
        """Whether the handler implements the dataframe protocol."""
        return True

    def load_columns(self) -> list[str]:
        """List all labels in the group."""
        return list(self.load_as_dataframe().columns)

    def load_as_anndata(self, columns: Collection[str] | None = None) -> AnnData:
        """Load the metadata in the store."""
        anndata = custom_read_zarr(self._group_handler._group)
        if columns is not None:
            raise NotImplementedError(
                "Selecting columns is not implemented for AnnData."
            )
        return anndata

    def load_as_dataframe(self, columns: Collection[str] | None = None) -> DataFrame:
        """List all labels in the group."""
        dataframe = anndata_to_dataframe(
            self.load_as_anndata(),
            index_key=self._index_key,
            index_type=self._index_type,
        )
        if columns is not None:
            dataframe = dataframe[columns]
        return dataframe

    def write_from_dataframe(
        self, table: DataFrame, metadata: dict | None = None
    ) -> None:
        """Consolidate the metadata in the store."""
        anndata = dataframe_to_anndata(table, index_key=self._index_key)
        self.write_from_anndata(anndata, metadata)

    def write_from_anndata(self, table: AnnData, metadata: dict | None = None) -> None:
        """Consolidate the metadata in the store."""
        store = self._group_handler.store
        if not isinstance(store, str | Path):
            raise NgioValueError(
                "To write an AnnData object the store must be a local path/str."
            )

        store = Path(store) / self._group_handler.group.path
        table.write_zarr(store)
        if metadata is not None:
            self._group_handler.write_attrs(metadata)

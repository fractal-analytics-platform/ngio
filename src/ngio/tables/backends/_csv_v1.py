import io
from collections.abc import Collection

import pandas as pd
from pandas import DataFrame
from zarr.storage import DirectoryStore, FSStore

from ngio.tables.backends._abstract_backend import AbstractTableBackend
from ngio.utils import NgioFileNotFoundError, NgioValueError


class CsvTableBackend(AbstractTableBackend):
    """A class to load and write small tables in the zarr group .attrs (json)."""

    csv_name = "table.csv"

    @staticmethod
    def backend_name() -> str:
        """The name of the backend."""
        return "experimental_csv_v1"

    @staticmethod
    def implements_anndata() -> bool:
        """Whether the handler implements the anndata protocol."""
        return False

    @staticmethod
    def implements_pandas() -> bool:
        """Whether the handler implements the dataframe protocol."""
        return True

    @staticmethod
    def implements_polars() -> bool:
        """Whether the handler implements the polars protocol."""
        return False

    def load_columns(self) -> list[str]:
        """List all labels in the group."""
        return list(self.load_as_dataframe().columns)

    def _load_from_directory_store(self) -> DataFrame:
        """Load the metadata in the store."""
        url = self._group_handler.full_url
        if url is None:
            raise NgioValueError(
                f"Ngio does not support reading a CSV file from a "
                f"store of type {type(self._group_handler)}. "
                "Please make sure to use a compatible "
                "store like a zarr.DirectoryStore."
            )
        csv_path = f"{url}/{self.csv_name}"
        dataframe = pd.read_csv(csv_path)
        return dataframe

    def _load_from_fs_store(self) -> DataFrame:
        """Load the metadata in the store."""
        bytes_table = self._group_handler.store.get(self.csv_name)
        if bytes_table is None:
            raise NgioFileNotFoundError(f"No table found at {self.csv_name}. ")
        dataframe = pd.read_csv(io.BytesIO(bytes_table))
        return dataframe

    def load_as_dataframe(self, columns: Collection[str] | None = None) -> DataFrame:
        """List all labels in the group."""
        store = self._group_handler.store
        if isinstance(store, DirectoryStore):
            dataframe = self._load_from_directory_store()
        elif isinstance(store, FSStore):
            dataframe = self._load_from_fs_store()
        else:
            raise NgioFileNotFoundError(
                f"Ngio does not support reading a CSV file from a "
                f"store of type {type(store)}. "
                "Please make sure to use a compatible "
                "store like a zarr.DirectoryStore or "
                "zarr.FSStore."
            )

        if columns is not None:
            dataframe = dataframe[columns]
        return dataframe

    def write_from_dataframe(
        self, table: DataFrame, metadata: dict | None = None
    ) -> None:
        """Consolidate the metadata in the store."""
        store = self._group_handler.store
        if isinstance(store, DirectoryStore):
            csv_path = f"{self._group_handler.full_url}/{self.csv_name}"
            table.to_csv(csv_path, index=False)

        else:
            raise NgioFileNotFoundError(
                f"Ngio does not support writing a CSV file from a "
                f"store of type {type(store)}. "
                "Please make sure to use a compatible "
                "store like a zarr.DirectoryStore"
            )
        if metadata is not None:
            self._group_handler.write_attrs(metadata)

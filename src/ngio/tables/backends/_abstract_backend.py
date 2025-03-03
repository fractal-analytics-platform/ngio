from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Literal

from anndata import AnnData
from pandas import DataFrame

from ngio.utils import ZarrGroupHandler


class AbstractTableBackend(ABC):
    """Abstract class for table backends."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        index_key: str | None = None,
        index_type: Literal["int", "str"] = "int",
    ):
        """Initialize the handler.

        Args:
            group_handler (ZarrGroupHandler): An object to handle the Zarr group
                containing the table data.
            index_key (str): The column name to use as the index of the DataFrame.
            index_type (str): The type of the index column in the DataFrame.
        """
        self._group_handler = group_handler
        self._index_key = index_key
        self._index_type = index_type

    @staticmethod
    @abstractmethod
    def backend_name() -> str:
        """The name of the backend."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def implements_anndata() -> bool:
        """Whether the handler implements the anndata protocol."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def implements_dataframe() -> bool:
        """Whether the handler implements the dataframe protocol."""
        raise NotImplementedError

    @abstractmethod
    def load_columns(self) -> list[str]:
        """List all labels in the group."""
        raise NotImplementedError

    def load_as_anndata(self, columns: Collection[str] | None = None) -> AnnData:
        """Load the metadata in the store."""
        raise NotImplementedError

    def load_as_dataframe(self, columns: Collection[str] | None = None) -> DataFrame:
        """List all labels in the group."""
        raise NotImplementedError

    def write_from_dataframe(
        self, table: DataFrame, metadata: dict | None = None
    ) -> None:
        """Consolidate the metadata in the store."""
        raise NotImplementedError

    def write_from_anndata(self, table: AnnData, metadata: dict | None = None) -> None:
        """Consolidate the metadata in the store."""
        raise NotImplementedError

"""Protocol for table backends handlers."""

from collections.abc import Collection
from typing import Literal, Protocol

from anndata import AnnData
from pandas import DataFrame

from ngio.tables.backends._anndata_v1 import AnnDataBackend
from ngio.tables.backends._json_v1 import JsonTableBackend
from ngio.utils import NgioValueError, ZarrGroupHandler


class TableBackendProtocol(Protocol):
    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        index_key: str | None = None,
        index_type: Literal["int", "str"] = "int",
    ): ...

    @staticmethod
    def backend_name() -> str: ...

    @staticmethod
    def implements_anndata() -> bool: ...

    @staticmethod
    def implements_dataframe() -> bool: ...

    def load_columns(self) -> list[str]: ...

    def load_as_anndata(self, columns: Collection[str] | None = None) -> AnnData: ...

    def load_as_dataframe(
        self, columns: Collection[str] | None = None
    ) -> DataFrame: ...

    def write_from_dataframe(
        self, table: DataFrame, metadata: dict | None = None
    ) -> None: ...

    def write_from_anndata(
        self, table: AnnData, metadata: dict | None = None
    ) -> None: ...


class ImplementedTableBackends:
    """A class to manage the available table backends."""

    _instance = None
    _implemented_backends: dict[str, type[TableBackendProtocol]]

    def __new__(cls):
        """Create a new instance of the class if it does not exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._implemented_backends = {}
        return cls._instance

    @property
    def available_backends(self) -> list[str]:
        """Return the available table backends."""
        return list(self._implemented_backends.keys())

    def get_backend(
        self,
        backend_name: str | None,
        group_handler: ZarrGroupHandler,
        index_key: str | None = None,
        index_type: Literal["int", "str"] = "int",
    ) -> TableBackendProtocol:
        """Try to get a handler for the given store based on the metadata version."""
        if backend_name is None:
            # Default to anndata since it is currently
            # the only backend in use.
            backend_name = "anndata_v1"

        if backend_name not in self._implemented_backends:
            raise NgioValueError(f"Table backend {backend_name} not implemented.")
        handler = self._implemented_backends[backend_name](
            group_handler=group_handler, index_key=index_key, index_type=index_type
        )
        return handler

    def add_backend(
        self,
        table_beckend: type[TableBackendProtocol],
        overwrite: bool = False,
    ):
        """Register a new handler."""
        backend_name = table_beckend.backend_name()
        if backend_name in self._implemented_backends and not overwrite:
            raise NgioValueError(
                f"Table backend {backend_name} already implemented. "
                "Use the `overwrite=True` parameter to overwrite it."
            )
        self._implemented_backends[backend_name] = table_beckend


ImplementedTableBackends().add_backend(AnnDataBackend)
ImplementedTableBackends().add_backend(JsonTableBackend)

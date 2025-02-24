"""Protocol for table backends handlers."""

from collections.abc import Collection
from typing import Literal, Protocol

from anndata import AnnData
from pandas import DataFrame

from ngio.tables.backends._anndata import AnnDataBackend
from ngio.tables.backends._json import JsonTableBackend
from ngio.utils import NgioValueError, ZarrGroupHandler


class TableBackendProtocol(Protocol):
    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        index_key: str | None = None,
        index_type: Literal["int", "str"] = "str",
    ): ...

    @property
    def backend_name(self) -> str: ...

    @property
    def implements_anndata(self) -> bool: ...

    @property
    def implements_dataframe(self) -> bool: ...

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


class TableBackendsManager:
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
        index_type: Literal["int", "str"] = "str",
    ) -> TableBackendProtocol:
        """Try to get a handler for the given store based on the metadata version."""
        if backend_name is None:
            # Default to anndata since it is currently
            # the only backend in use.
            backend_name = "anndata"

        if backend_name not in self._implemented_backends:
            raise NgioValueError(f"Table backend {backend_name} not implemented.")
        handler = self._implemented_backends[backend_name](
            group_handler=group_handler, index_key=index_key, index_type=index_type
        )
        return handler

    def add_handler(
        self,
        backend_name: str,
        table_beckend: type[TableBackendProtocol],
        overwrite: bool = False,
    ):
        """Register a new handler."""
        if backend_name in self._implemented_backends and not overwrite:
            raise NgioValueError(
                f"Table backend {backend_name} already implemented. "
                "Use the `overwrite=True` parameter to overwrite it."
            )
        self._implemented_backends[backend_name] = table_beckend


TableBackendsManager().add_handler("anndata", AnnDataBackend)
TableBackendsManager().add_handler("json", JsonTableBackend)

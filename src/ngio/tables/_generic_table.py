"""Implementation of the ROI Table class.

This class follows the roi_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

import pandas as pd
from pydantic import BaseModel

from ngio.tables.backends import ImplementedTableBackends
from ngio.utils import AccessModeLiteral, StoreOrGroup, ZarrGroupHandler


class GenericTableMeta(BaseModel):
    """Metadata for the ROI table."""

    fractal_table_version: str | None = None
    type: str | None = None
    backend: str | None = None


class GenericTable:
    """Class to a non-specific table.

    This can be used to load any table that does not have
    a specific definition.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Initialize the GenericTable."""
        self._meta = GenericTableMeta()
        self._dataframe = dataframe
        self._table_backend = None

    @staticmethod
    def type() -> str:
        """Return the type of the table."""
        return "generic"

    @staticmethod
    def version() -> str:
        """The generic table does not have a version.

        Since does not follow a specific schema.
        """
        return "None"

    @property
    def backend_name(self) -> str | None:
        """Return the name of the backend."""
        if self._table_backend is None:
            return None
        return self._table_backend.backend_name

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the table as a DataFrame."""
        return self._dataframe

    @dataframe.setter
    def dataframe(self, dataframe: pd.DataFrame) -> None:
        """Set the table as a DataFrame."""
        self._dataframe = dataframe

    @classmethod
    def from_store(
        cls,
        store: StoreOrGroup,
        cache: bool = False,
        mode: AccessModeLiteral = "a",
        parallel_safe: bool = False,
    ) -> "GenericTable":
        """Create a new ROI table from a Zarr store."""
        handler = ZarrGroupHandler(
            store=store, cache=cache, mode=mode, parallel_safe=parallel_safe
        )
        meta = GenericTableMeta(**handler.load_attrs())
        backend = ImplementedTableBackends().get_backend(
            backend_name=meta.backend,
            group_handler=handler,
            index_key=None,
        )

        if not backend.implements_dataframe:
            raise ValueError("The backend does not implement the dataframe protocol.")

        dataframe = backend.load_as_dataframe()

        table = cls(dataframe)
        table._meta = meta
        table._table_backend = backend
        return table

    def set_backend(
        self,
        store: StoreOrGroup,
        backend_name: str | None = None,
        cache: bool = False,
        mode: AccessModeLiteral = "a",
        parallel_safe: bool = False,
    ) -> None:
        """Set the backend of the table."""
        handler = ZarrGroupHandler(
            store=store, cache=cache, mode=mode, parallel_safe=parallel_safe
        )
        backend = ImplementedTableBackends().get_backend(
            backend_name=backend_name,
            group_handler=handler,
            index_key=None,
        )
        self._meta.backend = backend_name
        self._table_backend = backend

    def consolidate(self) -> None:
        """Write the current state of the table to the Zarr file."""
        if self._table_backend is None:
            raise ValueError(
                "No backend set for the table. "
                "Please add the table to a OME-Zarr Image before calling consolidate."
            )

        self._table_backend.write_from_dataframe(
            self._dataframe, metadata=self._meta.model_dump(exclude_none=True)
        )

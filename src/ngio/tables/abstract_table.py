"""Implementation of a generic table class."""

import builtins
from abc import ABC, abstractmethod
from typing import Literal, Self

import pandas as pd
import polars as pl
from anndata import AnnData

from ngio.tables.backends import (
    BackendMeta,
    ImplementedTableBackends,
    SupportedTables,
    TableBackendProtocol,
    convert_to_pandas,
    normalize_table,
)
from ngio.utils import NgioValueError, ZarrGroupHandler


class AbstractBaseTable(ABC):
    """Abstract base class for a table.

    This is used to define common methods and properties
    for all tables.

    This class is not meant to be used directly.
    """

    def __init__(
        self,
        table: SupportedTables | None = None,
        *,
        meta: BackendMeta | None = None,
        index_key: str | None = None,
        index_type: Literal["int", "str"] | None = None,
    ) -> None:
        """Initialize the table."""
        if meta is None:
            meta = BackendMeta()

        if index_key is not None:
            meta.index_key = index_key

        if index_type is not None:
            meta.index_type = index_type

        self._meta = meta
        if table is not None:
            table = normalize_table(
                table,
                index_key=meta.index_key,
                index_type=meta.index_type,
            )
        self._table = table
        self._table_backend = None

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        return f"{self.__class__.__name__}"

    @staticmethod
    @abstractmethod
    def type() -> str:
        """Return the type of the table."""
        ...

    @staticmethod
    @abstractmethod
    def version() -> str:
        """The generic table does not have a version.

        Since does not follow a specific schema.
        """
        ...

    @property
    def backend_name(self) -> str | None:
        """Return the name of the backend."""
        if self._table_backend is None:
            return None
        return self._table_backend.backend_name()

    @property
    def index_key(self) -> str | None:
        """Get the index key."""
        return self._meta.index_key

    @property
    def index_type(self) -> Literal["int", "str"] | None:
        """Get the index type."""
        return self._meta.index_type

    def load_as_anndata(self) -> AnnData:
        """Load the table as an AnnData object."""
        if self._table_backend is None:
            raise NgioValueError("No backend set for the table.")
        return self._table_backend.load_as_anndata()

    def load_as_pandas_df(self) -> pd.DataFrame:
        """Load the table as a pandas DataFrame."""
        if self._table_backend is None:
            raise NgioValueError("No backend set for the table.")
        return self._table_backend.load_as_pandas_df()

    def load_as_polars_lf(self) -> pl.LazyFrame:
        """Load the table as a polars LazyFrame."""
        if self._table_backend is None:
            raise NgioValueError("No backend set for the table.")
        return self._table_backend.load_as_polars_lf()

    @property
    def table(self) -> SupportedTables:
        """Return the table."""
        if self._table is not None:
            return self._table

        if self._table_backend is None:
            raise NgioValueError(
                "The table does not have a DataFrame in memory nor a backend."
            )

        if self._table_backend.implements_pandas():
            self._table = self._table_backend.load_as_pandas_df()
        elif self._table_backend.implements_polars():
            self._table = self._table_backend.load_as_polars_lf()
        elif self._table_backend.implements_anndata():
            self._table = self._table_backend.load_as_anndata()
        else:
            raise NgioValueError(
                "The backend does not implement any of the dataframe protocols."
            )

        return self._table

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the table as a DataFrame."""
        return convert_to_pandas(
            self.table, index_key=self.index_key, index_type=self.index_type
        )

    @staticmethod
    def _load_backend(
        meta: BackendMeta, handler: ZarrGroupHandler, backend_name: str | None = None
    ) -> TableBackendProtocol:
        """Create a new ROI table from a Zarr group handler."""
        backend_name = backend_name if backend_name else meta.backend

        backend = ImplementedTableBackends().get_backend(
            backend_name=backend_name,
            group_handler=handler,
            index_key=meta.index_key,
            index_type=meta.index_type,
        )
        return backend

    def set_table(
        self,
        table: SupportedTables | None = None,
        refresh: bool = False,
    ) -> None:
        """Set the table.

        If an object is passed, it will be used as the table.
        If None is passed, the table will be loaded from the backend.

        If refresh is True, the table will be reloaded from the backend.
            If table is not None, this will be ignored.
        """
        if table is not None:
            if not isinstance(table, SupportedTables):
                raise NgioValueError(
                    "The table must be a pandas DataFrame, polars LazyFrame, "
                    " or AnnData object."
                )

            self._table = normalize_table(
                table,
                index_key=self.index_key,
                index_type=self.index_type,
            )
            return None

        if self._table is not None and not refresh:
            return None

        if self._table_backend is None:
            raise NgioValueError(
                "The table does not have a DataFrame in memory nor a backend."
            )
        self._table = self._table_backend.load()

    def set_backend(
        self,
        handler: ZarrGroupHandler | None = None,
        backend_name: str | None = None,
        index_key: str | None = None,
        index_type: Literal["int", "str"] | None = None,
    ) -> None:
        """Set the backend of the table."""
        if handler is None:
            if self._table_backend is None:
                raise NgioValueError(
                    "No backend set for the table yet. "
                    "A ZarrGroupHandler must be provided."
                )
            handler = self._table_backend.group_handler

        meta = self._meta
        if backend_name is not None:
            meta.backend = backend_name
        if index_key is not None:
            meta.index_key = index_key
        if index_type is not None:
            meta.index_type = index_type

        backend = self._load_backend(
            meta=meta,
            handler=handler,
            backend_name=backend_name,
        )
        self._meta = meta
        self._table_backend = backend

    @classmethod
    def _from_handler(
        cls,
        handler: ZarrGroupHandler,
        meta_model: builtins.type[BackendMeta],
        backend_name: str | None = None,
    ) -> Self:
        """Create a new ROI table from a Zarr group handler."""
        meta = meta_model(**handler.load_attrs())
        table = cls(meta=meta)
        table.set_backend(handler=handler, backend_name=backend_name)
        return table

    @classmethod
    @abstractmethod
    def from_handler(
        cls,
        handler: ZarrGroupHandler,
        backend_name: str | None = None,
    ) -> Self:
        """Create a new ROI table from a Zarr group handler."""
        pass

    def consolidate(self) -> None:
        """Write the current state of the table to the Zarr file."""
        if self._table_backend is None:
            raise NgioValueError(
                "No backend set for the table. "
                "Please add the table to a OME-Zarr Image before calling consolidate."
            )

        self._table_backend.write(
            self.table,
            metadata=self._meta.model_dump(exclude_none=True),
        )

    def _concatenate(
        self,
        table: Self,
        src_columns: dict[str, str] | None = None,
        dst_columns: dict[str, str] | None = None,
        index_key: str = "index",
    ) -> Self:
        """Concatenate multiple tables into a single table."""
        table1 = self.dataframe
        table2 = table.dataframe

        if src_columns is not None:
            table1 = _reindex_dataframe(table1, src_columns, index_key)

        if dst_columns is not None:
            table2 = _reindex_dataframe(table2, dst_columns, index_key)

        if table1.index.name != index_key:
            raise NgioValueError(
                f"Table 1 index name {table1.index.name} "
                f"does not match the expected index key {index_key}"
            )

        if table2.index.name != index_key:
            raise NgioValueError(
                f"Table 2 index name {table2.index.name} "
                f"does not match the expected index key {index_key}"
            )

        if len(table1.columns) != len(table2.columns) or any(
            table1.columns != table2.columns
        ):
            raise NgioValueError(
                "The columns of the two tables do not match. "
                "Please make sure to use the same columns."
                f" Got {table1.columns} and {table2.columns}"
            )
        concatenated_df = pd.concat([table1, table2])
        return type(self)(
            meta=self._meta,
            table=concatenated_df,
            index_key=index_key,
            index_type="str",
        )


def _reindex_dataframe(
    df, new_cols: dict[str, str], index_key: str = "index"
) -> pd.DataFrame:
    """Reindex a dataframe."""
    old_index = df.index.name
    df = df.reset_index()
    for col, value in new_cols.items():
        df[col] = value

    index_cols = list(new_cols.keys())
    if old_index is not None:
        index_cols.append(old_index)
    df.index = df[index_cols].astype(str).agg("_".join, axis=1)
    df.index.name = index_key
    return df

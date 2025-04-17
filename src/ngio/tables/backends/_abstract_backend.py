from abc import ABC, abstractmethod
from typing import Literal

from anndata import AnnData
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame
from pydantic import BaseModel

from ngio.tables.backends._utils import (
    convert_anndata_to_pandas,
    convert_anndata_to_polars,
    convert_pandas_to_anndata,
    convert_pandas_to_polars,
    convert_polars_to_anndata,
    convert_polars_to_pandas,
)
from ngio.utils import NgioValueError, ZarrGroupHandler


class BackendMeta(BaseModel):
    """Metadata for the backend."""

    backend: str | None = None
    index_key: str | None = None
    index_type: Literal["int", "str"] | None = None


class AbstractTableBackend(ABC):
    """Abstract class for table backends."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        index_key: str | None = None,
        index_type: Literal["int", "str"] | None = None,
    ):
        """Initialize the handler.

        This is a base class for the table backends protocol.

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
        """Return the name of the backend.

        As a convention we set name as:
            {backend_name}_v{version}

        Where the version is a integer.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def implements_anndata() -> bool:
        """Check if the backend implements the anndata protocol.

        If this is True, the backend should implement the
        `load_as_anndata` and `write_from_anndata` methods.

        If this is False, these methods should raise a
        `NotImplementedError`.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def implements_pandas() -> bool:
        """Check if the backend implements the pandas protocol.

        If this is True, the backend should implement the
        `load_as_dataframe` and `write_from_dataframe` methods.

        If this is False, these methods should raise a
        `NotImplementedError`.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def implements_polars() -> bool:
        """Check if the backend implements the polars protocol.

        If this is True, the backend should implement the
        `load_as_polars` and `write_from_polars` methods.

        If this is False, these methods should raise a
        `NotImplementedError`.
        """
        raise NotImplementedError

    @property
    def group_handler(self) -> ZarrGroupHandler:
        """Get the group handler."""
        return self._group_handler

    @property
    def index_key(self) -> str | None:
        """Get the index key."""
        return self._index_key

    @property
    def index_type(self) -> Literal["int", "str"] | None:
        """Get the index type."""
        if self._index_type is None:
            return None

        if self._index_type not in ["int", "str"]:
            raise NgioValueError(
                f"Invalid index type {self._index_type}. Must be 'int' or 'str'."
            )
        return self._index_type  # type: ignore[return-value]

    def load_as_anndata(self) -> AnnData:
        """Load the table as an AnnData object.

        Since the AnnData object is more complex than a DataFrame,
        selecting columns is not implemented, because it is not
        straightforward to do so for an arbitrary AnnData object.
        """
        if self.implements_pandas():
            return convert_pandas_to_anndata(
                self.load_as_pandas_df(),
                index_key=self.index_key,
            )
        elif self.implements_polars():
            return convert_polars_to_anndata(
                self.load_as_polars_lf(),
                index_key=self.index_key,
            )
        else:
            raise NgioValueError(
                "Backend does not implement any of the protocols. "
                "A backend should implement at least one of the "
                "following protocols: anndata, pandas, polars."
            )

    def load_as_pandas_df(self) -> DataFrame:
        """Load the table as a pandas DataFrame.

        If columns are provided, the table should be filtered
        """
        if self.implements_anndata():
            return convert_anndata_to_pandas(
                self.load_as_anndata(),
                index_key=self.index_key,
                index_type=self.index_type,
            )
        elif self.implements_polars():
            return convert_polars_to_pandas(
                self.load_as_polars_lf(),
                index_key=self.index_key,
                index_type=self.index_type,
            )
        else:
            raise NgioValueError(
                "Backend does not implement any of the protocols. "
                "A backend should implement at least one of the "
                "following protocols: anndata, pandas, polars."
            )

    def load_as_polars_lf(self) -> LazyFrame:
        """Load the table as a polars LazyFrame.

        If columns are provided, the table should be filtered
        """
        if self.implements_anndata():
            return convert_anndata_to_polars(
                self.load_as_anndata(),
                index_key=self.index_key,
                index_type=self.index_type,
            ).lazy()
        elif self.implements_pandas():
            return convert_pandas_to_polars(
                self.load_as_pandas_df(),
                index_key=self.index_key,
                index_type=self.index_type,
            ).lazy()
        else:
            raise NgioValueError(
                "Backend does not implement any of the protocols. "
                "A backend should implement at least one of the "
                "following protocols: anndata, pandas, polars."
            )

    def write_from_pandas(self, table: DataFrame) -> None:
        """Serialize the table from a pandas DataFrame."""
        raise NotImplementedError(
            f"Backend {self.backend_name()} does not support "
            "serialization of DataFrame objects."
        )

    def write_from_anndata(self, table: AnnData) -> None:
        """Serialize the table from an AnnData object."""
        raise NotImplementedError(
            f"Backend {self.backend_name()} does not support "
            "serialization of AnnData objects."
        )

    def write_from_polars(self, table: PolarsDataFrame | LazyFrame) -> None:
        """Serialize the table from a polars DataFrame or LazyFrame."""
        raise NotImplementedError(
            f"Backend {self.backend_name()} does not support "
            "serialization of Polars objects."
        )

    def write_metadata(self, metadata: dict | None = None) -> None:
        """Write the metadata to the store."""
        if metadata is None:
            metadata = {}

        backend_metadata = BackendMeta(
            backend=self.backend_name(),
            index_key=self.index_key,
            index_type=self.index_type,
        ).model_dump(exclude_none=True)
        metadata.update(backend_metadata)
        self._group_handler.write_attrs(metadata)

    def write(
        self,
        table: DataFrame | AnnData | PolarsDataFrame | LazyFrame,
        metadata: dict | None = None,
        mode: Literal["pandas", "anndata", "polars"] | None = None,
    ) -> None:
        """Serialize the table to the store, and write the metadata.

        This is a generic write method.
        Based on the explicit mode or the type of the table,
        it will call the appropriate write method.
        """
        if mode == "pandas" or isinstance(table, DataFrame):
            self.write_from_pandas(table)  # type: ignore[arg-type]
        elif mode == "anndata" or isinstance(table, AnnData):
            self.write_from_anndata(table)  # type: ignore[arg-type]
        elif mode == "polars" or isinstance(table, PolarsDataFrame | LazyFrame):
            self.write_from_polars(table)
        else:
            raise NgioValueError(
                f"Unsupported table type {type(table)}. "
                "Please specify the mode explicitly. "
                "Supported serialization modes are: "
                "'pandas', 'anndata', 'polars'."
            )
        self.write_metadata(metadata)

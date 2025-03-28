"""Implementation of a generic table class."""

import pandas as pd
from anndata import AnnData
from pydantic import BaseModel

from ngio.tables.backends import ImplementedTableBackends
from ngio.tables.backends._anndata_utils import (
    anndata_to_dataframe,
    dataframe_to_anndata,
)
from ngio.utils import NgioValueError, ZarrGroupHandler


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

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        anndata: AnnData | None = None,
    ) -> None:
        """Initialize the GenericTable."""
        self._meta = GenericTableMeta()
        if dataframe is None and anndata is None:
            raise NgioValueError(
                "Either a DataFrame or an AnnData object must be provided."
            )

        if dataframe is not None and anndata is not None:
            raise NgioValueError(
                "Only one of DataFrame or AnnData object can be provided."
            )

        self._dataframe = dataframe
        self._anndata = anndata

        self.anndata_native = True if anndata is not None else False

        self._table_backend = None

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        if self._dataframe is not None:
            num_rows = len(self.dataframe)
            num_columns = len(self.dataframe.columns)
            prop = f"num_rows={num_rows}, num_columns={num_columns}, mode=dataframe"
        else:
            prop = "mode=anndata"
        return f"GenericTable({prop})"

    @staticmethod
    def type() -> str:
        """Return the type of the table."""
        return "generic"

    @staticmethod
    def version() -> str:
        """The generic table does not have a version.

        Since does not follow a specific schema.
        """
        return "1"

    @property
    def backend_name(self) -> str | None:
        """Return the name of the backend."""
        if self._table_backend is None:
            return None
        return self._table_backend.backend_name()

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the table as a DataFrame."""
        if self._dataframe is not None:
            return self._dataframe

        if self._anndata is not None:
            return anndata_to_dataframe(self._anndata)

        raise NgioValueError("No table loaded.")

    @dataframe.setter
    def dataframe(self, dataframe: pd.DataFrame) -> None:
        """Set the table as a DataFrame."""
        self._dataframe = dataframe
        self.anndata_native = False

    @property
    def anndata(self) -> AnnData:
        """Return the table as an AnnData object."""
        if self._anndata is not None:
            return self._anndata

        if self._dataframe is not None:
            return dataframe_to_anndata(self._dataframe)
        raise NgioValueError("No table loaded.")

    @anndata.setter
    def anndata(self, anndata: AnnData) -> None:
        """Set the table as an AnnData object."""
        self._anndata = anndata
        self.anndata_native = True

    @classmethod
    def _from_handler(
        cls, handler: ZarrGroupHandler, backend_name: str | None = None
    ) -> "GenericTable":
        """Create a new ROI table from a Zarr group handler."""
        meta = GenericTableMeta(**handler.load_attrs())
        if backend_name is None:
            backend = ImplementedTableBackends().get_backend(
                backend_name=meta.backend,
                group_handler=handler,
                index_key=None,
            )
        else:
            backend = ImplementedTableBackends().get_backend(
                backend_name=backend_name,
                group_handler=handler,
                index_key=None,
            )
            meta.backend = backend_name

        if backend.implements_anndata():
            anndata = backend.load_as_anndata()
            table = cls(anndata=anndata)

        elif backend.implements_dataframe():
            dataframe = backend.load_as_dataframe()
            table = cls(dataframe=dataframe)
        else:
            raise NgioValueError(
                "The backend does not implement the dataframe protocol."
            )

        table._meta = meta
        table._table_backend = backend
        return table

    def _set_backend(
        self,
        handler: ZarrGroupHandler,
        backend_name: str | None = None,
    ) -> None:
        """Set the backend of the table."""
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
            raise NgioValueError(
                "No backend set for the table. "
                "Please add the table to a OME-Zarr Image before calling consolidate."
            )

        if self.anndata_native:
            self._table_backend.write_from_anndata(
                self.anndata, metadata=self._meta.model_dump(exclude_none=True)
            )
        else:
            self._table_backend.write_from_dataframe(
                self.dataframe, metadata=self._meta.model_dump(exclude_none=True)
            )

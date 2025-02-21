from collections.abc import Iterable

from anndata import AnnData
from pandas import DataFrame

from ngio.tables._validators import TableValidator
from ngio.tables.tables_backends._anndata_utils import (
    anndata_to_dataframe,
    custom_read_zarr,
)
from ngio.utils import (
    AccessModeLiteral,
    StoreOrGroup,
    ZarrGroupHandler,
)


class AnnDataHandler:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(
        self,
        store: StoreOrGroup,
        cache: bool = False,
        mode: AccessModeLiteral = "a",
        load_validators: Iterable[TableValidator] | None = None,
        write_validators: Iterable[TableValidator] | None = None,
    ):
        """Initialize the handler.

        Args:
            meta_converter (MetaConverter): The metadata converter.
            store (StoreOrGroup): The Zarr store or group containing the image data.
            meta_mode (str): The mode of the metadata handler.
            cache (bool): Whether to cache the metadata.
            mode (str): The mode of the store.
            load_validators (Iterable[TableValidator]): Validators to use when loading
                the table.
            write_validators (Iterable[TableValidator]): Validators to use when writing
                the table.
        """
        self._group_handler = ZarrGroupHandler(store, cache, mode)

        self._load_validators = load_validators
        self._write_validators = write_validators

    def load_as_anndata(self) -> AnnData:
        """Load the metadata in the store."""
        anndata = self._group_handler.get_from_cache("anndata")
        if anndata is not None:
            if not isinstance(anndata, AnnData):
                raise TypeError("The cached 'anndata' is not an AnnData object.")
            return anndata

        anndata = custom_read_zarr(self._group_handler._group)
        self._group_handler.add_to_cache("anndata", anndata)
        return anndata

    def load_as_dataframe(self) -> DataFrame:
        """List all labels in the group."""
        dataframe = self._group_handler.get_from_cache("dataframe")
        if dataframe is not None:
            if not isinstance(dataframe, DataFrame):
                raise TypeError("The cached 'dataframe' is not a DataFrame object.")
            return dataframe

        dataframe = anndata_to_dataframe(
            self.load_as_anndata(), validators=self._load_validators
        )
        self._group_handler.add_to_cache("dataframe", dataframe)
        return dataframe

    def write_from_dataframe(self, table: DataFrame) -> None:
        """Consolidate the metadata in the store."""
        raise NotImplementedError

    def write_from_anndata(self, table: AnnData) -> None:
        """Consolidate the metadata in the store."""
        raise NotImplementedError

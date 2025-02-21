# %%
import pandas as pd
from anndata import AnnData
from pandas import DataFrame

from ngio.utils import (
    AccessModeLiteral,
    StoreOrGroup,
    ZarrGroupHandler,
)


class JsonTableBackend:
    """A class to handle stoing small tables in the zarr group .attrs (json)."""

    def __init__(
        self,
        store: StoreOrGroup,
        cache: bool = False,
        mode: AccessModeLiteral = "a",
    ):
        """Initialize the handler.

        Args:
            store (StoreOrGroup): The Zarr store or group containing the image data.
            cache (bool): Whether to cache the metadata.
            mode (str): The mode of the store.
        """
        self._group_handler = ZarrGroupHandler(store, cache, mode)

    def load_as_anndata(self) -> AnnData:
        """Load the metadata in the store."""
        raise NotImplementedError

    def load_as_dataframe(self) -> DataFrame:
        """List all labels in the group."""
        attrs = self._group_handler.load_attrs()
        table = attrs.get("table")
        if table is None:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(table)

    def write_from_dataframe(self, table: DataFrame) -> None:
        """Consolidate the metadata in the store."""
        attrs = self._group_handler.load_attrs()
        attrs["table"] = table.to_dict()
        self._group_handler.write_attrs(attrs)

    def write_from_anndata(self, table: AnnData) -> None:
        """Consolidate the metadata in the store."""
        raise NotImplementedError

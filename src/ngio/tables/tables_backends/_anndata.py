from __future__ import annotations

from typing import TYPE_CHECKING

from ngio.utils import (
    AccessModeLiteral,
    StoreOrGroup,
    ZarrGroupHandler,
)

if TYPE_CHECKING:
    from pandas import DataFrame


class AnnDataHandler:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(
        self,
        store: StoreOrGroup,
        cache: bool = False,
        mode: AccessModeLiteral = "a",
    ):
        """Initialize the handler.

        Args:
            meta_converter (MetaConverter): The metadata converter.
            store (StoreOrGroup): The Zarr store or group containing the image data.
            meta_mode (str): The mode of the metadata handler.
            cache (bool): Whether to cache the metadata.
            mode (str): The mode of the store.
        """
        self._group_handler = ZarrGroupHandler(store, cache, mode)
        self.cache = cache
        self._attrs: dict | None = None

    def load(self) -> DataFrame:
        """List all labels in the group."""
        raise NotImplementedError

    def write(self, table: DataFrame) -> None:
        """Consolidate the metadata in the store."""
        raise NotImplementedError

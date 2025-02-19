"""Concrete implementation of the OME-Zarr metadata handlers for version 0.4."""

from collections.abc import Callable

from pydantic import ValidationError
from zarr import Group

from ngio.ome_zarr_meta.ngio_specs import (
    NgioImageLabelMeta,
    NgioImageMeta,
    NgioLabelMeta,
)
from ngio.ome_zarr_meta.v04._v04_spec_utils import (
    ngio_to_v04_image_meta,
    ngio_to_v04_label_meta,
    v04_to_ngio_image_meta,
    v04_to_ngio_label_meta,
)
from ngio.utils import (
    AccessModeLiteral,
    NgioValueError,
    StoreOrGroup,
    open_group_wrapper,
)


class OmeZarrV04BaseHandler:
    """Class for loading and writing OME-NGFF 0.4 metadata."""

    def __init__(
        self,
        meta_importer: Callable,
        meta_exporter: Callable,
        store: StoreOrGroup,
        cache: bool = False,
        mode: AccessModeLiteral = "a",
    ):
        """Initialize the handler.

        Args:
            meta_importer (Callable): The function to import the metadata.
            meta_exporter (Callable): The function to export the metadata.
            store (StoreOrGroup): The Zarr store or group containing the image data.
            meta_mode (str): The mode of the metadata handler.
            cache (bool): Whether to cache the metadata.
            mode (str): The mode of the store.
        """
        if isinstance(store, Group):
            if hasattr(store, "store_path"):
                self._store = store.store_path
            else:
                self._store = store.store
            self._group = store

        else:
            self._store = store
            self._group = open_group_wrapper(store=store, mode=mode)

        self.cache = cache
        self._attrs: dict | None = None

        self._meta_importer = meta_importer
        self._meta_exporter = meta_exporter

    def _load_attrs(self) -> dict:
        """Load the attributes of the group."""
        if self._attrs is not None:
            return self._attrs

        self._attrs = dict(self._group.attrs)
        return self._attrs

    def load(self, return_error: bool = False) -> NgioImageLabelMeta | ValidationError:
        attrs = self._load_attrs()
        meta = self._meta_importer(attrs)
        if isinstance(meta, NgioImageLabelMeta):
            return meta

        if return_error:
            return meta
        raise meta

    def write(self, meta: NgioImageLabelMeta) -> None:
        if not self._group.store.is_writeable():
            raise NgioValueError("The store is not writeable. Cannot write metadata.")

        v04_meta = self._meta_exporter(meta)
        self._group.attrs.update(v04_meta)

    def clean_cache(self) -> None:
        """Clear the cached metadata."""
        self._attrs = None


class OmeZarrV04ImageHandler(OmeZarrV04BaseHandler):
    """Class for loading and writing OME-NGFF 0.4 metadata."""

    def __init__(
        self, store: StoreOrGroup, cache: bool = False, mode: AccessModeLiteral = "a"
    ):
        """Initialize the handler.

        Args:
            store (StoreOrGroup): The Zarr store or group containing the image data.
            cache (bool): Whether to cache the metadata.
            mode (str): The mode of the store.
        """
        super().__init__(
            meta_importer=v04_to_ngio_image_meta,
            meta_exporter=ngio_to_v04_image_meta,
            store=store,
            cache=cache,
            mode=mode,
        )

    def load(self, return_error: bool = False) -> NgioImageMeta | ValidationError:
        """Load the metadata of the group."""
        return super().load(return_error=return_error)

    def write(self, meta: NgioImageMeta) -> None:
        """Write the metadata to the store."""
        super().write(meta)

    def clean_cache(self) -> None:
        """Clear the cached metadata."""
        super().clean_cache()


class OmeZarrV04LabelHandler(OmeZarrV04BaseHandler):
    """Class for loading and writing OME-NGFF 0.4 metadata."""

    def __init__(
        self, store: StoreOrGroup, cache: bool = False, mode: AccessModeLiteral = "a"
    ):
        """Initialize the handler.

        Args:
            store (StoreOrGroup): The Zarr store or group containing the image data.
            cache (bool): Whether to cache the metadata.
            mode (str): The mode of the store.
        """
        super().__init__(
            meta_importer=v04_to_ngio_label_meta,
            meta_exporter=ngio_to_v04_label_meta,
            store=store,
            cache=cache,
            mode=mode,
        )

    def load(self, return_error: bool = False) -> NgioLabelMeta | ValidationError:
        """Load the metadata of the group."""
        return super().load(return_error=return_error)

    def write(self, meta: NgioLabelMeta) -> None:
        """Write the metadata to the store."""
        super().write(meta)

    def clean_cache(self) -> None:
        """Clear the cached metadata."""
        super().clean_cache()

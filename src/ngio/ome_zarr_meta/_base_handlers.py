"""Base class for handling OME-NGFF metadata in Zarr groups."""

from typing import Generic, TypeVar

from zarr import Group

from ngio.ome_zarr_meta._meta_converter_prototypes import (
    ConverterError,
    ImageMetaConverter,
    LabelMetaConverter,
)
from ngio.ome_zarr_meta.ngio_specs import (
    NgioImageMeta,
    NgioLabelMeta,
)
from ngio.utils import (
    AccessModeLiteral,
    NgioValueError,
    StoreOrGroup,
    open_group_wrapper,
)

_Image_or_Label = TypeVar("_Image_or_Label", NgioImageMeta, NgioLabelMeta)
_Image_or_Label_Converter = TypeVar(
    "_Image_or_Label_Converter", ImageMetaConverter, LabelMetaConverter
)


class GenericOmeZarrHandler(Generic[_Image_or_Label, _Image_or_Label_Converter]):
    """Generic class for handling OME-Zarr metadata in Zarr groups."""

    def __init__(
        self,
        meta_converter: _Image_or_Label_Converter,
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

        self._meta_converter = meta_converter

    def _load_attrs(self) -> dict:
        """Load the attributes of the group."""
        if self._attrs is not None:
            return self._attrs

        self._attrs = dict(self._group.attrs)
        return self._attrs

    def _load_meta(self, return_error: bool = False):
        """Load the metadata from the store."""
        attrs = self._load_attrs()
        is_valid, meta_or_error = self._meta_converter.from_dict(attrs)
        if is_valid:
            return meta_or_error

        if return_error:
            return meta_or_error

        raise NgioValueError(f"Could not load metadata: {meta_or_error}")

    def load(self, return_error: bool = False) -> _Image_or_Label | ConverterError:
        """Load the metadata from the store."""
        raise NotImplementedError

    def _write_meta(self, meta) -> None:
        """Write the metadata to the store."""
        if not self._group.store.is_writeable():
            raise NgioValueError("The store is not writeable. Cannot write metadata.")

        v04_meta = self._meta_converter.to_dict(meta)
        self._group.attrs.update(v04_meta)

    def write(self, meta: _Image_or_Label) -> None:
        """Write the metadata to the store."""
        raise NotImplementedError

    def clean_cache(self) -> None:
        """Clear the cached metadata."""
        self._attrs = None


class BaseOmeZarrImageHandler(GenericOmeZarrHandler[NgioImageMeta, ImageMetaConverter]):
    """Generic class for handling OME-Zarr metadata in Zarr groups."""

    def __init__(
        self,
        meta_converter: ImageMetaConverter,
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
        super().__init__(meta_converter, store, cache, mode)

    def load(self, return_error: bool = False) -> NgioImageMeta | ConverterError:
        """Load the metadata from the store."""
        return self._load_meta(return_error)

    def write(self, meta: NgioImageMeta) -> None:
        self._write_meta(meta)


class BaseOmeZarrLabelHandler(GenericOmeZarrHandler[NgioLabelMeta, LabelMetaConverter]):
    """Generic class for handling OME-Zarr metadata in Zarr groups."""

    def __init__(
        self,
        meta_converter: LabelMetaConverter,
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
        super().__init__(meta_converter, store, cache, mode)

    def load(self, return_error: bool = False) -> NgioLabelMeta | ConverterError:
        """Load the metadata from the store."""
        return self._load_meta(return_error)

    def write(self, meta: NgioLabelMeta) -> None:
        self._write_meta(meta)

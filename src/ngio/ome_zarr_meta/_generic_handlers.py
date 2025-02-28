"""Base class for handling OME-NGFF metadata in Zarr groups."""

from typing import Generic, Protocol, TypeVar

from pydantic import ValidationError

from ngio.ome_zarr_meta.ngio_specs import (
    NgioImageMeta,
    NgioLabelMeta,
)
from ngio.utils import (
    AccessModeLiteral,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)

ConverterError = ValidationError | Exception | None


class ImageMetaHandler(Protocol):
    """Protocol for OME-Zarr image handlers."""

    def __init__(
        self, store: StoreOrGroup, cache: bool = False, mode: AccessModeLiteral = "a"
    ):
        """Initialize the handler."""
        ...

    def safe_load_meta(self) -> NgioImageMeta | ConverterError:
        """Load the metadata from the store."""
        ...

    @property
    def meta(self) -> NgioImageMeta:
        """Return the metadata."""
        ...

    def write_meta(self, meta: NgioImageMeta) -> None:
        """Write the metadata to the store."""
        ...

    @property
    def group_handler(self) -> ZarrGroupHandler:
        """Return the group handler."""
        ...


class LabelMetaHandler(Protocol):
    """Protocol for OME-Zarr label handlers."""

    def __init__(
        self, store: StoreOrGroup, cache: bool = False, mode: AccessModeLiteral = "a"
    ):
        """Initialize the handler."""
        ...

    def safe_load_meta(self) -> NgioLabelMeta | ConverterError:
        """Load the metadata from the store."""
        ...

    @property
    def meta(self) -> NgioLabelMeta:
        """Return the metadata."""
        ...

    def write_meta(self, meta: NgioLabelMeta) -> None:
        """Write the metadata to the store."""
        ...

    @property
    def group_handler(self) -> ZarrGroupHandler:
        """Return the group handler."""
        ...


###########################################################################
#
# The code below implements a generic class for handling OME-Zarr metadata
# in Zarr groups.
#
###########################################################################


class ImageMetaConverter(Protocol):
    """Protocol for metadata converters."""

    def from_dict(self, meta: dict) -> tuple[bool, NgioImageMeta | ConverterError]:
        """Convert the metadata from a dictionary to the ngio image metadata.

        This function should return a monadic style return:
            * is_success = True -> converted_meta
            * is_success = False -> error


        Ideally if the metadata is valid but the conversion fails for other reasons,
            the error should be raised.

        Args:
            meta (dict): The metadata in dictionary form
                (usually the Zarr group attributes).

        Returns:
            tuple[bool, NgioImageLabelMeta | ConverterError]: Monadic style return.
        """
        ...

    def to_dict(self, meta: NgioImageMeta) -> dict:
        """Convert the ngio image metadata to a dictionary."""
        ...


class LabelMetaConverter(Protocol):
    """Protocol for metadata converters."""

    def from_dict(self, meta: dict) -> tuple[bool, NgioLabelMeta | ConverterError]:
        """Convert the metadata from a dictionary to the ngio label metadata.

        This function should return a monadic style return:
            * is_success = True -> converted_meta
            * is_success = False -> error


        Ideally if the metadata is valid but the conversion fails for other reasons,
            the error should be raised.

        Args:
            meta (dict): The metadata in dictionary form
                (usually the Zarr group attributes).

        Returns:
            tuple[bool, NgioImageLabelMeta | ConverterError]: Monadic style return.
        """
        ...

    def to_dict(self, meta: NgioLabelMeta) -> dict:
        """Convert the ngio label metadata to a dictionary."""
        ...


_Image_or_Label = TypeVar("_Image_or_Label", NgioImageMeta, NgioLabelMeta)
_Image_or_Label_Converter = TypeVar(
    "_Image_or_Label_Converter", ImageMetaConverter, LabelMetaConverter
)


class GenericMetaHandler(Generic[_Image_or_Label, _Image_or_Label_Converter]):
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
        self._group_handler = ZarrGroupHandler(store=store, cache=cache, mode=mode)
        self._meta_converter = meta_converter

    def _load_meta(self, return_error: bool = False):
        """Load the metadata from the store."""
        attrs = self._group_handler.load_attrs()
        is_valid, meta_or_error = self._meta_converter.from_dict(attrs)
        if is_valid:
            return meta_or_error

        if return_error:
            return meta_or_error

        raise NgioValueError(f"Could not load metadata: {meta_or_error}")

    def safe_load_meta(self) -> _Image_or_Label | ConverterError:
        """Load the metadata from the store."""
        return self._load_meta(return_error=True)

    def _write_meta(self, meta) -> None:
        """Write the metadata to the store."""
        v04_meta = self._meta_converter.to_dict(meta)
        self._group_handler.write_attrs(v04_meta)

    def write_meta(self, meta: _Image_or_Label) -> None:
        """Write the metadata to the store."""
        raise NotImplementedError

    def clean_cache(self) -> None:
        """Clear the cached metadata."""
        self._attrs = None

    @property
    def meta(self) -> _Image_or_Label:
        """Return the metadata."""
        raise NotImplementedError

    @property
    def group_handler(self) -> ZarrGroupHandler:
        """Return the group handler."""
        return self._group_handler


class BaseImageMetaHandler(GenericMetaHandler[NgioImageMeta, ImageMetaConverter]):
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

    def safe_load_meta(
        self, return_error: bool = False
    ) -> NgioImageMeta | ConverterError:
        """Load the metadata from the store."""
        return self._load_meta(return_error)

    @property
    def meta(self) -> NgioImageMeta:
        """Load the metadata from the store."""
        meta = self._load_meta()
        if isinstance(meta, NgioImageMeta):
            return meta
        raise NgioValueError(f"Could not load metadata: {meta}")

    def write_meta(self, meta: NgioImageMeta) -> None:
        self._write_meta(meta)


class BaseLabelMetaHandler(GenericMetaHandler[NgioLabelMeta, LabelMetaConverter]):
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

    def safe_load_meta(
        self, return_error: bool = False
    ) -> NgioLabelMeta | ConverterError:
        """Load the metadata from the store."""
        return self._load_meta(return_error)

    @property
    def meta(self) -> NgioLabelMeta:
        """Load the metadata from the store."""
        meta = self._load_meta()
        if isinstance(meta, NgioLabelMeta):
            return meta
        raise NgioValueError(f"Could not load metadata: {meta}")

    def write_meta(self, meta: NgioLabelMeta) -> None:
        self._write_meta(meta)

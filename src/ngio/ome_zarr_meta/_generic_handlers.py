"""Base class for handling OME-NGFF metadata in Zarr groups."""

from typing import Generic, Protocol, TypeVar

from pydantic import ValidationError

from ngio.ome_zarr_meta.ngio_specs import AxesSetup, NgioImageMeta, NgioLabelMeta
from ngio.utils import (
    NgioValueError,
    ZarrGroupHandler,
)

ConverterError = ValidationError | Exception | None


class ImageMetaHandler(Protocol):
    """Protocol for OME-Zarr image handlers."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
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


class LabelMetaHandler(Protocol):
    """Protocol for OME-Zarr label handlers."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
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


###########################################################################
#
# The code below implements a generic class for handling OME-Zarr metadata
# in Zarr groups.
#
###########################################################################


class ImageMetaImporter(Protocol):
    @staticmethod
    def __call__(
        metadata: dict,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ) -> tuple[bool, NgioImageMeta | ConverterError]:
        """Convert the metadata to a NgioImageMeta object.

        Args:
            metadata (dict): The metadata (typically from a Zarr group .attrs).
            axes_setup (AxesSetup, optional): The axes setup.
                This is used to map axes with non-canonical names.
            allow_non_canonical_axes (bool, optional): Whether to allow non-canonical
                axes.
            strict_canonical_order (bool, optional): Whether to enforce a strict
                canonical order.

        Returns:
            tuple[bool, NgioImageMeta | ConverterError]: A tuple with a boolean
                indicating whether the conversion was successful and the
                NgioImageMeta object or an error.

        """
        ...


class ImageMetaExporter(Protocol):
    def __call__(self, metadata: NgioImageMeta) -> dict: ...


class LabelMetaImporter(Protocol):
    @staticmethod
    def __call__(
        metadata: dict,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ) -> tuple[bool, NgioLabelMeta | ConverterError]:
        """Convert the metadata to a NgioLabelMeta object.

        Args:
            metadata (dict): The metadata (typically from a Zarr group .attrs).
            axes_setup (AxesSetup, optional): The axes setup.
                This is used to map axes with non-canonical names.
            allow_non_canonical_axes (bool, optional): Whether to allow non-canonical
                axes.
            strict_canonical_order (bool, optional): Whether to enforce a strict
                canonical order.

        Returns:
            tuple[bool, NgioLabelMeta | ConverterError]: A tuple with a boolean
                indicating whether the conversion was successful and the
                NgioLabelMeta object or an error.

        """
        ...


class LabelMetaExporter(Protocol):
    def __call__(self, metadata: NgioLabelMeta) -> dict: ...


_meta = TypeVar("_meta", NgioImageMeta, NgioLabelMeta)
_meta_importer = TypeVar("_meta_importer", ImageMetaImporter, LabelMetaImporter)
_meta_exporter = TypeVar("_meta_exporter", ImageMetaExporter, LabelMetaExporter)


class GenericMetaHandler(Generic[_meta, _meta_importer, _meta_exporter]):
    """Generic class for handling OME-Zarr metadata in Zarr groups."""

    def __init__(
        self,
        meta_importer: _meta_importer,
        meta_exporter: _meta_exporter,
        group_handler: ZarrGroupHandler,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ):
        """Initialize the handler.

        Args:
            meta_importer (MetaImporter): The metadata importer.
            meta_exporter (MetaExporter): The metadata exporter.
            group_handler (ZarrGroupHandler): The Zarr group handler.
            axes_setup (AxesSetup, optional): The axes setup.
                This is used to map axes with non-canonical names.
            allow_non_canonical_axes (bool, optional): Whether to allow non-canonical
                axes.
            strict_canonical_order (bool, optional): Whether to enforce a strict
                canonical order.
        """
        self._group_handler = group_handler
        self._meta_importer = meta_importer
        self._meta_exporter = meta_exporter
        self._axes_setup = axes_setup
        self._allow_non_canonical_axes = allow_non_canonical_axes
        self._strict_canonical_order = strict_canonical_order

    def _load_meta(self, return_error: bool = False):
        """Load the metadata from the store."""
        attrs = self._group_handler.load_attrs()
        is_valid, meta_or_error = self._meta_importer(
            metadata=attrs,
            axes_setup=self._axes_setup,
            allow_non_canonical_axes=self._allow_non_canonical_axes,
            strict_canonical_order=self._strict_canonical_order,
        )
        if is_valid:
            return meta_or_error

        if return_error:
            return meta_or_error

        raise NgioValueError(f"Could not load metadata: {meta_or_error}")

    def safe_load_meta(self) -> _meta | ConverterError:
        """Load the metadata from the store."""
        return self._load_meta(return_error=True)

    def _write_meta(self, meta) -> None:
        """Write the metadata to the store."""
        v04_meta = self._meta_exporter(metadata=meta)
        self._group_handler.write_attrs(v04_meta)

    def write_meta(self, meta: _meta) -> None:
        """Write the metadata to the store."""
        raise NotImplementedError

    @property
    def meta(self) -> _meta:
        """Return the metadata."""
        raise NotImplementedError


class BaseImageMetaHandler(
    GenericMetaHandler[NgioImageMeta, ImageMetaImporter, ImageMetaExporter]
):
    """Generic class for handling OME-Zarr metadata in Zarr groups."""

    def __init__(
        self,
        meta_importer: ImageMetaImporter,
        meta_exporter: ImageMetaExporter,
        group_handler: ZarrGroupHandler,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ):
        """Initialize the handler.

        Args:
            meta_importer (ImageMetaImporter): The metadata importer.
            meta_exporter (ImageMetaExporter): The metadata exporter.
            group_handler (ZarrGroupHandler): The Zarr group handler.
            axes_setup (AxesSetup, optional): The axes setup.
                This is used to map axes with non-canonical names.
            allow_non_canonical_axes (bool, optional): Whether to allow non-canonical
                axes.
            strict_canonical_order (bool, optional): Whether to enforce a strict
                canonical order.
        """
        super().__init__(
            meta_importer=meta_importer,
            meta_exporter=meta_exporter,
            group_handler=group_handler,
            axes_setup=axes_setup,
            allow_non_canonical_axes=allow_non_canonical_axes,
            strict_canonical_order=strict_canonical_order,
        )

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


class BaseLabelMetaHandler(
    GenericMetaHandler[NgioLabelMeta, LabelMetaImporter, LabelMetaExporter]
):
    """Generic class for handling OME-Zarr metadata in Zarr groups."""

    def __init__(
        self,
        meta_importer: LabelMetaImporter,
        meta_exporter: LabelMetaExporter,
        group_handler: ZarrGroupHandler,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ):
        """Initialize the handler.

        Args:
            meta_importer (LabelMetaImporter): The metadata importer.
            meta_exporter (LabelMetaExporter): The metadata exporter.
            group_handler (ZarrGroupHandler): The Zarr group handler.
            axes_setup (AxesSetup, optional): The axes setup.
                This is used to map axes with non-canonical names.
            allow_non_canonical_axes (bool, optional): Whether to allow non-canonical
                axes.
            strict_canonical_order (bool, optional): Whether to enforce a strict
                canonical order.
        """
        super().__init__(
            meta_importer=meta_importer,
            meta_exporter=meta_exporter,
            group_handler=group_handler,
            axes_setup=axes_setup,
            allow_non_canonical_axes=allow_non_canonical_axes,
            strict_canonical_order=strict_canonical_order,
        )

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

from typing import Any, Generic, TypeVar

from pydantic import ValidationError

from ngio.ome_zarr_meta._generic_handlers import (
    ImageMetaHandler,
    LabelMetaHandler,
)
from ngio.ome_zarr_meta.ngio_specs import AxesSetup
from ngio.ome_zarr_meta.v04 import V04ImageMetaHandler, V04LabelMetaHandler
from ngio.utils import (
    AccessModeLiteral,
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)

_Image_or_Label_Plugin = TypeVar(
    "_Image_or_Label_Plugin", ImageMetaHandler, LabelMetaHandler
)


class _ImplementedMetaHandlers(Generic[_Image_or_Label_Plugin]):
    """This class is a singleton that manages the available image handler plugins."""

    _instance = None
    _implemented_handlers: dict[str, Any]

    def __new__(cls):
        """Create a new instance of the class if it does not exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._implemented_handlers = {}
        return cls._instance

    def available_handlers(self) -> list[str]:
        """Get the available image handler versions.

        The versions are returned in descending order.
            such that the latest version is the first in the list and the fist to be
            checked.
        """
        return list(reversed(self._implemented_handlers.keys()))

    def find_meta_handler(
        self,
        group_handler: ZarrGroupHandler,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ) -> _Image_or_Label_Plugin:
        """Try to get a handler for the given store based on the metadata version."""
        _errors = {}

        for version, handler in reversed(self._implemented_handlers.items()):
            handler = handler(
                group_handler=group_handler,
                axes_setup=axes_setup,
                allow_non_canonical_axes=allow_non_canonical_axes,
                strict_canonical_order=strict_canonical_order,
            )
            meta = handler.safe_load_meta()
            if isinstance(meta, ValidationError):
                _errors[version] = meta
                continue
            return handler

        raise NgioValidationError(
            f"Could not load OME-Zarr metadata from any known version. "
            f"Errors: {_errors}"
        )

    def get_handler(
        self,
        version: str,
        group_handler: ZarrGroupHandler,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ) -> _Image_or_Label_Plugin:
        """Get a handler for a specific version."""
        if version not in self._implemented_handlers:
            raise NgioValueError(f"Image handler for version {version} does not exist.")
        return self._implemented_handlers[version](
            group_handler=group_handler,
            axes_setup=axes_setup,
            allow_non_canonical_axes=allow_non_canonical_axes,
            strict_canonical_order=strict_canonical_order,
        )

    def add_handler(
        self, key: str, handler: type[_Image_or_Label_Plugin], overwrite: bool = False
    ):
        """Register a new handler."""
        if key in self._implemented_handlers and not overwrite:
            raise NgioValueError(f"Image handler for version {key} already exists.")
        self._implemented_handlers[key] = handler


class ImplementedImageMetaHandlers(_ImplementedMetaHandlers[ImageMetaHandler]):
    def __init__(self):
        super().__init__()


ImplementedImageMetaHandlers().add_handler("0.4", V04ImageMetaHandler)


class ImplementedLabelMetaHandlers(_ImplementedMetaHandlers[LabelMetaHandler]):
    def __init__(self):
        super().__init__()


ImplementedLabelMetaHandlers().add_handler("0.4", V04LabelMetaHandler)


def open_image_meta_handler(
    store: StoreOrGroup,
    cache: bool = False,
    mode: AccessModeLiteral = "a",
    axes_setup: AxesSetup | None = None,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = True,
) -> ImageMetaHandler:
    """Open the metadata of an OME-Zarr image.

    Args:
        store: The Zarr store or group where the model is
        cache: Whether to cache the metadata.
        mode: The store access mode.
        axes_setup: The axes setup. This is used to map axes with
            a non-canonical name to a canonical name.
        allow_non_canonical_axes: Whether to allow non-canonical axes.
        strict_canonical_order: Whether to enforce strict canonical order.
    """
    zarr_group_handler = ZarrGroupHandler(store, cache, mode)
    return ImplementedImageMetaHandlers().find_meta_handler(
        zarr_group_handler,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )

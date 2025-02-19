"""Image handler for OME-Zarr files."""

from typing import Protocol

from pydantic import ValidationError

from ngio.ome_zarr_meta.ngio_specs import NgioImageMeta, NgioLabelMeta
from ngio.ome_zarr_meta.v04 import OmeZarrV04ImageHandler, OmeZarrV04LabelHandler
from ngio.utils import AccessModeLiteral, NgioValidationError, StoreOrGroup


class OmeZarrImageHandler(Protocol):
    """Protocol for OME-Zarr image metadata handlers."""

    def load(self, return_error: bool = False) -> NgioImageMeta | ValidationError:
        """Load the metadata from the store."""
        ...

    def write(self, meta: NgioImageMeta) -> None:
        """Write the metadata to the store."""
        ...

    def cleat_cache(self) -> None:
        """Clear the cached metadata."""
        ...


class OmeZarrLabelHandler(Protocol):
    """Protocol for OME-Zarr label metadata handlers."""

    def __init__(
        self, store: StoreOrGroup, cache: bool = False, mode: AccessModeLiteral = "a"
    ):
        """Initialize the handler."""
        ...

    def load(self, return_error: bool = False) -> NgioLabelMeta | ValidationError:
        """Load the metadata from the store."""
        ...

    def write(self, meta: NgioLabelMeta) -> None:
        """Write the metadata to the store."""
        ...

    def clean_cache(self) -> None:
        """Clear the cached metadata."""
        ...


def _get_handler(
    _implemented_handlers: dict[str, OmeZarrImageHandler | OmeZarrLabelHandler],
    store: StoreOrGroup,
    cache: bool = False,
    mode: AccessModeLiteral = "a",
) -> OmeZarrImageHandler | OmeZarrLabelHandler:
    _errors = {}
    for version, handler in _implemented_handlers.items():
        handler = handler(store, cache=cache, mode=mode)
        meta = handler.load(return_error=True)
        if isinstance(meta, ValidationError):
            _errors[version] = meta
            continue
        return handler

    raise NgioValidationError(
        f"Could not load OME-Zarr metadata from any known version. Errors: {_errors}"
    )


def get_image_handler(
    store: StoreOrGroup, cache: bool = False, mode: AccessModeLiteral = "a"
) -> OmeZarrImageHandler:
    """Try to get an image handler for the given store based on the metadata version."""
    _implemented_handlers = {
        "0.4": OmeZarrV04ImageHandler,
    }
    return _get_handler(_implemented_handlers, store, cache, mode)


def get_label_handler(
    store: StoreOrGroup, cache: bool = False, mode: AccessModeLiteral = "a"
) -> OmeZarrLabelHandler:
    """Try to get an label handler for the given store based on the metadata version."""
    _implemented_handlers = {
        "0.4": OmeZarrV04LabelHandler,
    }
    return _get_handler(_implemented_handlers, store, cache, mode)

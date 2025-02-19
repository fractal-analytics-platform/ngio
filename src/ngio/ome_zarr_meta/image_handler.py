"""Image handler for OME-Zarr files."""

from pydantic import ValidationError

from ngio.ome_zarr_meta.v04 import OmeZarrV04ImageHandler, OmeZarrV04LabelHandler
from ngio.utils import NgioValidationError


def _get_handler(_implemented_handlers: dict, store, cache=False, mode="a"):
    _errors = {}
    for version, handler in _implemented_handlers.items():
        handler = handler(store, cache=cache, mode=mode)
        meta = handler.load(return_errors=True)
        if isinstance(meta, ValidationError):
            _errors[version] = meta
            continue
        return handler

    raise NgioValidationError(
        f"Could not load OME-Zarr metadata from any known version. Errors: {_errors}"
    )


def get_image_handler(store, cache=False, mode="a"):
    """Try to get an image handler for the given store based on the metadata version."""
    _implemented_handlers = {
        "0.4": OmeZarrV04ImageHandler,
    }
    return _get_handler(_implemented_handlers, store, cache, mode)


def get_label_handler(store, cache=False, mode="a"):
    """Try to get an label handler for the given store based on the metadata version."""
    _implemented_handlers = {
        "0.4": OmeZarrV04LabelHandler,
    }
    return _get_handler(_implemented_handlers, store, cache, mode)

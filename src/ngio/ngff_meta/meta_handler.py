"""Handler for reading and writing NGFF image metadata."""

from typing import Literal

from zarr.store.common import StoreLike

from ngio.ngff_meta._meta_handler_protocol import NgffImageMetaHandler
from ngio.ngff_meta.v04.zarr_utils import (
    NgffImageMetaZarrHandlerV04,
)

_available_load_ngff_image_meta_handlers = {
    "0.4": NgffImageMetaZarrHandlerV04,
}


def find_ngff_image_meta_handler_version(store: StoreLike) -> str:
    """Find the version of the NGFF image metadata."""
    for version, handler in _available_load_ngff_image_meta_handlers.items():
        if handler.check_version(store=store):
            return version

    supported_versions = ", ".join(_available_load_ngff_image_meta_handlers.keys())
    raise ValueError(
        f"The Zarr store does not contain a supported version. \
            Supported OME-Ngff versions are: {supported_versions}."
    )


def get_ngff_image_meta_handler(
    store: StoreLike, meta_mode: Literal["image", "label"], cache: bool = False
) -> NgffImageMetaHandler:
    """Load the NGFF image metadata handler."""
    version = find_ngff_image_meta_handler_version(store)
    handler = _available_load_ngff_image_meta_handlers[version]
    return handler(store=store, meta_mode=meta_mode, cache=cache)

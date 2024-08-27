"""Handler for reading and writing NGFF image metadata."""

from typing import Literal, Protocol

from ngio.ngff_meta.fractal_image_meta import FractalImageLabelMeta
from ngio.ngff_meta.v04.zarr_utils import (
    NgffImageMetaZarrHandlerV04,
)


class NgffImageMetaHandler(Protocol):
    """Handler for NGFF image metadata."""

    def __init__(
        self, zarr_path: str, meta_mode: Literal["image", "label"], cache: bool = False
    ):
        """Initialize the handler."""
        ...

    def load_meta(self) -> FractalImageLabelMeta:
        """Load the OME-NGFF 0.4 metadata."""
        ...

    def write_meta(self, meta: FractalImageLabelMeta) -> None:
        """Write the OME-NGFF 0.4 metadata."""
        ...

    def update_cache(self, meta: FractalImageLabelMeta) -> None:
        """Update the cached metadata."""
        ...

    def clear_cache(self) -> None:
        """Clear the cached metadata."""
        ...


_available_load_ngff_image_meta_handlers = {
    "0.4": NgffImageMetaZarrHandlerV04,
}


def find_ngff_image_meta_handler_version(zarr_path: str) -> str:
    """Find the version of the NGFF image metadata."""
    for version, handler in _available_load_ngff_image_meta_handlers.items():
        if handler.check_version(zarr_path):
            return version

    supported_versions = ", ".join(_available_load_ngff_image_meta_handlers.keys())
    raise ValueError(
        f"The Zarr store does not contain a supported version. \
            Supported OME-Ngff versions are: {supported_versions}."
    )


def get_ngff_image_meta_handler(
    zarr_path: str, meta_mode: Literal["image", "label"]
) -> NgffImageMetaHandler:
    """Load the NGFF image metadata handler."""
    version = find_ngff_image_meta_handler_version(zarr_path)
    handler = _available_load_ngff_image_meta_handlers[version]
    return handler(zarr_path, meta_mode=meta_mode)

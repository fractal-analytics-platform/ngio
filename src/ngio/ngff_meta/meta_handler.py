"""Handler for reading and writing NGFF image metadata."""

from typing import Literal, Protocol

from ngio.io import AccessModeLiteral, Group, StoreOrGroup
from ngio.ngff_meta.fractal_image_meta import ImageLabelMeta
from ngio.ngff_meta.v04.zarr_utils import (
    NgffImageMetaZarrHandlerV04,
)


class NgffImageMetaHandler(Protocol):
    """Handler for NGFF image metadata."""

    def __init__(
        self,
        store: StoreOrGroup,
        meta_mode: Literal["image", "label"],
        cache: bool = False,
        mode: AccessModeLiteral = "a",
    ):
        """Initialize the handler."""
        ...

    @property
    def group(self) -> Group:
        """Return the Zarr group."""
        ...

    @property
    def store(self) -> StoreOrGroup:
        """Return the Zarr store."""
        ...

    @property
    def zarr_version(self) -> int:
        """Return the Zarr version."""
        ...

    @staticmethod
    def check_version(store: StoreOrGroup) -> bool:
        """Check if the version of the metadata is supported."""
        ...

    def load_meta(self) -> ImageLabelMeta:
        """Load the OME-NGFF 0.4 metadata."""
        ...

    def write_meta(self, meta: ImageLabelMeta) -> None:
        """Write the OME-NGFF 0.4 metadata."""
        ...

    def update_cache(self, meta: ImageLabelMeta) -> None:
        """Update the cached metadata."""
        ...

    def clear_cache(self) -> None:
        """Clear the cached metadata."""
        ...


_available_load_ngff_image_meta_handlers = {
    "0.4": NgffImageMetaZarrHandlerV04,
}


def find_ngff_image_meta_handler_version(store: StoreOrGroup) -> str:
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
    store: StoreOrGroup,
    meta_mode: Literal["image", "label"],
    version: str | None = None,
    cache: bool = False,
    mode: AccessModeLiteral = "a",
) -> NgffImageMetaHandler:
    """Load the NGFF image metadata handler."""
    if version is None:
        version = find_ngff_image_meta_handler_version(store)

    handler = _available_load_ngff_image_meta_handlers[version]
    return handler(store=store, meta_mode=meta_mode, cache=cache, mode=mode)

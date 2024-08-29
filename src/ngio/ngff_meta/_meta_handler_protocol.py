from typing import Literal, Protocol

from zarr.store.common import StoreLike

from ngio.ngff_meta.fractal_image_meta import FractalImageLabelMeta


class NgffImageMetaHandler(Protocol):
    """Handler for NGFF image metadata."""

    def __init__(
        self,
        store: StoreLike,
        meta_mode: Literal["image", "label"],
        cache: bool = False,
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

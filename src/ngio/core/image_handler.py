from zarr.store.common import StoreLike

from ngio.ngff_meta import (
    FractalImageMeta,
    get_ngff_image_meta_handler,
)


class ImageHandler:
    """A class to handle OME-NGFF images stored in Zarr format.

    This class provides methods to access image data and ROI tables.
    """

    def __init__(
        self,
        store: StoreLike,
        *,
        level: int | str | None = None,
        pixel_size: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        """Initialize the MultiscaleHandler in read mode."""
        pass

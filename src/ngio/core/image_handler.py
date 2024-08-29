from ngio.ngff_meta import (
    FractalImageMeta,
    NgffImageMetaHandler,
    get_ngff_image_meta_handler,
)


class ImageHandler:
    """A class to handle OME-NGFF images stored in Zarr format.

    This class provides methods to access image data and ROI tables.
    """

    def __init__(self, zarr_url: str, path: str | None = None, level: int = 0) -> None:
        """Initialize the MultiscaleHandler in read mode."""
        self._metadata_handler = get_ngff_image_meta_handler(zarr_url, path, level)

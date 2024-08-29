"""Generic class to handle Image-like data in a OME-NGFF file."""

from zarr.store.common import StoreLike

from ngio.ngff_meta import FractalImageMeta, PixelSize, get_ngff_image_meta_handler


class ImageLikeHandler:
    """A class to handle OME-NGFF images stored in Zarr format.

    This class provides methods to access image data and ROI tables.
    """

    def __init__(
        self,
        store: StoreLike,
        *,
        level_path: str | int | None = None,
        pixel_size: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        """Initialize the MultiscaleHandler in read mode."""
        self._metadata_handler = get_ngff_image_meta_handler(
            store=store, meta_mode="image", cache=False
        )

        if level_path is None and pixel_size is None:
            raise ValueError("Either level_path or pixel_size must be provided.")

        if level_path is not None and pixel_size is not None:
            raise ValueError("Only one of level_path or pixel_size must be provided.")

        # Find the level / resolution index
        self.level_path = self._find_level(level_path, pixel_size)

    def _find_level(
        self,
        level_path: int | str | None,
        pixel_size: tuple[float, ...] | list[float] | None,
    ) -> str:
        """Find the index of the level."""
        if pixel_size is None:
            dataset = self.metadata.get_dataset(level_path=level_path)

        else:
            dataset = self.metadata.get_dataset_from_pixel_size(pixel_size, strict=True)

        return dataset.path

    @property
    def metadata(self) -> FractalImageMeta:
        """Return the metadata of the image."""
        return self._metadata_handler.load_meta()

    @property
    def axes_names(self) -> list[str]:
        """Return the names of the axes in the image."""
        return self.metadata.axes_names

    @property
    def pixel_size(self) -> PixelSize:
        """Return the pixel resolution of the image."""
        return self.metadata.pixel_size(level_path=self.level_path)

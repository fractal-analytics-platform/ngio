"""Generic class to handle Image-like data in a OME-NGFF file."""

from zarr.store.common import StoreLike

from ngio.ngff_meta import (
    FractalImageMeta,
    PixelSize,
    get_ngff_image_meta_handler,
    SpaceUnits,
)


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
        highest_resolution: bool = False,
    ) -> None:
        """Initialize the MultiscaleHandler in read mode."""
        self._metadata_handler = get_ngff_image_meta_handler(
            store=store, meta_mode="image", cache=False
        )

        # Find the level / resolution index
        self.level_path = self._find_level(level_path, pixel_size, highest_resolution)

    def _find_level(
        self,
        level_path: int | str | None,
        pixel_size: tuple[float, ...] | list[float] | None,
        highest_resolution: bool,
    ) -> str:
        """Find the index of the level."""
        args_valid = [
            level_path is not None,
            pixel_size is not None,
            highest_resolution,
        ]

        if sum(args_valid) != 1:
            raise ValueError(
                "One and only one of level_path, pixel_size, "
                "or highest_resolution=True can be used. "
                f"Received: {level_path=}, {pixel_size=}, {highest_resolution=}"
            )
        meta = self._metadata_handler.load_meta()
        if level_path is not None:
            return meta.get_dataset(level_path).path

        if pixel_size is not None:
            return meta.get_dataset_from_pixel_size(pixel_size, strict=True).path

        return meta.get_highest_resolution_dataset().path

    @property
    def metadata(self) -> FractalImageMeta:
        """Return the metadata of the image."""
        return self._metadata_handler.load_meta()

    @property
    def axes_names(self) -> list[str]:
        """Return the names of the axes in the image."""
        return self.metadata.axes_names

    @property
    def space_axes_names(self) -> list[str]:
        """Return the names of the space axes in the image."""
        return self.metadata.space_axes_names

    @property
    def space_axes_unit(self) -> SpaceUnits:
        """Return the units of the space axes in the image."""
        return self.metadata.space_axes_unit

    @property
    def pixel_size(self) -> PixelSize:
        """Return the pixel resolution of the image."""
        return self.metadata.pixel_size(level_path=self.level_path)

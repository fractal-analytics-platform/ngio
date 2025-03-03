"""Generic class to handle Image-like data in a OME-NGFF file."""

from typing import Literal

from ngio.images.abstract_image import AbstractImage
from ngio.ome_zarr_meta import (
    ImageMetaHandler,
    ImplementedImageMetaHandlers,
    NgioImageMeta,
    PixelSize,
)
from ngio.utils import ZarrGroupHandler


class Image(AbstractImage):
    """A class to handle a single image (or level) in an OME-Zarr image.

    This class is meant to be subclassed by specific image types.
    """

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: ImageMetaHandler | None,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the omezarr file.
            meta_handler: The image metadata handler.

        """
        if meta_handler is None:
            meta_handler = ImplementedImageMetaHandlers().find_meta_handler(
                group_handler
            )
        super().__init__(
            group_handler=group_handler, path=path, meta_handler=meta_handler
        )

    def consolidate(
        self,
        order: Literal[0, 1, 2] = 1,
        mode: Literal["dask", "numpy", "coarsen"] = "dask",
    ) -> None:
        """Consolidate the label on disk."""
        super()._consolidate(order=order, mode=mode)


class ImagesContainer:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(self, group_handler: ZarrGroupHandler) -> None:
        """Initialize the LabelGroupHandler."""
        self._group_handler = group_handler
        self._meta_handler = ImplementedImageMetaHandlers().find_meta_handler(
            group_handler
        )

    def meta(self) -> NgioImageMeta:
        """Return the metadata."""
        return self._meta_handler.meta

    @property
    def levels(self) -> int:
        """Return the number of levels in the image."""
        return self._meta_handler.meta.levels

    @property
    def levels_paths(self) -> list[str]:
        """Return the paths of the levels in the image."""
        return self._meta_handler.meta.paths

    def get(
        self,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        highest_resolution: bool = True,
    ) -> Image:
        """Get an image at a specific level."""
        if path is not None or pixel_size is not None:
            highest_resolution = False
        dataset = self._meta_handler.meta.get_dataset(
            path=path, pixel_size=pixel_size, highest_resolution=highest_resolution
        )
        return Image(
            group_handler=self._group_handler,
            path=dataset.path,
            meta_handler=self._meta_handler,
        )

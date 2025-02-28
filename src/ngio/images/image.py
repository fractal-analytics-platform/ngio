"""Generic class to handle Image-like data in a OME-NGFF file."""

from ngio.images.abstract_image import Image
from ngio.ome_zarr_meta import (
    ImplementedImageMetaHandlers,
    NgioImageMeta,
    PixelSize,
)
from ngio.utils import ZarrGroupHandler


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
            meta_handler=self._meta_handler,
            path=dataset.path,
        )

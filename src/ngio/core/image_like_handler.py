"""Generic class to handle Image-like data in a OME-NGFF file."""

from typing import Literal

import zarr

from ngio.io import StoreOrGroup, open_group
from ngio.ngff_meta import (
    Dataset,
    ImageLabelMeta,
    PixelSize,
    SpaceUnits,
    get_ngff_image_meta_handler,
)


class ImageLike:
    """A class to handle OME-NGFF images stored in Zarr format.

    This class provides methods to access image data and ROI tables.
    """

    def __init__(
        self,
        store: StoreOrGroup,
        path: str | None = None,
        idx: int | None = None,
        pixel_size: PixelSize | None = None,
        highest_resolution: bool = False,
        strict: bool = True,
        meta_mode: Literal["image", "label"] = "image",
        cache: bool = True,
    ) -> None:
        """Initialize the MultiscaleHandler in read mode.

        Note: Only one of `path`, `idx`, 'pixel_size' or 'highest_resolution'
        should be provided.

        store (StoreOrGroup): The Zarr store or group containing the image data.
        path (str | None): The path to the level.
        idx (int | None): The index of the level.
        pixel_size (PixelSize | None): The pixel size of the level.
        highest_resolution (bool): Whether to get the highest resolution level.
        strict (bool): Whether to raise an error where a pixel size is not found
            to match the requested "pixel_size".
        meta_mode (str): The mode of the metadata handler.
        cache (bool): Whether to cache the metadata.
        """
        if not isinstance(store, zarr.Group):
            store = open_group(store=store, mode="r")

        self._metadata_handler = get_ngff_image_meta_handler(
            store=store, meta_mode=meta_mode, cache=cache
        )

        # Find the level / resolution index
        metadata = self._metadata_handler.load_meta()
        self._dataset = metadata.get_dataset(
            path=path,
            idx=idx,
            pixel_size=pixel_size,
            highest_resolution=highest_resolution,
            strict=strict,
        )
        self._group = store

    @property
    def group(self) -> zarr.Group:
        """Return the Zarr group containing the image data."""
        return self._group

    @property
    def metadata(self) -> ImageLabelMeta:
        """Return the metadata of the image."""
        return self._metadata_handler.load_meta()

    @property
    def dataset(self) -> Dataset:
        """Return the dataset of the image."""
        return self._dataset

    @property
    def path(self) -> str:
        """Return the path of the dataset (relative to the root group)."""
        return self.dataset.path

    @property
    def channel_labels(self) -> list[str]:
        """Return the names of the channels in the image."""
        return self.dataset.path

    @property
    def axes_names(self) -> list[str]:
        """Return the names of the axes in the image."""
        return self.dataset.axes_names

    @property
    def space_axes_names(self) -> list[str]:
        """Return the names of the space axes in the image."""
        return self.dataset.space_axes_names

    @property
    def space_axes_unit(self) -> SpaceUnits:
        """Return the units of the space axes in the image."""
        return self.dataset.space_axes_unit

    @property
    def pixel_size(self) -> PixelSize:
        """Return the pixel resolution of the image."""
        return self.dataset.pixel_size

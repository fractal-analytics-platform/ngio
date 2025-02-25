"""Generic class to handle Image-like data in a OME-NGFF file."""

import numpy as np
import zarr

from ngio.common import Dimensions
from ngio.ome_zarr_meta import (
    BaseOmeZarrImageHandler,
    Dataset,
    PixelSize,
)
from ngio.utils import (
    NgioFileExistsError,
)


class Image:
    """A class to handle OME-NGFF images stored in Zarr format.

    This class provides methods to access image data and ROI tables.
    """

    def __init__(
        self,
        ome_zarr_handler: BaseOmeZarrImageHandler,
        path: str,
    ) -> None:
        """Initialize the MultiscaleHandler in read mode.

        Args:
            ome_zarr_handler (BaseOmeZarrImageHandler): The OME-Zarr image handler.
            path (str): The path to the image in the Zarr group.
        """
        self._path = path
        self._ome_zarr_handler = ome_zarr_handler

        self._dataset = self._ome_zarr_handler.meta.get_dataset(path=path)
        self._pixel_size = self._dataset.pixel_size

        try:
            self._zarr_array = self._ome_zarr_handler.group_handler.get_array(path)
        except NgioFileExistsError as e:
            raise NgioFileExistsError(f"Could not find the dataset at {path}.") from e

        self._dimensions = Dimensions(
            shape=self._zarr_array.shape, axes_mapper=self._dataset.axes_mapper
        )

        self._axer_mapper = self._dataset.axes_mapper

    def __repr__(self) -> str:
        """Return a string representation of the image."""
        return f"Image(path={self.path}, {self.dimensions})"

    @property
    def zarr_array(self) -> zarr.Array:
        """Return the Zarr array."""
        return self._zarr_array

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the image."""
        return self.zarr_array.shape

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype of the image."""
        return self.zarr_array.dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        """Return the chunks of the image."""
        return self.zarr_array.chunks

    @property
    def dimensions(self) -> Dimensions:
        """Return the dimensions of the image."""
        return self._dimensions

    @property
    def pixel_size(self) -> PixelSize:
        """Return the pixel size of the image."""
        return self._pixel_size

    @property
    def dataset(self) -> Dataset:
        """Return the dataset of the image."""
        return self._dataset

    @property
    def path(self) -> str:
        """Return the path of the image."""
        return self._dataset.path

    def consolidate(self) -> None:
        """Consolidate the Zarr array."""
        raise NotImplementedError

"""Generic class to handle Image-like data in a OME-NGFF file."""

# %%
from collections.abc import Collection, Iterable
from typing import Literal

import numpy as np
import zarr

from ngio.common import (
    ArrayLike,
    Dimensions,
    WorldCooROI,
    consolidate_pyramid,
    get_pipe,
    set_pipe,
)
from ngio.ome_zarr_meta import (
    Dataset,
    ImageMetaHandler,
    LabelMetaHandler,
    PixelSize,
)
from ngio.utils import (
    NgioFileExistsError,
    ZarrGroupHandler,
)


class AbstractImage:
    """A class to handle a single image (or level) in an OME-Zarr image.

    This class is meant to be subclassed by specific image types.
    """

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: ImageMetaHandler | LabelMetaHandler,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the omezarr file.
            meta_handler: The image metadata handler.

        """
        self._path = path
        self._group_handler = group_handler
        self._meta_handler = meta_handler

        self._dataset = self._meta_handler.meta.get_dataset(path=path)
        self._pixel_size = self._dataset.pixel_size

        try:
            self._zarr_array = self._group_handler.get_array(self._dataset.path)
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

    def get_array(
        self,
        axes_order: Collection[str] | None = None,
        mode: Literal["numpy", "dask", "delayed"] = "numpy",
        **slice_kwargs: slice | int | Iterable[int],
    ) -> ArrayLike:
        """Get a slice of the image.

        Args:
            axes_order: The order of the axes to return the array.
            mode: The mode to return the array.
            **slice_kwargs: The slices to get the array.

        Returns:
            The array of the region of interest.
        """
        return get_pipe(
            array=self.zarr_array,
            dimensions=self.dimensions,
            axes_order=axes_order,
            mode=mode,
            **slice_kwargs,
        )

    def get_roi(
        self,
        roi: WorldCooROI,
        axes_order: Collection[str] | None = None,
        mode: Literal["numpy", "dask", "delayed"] = "numpy",
        **slice_kwargs: slice | int | Iterable[int],
    ) -> ArrayLike:
        """Get a slice of the image.

        Args:
            roi: The region of interest to get the array.
            axes_order: The order of the axes to return the array.
            mode: The mode to return the array.
            **slice_kwargs: The slices to get the array.

        Returns:
            The array of the region of interest.
        """
        raster_roi = roi.to_raster_coo(
            pixel_size=self.pixel_size, dimensions=self.dimensions
        ).to_slices()

        for key in slice_kwargs.keys():
            if key in raster_roi:
                raise ValueError(
                    f"Key {key} is already in the slice_kwargs. "
                    "Ambiguous which one to use: "
                    f"{key}={slice_kwargs[key]} or roi_{key}={raster_roi[key]}"
                )
        return self.get_array(
            axes_order=axes_order, mode=mode, **raster_roi, **slice_kwargs
        )

    def set_array(
        self,
        patch: ArrayLike,
        axes_order: Collection[str] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> None:
        """Set a slice of the image.

        Args:
            patch: The patch to set.
            axes_order: The order of the axes to set the patch.
            **slice_kwargs: The slices to set the patch.

        """
        set_pipe(
            array=self.zarr_array,
            patch=patch,
            dimensions=self.dimensions,
            axes_order=axes_order,
            **slice_kwargs,
        )

    def set_roi(
        self,
        roi: WorldCooROI,
        patch: ArrayLike,
        axes_order: Collection[str] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> None:
        """Set a slice of the image.

        Args:
            roi: The region of interest to set the patch.
            patch: The patch to set.
            axes_order: The order of the axes to set the patch.
            **slice_kwargs: The slices to set the patch.

        """
        raster_roi = roi.to_raster_coo(
            pixel_size=self.pixel_size, dimensions=self.dimensions
        ).to_slices()

        for key in slice_kwargs.keys():
            if key in raster_roi:
                raise ValueError(
                    f"Key {key} is already in the slice_kwargs. "
                    "Ambiguous which one to use: "
                    f"{key}={slice_kwargs[key]} or roi_{key}={raster_roi[key]}"
                )
        self.set_array(patch=patch, axes_order=axes_order, **raster_roi, **slice_kwargs)

    def _consolidate(
        self,
        order: Literal[0, 1, 2],
        mode: Literal["dask", "numpy", "coarsen"],
    ) -> None:
        """Consolidate the Zarr array."""
        target_paths = self._meta_handler.meta.paths
        targets = [
            self._group_handler.get_array(path)
            for path in target_paths
            if path != self.path
        ]
        consolidate_pyramid(
            source=self.zarr_array, targets=targets, order=order, mode=mode
        )

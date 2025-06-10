"""Generic class to handle Image-like data in a OME-NGFF file."""

from collections.abc import Collection, Iterable
from typing import Generic, Literal, TypeVar

import zarr

from ngio.common import (
    ArrayLike,
    Dimensions,
    Roi,
    RoiPixels,
    consolidate_pyramid,
    get_pipe,
    roi_to_slice_kwargs,
    set_pipe,
)
from ngio.ome_zarr_meta import (
    AxesMapper,
    Dataset,
    ImageMetaHandler,
    LabelMetaHandler,
    PixelSize,
)
from ngio.tables import RoiTable
from ngio.utils import NgioFileExistsError, ZarrGroupHandler

_image_handler = TypeVar("_image_handler", ImageMetaHandler, LabelMetaHandler)


class AbstractImage(Generic[_image_handler]):
    """A class to handle a single image (or level) in an OME-Zarr image.

    This class is meant to be subclassed by specific image types.
    """

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: _image_handler,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the ome_zarr file.
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

        self._axes_mapper = self._dataset.axes_mapper

    def __repr__(self) -> str:
        """Return a string representation of the image."""
        return f"Image(path={self.path}, {self.dimensions})"

    @property
    def meta_handler(self) -> _image_handler:
        """Return the metadata."""
        return self._meta_handler

    @property
    def zarr_array(self) -> zarr.Array:
        """Return the Zarr array."""
        return self._zarr_array

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the image."""
        return self.zarr_array.shape

    @property
    def dtype(self) -> str:
        """Return the dtype of the image."""
        return str(self.zarr_array.dtype)

    @property
    def chunks(self) -> tuple[int, ...]:
        """Return the chunks of the image."""
        return self.zarr_array.chunks

    @property
    def dimensions(self) -> Dimensions:
        """Return the dimensions of the image."""
        return self._dimensions

    @property
    def axes_mapper(self) -> AxesMapper:
        """Return the axes mapper of the image."""
        return self._axes_mapper

    @property
    def is_3d(self) -> bool:
        """Return True if the image is 3D."""
        return self.dimensions.is_3d

    @property
    def is_2d(self) -> bool:
        """Return True if the image is 2D."""
        return self.dimensions.is_2d

    @property
    def is_time_series(self) -> bool:
        """Return True if the image is a time series."""
        return self.dimensions.is_time_series

    @property
    def is_2d_time_series(self) -> bool:
        """Return True if the image is a 2D time series."""
        return self.dimensions.is_2d_time_series

    @property
    def is_3d_time_series(self) -> bool:
        """Return True if the image is a 3D time series."""
        return self.dimensions.is_3d_time_series

    @property
    def is_multi_channels(self) -> bool:
        """Return True if the image is multichannel."""
        return self.dimensions.is_multi_channels

    @property
    def space_unit(self) -> str | None:
        """Return the space unit of the image."""
        return self.meta_handler.meta.space_unit

    @property
    def time_unit(self) -> str | None:
        """Return the time unit of the image."""
        return self.meta_handler.meta.time_unit

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

    def has_axis(self, axis: str) -> bool:
        """Return True if the image has the given axis."""
        self.axes_mapper.get_index("x")
        return self.dimensions.has_axis(axis)

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
        roi: Roi,
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
        return get_roi_pipe(
            image=self, roi=roi, axes_order=axes_order, mode=mode, **slice_kwargs
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
        roi: Roi,
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
        return set_roi_pipe(
            image=self, roi=roi, patch=patch, axes_order=axes_order, **slice_kwargs
        )

    def build_image_roi_table(self, name: str = "image") -> RoiTable:
        """Build the ROI table for an image."""
        return build_image_roi_table(image=self, name=name)


def consolidate_image(
    image: AbstractImage,
    order: Literal[0, 1, 2] = 1,
    mode: Literal["dask", "numpy", "coarsen"] = "dask",
) -> None:
    """Consolidate the image on disk."""
    target_paths = image._meta_handler.meta.paths
    targets = [
        image._group_handler.get_array(path)
        for path in target_paths
        if path != image.path
    ]
    consolidate_pyramid(
        source=image.zarr_array, targets=targets, order=order, mode=mode
    )


def get_roi_pipe(
    image: AbstractImage,
    roi: Roi,
    axes_order: Collection[str] | None = None,
    mode: Literal["numpy", "dask", "delayed"] = "numpy",
    **slice_kwargs: slice | int | Iterable[int],
) -> ArrayLike:
    """Get a slice of the image.

    Args:
        image: The image to get the ROI.
        roi: The region of interest to get the array.
        axes_order: The order of the axes to return the array.
        mode: The mode to return the array.
        **slice_kwargs: The slices to get the array.

    Returns:
        The array of the region of interest.
    """
    slice_kwargs = roi_to_slice_kwargs(
        roi=roi,
        pixel_size=image.pixel_size,
        dimensions=image.dimensions,
        **slice_kwargs,
    )
    return get_pipe(
        array=image.zarr_array,
        dimensions=image.dimensions,
        axes_order=axes_order,
        mode=mode,
        **slice_kwargs,
    )


def set_roi_pipe(
    image: AbstractImage,
    roi: Roi,
    patch: ArrayLike,
    axes_order: Collection[str] | None = None,
    **slice_kwargs: slice | int | Iterable[int],
) -> None:
    """Set a slice of the image.

    Args:
        image: The image to set the ROI.
        roi: The region of interest to set the patch.
        patch: The patch to set.
        axes_order: The order of the axes to set the patch.
        **slice_kwargs: The slices to set the patch.

    """
    slice_kwargs = roi_to_slice_kwargs(
        roi=roi,
        pixel_size=image.pixel_size,
        dimensions=image.dimensions,
        **slice_kwargs,
    )
    set_pipe(
        array=image.zarr_array,
        patch=patch,
        dimensions=image.dimensions,
        axes_order=axes_order,
        **slice_kwargs,
    )


def build_image_roi_table(image: AbstractImage, name: str = "image") -> RoiTable:
    """Build the ROI table for an image."""
    dim_z, dim_y, dim_x = (
        image.dimensions.get("z", strict=False),
        image.dimensions.get("y"),
        image.dimensions.get("x"),
    )
    image_roi = RoiPixels(
        name=name,
        x=0,
        y=0,
        z=0,
        x_length=dim_x,
        y_length=dim_y,
        z_length=dim_z,
    )
    return RoiTable(rois=[image_roi.to_roi(pixel_size=image.pixel_size)])

from collections.abc import Callable, Collection, Generator

import dask.array as da
import numpy as np

from ngio import PixelSize
from ngio.common import (
    Dimensions,
    Roi,
    add_channel_label_to_slice_kwargs,
    roi_to_slice_kwargs,
)
from ngio.common._io_transforms import TransformProtocol
from ngio.experimental.iterators._abstract_iterator import AbstractIteratorFactory
from ngio.experimental.iterators._read_and_write import (
    DaskReader,
    DaskWriter,
    NumpyReader,
    NumpyWriter,
    build_dask_reader,
    build_dask_writer,
    build_numpy_reader,
    build_numpy_writer,
)
from ngio.experimental.iterators._rois_handler import RoisHandler
from ngio.images import Image, Label
from ngio.tables import RoiTable
from ngio.utils._errors import NgioValidationError


def make_unique_label_np(x: np.ndarray, p: int, n: int) -> np.ndarray:
    """Make a unique label for the patch."""
    x = np.where(x > 0, (1 + p - n) + x * n, 0)
    return x


def make_unique_label_da(x: da.Array, p: int, n: int) -> da.Array:
    """Make a unique label for the patch."""
    x = da.where(x > 0, (1 + p - n) + x * n, 0)
    return x


class SegmentationIterator(AbstractIteratorFactory):
    """Base class for iterators over ROIs."""

    def __init__(
        self,
        input_image: Image,
        output_label: Label,
        roi_base: RoiTable | list[Roi] | None = None,
    ) -> None:
        """Initialize the iterator with a ROI table and input/output images.

        Args:
            input_image (Image): The input image to be used as input for the
                segmentation.
            output_label (Label): The label image where the ROIs will be written.
            roi_base (RoiTable | None): Optional table containing ROI definitions.
        """
        self._input = input_image
        self._output = output_label
        self._set_rois_handler(RoisHandler(ref_image=input_image, rois_base=roi_base))

        # Check compatibility between input and output images
        if not self._input.dimensions.is_compatible_with(self._output.dimensions):
            raise NgioValidationError(
                "Input image and output label have incompatible dimensions. "
                f"Input: {self._input.dimensions}, Output: {self._output.dimensions}."
            )

    def __repr__(self) -> str:
        return f"SegmentationIterator(regions={len(self._rois_handler.rois)})"

    def get_init_kwargs(self) -> dict:
        """Return the initialization arguments for the iterator."""
        return {
            "input_image": self._input,
            "output_label": self._output,
            "roi_base": self._rois_handler.rois,
        }

    def by_masking_label(
        self,
        masking_label: Label,
        masking_roi: RoiTable | None = None,
    ) -> "SegmentationIterator":
        """Return a new iterator that iterates over ROIs by masking a label."""
        self._masking_label = masking_label
        if masking_roi is not None:
            rois = masking_roi.rois()
        else:
            rois = masking_label.build_masking_roi_table().rois()
        rois_handler = self._rois_handler.product(rois)
        return self._new_from_rois_handler(rois_handler)

    def _setup_slice_kwargs(
        self,
        roi: Roi,
        dimensions: Dimensions,
        pixel_size: PixelSize,
        channel_label: str | None = None,
        channel_idx: int | None = None,
        **slice_kwargs: slice | int | Collection[int],
    ) -> dict[str, slice | int | Collection[int]]:
        """Prepare slice kwargs for the ROI."""
        input_slice_kwargs = roi_to_slice_kwargs(
            roi=roi,
            dimensions=dimensions,
            pixel_size=pixel_size,
            **slice_kwargs,
        )
        input_slice_kwargs = add_channel_label_to_slice_kwargs(
            channel_idx=channel_idx,
            channel_label=channel_label,
            **input_slice_kwargs,
        )
        return input_slice_kwargs

    def _iter_as_numpy(
        self,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        input_transforms: Collection[TransformProtocol] | None = None,
        output_transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Collection[int],
    ) -> Generator[tuple[NumpyReader, NumpyWriter]]:
        """Create an iterator over the pixels of the ROIs.

        Args:
            channel_label: Select a specific channel by label.
                If None, all channels are returned.
                Alternatively, you can slice arbitrary channels
                using the slice_kwargs (c=[0, 2]).
            axes_order: The order of the axes to return the array.
            input_transforms: The transforms to apply to the input array.
            output_transforms: The transforms to apply to the output array.
            **slice_kwargs: The slices to get the array.

        Returns:
            RoiPixels: An iterator over the pixels of the ROIs.
        """
        for roi in self._rois_handler.rois:
            input_slice_kwargs = self._setup_slice_kwargs(
                roi=roi,
                dimensions=self._input.dimensions,
                pixel_size=self._input.pixel_size,
                channel_label=channel_label,
                channel_idx=self._input.get_channel_idx(channel_label=channel_label),
                **slice_kwargs,
            )

            output_slice_kwargs = self._setup_slice_kwargs(
                roi=roi,
                dimensions=self._output.dimensions,
                pixel_size=self._output.pixel_size,
                channel_label=None,
                channel_idx=None,
                **slice_kwargs,
            )

            reader = build_numpy_reader(
                array=self._input.zarr_array,
                dimensions=self._input.dimensions,
                axes_order=axes_order,
                transforms=input_transforms,
                **input_slice_kwargs,
            )
            writer = build_numpy_writer(
                array=self._output.zarr_array,
                dimensions=self._output.dimensions,
                axes_order=axes_order,
                transforms=output_transforms,
                **output_slice_kwargs,
            )

            yield (reader, writer)

        self._output.consolidate()

    def iter_as_numpy(
        self,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        input_transforms: Collection[TransformProtocol] | None = None,
        output_transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Collection[int],
    ) -> Generator[tuple[np.ndarray, NumpyWriter]]:
        """Create an iterator over the pixels of the ROIs as NumPy arrays.

        Args:
            channel_label: Select a specific channel by label.
                If None, all channels are returned.
                Alternatively, you can slice arbitrary channels
                using the slice_kwargs (c=[0, 2]).
            axes_order: The order of the axes to return the array.
            input_transforms: The transforms to apply to the input array.
            output_transforms: The transforms to apply to the output array.
            **slice_kwargs: The slices to get the array.

        Returns:
            RoiPixels: An iterator over the pixels of the ROIs as NumPy arrays.
        """
        for reader, writer in self._iter_as_numpy(
            channel_label=channel_label,
            axes_order=axes_order,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
            **slice_kwargs,
        ):
            yield (reader(), writer)

    def _iter_as_dask(
        self,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        input_transforms: Collection[TransformProtocol] | None = None,
        output_transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Collection[int],
    ) -> Generator[tuple[DaskReader, DaskWriter]]:
        """Create an iterator over the pixels of the ROIs as Dask arrays.

        Args:
            channel_label: Select a specific channel by label.
                If None, all channels are returned.
                Alternatively, you can slice arbitrary channels
                using the slice_kwargs (c=[0, 2]).
            axes_order: The order of the axes to return the array.
            input_transforms: The transforms to apply to the input array.
            output_transforms: The transforms to apply to the output array.
            **slice_kwargs: The slices to get the array.

        Returns:
            RoiPixels: An iterator over the pixels of the ROIs as Dask arrays.
        """
        for roi in self._rois_handler.rois:
            input_slice_kwargs = roi_to_slice_kwargs(
                roi=roi,
                dimensions=self._input.dimensions,
                pixel_size=self._input.pixel_size,
                **slice_kwargs,
            )
            input_slice_kwargs = add_channel_label_to_slice_kwargs(
                channel_idx=self._input.get_channel_idx(channel_label=channel_label),
                channel_label=channel_label,
                **input_slice_kwargs,
            )

            output_slice_kwargs = roi_to_slice_kwargs(
                roi=roi,
                dimensions=self._output.dimensions,
                pixel_size=self._output.pixel_size,
                **slice_kwargs,
            )

            reader = build_dask_reader(
                array=self._input.zarr_array,
                dimensions=self._input.dimensions,
                axes_order=axes_order,
                transforms=input_transforms,
                **input_slice_kwargs,
            )
            writer = build_dask_writer(
                array=self._output.zarr_array,
                dimensions=self._output.dimensions,
                axes_order=axes_order,
                transforms=output_transforms,
                **output_slice_kwargs,
            )

            yield (reader, writer)

        self._output.consolidate()

    def iter_as_dask(
        self,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        input_transforms: Collection[TransformProtocol] | None = None,
        output_transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Collection[int],
    ) -> Generator[tuple[da.Array, DaskWriter]]:
        """Create an iterator over the pixels of the ROIs as Dask arrays.

        Args:
            channel_label: Select a specific channel by label.
                If None, all channels are returned.
                Alternatively, you can slice arbitrary channels
                using the slice_kwargs (c=[0, 2]).
            axes_order: The order of the axes to return the array.
            input_transforms: The transforms to apply to the input array.
            output_transforms: The transforms to apply to the output array.
            **slice_kwargs: The slices to get the array.
        """
        for reader, writer in self._iter_as_dask(
            channel_label=channel_label,
            axes_order=axes_order,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
            **slice_kwargs,
        ):
            yield (reader(), writer)

    def map_as_numpy(
        self,
        func: Callable,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        input_transforms: Collection[TransformProtocol] | None = None,
        output_transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Collection[int],
    ) -> Label:
        """Apply a transformation function to the ROI pixels."""
        for reader, writer in self._iter_as_numpy(
            channel_label=channel_label,
            axes_order=axes_order,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
            **slice_kwargs,
        ):
            input_patch = reader()
            transformed_patch = func(input_patch)
            writer(transformed_patch)
        return self._output

    def map_as_dask(
        self,
        func: Callable,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        input_transforms: Collection[TransformProtocol] | None = None,
        output_transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Collection[int],
    ) -> Label:
        """Apply a transformation function to the ROI pixels."""
        for reader, writer in self._iter_as_dask(
            channel_label=channel_label,
            axes_order=axes_order,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
            **slice_kwargs,
        ):
            input_patch = reader()
            transformed_patch = func(input_patch)
            writer(transformed_patch)
        return self._output

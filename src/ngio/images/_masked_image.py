"""A module for handling label images in OME-NGFF files."""

from collections.abc import Collection, Iterable
from typing import Literal

import dask.array as da
import numpy as np
from dask.delayed import Delayed

from ngio.common import (
    ArrayLike,
    TransformProtocol,
    get_masked_as_dask,
    get_masked_as_numpy,
    roi_to_slice_kwargs,
    set_dask_masked,
    set_numpy_masked,
)
from ngio.images._image import Image
from ngio.images._label import Label
from ngio.ome_zarr_meta import ImageMetaHandler, LabelMetaHandler
from ngio.tables import MaskingRoiTable
from ngio.utils import (
    ZarrGroupHandler,
)


class MaskedImage(Image):
    """Placeholder class for a label."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: ImageMetaHandler | None,
        label: Label,
        masking_roi_table: MaskingRoiTable,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the ome_zarr file.
            meta_handler: The image metadata handler.
            label: The label image.
            masking_roi_table: The masking ROI table.

        """
        super().__init__(
            group_handler=group_handler, path=path, meta_handler=meta_handler
        )
        self._label = label
        self._masking_roi_table = masking_roi_table

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        label_name = self._label.meta.name
        if label_name is None:
            label_name = self._masking_roi_table.reference_label
        return f"MaskedImage(path={self.path}, {self.dimensions}, {label_name})"

    def get_roi_as_numpy( # type: ignore (this ignore the method override issue)
        self,
        label: int,
        channel_label: str | None = None,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> ArrayLike:
        """Return the array for a given ROI."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi_as_numpy(
            roi=roi,
            channel_label=channel_label,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    def get_roi_as_dask( # type: ignore (this ignore the method override issue)
        self,
        label: int,
        channel_label: str | None = None,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> da.Array:
        """Return the array for a given ROI as a Dask array."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi_as_dask(
            roi=roi,
            channel_label=channel_label,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    def get_roi_as_delayed( # type: ignore (this ignore the method override issue)
        self,
        label: int,
        channel_label: str | None = None,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> Delayed:
        """Return the array for a given ROI as a delayed object."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi_as_delayed(
            roi=roi,
            channel_label=channel_label,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    def get_roi( # type: ignore (this ignore the method override issue)
        self,
        label: int,
        zoom_factor: float = 1.0,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        mode: Literal["numpy", "dask", "delayed"] = "numpy",
        **slice_kwargs: slice | int | Iterable[int],
    ) -> ArrayLike:
        """Return the array for a given ROI."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi(
            roi=roi,
            channel_label=channel_label,
            axes_order=axes_order,
            transforms=transforms,
            mode=mode,
            **slice_kwargs,
        )

    def set_roi( # type: ignore (this ignore the method override issue)
        self,
        label: int,
        patch: ArrayLike,
        zoom_factor: float = 1.0,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> None:
        """Set the array for a given ROI."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        return super().set_roi(
            roi=roi,
            patch=patch,
            channel_label=channel_label,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    def _build_slice_kwargs(
        self,
        label: int,
        channel_label: str | None = None,
        zoom_factor: float = 1.0,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> dict[str, slice | int | Iterable[int]]:
        """Build the slice kwargs for the ROI."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        slice_kwargs = roi_to_slice_kwargs(
            roi=roi,
            pixel_size=self.pixel_size,
            dimensions=self.dimensions,
            **slice_kwargs,
        )
        slice_kwargs = self._add_channel_label(
            channel_label=channel_label, **slice_kwargs
        )
        return slice_kwargs

    def get_roi_masked_as_numpy(
        self,
        label: int,
        channel_label: str | None = None,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> np.ndarray:
        """Return the masked array for a given label as a NumPy array."""
        slice_kwargs = self._build_slice_kwargs(
            channel_label=channel_label,
            label=label,
            zoom_factor=zoom_factor,
            **slice_kwargs,
        )
        return get_masked_as_numpy(
            array=self.zarr_array,
            label_array=self._label.zarr_array,
            label=label,
            dimensions_array=self.dimensions,
            dimensions_label=self._label.dimensions,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    def get_roi_masked_as_dask(
        self,
        label: int,
        channel_label: str | None = None,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> da.Array:
        """Return the masked array for a given label as a Dask array."""
        slice_kwargs = self._build_slice_kwargs(
            channel_label=channel_label,
            label=label,
            zoom_factor=zoom_factor,
            **slice_kwargs,
        )
        return get_masked_as_dask(
            array=self.zarr_array,
            label_array=self._label.zarr_array,
            label=label,
            dimensions_array=self.dimensions,
            dimensions_label=self._label.dimensions,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    def get_roi_masked(
        self,
        label: int,
        channel_label: str | None = None,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        **slice_kwargs: slice | int | Iterable[int],
    ) -> ArrayLike:
        """Return the masked array for a given label."""
        if mode == "numpy":
            return self.get_roi_masked_as_numpy(
                label=label,
                channel_label=channel_label,
                zoom_factor=zoom_factor,
                axes_order=axes_order,
                transforms=transforms,
                **slice_kwargs,
            )
        elif mode == "dask":
            return self.get_roi_masked_as_dask(
                label=label,
                channel_label=channel_label,
                zoom_factor=zoom_factor,
                axes_order=axes_order,
                transforms=transforms,
                **slice_kwargs,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def set_roi_masked(
        self,
        label: int,
        patch: ArrayLike,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        zoom_factor: float = 1.0,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> None:
        """Set the masked array for a given label."""
        slice_kwargs = self._build_slice_kwargs(
            label=label,
            channel_label=channel_label,
            zoom_factor=zoom_factor,
            **slice_kwargs,
        )
        if isinstance(patch, da.Array):
            set_dask_masked(
                array=self.zarr_array,
                label_array=self._label.zarr_array,
                label=label,
                patch=patch,
                dimensions_array=self.dimensions,
                dimensions_label=self._label.dimensions,
                axes_order=axes_order,
                transforms=transforms,
                **slice_kwargs,
            )
        elif isinstance(patch, np.ndarray):
            set_numpy_masked(
                array=self.zarr_array,
                label_array=self._label.zarr_array,
                label=label,
                patch=patch,
                dimensions_array=self.dimensions,
                dimensions_label=self._label.dimensions,
                axes_order=axes_order,
                transforms=transforms,
                **slice_kwargs,
            )
        else:
            raise TypeError(
                f"Unsupported patch type: {type(patch)}. "
                "Expected numpy.ndarray or dask.array.Array."
            )


class MaskedLabel(Label):
    """Placeholder class for a label."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: LabelMetaHandler | None,
        label: Label,
        masking_roi_table: MaskingRoiTable,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the ome_zarr file.
            meta_handler: The image metadata handler.
            label: The label image.
            masking_roi_table: The masking ROI table.

        """
        super().__init__(
            group_handler=group_handler, path=path, meta_handler=meta_handler
        )
        self._label = label
        self._masking_roi_table = masking_roi_table

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        label_name = self._label.meta.name
        if label_name is None:
            label_name = self._masking_roi_table.reference_label
        return f"MaskedLabel(path={self.path}, {self.dimensions}, {label_name})"

    def get_roi_as_numpy(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> np.ndarray:
        """Return the ROI as a NumPy array."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi_as_numpy(
            roi=roi,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    def get_roi_as_dask(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> da.Array:
        """Return the ROI as a Dask array."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi_as_dask(
            roi=roi,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    def get_roi_as_delayed(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> Delayed:
        """Return the ROI as a delayed object."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi_as_delayed(
            roi=roi,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    def get_roi(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        mode: Literal["numpy", "dask", "delayed"] = "numpy",
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> ArrayLike:
        """Return the array for a given ROI."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi(
            roi=roi,
            axes_order=axes_order,
            mode=mode,
            transforms=transforms,
            **slice_kwargs,
        )

    def set_roi(
        self,
        label: int,
        patch: ArrayLike,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> None:
        """Set the array for a given ROI."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        return super().set_roi(
            roi=roi,
            patch=patch,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    def _build_slice_kwargs(
        self,
        label: int,
        zoom_factor: float = 1.0,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> dict[str, slice | int | Iterable[int]]:
        """Build the slice kwargs for the ROI."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        slice_kwargs = roi_to_slice_kwargs(
            roi=roi,
            pixel_size=self.pixel_size,
            dimensions=self.dimensions,
            **slice_kwargs,
        )
        return slice_kwargs

    def get_roi_masked_as_numpy(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> np.ndarray:
        """Return the masked array for a given label as a NumPy array."""
        slice_kwargs = self._build_slice_kwargs(
            label=label,
            zoom_factor=zoom_factor,
            **slice_kwargs,
        )
        return get_masked_as_numpy(
            array=self.zarr_array,
            label_array=self._label.zarr_array,
            label=label,
            dimensions_array=self.dimensions,
            dimensions_label=self._label.dimensions,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    def get_roi_masked_as_dask(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> da.Array:
        """Return the masked array for a given label as a Dask array."""
        slice_kwargs = self._build_slice_kwargs(
            label=label,
            zoom_factor=zoom_factor,
            **slice_kwargs,
        )
        return get_masked_as_dask(
            array=self.zarr_array,
            label_array=self._label.zarr_array,
            label=label,
            dimensions_array=self.dimensions,
            dimensions_label=self._label.dimensions,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    def get_roi_masked(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> ArrayLike:
        """Return the masked array for a given label."""
        slice_kwargs = self._build_slice_kwargs(
            label=label,
            zoom_factor=zoom_factor,
            **slice_kwargs,
        )
        if mode == "numpy":
            return self.get_roi_masked_as_numpy(
                label=label,
                zoom_factor=zoom_factor,
                axes_order=axes_order,
                transforms=transforms,
                **slice_kwargs,
            )
        elif mode == "dask":
            return self.get_roi_masked_as_dask(
                label=label,
                zoom_factor=zoom_factor,
                axes_order=axes_order,
                transforms=transforms,
                **slice_kwargs,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def set_roi_masked(
        self,
        label: int,
        patch: ArrayLike,
        axes_order: Collection[str] | None = None,
        zoom_factor: float = 1.0,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> None:
        """Set the masked array for a given label."""
        slice_kwargs = self._build_slice_kwargs(
            label=label,
            zoom_factor=zoom_factor,
            **slice_kwargs,
        )
        if isinstance(patch, da.Array):
            set_dask_masked(
                array=self.zarr_array,
                label_array=self._label.zarr_array,
                label=label,
                patch=patch,
                dimensions_array=self.dimensions,
                dimensions_label=self._label.dimensions,
                axes_order=axes_order,
                transforms=transforms,
                **slice_kwargs,
            )
        elif isinstance(patch, np.ndarray):
            set_numpy_masked(
                array=self.zarr_array,
                label_array=self._label.zarr_array,
                label=label,
                patch=patch,
                dimensions_array=self.dimensions,
                dimensions_label=self._label.dimensions,
                axes_order=axes_order,
                transforms=transforms,
                **slice_kwargs,
            )
        else:
            raise TypeError(
                f"Unsupported patch type: {type(patch)}. "
                "Expected numpy.ndarray or dask.array.Array."
            )

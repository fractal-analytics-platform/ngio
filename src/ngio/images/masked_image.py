"""A module for handling label images in OME-NGFF files."""

from collections.abc import Collection, Iterable
from typing import Literal

from ngio.common import ArrayLike, get_masked_pipe, roi_to_slice_kwargs, set_masked_pipe
from ngio.images.image import Image
from ngio.images.label import Label
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

    def get_roi(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        mode: Literal["numpy", "dask", "delayed"] = "numpy",
        **slice_kwargs: slice | int | Iterable[int],
    ) -> ArrayLike:
        """Return the array for a given ROI."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi(
            roi=roi, axes_order=axes_order, mode=mode, **slice_kwargs
        )

    def set_roi(
        self,
        label: int,
        patch: ArrayLike,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> None:
        """Set the array for a given ROI."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        return super().set_roi(
            roi=roi, patch=patch, axes_order=axes_order, **slice_kwargs
        )

    def get_roi_masked(
        self,
        label: int,
        axes_order: Collection[str] | None = None,
        mode: Literal["numpy", "dask", "delayed"] = "numpy",
        zoom_factor: float = 1.0,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> ArrayLike:
        """Return the masked array for a given label."""
        return get_masked_roi_pipe(
            image=self,
            label=label,
            axes_order=axes_order,
            mode=mode,
            zoom_factor=zoom_factor,
            **slice_kwargs,
        )

    def set_roi_masked(
        self,
        label: int,
        patch: ArrayLike,
        axes_order: Collection[str] | None = None,
        zoom_factor: float = 1.0,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> None:
        """Set the masked array for a given label."""
        return set_masked_roi_pipe(
            image=self,
            label=label,
            patch=patch,
            axes_order=axes_order,
            zoom_factor=zoom_factor,
            **slice_kwargs,
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

    def get_roi(
        self,
        label: int,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        mode: Literal["numpy", "dask", "delayed"] = "numpy",
        **slice_kwargs: slice | int | Iterable[int],
    ) -> ArrayLike:
        """Return the array for a given ROI."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        return super().get_roi(
            roi=roi, axes_order=axes_order, mode=mode, **slice_kwargs
        )

    def set_roi(
        self,
        label: int,
        patch: ArrayLike,
        zoom_factor: float = 1.0,
        axes_order: Collection[str] | None = None,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> None:
        """Set the array for a given ROI."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        return super().set_roi(
            roi=roi, patch=patch, axes_order=axes_order, **slice_kwargs
        )

    def get_roi_masked(
        self,
        label: int,
        axes_order: Collection[str] | None = None,
        mode: Literal["numpy", "dask", "delayed"] = "numpy",
        zoom_factor: float = 1.0,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> ArrayLike:
        """Return the masked array for a given label."""
        return get_masked_roi_pipe(
            image=self,
            label=label,
            axes_order=axes_order,
            mode=mode,
            zoom_factor=zoom_factor,
            **slice_kwargs,
        )

    def set_roi_masked(
        self,
        label: int,
        patch: ArrayLike,
        axes_order: Collection[str] | None = None,
        zoom_factor: float = 1.0,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> None:
        """Set the masked array for a given label."""
        return set_masked_roi_pipe(
            image=self,
            label=label,
            patch=patch,
            axes_order=axes_order,
            zoom_factor=zoom_factor,
            **slice_kwargs,
        )


def get_masked_roi_pipe(
    image: MaskedImage | MaskedLabel,
    label: int,
    axes_order: Collection[str] | None = None,
    mode: Literal["numpy", "dask", "delayed"] = "numpy",
    zoom_factor: float = 1.0,
    **slice_kwargs: slice | int | Iterable[int],
) -> ArrayLike:
    """Return the masked array for a given label."""
    roi = image._masking_roi_table.get(label)
    roi = roi.zoom(zoom_factor)
    slice_kwargs = roi_to_slice_kwargs(
        roi=roi,
        pixel_size=image.pixel_size,
        dimensions=image.dimensions,
        **slice_kwargs,
    )
    return get_masked_pipe(
        array=image.zarr_array,
        label_array=image._label.zarr_array,
        label=label,
        dimensions_array=image.dimensions,
        dimensions_label=image._label.dimensions,
        axes_order=axes_order,
        mode=mode,
        **slice_kwargs,
    )


def set_masked_roi_pipe(
    image: MaskedImage | MaskedLabel,
    label: int,
    patch: ArrayLike,
    axes_order: Collection[str] | None = None,
    zoom_factor: float = 1.0,
    **slice_kwargs: slice | int | Iterable[int],
) -> None:
    """Set the masked array for a given label."""
    roi = image._masking_roi_table.get(label)
    roi = roi.zoom(zoom_factor)
    slice_kwargs = roi_to_slice_kwargs(
        roi=roi,
        pixel_size=image.pixel_size,
        dimensions=image.dimensions,
        **slice_kwargs,
    )
    return set_masked_pipe(
        array=image.zarr_array,
        label_array=image._label.zarr_array,
        label=label,
        patch=patch,
        dimensions_array=image.dimensions,
        dimensions_label=image._label.dimensions,
        axes_order=axes_order,
        **slice_kwargs,
    )

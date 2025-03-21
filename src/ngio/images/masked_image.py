"""A module for handling label images in OME-NGFF files."""

# %%
from collections.abc import Collection, Iterable
from typing import Literal

import dask.array as da
import numpy as np

from ngio.common import (
    ArrayLike,
)
from ngio.images.image import Image
from ngio.images.label import Label
from ngio.ome_zarr_meta import (
    ImageMetaHandler,
)
from ngio.tables import MaskingROITable
from ngio.utils import (
    NgioValueError,
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
        masking_roi_table: MaskingROITable,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the omezarr file.
            meta_handler: The image metadata handler.
            label: The label image.
            masking_roi_table: The masking ROI table.

        """
        super().__init__(
            group_handler=group_handler, path=path, meta_handler=meta_handler
        )
        self._label = label
        self._masking_roi_table = masking_roi_table

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

    def get_array_masked(
        self,
        label: int,
        axes_order: Collection[str] | None = None,
        mode: Literal["numpy", "dask", "delayed"] = "numpy",
        zoom_factor: float = 1.0,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> ArrayLike:
        """Return the masked array for a given label."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)
        array = super().get_roi(
            roi=roi, axes_order=axes_order, mode=mode, **slice_kwargs
        )

        if "c" in slice_kwargs.keys():
            # This makes the strong assumption that the
            # user is passing the channel axis as "c"
            # This will fail if the channel axis is queried
            # with a different on-disk name
            slice_kwargs.pop("c")
        label_array = self._label.get_roi(
            roi=roi, axes_order=axes_order, mode=mode, **slice_kwargs
        )

        if isinstance(array, np.ndarray):
            label_array = np.broadcast_to(label_array, array.shape)
        elif isinstance(array, da.Array):
            label_array = da.broadcast_to(label_array, array.shape)
        else:
            raise NgioValueError(f"Mode {mode} not yet supported for masked array.")

        array[label_array != label] = 0
        return array

    def set_array_masked(
        self,
        label: int,
        patch: ArrayLike,
        axes_order: Collection[str] | None = None,
        zoom_factor: float = 1.0,
        **slice_kwargs: slice | int | Iterable[int],
    ) -> None:
        """Set the masked array for a given label."""
        roi = self._masking_roi_table.get(label)
        roi = roi.zoom(zoom_factor)

        if "c" in slice_kwargs.keys():
            # This makes the strong assumption that the
            # user is passing the channel axis as "c"
            # This will fail if the channel axis is queried
            # with a different on-disk name
            slice_kwargs.pop("c")

        if isinstance(patch, np.ndarray):
            mode = "numpy"
        elif isinstance(patch, da.Array):
            mode = "dask"
        else:
            raise NgioValueError("Only numpy and dask arrays are supported.")

        array = self.get_roi(roi=roi, axes_order=axes_order, mode=mode, **slice_kwargs)
        label_array = self._label.get_roi(
            roi=roi, axes_order=axes_order, mode=mode, **slice_kwargs
        )

        if mode == "numpy":
            label_array = np.broadcast_to(label_array, patch.shape)
        elif mode == "dask":
            label_array = da.broadcast_to(label_array, patch.shape)
        else:
            raise NgioValueError(f"Mode {mode} not yet supported for masked array.")

        array[label_array == label] = patch[label_array == label]
        self.set_roi(roi=roi, patch=array, axes_order=axes_order, **slice_kwargs)
        return array

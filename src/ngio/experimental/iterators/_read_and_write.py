from collections.abc import Callable, Collection

import dask.array as da
import numpy as np
import zarr

from ngio.common import (
    Dimensions,
    get_as_dask,
    get_as_numpy,
    set_dask,
    set_numpy,
)
from ngio.common._io_transforms import TransformProtocol

NumpyReader = Callable[[], np.ndarray]
NumpyWriter = Callable[[np.ndarray], None]

DaskReader = Callable[[], da.Array]
DaskWriter = Callable[[da.Array], None]


def build_numpy_reader(
    array: zarr.Array,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> Callable[[], np.ndarray]:
    """Create a reader function for the ROI."""

    def reader() -> np.ndarray:
        """Read the patch from the input image."""
        return get_as_numpy(
            array=array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    return reader


def build_dask_reader(
    array: zarr.Array,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> Callable[[], da.Array]:
    """Create a reader function for the ROI."""

    def reader() -> da.Array:
        """Read the patch from the input image."""
        return get_as_dask(
            array=array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    return reader


def build_dask_writer(
    array: zarr.Array,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> Callable[[da.Array], None]:
    """Create a writer function for the ROI."""

    def writer(patch: da.Array) -> None:
        """Write the patch to the output label."""
        set_dask(
            array=array,
            patch=patch,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    return writer


def build_numpy_writer(
    array: zarr.Array,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> Callable[[np.ndarray], None]:
    """Create a writer function for the ROI."""

    def writer(patch: np.ndarray) -> None:
        """Write the patch to the output label."""
        set_numpy(
            array=array,
            patch=patch,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    return writer

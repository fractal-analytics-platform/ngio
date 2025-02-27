from collections.abc import Collection, Iterable
from typing import Literal

import dask
import dask.delayed
import numpy as np
import zarr

from ngio.common import Dimensions
from ngio.common._axes_transforms import transform_dask_array, transform_numpy_array
from ngio.common._slicer import (
    SliceTransform,
    compute_and_slices,
    dask_get_slice,
    dask_set_slice,
    numpy_get_slice,
    numpy_set_slice,
)
from ngio.ome_zarr_meta.ngio_specs import AxesTransformation
from ngio.utils import NgioValueError


def _compute_from_disk_transforms(
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    **slice_kwargs: slice | int | Iterable[int],
) -> tuple[SliceTransform, tuple[AxesTransformation, ...]]:
    slices = compute_and_slices(dimensions=dimensions, **slice_kwargs)

    if axes_order is None:
        return slices, ()

    additional_transformations = dimensions._axes_mapper.to_order(axes_order)
    return slices, additional_transformations


def _compute_to_disk_transforms(
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    **slice_kwargs: slice | int | Iterable[int],
) -> tuple[SliceTransform, tuple[AxesTransformation, ...]]:
    slices = compute_and_slices(dimensions=dimensions, **slice_kwargs)
    if axes_order is None:
        return slices, ()

    additional_transformations = dimensions._axes_mapper.from_order(axes_order)
    return slices, additional_transformations


def _numpy_get_pipe(
    array: zarr.Array,
    slices: SliceTransform,
    transformations: tuple[AxesTransformation, ...],
) -> np.ndarray:
    array = numpy_get_slice(array, slices)
    return transform_numpy_array(array, transformations)


def _delayed_numpy_get_pipe(
    array: zarr.Array,
    slices: SliceTransform,
    transformations: tuple[AxesTransformation, ...],
) -> dask.delayed:
    array = dask.delayed(numpy_get_slice)(array, slices)
    return dask.delayed(transform_numpy_array)(array, transformations)


def _dask_get_pipe(
    array: zarr.Array,
    slices: SliceTransform,
    transformations: tuple[AxesTransformation, ...],
) -> dask.array:
    array = dask_get_slice(array, slices)
    return transform_dask_array(array, transformations)


def _numpy_set_pipe(
    array: zarr.Array,
    patch: np.ndarray,
    slices: SliceTransform,
    transformations: tuple[AxesTransformation, ...],
) -> None:
    patch = transform_numpy_array(patch, transformations)
    numpy_set_slice(array, patch, slices)


def _dask_set_pipe(
    array: zarr.Array,
    patch: np.ndarray,
    slices: SliceTransform,
    transformations: tuple[AxesTransformation, ...],
) -> None:
    patch = transform_dask_array(patch, transformations)
    dask_set_slice(array, patch, slices)


def _delayed_numpy_set_pipe(
    array: zarr.Array,
    patch: np.ndarray,
    slices: SliceTransform,
    transformations: tuple[AxesTransformation, ...],
) -> dask.delayed:
    patch = dask.delayed(transform_numpy_array)(patch, transformations)
    return dask.delayed(numpy_set_slice)(array, patch, slices)


def get_pipe(
    array: zarr.Array,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    mode: Literal["numpy", "dask", "delayed"] = "numpy",
    **slice_kwargs: slice | int | Iterable[int],
):
    slices, transformations = _compute_from_disk_transforms(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    print(slices, transformations)
    match mode:
        case "numpy":
            return _numpy_get_pipe(array, slices, transformations)
        case "dask":
            return _dask_get_pipe(array, slices, transformations)

        case "delayed_numpy":
            return _delayed_numpy_get_pipe(array, slices, transformations)

        case _:
            raise NgioValueError(
                f"Unknown get pipe mode {mode}, expected 'numpy', 'dask' or 'delayed'."
            )


def set_pipe(
    array: zarr.Array,
    patch: np.ndarray,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    mode: Literal["numpy", "dask", "delayed"] = "numpy",
    **slice_kwargs: slice | int | Iterable[int],
):
    slices, transformations = _compute_to_disk_transforms(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    print(slices, transformations)
    match mode:
        case "numpy":
            _numpy_set_pipe(array, patch, slices, transformations)
        case "dask":
            _dask_set_pipe(array, patch, slices, transformations)
        case "delayed_numpy":
            # This is the only case where we return the delayed object
            # because we want to allow the user to control when the computation
            # is started.
            return _delayed_numpy_set_pipe(array, patch, slices, transformations)
        case _:
            raise NgioValueError(
                f"Unknown set pipe mode {mode}, expected 'numpy', 'dask' or 'delayed'."
            )

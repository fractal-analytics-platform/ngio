from collections.abc import Iterable

import dask.array as da
import numpy as np
import zarr

from ngio.common._dimensions import Dimensions
from ngio.ome_zarr_meta.ngio_specs import AxesTransformation
from ngio.utils import NgioValueError


def _validate_int(value: int, shape: int) -> int:
    if not isinstance(value, int):
        raise NgioValueError(f"Invalid value {value} of type {type(value)}")
    if value < 0 or value >= shape:
        raise NgioValueError(
            f"Invalid value {value}. Index out of bounds for axis of shape {shape}"
        )
    return value


def _validate_iter_of_ints(value: Iterable[int], shape: int) -> list[int]:
    if not isinstance(value, list):
        raise NgioValueError(f"Invalid value {value} of type {type(value)}")
    value = [_validate_int(v, shape=shape) for v in value]
    return value


def _validate_slice(value: slice, shape: int) -> slice:
    start = value.start if value.start is not None else 0
    start = max(start, 0)
    stop = value.stop if value.stop is not None else shape
    return slice(start, stop)


class SliceTransform(AxesTransformation):
    slices: tuple[slice | tuple[int, ...], ...]


def compute_and_slices(
    *,
    dimensions: Dimensions,
    **slice_kwargs: slice | int | Iterable[int],
) -> SliceTransform:
    _slices = {}
    axes_names = dimensions._axes_mapper.on_disk_axes_names
    for axis_name, slice_ in slice_kwargs.items():
        axis = dimensions._axes_mapper.get_axis(axis_name)
        if axis is None:
            raise NgioValueError(
                f"Invalid axis {axis_name}. "
                f"Not found on the on-disk axes {axes_names}. "
                "If you want to get/set a singletorn value include "
                "it in the axes_order parameter."
            )

        shape = dimensions.get(axis.on_disk_name)

        if isinstance(slice_, int):
            slice_ = _validate_int(slice_, shape)
            slice_ = slice(slice_, slice_ + 1)

        elif isinstance(slice_, Iterable):
            slice_ = _validate_iter_of_ints(slice_, shape)
            slice_ = tuple(slice_)

        elif isinstance(slice_, slice):
            slice_ = _validate_slice(slice_, shape)

        elif not isinstance(slice_, slice):
            raise NgioValueError(
                f"Invalid slice definition {slice_} of type {type(slice_)}"
            )
        _slices[axis.on_disk_name] = slice_

    slices = tuple(_slices.get(axis, slice(None)) for axis in axes_names)
    return SliceTransform(slices=slices)


def numpy_get_slice(array: zarr.Array, slices: SliceTransform) -> np.ndarray:
    return array[slices.slices]


def dask_get_slice(array: zarr.Array, slices: SliceTransform) -> da.Array:
    da_array = da.from_zarr(array)
    return da_array[slices.slices]


def numpy_set_slice(
    array: zarr.Array, patch: np.ndarray, slices: SliceTransform
) -> None:
    array[slices.slices] = patch


def dask_set_slice(array: zarr.Array, patch: da.Array, slices: SliceTransform) -> None:
    da.to_zarr(arr=patch, url=array, region=slices.slices)

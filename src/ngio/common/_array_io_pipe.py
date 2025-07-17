from collections.abc import Collection
from typing import TypeAlias

import dask.array as da
import numpy as np
import zarr
from dask.array import Array as DaskArray
from dask.delayed import Delayed, delayed

from ngio.common._dimensions import Dimensions
from ngio.common._io_transforms import (
    TransformProtocol,
    apply_dask_transforms,
    apply_delayed_transforms,
    apply_numpy_transforms,
)
from ngio.ome_zarr_meta.ngio_specs import AxesOps
from ngio.utils import NgioValueError

##############################################################
#
# Slicing Operations
#
##############################################################

SliceDefinition: TypeAlias = tuple[slice | tuple[int, ...] | int, ...]
ArrayLike: TypeAlias = np.ndarray | DaskArray | Delayed


def _validate_int(value: int, shape: int) -> int:
    if not isinstance(value, int):
        raise NgioValueError(f"Invalid value {value} of type {type(value)}")
    if value < 0 or value >= shape:
        raise NgioValueError(
            f"Invalid value {value}. Index out of bounds for axis of shape {shape}"
        )
    return value


def _try_to_slice(input: Collection[int]) -> slice | tuple[int, ...]:
    """Try to convert a list of integers into a slice if they are contiguous.

    - If the input is empty, return an empty tuple.
    - If the input is sorted, and contains contiguous integers,
      return a slice from the minimum to the maximum integer.
    - Otherwise, return the input as a tuple.
    """
    if not input:
        return ()

    # If the input is not sorted, return it as a tuple
    max_input = max(input)
    min_input = min(input)
    assert min_input >= 0, "Input must contain non-negative integers"
    assert max_input >= 0, "Input must contain non-negative integers"

    if sorted(input) == list(range(min_input, max_input + 1)):
        return slice(min_input, max_input + 1)

    return tuple(input)


def _validate_iter_of_ints(
    value: Collection[int], shape: int
) -> slice | tuple[int, ...]:
    value = [_validate_int(v, shape=shape) for v in value]
    return _try_to_slice(value)


def _validate_slice(value: slice, shape: int) -> slice:
    start = value.start if value.start is not None else 0
    start = max(start, 0)
    stop = value.stop if value.stop is not None else shape
    return slice(start, stop)


def _build_slices(
    *,
    dimensions: Dimensions,
    axes_ops: AxesOps | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> SliceDefinition | None:
    _slices = {}
    if not slice_kwargs:
        # Skip unnecessary computation if no slicing is requested
        return None

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
            if axes_ops is not None:
                slice_ = slice(slice_, slice_ + 1)

        elif isinstance(slice_, Collection):
            slice_ = _validate_iter_of_ints(slice_, shape)

        elif isinstance(slice_, slice):
            slice_ = _validate_slice(slice_, shape)

        elif not isinstance(slice_, slice):
            raise NgioValueError(
                f"Invalid slice definition {slice_} of type {type(slice_)}"
            )
        _slices[axis.on_disk_name] = slice_

    slices = tuple(_slices.get(axis, slice(None)) for axis in axes_names)
    return slices


def _get_slice_as_numpy(
    array: zarr.Array, slices: SliceDefinition | None
) -> np.ndarray:
    if slices is None:
        return array[...]

    if all(not isinstance(s, tuple) for s in slices):
        return array[slices]

    fist_slices = []
    for s in slices:
        if isinstance(s, tuple):
            fist_slices.append(slice(None))
        else:
            fist_slices.append(s)
    second_slices = []
    for s in slices:
        if isinstance(s, tuple):
            second_slices.append(s)
        else:
            second_slices.append(slice(None))

    return array[tuple(fist_slices)][tuple(second_slices)]


def _get_slice_as_dask(array: zarr.Array, slices: SliceDefinition | None) -> da.Array:
    da_array = da.from_zarr(array)
    if slices is None:
        return da_array

    if any(isinstance(s, tuple) for s in slices):
        raise NgioValueError(
            "Slicing with non-contiguous tuples/lists "
            "is not supported for Dask arrays. Use the "
            "numpy api to get the correct array slice."
        )
    return da_array[slices]


def _set_numpy_slice(
    array: zarr.Array, patch: np.ndarray, slices: SliceDefinition | None
) -> None:
    if slices is None:
        array[...] = patch
        return
    array[slices] = patch


def _set_dask_slice(
    array: zarr.Array, patch: da.Array, slices: SliceDefinition | None
) -> None:
    da.to_zarr(arr=patch, url=array, region=slices)


##############################################################
#
# Array Axes Operations
#
##############################################################


def _numpy_apply_axes_ops(array: np.ndarray, axes_ops: AxesOps | None) -> np.ndarray:
    if axes_ops is None:
        return array
    if axes_ops.squeeze_axes is not None:
        array = np.squeeze(array, axis=axes_ops.squeeze_axes)
    if axes_ops.transpose_axes is not None:
        array = np.transpose(array, axes=axes_ops.transpose_axes)
    if axes_ops.expand_axes is not None:
        array = np.expand_dims(array, axis=axes_ops.expand_axes)
    return array


def _dask_apply_axes_ops(array: da.Array, axes_ops: AxesOps | None) -> da.Array:
    if axes_ops is None:
        return array
    if axes_ops.squeeze_axes is not None:
        array = da.squeeze(array, axis=axes_ops.squeeze_axes)
    if axes_ops.transpose_axes is not None:
        array = da.transpose(array, axes=axes_ops.transpose_axes)
    if axes_ops.expand_axes is not None:
        array = da.expand_dims(array, axis=axes_ops.expand_axes)
    return array


##############################################################
#
# Concrete "From Disk" Pipes
#
##############################################################


def _setup_from_disk_pipe(
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> tuple[SliceDefinition | None, AxesOps | None]:
    axes_ops = (
        dimensions.axes_mapper.to_order(axes_order) if axes_order is not None else None
    )

    slices = _build_slices(dimensions=dimensions, axes_ops=axes_ops, **slice_kwargs)
    return slices, axes_ops


def _numpy_get_pipe(
    array: zarr.Array,
    slices: SliceDefinition | None,
    axes_ops: AxesOps | None,
    transforms: Collection[TransformProtocol] | None,
) -> np.ndarray:
    _array = _get_slice_as_numpy(array, slices)
    _array = _numpy_apply_axes_ops(_array, axes_ops)
    _array = apply_numpy_transforms(_array, transforms)
    return _array


def _delayed_numpy_get_pipe(
    array: zarr.Array,
    slices: SliceDefinition | None,
    axes_ops: AxesOps | None,
    transforms: Collection[TransformProtocol] | None,
) -> Delayed:
    _array = delayed(_get_slice_as_numpy)(array, slices)
    _array = delayed(_numpy_apply_axes_ops)(_array, axes_ops)
    _array = apply_delayed_transforms(_array, transforms)
    return _array


def _dask_get_pipe(
    array: zarr.Array,
    slices: SliceDefinition | None,
    axes_ops: AxesOps | None,
    transforms: Collection[TransformProtocol] | None,
) -> DaskArray:
    _array = _get_slice_as_dask(array, slices)
    _array = _dask_apply_axes_ops(_array, axes_ops)
    _array = apply_dask_transforms(_array, transforms)
    return _array


def get_as_numpy(
    array: zarr.Array,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> np.ndarray:
    slices, axes_ops = _setup_from_disk_pipe(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    return _numpy_get_pipe(
        array=array, slices=slices, axes_ops=axes_ops, transforms=transforms
    )


def get_as_dask(
    array: zarr.Array,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> DaskArray:
    slices, axes_ops = _setup_from_disk_pipe(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    return _dask_get_pipe(
        array=array, slices=slices, axes_ops=axes_ops, transforms=transforms
    )


def get_as_delayed(
    array: zarr.Array,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> Delayed:
    slices, axes_ops = _setup_from_disk_pipe(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    return _delayed_numpy_get_pipe(
        array=array, slices=slices, axes_ops=axes_ops, transforms=transforms
    )


##############################################################
#
# Concrete "To Disk" Pipes
#
##############################################################


def _setup_to_disk_pipe(
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> tuple[SliceDefinition | None, AxesOps | None]:
    axes_ops = (
        dimensions.axes_mapper.from_order(axes_order)
        if axes_order is not None
        else None
    )
    slices = _build_slices(dimensions=dimensions, axes_ops=axes_ops, **slice_kwargs)
    return slices, axes_ops


def _numpy_set_pipe(
    array: zarr.Array,
    patch: np.ndarray,
    slices: SliceDefinition | None,
    axes_ops: AxesOps | None,
    transforms: Collection[TransformProtocol] | None,
) -> None:
    _patch = apply_numpy_transforms(patch, transforms)
    _patch = _numpy_apply_axes_ops(_patch, axes_ops)
    _set_numpy_slice(array, _patch, slices)


def _dask_set_pipe(
    array: zarr.Array,
    patch: DaskArray,
    slices: SliceDefinition | None,
    axes_ops: AxesOps | None,
    transforms: Collection[TransformProtocol] | None,
) -> None:
    _patch = apply_dask_transforms(patch, transforms)
    _patch = _dask_apply_axes_ops(_patch, axes_ops)
    _set_dask_slice(array, _patch, slices)


def _delayed_numpy_set_pipe(
    array: zarr.Array,
    patch: np.ndarray | Delayed,
    slices: SliceDefinition | None,
    axes_ops: AxesOps | None,
    transforms: Collection[TransformProtocol] | None,
) -> Delayed:
    if isinstance(patch, np.ndarray):
        patch = delayed(patch)
    _patch = apply_delayed_transforms(patch, transforms)
    _patch = delayed(_numpy_apply_axes_ops)(_patch, axes_ops)
    return delayed(_set_numpy_slice)(array, _patch, slices)


def set_numpy(
    array: zarr.Array,
    patch: np.ndarray,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
):
    slices, axes_ops = _setup_to_disk_pipe(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    _numpy_set_pipe(
        array=array,
        patch=patch,
        slices=slices,
        axes_ops=axes_ops,
        transforms=transforms,
    )


def set_dask(
    array: zarr.Array,
    patch: DaskArray,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
):
    slices, axes_ops = _setup_to_disk_pipe(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    _dask_set_pipe(
        array=array,
        patch=patch,
        slices=slices,
        axes_ops=axes_ops,
        transforms=transforms,
    )


def set_delayed(
    array: zarr.Array,
    patch: np.ndarray | Delayed,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
):
    slices, axes_ops = _setup_to_disk_pipe(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    _delayed_numpy_set_pipe(
        array=array,
        patch=patch,
        slices=slices,
        axes_ops=axes_ops,
        transforms=transforms,
    )


################################################################
#
# Masked Array Pipes
#
################################################################


def _mask_pipe_common_numpy(
    array: zarr.Array,
    label_array: zarr.Array,
    label: int,
    *,
    dimensions_array: Dimensions,
    dimensions_label: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> tuple[np.ndarray, np.ndarray]:
    array_patch = get_as_numpy(
        array,
        dimensions=dimensions_array,
        axes_order=axes_order,
        transforms=transforms,
        **slice_kwargs,
    )

    label_slice_kwargs = {}
    for key, value in slice_kwargs.items():
        if dimensions_label.get(key, -1) != -1:
            label_slice_kwargs[key] = value

    label_patch = get_as_numpy(
        label_array,
        dimensions=dimensions_label,
        axes_order=axes_order,
        transforms=transforms,
        **label_slice_kwargs,
    )

    if label_patch.shape != array_patch.shape:
        label_patch = np.broadcast_to(label_patch, array_patch.shape)

    mask = label_patch == label
    return array_patch, mask


def _mask_pipe_common_dask(
    array: zarr.Array,
    label_array: zarr.Array,
    label: int,
    *,
    dimensions_array: Dimensions,
    dimensions_label: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> tuple[DaskArray, DaskArray]:
    array_patch = get_as_dask(
        array,
        dimensions=dimensions_array,
        axes_order=axes_order,
        transforms=transforms,
        **slice_kwargs,
    )

    label_slice_kwargs = {}
    for key, value in slice_kwargs.items():
        if dimensions_label.get(key, -1) != -1:
            label_slice_kwargs[key] = value

    label_patch = get_as_numpy(
        label_array,
        dimensions=dimensions_label,
        axes_order=axes_order,
        transforms=transforms,
        **label_slice_kwargs,
    )

    label_patch = get_as_dask(
        label_array,
        dimensions=dimensions_label,
        axes_order=axes_order,
        transforms=transforms,
        **slice_kwargs,
    )

    if label_patch.shape != array_patch.shape:
        label_patch = da.broadcast_to(label_patch, array_patch.shape)

    mask = label_patch == label
    return array_patch, mask


def get_masked_as_numpy(
    array: zarr.Array,
    label_array: zarr.Array,
    label: int,
    *,
    dimensions_array: Dimensions,
    dimensions_label: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> np.ndarray:
    array_patch, mask = _mask_pipe_common_numpy(
        array=array,
        label_array=label_array,
        label=label,
        dimensions_array=dimensions_array,
        dimensions_label=dimensions_label,
        axes_order=axes_order,
        transforms=transforms,
        **slice_kwargs,
    )
    array_patch[~mask] = 0
    return array_patch


def get_masked_as_dask(
    array: zarr.Array,
    label_array: zarr.Array,
    label: int,
    *,
    dimensions_array: Dimensions,
    dimensions_label: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> DaskArray:
    array_patch, mask = _mask_pipe_common_dask(
        array=array,
        label_array=label_array,
        label=label,
        dimensions_array=dimensions_array,
        dimensions_label=dimensions_label,
        axes_order=axes_order,
        transforms=transforms,
        **slice_kwargs,
    )
    array_patch = da.where(mask, array_patch, 0)
    return array_patch


def set_numpy_masked(
    array: zarr.Array,
    label_array: zarr.Array,
    label: int,
    patch: np.ndarray,
    *,
    dimensions_array: Dimensions,
    dimensions_label: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
):
    array_patch, mask = _mask_pipe_common_numpy(
        array=array,
        label_array=label_array,
        label=label,
        dimensions_array=dimensions_array,
        dimensions_label=dimensions_label,
        axes_order=axes_order,
        transforms=transforms,
        **slice_kwargs,
    )
    _patch = np.where(mask, patch, array_patch)

    set_numpy(
        array,
        _patch,
        dimensions=dimensions_array,
        axes_order=axes_order,
        transforms=transforms,
        **slice_kwargs,
    )


def set_dask_masked(
    array: zarr.Array,
    label_array: zarr.Array,
    label: int,
    patch: DaskArray,
    *,
    dimensions_array: Dimensions,
    dimensions_label: Dimensions,
    axes_order: Collection[str] | None = None,
    transforms: Collection[TransformProtocol] | None = None,
    **slice_kwargs: slice | int | Collection[int],
):
    array_patch, mask = _mask_pipe_common_dask(
        array=array,
        label_array=label_array,
        label=label,
        dimensions_array=dimensions_array,
        dimensions_label=dimensions_label,
        axes_order=axes_order,
        transforms=transforms,
        **slice_kwargs,
    )
    _patch = da.where(mask, patch, array_patch)

    set_dask(
        array,
        _patch,
        dimensions=dimensions_array,
        axes_order=axes_order,
        transforms=transforms,
        **slice_kwargs,
    )

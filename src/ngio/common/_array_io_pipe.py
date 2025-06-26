from collections.abc import Collection, Iterable
from typing import Literal

import dask.array as da
import numpy as np
import zarr
from dask.array import Array as DaskArray
from dask.delayed import Delayed, delayed

from ngio.common._common_types import ArrayLike
from ngio.common._dimensions import Dimensions
from ngio.ome_zarr_meta.ngio_specs import AxesOps
from ngio.utils import NgioValueError

##############################################################
#
# Slicing Operations
#
##############################################################

SliceDefinition = tuple[slice | tuple[int, ...], ...]


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


def _build_slices(
    *,
    dimensions: Dimensions,
    **slice_kwargs: slice | int | Iterable[int],
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
    return slices


def _get_slice_as_numpy(
    array: zarr.Array, slices: SliceDefinition | None
) -> np.ndarray:
    if slices is None:
        return array[...]
    return array[slices]


def _get_slice_as_dask(array: zarr.Array, slices: SliceDefinition | None) -> da.Array:
    da_array = da.from_zarr(array)
    if slices is None:
        return da_array
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
# Array Axes Transformations
#
##############################################################


def _numpy_apply_axes_ops(array: np.ndarray, axes_ops: AxesOps) -> np.ndarray:
    if axes_ops.squeeze_axes is not None:
        array = np.squeeze(array, axis=axes_ops.squeeze_axes)
    if axes_ops.transpose_axes is not None:
        array = np.transpose(array, axes=axes_ops.transpose_axes)
    if axes_ops.expand_axes is not None:
        array = np.expand_dims(array, axis=axes_ops.expand_axes)
    return array


def _dask_apply_axes_ops(array: da.Array, axes_ops: AxesOps) -> da.Array:
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
    **slice_kwargs: slice | int | Iterable[int],
) -> tuple[SliceDefinition | None, AxesOps]:
    slices = _build_slices(dimensions=dimensions, **slice_kwargs)

    if axes_order is None:
        return slices, AxesOps(transpose_axes=None, expand_axes=None, squeeze_axes=None)

    additional_transformations = dimensions.axes_mapper.to_order(axes_order)
    return slices, additional_transformations


def _numpy_get_pipe(
    array: zarr.Array,
    slices: SliceDefinition | None,
    transformations: AxesOps,
) -> np.ndarray:
    _array = _get_slice_as_numpy(array, slices)
    return _numpy_apply_axes_ops(_array, transformations)


def _delayed_numpy_get_pipe(
    array: zarr.Array,
    slices: SliceDefinition | None,
    transformations: AxesOps,
) -> Delayed:
    _array = delayed(_get_slice_as_numpy)(array, slices)
    return delayed(_numpy_apply_axes_ops)(_array, transformations)


def _dask_get_pipe(
    array: zarr.Array,
    slices: SliceDefinition | None,
    transformations: AxesOps,
) -> DaskArray:
    _array = _get_slice_as_dask(array, slices)
    return _dask_apply_axes_ops(_array, transformations)


def get_as_numpy(
    array: zarr.Array,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    **slice_kwargs: slice | int | Iterable[int],
) -> np.ndarray:
    slices, transformations = _setup_from_disk_pipe(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    return _numpy_get_pipe(array, slices, transformations)


def get_as_dask(
    array: zarr.Array,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    **slice_kwargs: slice | int | Iterable[int],
) -> DaskArray:
    slices, transformations = _setup_from_disk_pipe(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    return _dask_get_pipe(array, slices, transformations)


def get_as_delayed(
    array: zarr.Array,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    **slice_kwargs: slice | int | Iterable[int],
) -> Delayed:
    slices, transformations = _setup_from_disk_pipe(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    return _delayed_numpy_get_pipe(array, slices, transformations)


def get_pipe(
    array: zarr.Array,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    mode: Literal["numpy", "dask", "delayed"] = "numpy",
    **slice_kwargs: slice | int | Iterable[int],
):
    match mode:
        case "numpy":
            return get_as_numpy(
                array, dimensions=dimensions, axes_order=axes_order, **slice_kwargs
            )
        case "dask":
            return get_as_dask(
                array, dimensions=dimensions, axes_order=axes_order, **slice_kwargs
            )

        case "delayed":
            return get_as_delayed(
                array, dimensions=dimensions, axes_order=axes_order, **slice_kwargs
            )

        case _:
            raise NgioValueError(
                f"Unknown get pipe mode {mode}, expected 'numpy', 'dask' or 'delayed'."
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
    **slice_kwargs: slice | int | Iterable[int],
) -> tuple[SliceDefinition | None, AxesOps]:
    slices = _build_slices(dimensions=dimensions, **slice_kwargs)
    if axes_order is None:
        return slices, AxesOps(transpose_axes=None, expand_axes=None, squeeze_axes=None)

    additional_transformations = dimensions.axes_mapper.from_order(axes_order)
    return slices, additional_transformations


def _numpy_set_pipe(
    array: zarr.Array,
    patch: np.ndarray,
    slices: SliceDefinition | None,
    transformations: AxesOps,
) -> None:
    patch = _numpy_apply_axes_ops(patch, transformations)
    _set_numpy_slice(array, patch, slices)


def _dask_set_pipe(
    array: zarr.Array,
    patch: DaskArray,
    slices: SliceDefinition | None,
    transformations: AxesOps,
) -> None:
    _patch = _dask_apply_axes_ops(patch, transformations)
    _set_dask_slice(array, _patch, slices)


def _delayed_numpy_set_pipe(
    array: zarr.Array,
    patch: np.ndarray | Delayed,
    slices: SliceDefinition | None,
    transformations: AxesOps,
) -> Delayed:
    _patch = delayed(_numpy_apply_axes_ops)(patch, transformations)
    return delayed(_set_numpy_slice)(array, _patch, slices)


def set_numpy(
    array: zarr.Array,
    patch: np.ndarray,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    **slice_kwargs: slice | int | Iterable[int],
):
    slices, transformations = _setup_to_disk_pipe(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    _numpy_set_pipe(
        array=array, patch=patch, slices=slices, transformations=transformations
    )


def set_dask(
    array: zarr.Array,
    patch: DaskArray,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    **slice_kwargs: slice | int | Iterable[int],
):
    slices, transformations = _setup_to_disk_pipe(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    _dask_set_pipe(
        array=array, patch=patch, slices=slices, transformations=transformations
    )


def set_delayed(
    array: zarr.Array,
    patch: np.ndarray | Delayed,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    **slice_kwargs: slice | int | Iterable[int],
):
    slices, transformations = _setup_to_disk_pipe(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    _delayed_numpy_set_pipe(
        array=array, patch=patch, slices=slices, transformations=transformations
    )


def set_pipe(
    array: zarr.Array,
    patch: ArrayLike,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    **slice_kwargs: slice | int | Iterable[int],
):
    if isinstance(patch, np.ndarray):
        set_numpy(
            array,
            patch,
            dimensions=dimensions,
            axes_order=axes_order,
            **slice_kwargs,
        )
    elif isinstance(patch, DaskArray):
        set_dask(
            array,
            patch,
            dimensions=dimensions,
            axes_order=axes_order,
            **slice_kwargs,
        )
    elif isinstance(patch, Delayed):
        set_delayed(
            array,
            patch,
            dimensions=dimensions,
            axes_order=axes_order,
            **slice_kwargs,
        )
    else:
        raise NgioValueError(
            f"Unknown patch type, expected numpy, dask or delayed. Got {type(patch)}."
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
    **slice_kwargs: slice | int | Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    array_patch = get_as_numpy(
        array,
        dimensions=dimensions_array,
        axes_order=axes_order,
        **slice_kwargs,
    )

    label_patch = get_as_numpy(
        label_array,
        dimensions=dimensions_label,
        axes_order=axes_order,
        **slice_kwargs,
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
    **slice_kwargs: slice | int | Iterable[int],
) -> tuple[DaskArray, DaskArray]:
    array_patch = get_as_dask(
        array,
        dimensions=dimensions_array,
        axes_order=axes_order,
        **slice_kwargs,
    )

    label_patch = get_as_dask(
        label_array,
        dimensions=dimensions_label,
        axes_order=axes_order,
        **slice_kwargs,
    )

    if label_patch.shape != array_patch.shape:
        label_patch = da.broadcast_to(label_patch, array_patch.shape)

    mask = label_patch == label
    return array_patch, mask


def get_masked_pipe(
    array: zarr.Array,
    label_array: zarr.Array,
    label: int,
    *,
    dimensions_array: Dimensions,
    dimensions_label: Dimensions,
    axes_order: Collection[str] | None = None,
    mode: Literal["numpy", "dask", "delayed"] = "numpy",
    **slice_kwargs: slice | int | Iterable[int],
):
    if mode == "numpy":
        array_patch, mask = _mask_pipe_common_numpy(
            array=array,
            label_array=label_array,
            label=label,
            dimensions_array=dimensions_array,
            dimensions_label=dimensions_label,
            axes_order=axes_order,
            **slice_kwargs,
        )
        array_patch[~mask] = 0
        return array_patch
    elif mode == "dask":
        array_patch, mask = _mask_pipe_common_dask(
            array=array,
            label_array=label_array,
            label=label,
            dimensions_array=dimensions_array,
            dimensions_label=dimensions_label,
            axes_order=axes_order,
            **slice_kwargs,
        )
        array_patch = da.where(mask, array_patch, 0)
        return array_patch

    raise NgioValueError(f"Unknown mode {mode}, expected 'numpy' or 'dask'.")


def set_masked_pipe(
    array: zarr.Array,
    label_array: zarr.Array,
    label: int,
    patch: ArrayLike,
    *,
    dimensions_array: Dimensions,
    dimensions_label: Dimensions,
    axes_order: Collection[str] | None = None,
    **slice_kwargs: slice | int | Iterable[int],
):
    if isinstance(patch, DaskArray):
        array_patch, mask = _mask_pipe_common_dask(
            array=array,
            label_array=label_array,
            label=label,
            dimensions_array=dimensions_array,
            dimensions_label=dimensions_label,
            axes_order=axes_order,
            **slice_kwargs,
        )
        _patch = da.where(mask, patch, array_patch)
    elif isinstance(patch, np.ndarray):
        array_patch, mask = _mask_pipe_common_numpy(
            array=array,
            label_array=label_array,
            label=label,
            dimensions_array=dimensions_array,
            dimensions_label=dimensions_label,
            axes_order=axes_order,
            **slice_kwargs,
        )
        _patch = np.where(mask, patch, array_patch)
    else:
        raise NgioValueError(
            "Mode not yet supported for masked array. Expected a numpy or dask array."
        )

    set_pipe(
        array,
        _patch,
        dimensions=dimensions_array,
        axes_order=axes_order,
        **slice_kwargs,
    )

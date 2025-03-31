from collections.abc import Collection, Iterable
from typing import Literal

import dask
import dask.delayed
import numpy as np
import zarr

from ngio.common._axes_transforms import transform_dask_array, transform_numpy_array
from ngio.common._common_types import ArrayLike
from ngio.common._dimensions import Dimensions
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
    match mode:
        case "numpy":
            return _numpy_get_pipe(array, slices, transformations)
        case "dask":
            return _dask_get_pipe(array, slices, transformations)

        case "delayed":
            return _delayed_numpy_get_pipe(array, slices, transformations)

        case _:
            raise NgioValueError(
                f"Unknown get pipe mode {mode}, expected 'numpy', 'dask' or 'delayed'."
            )


def set_pipe(
    array: zarr.Array,
    patch: ArrayLike,
    *,
    dimensions: Dimensions,
    axes_order: Collection[str] | None = None,
    **slice_kwargs: slice | int | Iterable[int],
):
    slices, transformations = _compute_to_disk_transforms(
        dimensions=dimensions, axes_order=axes_order, **slice_kwargs
    )
    if isinstance(patch, dask.array.Array):
        _dask_set_pipe(
            array=array, patch=patch, slices=slices, transformations=transformations
        )
    elif isinstance(patch, np.ndarray):
        _numpy_set_pipe(
            array=array, patch=patch, slices=slices, transformations=transformations
        )
    elif isinstance(patch, dask.delayed.Delayed):
        _delayed_numpy_set_pipe(
            array=array, patch=patch, slices=slices, transformations=transformations
        )
    else:
        raise NgioValueError("Unknown patch type, expected numpy, dask or delayed.")


def _mask_pipe_common(
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
    array_patch = get_pipe(
        array,
        dimensions=dimensions_array,
        axes_order=axes_order,
        mode=mode,
        **slice_kwargs,
    )

    if not dimensions_label.has_axis("c"):
        # Remove the 'c' from the slice_kwargs
        # This will not work if the query uses non-default
        # axes names for channel
        slice_kwargs = {k: v for k, v in slice_kwargs.items() if k != "c"}

    label_patch = get_pipe(
        label_array,
        dimensions=dimensions_label,
        axes_order=axes_order,
        mode=mode,
        **slice_kwargs,
    )

    if isinstance(array_patch, np.ndarray):
        label_patch = np.broadcast_to(label_patch, array_patch.shape)
    elif isinstance(array_patch, dask.array.Array):
        label_patch = dask.array.broadcast_to(label_patch, array_patch.shape)
    else:
        raise NgioValueError(f"Mode {mode} not yet supported for masked array.")

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
    array_patch, mask = _mask_pipe_common(
        array=array,
        label_array=label_array,
        label=label,
        dimensions_array=dimensions_array,
        dimensions_label=dimensions_label,
        axes_order=axes_order,
        mode=mode,
        **slice_kwargs,
    )
    array_patch[~mask] = 0
    return array_patch


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
    if isinstance(patch, dask.array.Array):
        mode = "dask"
    elif isinstance(patch, np.ndarray):
        mode = "numpy"
    else:
        raise NgioValueError(
            "Mode not yet supported for masked array. Expected a numpy or dask array."
        )

    array_patch, mask = _mask_pipe_common(
        array=array,
        label_array=label_array,
        label=label,
        dimensions_array=dimensions_array,
        dimensions_label=dimensions_label,
        axes_order=axes_order,
        mode=mode,
        **slice_kwargs,
    )
    patch = np.where(mask, patch, array_patch)
    set_pipe(
        array, patch, dimensions=dimensions_array, axes_order=axes_order, **slice_kwargs
    )

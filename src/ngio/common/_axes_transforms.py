from typing import TypeVar

import dask.array as da
import numpy as np

from ngio.ome_zarr_meta.ngio_specs._axes import (
    AxesExpand,
    AxesSqueeze,
    AxesTransformation,
    AxesTranspose,
)
from ngio.utils import NgioValueError

T = TypeVar("T")


def transform_list(
    input_list: list[T], default: T, operations: tuple[AxesTransformation, ...]
) -> list[T]:
    if isinstance(input_list, tuple):
        input_list = list(input_list)

    for operation in operations:
        if isinstance(operation, AxesTranspose):
            input_list = [input_list[i] for i in operation.axes]

        if isinstance(operation, AxesExpand):
            for ax in operation.axes:
                input_list.insert(ax, default)
        elif isinstance(operation, AxesSqueeze):
            for offset, ax in enumerate(operation.axes):
                input_list.pop(ax - offset)

    return input_list


def transform_numpy_array(
    array: np.ndarray, operations: tuple[AxesTransformation, ...]
) -> np.ndarray:
    for operation in operations:
        if isinstance(operation, AxesTranspose):
            array = np.transpose(array, operation.axes)
        elif isinstance(operation, AxesExpand):
            array = np.expand_dims(array, axis=operation.axes)
        elif isinstance(operation, AxesSqueeze):
            array = np.squeeze(array, axis=operation.axes)
        else:
            raise NgioValueError(f"Unknown operation {operation}")
    return array


def transform_dask_array(
    array: da.Array, operations: tuple[AxesTransformation, ...]
) -> da.Array:
    for operation in operations:
        if isinstance(operation, AxesTranspose):
            array = da.transpose(array, axes=operation.axes)
        elif isinstance(operation, AxesExpand):
            array = da.expand_dims(array, axis=operation.axes)
        elif isinstance(operation, AxesSqueeze):
            array = da.squeeze(array, axis=operation.axes)
        else:
            raise NgioValueError(f"Unknown operation {operation}")
    return array

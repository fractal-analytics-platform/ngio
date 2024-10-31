from functools import partial
from typing import Literal

import dask.array as da
import numpy as np
import zarr
from scipy.ndimage import zoom


def _zoom_inputs_check(
    source_array: np.ndarray | da.Array,
    scale: tuple[int, ...] | None = None,
    target_shape: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    if scale is None and target_shape is None:
        raise ValueError("Either scale or target_shape must be provided")

    if scale is not None and target_shape is not None:
        raise ValueError("Only one of scale or target_shape must be provided")

    if scale is None:
        assert target_shape is not None, "Target shape must be provided"
        assert len(target_shape) == source_array.ndim, (
            "Target shape must have the "
            "same number of dimensions as "
            "the source array"
        )
        _scale = np.array(target_shape) / np.array(source_array.shape)
        _target_shape = target_shape
    else:
        _scale = np.array(scale)
        _target_shape = tuple(np.array(source_array.shape) * scale)

    return _scale, _target_shape


def _dask_zoom(
    source_array: da.Array,
    scale: tuple[int, ...] | None = None,
    target_shape: tuple[int, ...] | None = None,
    order: Literal[0, 1, 2] = 1,
) -> da.Array:
    """Dask implementation of zooming an array.

    Only one of scale or target_shape must be provided.

    Args:
        source_array (da.Array): The source array to zoom.
        scale (tuple[int, ...] | None): The scale factor to zoom by.
        target_shape (tuple[int, ...], None): The target shape to zoom to.
        order (Literal[0, 1, 2]): The order of interpolation. Defaults to 1.

    Returns:
        da.Array: The zoomed array.
    """
    # This function follow the implementation from:
    # https://github.com/ome/ome-zarr-py/blob/master/ome_zarr/dask_utils.py
    # The module was contributed by Andreas Eisenbarth @aeisenbarth
    # See https://github.com/toloudis/ome-zarr-py/pull/

    _scale, _target_shape = _zoom_inputs_check(
        source_array=source_array, scale=scale, target_shape=target_shape
    )

    # Rechunk to better match the scaling operation
    source_chunks = np.array(source_array.chunksize)
    better_source_chunks = np.maximum(1, np.round(source_chunks * _scale) / _scale)
    better_source_chunks = better_source_chunks.astype(int)
    source_array = source_array.rechunk(better_source_chunks)  # type: ignore

    # Calculate the block output shape
    block_output_shape = tuple(np.ceil(better_source_chunks * _scale).astype(int))

    zoom_wrapper = partial(
        zoom, zoom=_scale, order=order, mode="grid-constant", grid_mode=True
    )

    out_array = da.map_blocks(
        zoom_wrapper, source_array, chunks=block_output_shape, dtype=source_array.dtype
    )

    # Slice and rechunk to target
    slices = tuple(slice(0, ts, 1) for ts in _target_shape)
    out_array = out_array[slices]
    return out_array


def _numpy_zoom(
    source_array: np.ndarray,
    scale: tuple[int, ...] | None = None,
    target_shape: tuple[int, ...] | None = None,
    order: Literal[0, 1, 2] = 1,
) -> np.ndarray:
    """Numpy implementation of zooming an array.

    Only one of scale or target_shape must be provided.

    Args:
        source_array (np.ndarray): The source array to zoom.
        scale (tuple[int, ...] | None): The scale factor to zoom by.
        target_shape (tuple[int, ...], None): The target shape to zoom to.
        order (Literal[0, 1, 2]): The order of interpolation. Defaults to 1.

    Returns:
        np.ndarray: The zoomed array
    """
    _scale, _ = _zoom_inputs_check(
        source_array=source_array, scale=scale, target_shape=target_shape
    )

    out_array = zoom(
        source_array, zoom=_scale, order=order, mode="grid-constant", grid_mode=True
    )
    assert isinstance(out_array, np.ndarray)
    return out_array


def on_disk_zoom(
    source: zarr.Array,
    target: zarr.Array,
    order: Literal[0, 1, 2] = 1,
    mode: Literal["dask", "numpy"] = "dask",
) -> None:
    """Apply a zoom operation from a source zarr array to a target zarr array.

    Args:
        source (zarr.Array): The source array to zoom.
        target (zarr.Array): The target array to save the zoomed result to.
        order (Literal[0, 1, 2]): The order of interpolation. Defaults to 1.
        mode (Literal["dask", "numpy"]): The mode to use. Defaults to "dask".
    """
    if not isinstance(source, zarr.Array):
        raise ValueError("source must be a zarr array")

    if not isinstance(target, zarr.Array):
        raise ValueError("target must be a zarr array")

    if source.dtype != target.dtype:
        raise ValueError("source and target must have the same dtype")

    assert mode in ["dask", "numpy"], "mode must be either 'dask' or 'numpy'"

    if mode == "numpy":
        target[...] = _numpy_zoom(source[...], target_shape=target.shape, order=order)

    source_array = da.from_zarr(source)
    target_array = _dask_zoom(source_array, target_shape=target.shape, order=order)

    target_array = target_array.rechunk(target.chunks)
    target_array.to_zarr(target)


def on_disk_coarsen(
    source: zarr.Array,
    target: zarr.Array,
    aggregation_function: np.ufunc,
    coarsening_setup: dict[int, int],
) -> None:
    """Apply a coarsening operation from a source zarr array to a target zarr array.

    Args:
        source (zarr.Array): The source array to coarsen.
        target (zarr.Array): The target array to save the coarsened result to.
        aggregation_function (np.ufunc): The aggregation function to use.
        coarsening_setup (dict[int, int]): The coarsening setup to use.
    """
    source_array = da.from_zarr(source)

    for ax, factor in coarsening_setup.items():
        if ax >= source_array.ndim:
            raise ValueError(
                "Coarsening axis must be less than the number of dimensions"
            )
        if factor <= 0:
            raise ValueError("Coarsening factor must be greater than 0")

        assert isinstance(factor, int), "Coarsening factor must be an integer"

    out_target = da.coarsen(
        aggregation_function, source_array, coarsening_setup, trim_excess=True
    )
    out_target = out_target.rechunk(target.chunks)
    out_target.to_zarr(target)

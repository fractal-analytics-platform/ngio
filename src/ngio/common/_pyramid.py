from typing import Literal

import dask.array as da
import numpy as np
import zarr

from ngio.common._zoom import _zoom_inputs_check, dask_zoom, numpy_zoom


def _on_disk_numpy_zoom(
    source: zarr.Array,
    target: zarr.Array,
    order: Literal[0, 1, 2] = 1,
) -> None:
    target[...] = numpy_zoom(source[...], target_shape=target.shape, order=order)


def _on_disk_dask_zoom(
    source: zarr.Array,
    target: zarr.Array,
    order: Literal[0, 1, 2] = 1,
) -> None:
    source_array = da.from_zarr(source)
    target_array = dask_zoom(source_array, target_shape=target.shape, order=order)

    target_array = target_array.rechunk(target.chunks)
    target_array.compute_chunk_sizes()
    target_array.to_zarr(target)


def _on_disk_coarsen(
    source: zarr.Array,
    target: zarr.Array,
    _order: Literal[0, 1] = 1,
    aggregation_function: np.ufunc | None = None,
) -> None:
    """Apply a coarsening operation from a source zarr array to a target zarr array.

    Args:
        source (zarr.Array): The source array to coarsen.
        target (zarr.Array): The target array to save the coarsened result to.
        _order (Literal[0, 1]): The order of interpolation is not really implemented
            for coarsening, but it is kept for compatibility with the zoom function.
            _order=1 -> linear interpolation ~ np.mean
            _order=0 -> nearest interpolation ~ np.max
        aggregation_function (np.ufunc): The aggregation function to use.
    """
    source_array = da.from_zarr(source)

    _scale, _target_shape = _zoom_inputs_check(
        source_array=source_array, scale=None, target_shape=target.shape
    )

    assert _target_shape == target.shape, (
        "Target shape must match the target array shape"
    )

    if aggregation_function is None:
        if _order == 1:
            aggregation_function = np.mean
        elif _order == 0:
            aggregation_function = np.max
        else:
            raise ValueError(
                f"Aggregation function must be provided for order {_order}"
            )

    coarsening_setup = {}
    for i, s in enumerate(_scale):
        factor = 1 / s
        if factor.is_integer():
            coarsening_setup[i] = int(factor)
        else:
            raise ValueError(
                f"Coarsening factor must be an integer, got {factor} on axis {i}"
            )

    out_target = da.coarsen(
        aggregation_function, source_array, coarsening_setup, trim_excess=True
    )
    out_target = out_target.rechunk(target.chunks)
    out_target.to_zarr(target)


def on_disk_zoom(
    source: zarr.Array,
    target: zarr.Array,
    order: Literal[0, 1, 2] = 1,
    mode: Literal["dask", "numpy", "coarsen"] = "dask",
) -> None:
    """Apply a zoom operation from a source zarr array to a target zarr array.

    Args:
        source (zarr.Array): The source array to zoom.
        target (zarr.Array): The target array to save the zoomed result to.
        order (Literal[0, 1, 2]): The order of interpolation. Defaults to 1.
        mode (Literal["dask", "numpy", "coarsen"]): The mode to use. Defaults to "dask".
    """
    if not isinstance(source, zarr.Array):
        raise ValueError("source must be a zarr array")

    if not isinstance(target, zarr.Array):
        raise ValueError("target must be a zarr array")

    if source.dtype != target.dtype:
        raise ValueError("source and target must have the same dtype")

    match mode:
        case "numpy":
            return _on_disk_numpy_zoom(source, target, order)
        case "dask":
            return _on_disk_dask_zoom(source, target, order)
        case "coarsen":
            return _on_disk_coarsen(
                source,
                target,
            )
        case _:
            raise ValueError("mode must be either 'dask', 'numpy' or 'coarsen'")

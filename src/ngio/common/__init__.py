"""Common classes and functions that are used across the package."""

from ngio.common._array_pipe import get_pipe, set_pipe
from ngio.common._axes_transforms import (
    transform_dask_array,
    transform_list,
    transform_numpy_array,
)
from ngio.common._common_types import ArrayLike
from ngio.common._dimensions import Dimensions
from ngio.common._pyramid import consolidate_pyramid, init_empty_pyramid, on_disk_zoom
from ngio.common._roi import RasterCooROI, WorldCooROI
from ngio.common._slicer import (
    SliceTransform,
    compute_and_slices,
    dask_get_slice,
    dask_set_slice,
    numpy_get_slice,
    numpy_set_slice,
)
from ngio.common._zoom import dask_zoom, numpy_zoom

__all__ = [
    "ArrayLike",
    "Dimensions",
    "RasterCooROI",
    "SliceTransform",
    "WorldCooROI",
    "compute_and_slices",
    "consolidate_pyramid",
    "dask_get_slice",
    "dask_set_slice",
    "dask_zoom",
    "get_pipe",
    "init_empty_pyramid",
    "numpy_get_slice",
    "numpy_set_slice",
    "numpy_zoom",
    "on_disk_zoom",
    "set_pipe",
    "transform_dask_array",
    "transform_list",
    "transform_numpy_array",
]

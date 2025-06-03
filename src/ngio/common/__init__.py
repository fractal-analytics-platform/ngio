"""Common classes and functions that are used across the package."""

from ngio.common._array_pipe import (
    get_masked_pipe,
    get_pipe,
    set_masked_pipe,
    set_pipe,
)
from ngio.common._axes_transforms import (
    transform_dask_array,
    transform_list,
    transform_numpy_array,
)
from ngio.common._common_types import ArrayLike
from ngio.common._dimensions import Dimensions
from ngio.common._masking_roi import compute_masking_roi
from ngio.common._pyramid import consolidate_pyramid, init_empty_pyramid, on_disk_zoom
from ngio.common._roi import Roi, RoiPixels, roi_to_slice_kwargs
from ngio.common._slicer import (
    SliceTransform,
    compute_and_slices,
    dask_get_slice,
    dask_set_slice,
    numpy_get_slice,
    numpy_set_slice,
)
from ngio.common._table_ops import (
    concatenate_image_tables,
    concatenate_image_tables_as,
    concatenate_image_tables_as_async,
    concatenate_image_tables_async,
    conctatenate_tables,
    list_image_tables,
    list_image_tables_async,
)
from ngio.common._zoom import dask_zoom, numpy_zoom

__all__ = [
    "ArrayLike",
    "Dimensions",
    "Roi",
    "RoiPixels",
    "SliceTransform",
    "compute_and_slices",
    "compute_masking_roi",
    "concatenate_image_tables",
    "concatenate_image_tables_as",
    "concatenate_image_tables_as_async",
    "concatenate_image_tables_async",
    "conctatenate_tables",
    "consolidate_pyramid",
    "dask_get_slice",
    "dask_set_slice",
    "dask_zoom",
    "get_masked_pipe",
    "get_pipe",
    "init_empty_pyramid",
    "list_image_tables",
    "list_image_tables_async",
    "numpy_get_slice",
    "numpy_set_slice",
    "numpy_zoom",
    "on_disk_zoom",
    "roi_to_slice_kwargs",
    "set_masked_pipe",
    "set_pipe",
    "transform_dask_array",
    "transform_list",
    "transform_numpy_array",
]

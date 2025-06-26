"""Common classes and functions that are used across the package."""

# from ngio.common._axes_ops import (
#    transform_dask_array,
#    transform_numpy_array,
# )
from ngio.common._array_io_pipe import (
    get_masked_pipe,
    get_pipe,
    set_masked_pipe,
    set_pipe,
)
from ngio.common._common_types import ArrayLike
from ngio.common._dimensions import Dimensions
from ngio.common._masking_roi import compute_masking_roi

# from ngio.common._pipes import (
#    get_masked_pipe,
#    get_pipe,
#    set_masked_pipe,
#    set_pipe,
# )
from ngio.common._pyramid import consolidate_pyramid, init_empty_pyramid, on_disk_zoom
from ngio.common._roi import Roi, RoiPixels, roi_to_slice_kwargs

# from ngio.common._slicing_ops import (
#    SliceDefinition,
#    build_slices,
#    get_slice_as_dask,
#    get_slice_as_numpy,
#    set_dask_slice,
#    set_numpy_slice,
# )
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
    "compute_masking_roi",
    "concatenate_image_tables",
    "concatenate_image_tables_as",
    "concatenate_image_tables_as_async",
    "concatenate_image_tables_async",
    "conctatenate_tables",
    "consolidate_pyramid",
    "dask_zoom",
    "get_masked_pipe",
    "get_pipe",
    "init_empty_pyramid",
    "list_image_tables",
    "list_image_tables_async",
    "numpy_zoom",
    "on_disk_zoom",
    "roi_to_slice_kwargs",
    "set_masked_pipe",
    "set_pipe",
]

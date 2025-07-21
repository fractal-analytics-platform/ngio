"""Common classes and functions that are used across the package."""

from ngio.common._array_io_pipe import (
    ArrayLike,
    get_as_dask,
    get_as_numpy,
    get_masked_as_dask,
    get_masked_as_numpy,
    set_dask,
    set_dask_masked,
    set_numpy,
    set_numpy_masked,
)
from ngio.common._dimensions import Dimensions
from ngio.common._io_transforms import (
    AbstractTransform,
    TransformProtocol,
)
from ngio.common._masking_roi import compute_masking_roi
from ngio.common._pyramid import consolidate_pyramid, init_empty_pyramid, on_disk_zoom
from ngio.common._roi import (
    Roi,
    RoiPixels,
    add_channel_label_to_slice_kwargs,
    roi_to_slice_kwargs,
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
    "AbstractTransform",
    "ArrayLike",
    "Dimensions",
    "Roi",
    "RoiPixels",
    "TransformProtocol",
    "add_channel_label_to_slice_kwargs",
    "compute_masking_roi",
    "concatenate_image_tables",
    "concatenate_image_tables_as",
    "concatenate_image_tables_as_async",
    "concatenate_image_tables_async",
    "conctatenate_tables",
    "consolidate_pyramid",
    "dask_zoom",
    "get_as_dask",
    "get_as_numpy",
    "get_masked_as_dask",
    "get_masked_as_numpy",
    "init_empty_pyramid",
    "list_image_tables",
    "list_image_tables_async",
    "numpy_zoom",
    "on_disk_zoom",
    "roi_to_slice_kwargs",
    "set_dask",
    "set_dask_masked",
    "set_numpy",
    "set_numpy_masked",
]

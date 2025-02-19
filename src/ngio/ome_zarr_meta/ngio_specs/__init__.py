"""ngio internal specs module.

Since the OME-Zarr specification are still evolving, this module provides a
set of classes to internally handle the metadata.

This models can be tr
"""

from ngio.ome_zarr_meta.ngio_specs._axes import (
    AxesExpand,
    AxesMapper,
    AxesSetup,
    AxesSqueeze,
    AxesTransformation,
    AxesTranspose,
    Axis,
    AxisType,
    SpaceUnits,
    TimeUnits,
)
from ngio.ome_zarr_meta.ngio_specs._channels import (
    Channel,
    ChannelsMeta,
    ChannelVisualisation,
    NgioColors,
)
from ngio.ome_zarr_meta.ngio_specs._dataset import Dataset
from ngio.ome_zarr_meta.ngio_specs._ngio_image_meta import ImageMeta, LabelMeta
from ngio.ome_zarr_meta.ngio_specs._pixel_size import PixelSize

__all__ = [
    "AxesExpand",
    "AxesMapper",
    "AxesSetup",
    "AxesSqueeze",
    "AxesTransformation",
    "AxesTranspose",
    "Axis",
    "AxisType",
    "Channel",
    "ChannelVisualisation",
    "ChannelsMeta",
    "Dataset",
    "ImageMeta",
    "LabelMeta",
    "NgioColors",
    "PixelSize",
    "SpaceUnits",
    "TimeUnits",
]

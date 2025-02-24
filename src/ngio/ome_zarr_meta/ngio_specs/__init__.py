"""ngio internal specs module.

Since the OME-Zarr specification are still evolving, this module provides a
set of classes to internally handle the metadata.

This models can be tr
"""

from ngio.common._pixel_size import PixelSize
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
    default_channel_name,
)
from ngio.ome_zarr_meta.ngio_specs._dataset import Dataset
from ngio.ome_zarr_meta.ngio_specs._ngio_image_meta import (
    ImageLabelSource,
    NgioImageLabelMeta,
    NgioImageMeta,
    NgioLabelMeta,
)

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
    "ImageLabelSource",
    "NgioColors",
    "NgioImageLabelMeta",
    "NgioImageMeta",
    "NgioLabelMeta",
    "PixelSize",
    "SpaceUnits",
    "TimeUnits",
    "default_channel_name",
]

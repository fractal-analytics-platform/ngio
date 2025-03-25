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
    canonical_axes_order,
    canonical_label_axes_order,
)
from ngio.ome_zarr_meta.ngio_specs._channels import (
    Channel,
    ChannelsMeta,
    ChannelVisualisation,
    NgioColors,
    default_channel_name,
)
from ngio.ome_zarr_meta.ngio_specs._dataset import Dataset
from ngio.ome_zarr_meta.ngio_specs._ngio_hcs import (
    ImageInWellPath,
    NgioPlateMeta,
    NgioWellMeta,
)
from ngio.ome_zarr_meta.ngio_specs._ngio_image import (
    ImageLabelSource,
    NgioImageLabelMeta,
    NgioImageMeta,
    NgioLabelMeta,
)
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
    "ImageInWellPath",
    "ImageLabelSource",
    "NgioColors",
    "NgioImageLabelMeta",
    "NgioImageMeta",
    "NgioLabelMeta",
    "NgioPlateMeta",
    "NgioWellMeta",
    "PixelSize",
    "SpaceUnits",
    "TimeUnits",
    "canonical_axes_order",
    "canonical_label_axes_order",
    "default_channel_name",
]

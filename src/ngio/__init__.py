"""Next Generation file format IO."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ngio")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

from ngio.common import ArrayLike, Dimensions, Roi, RoiPixels
from ngio.hcs import OmeZarrPlate, create_empty_plate, open_ome_zarr_plate
from ngio.images import (
    Image,
    Label,
    OmeZarrContainer,
    create_empty_ome_zarr,
    create_ome_zarr_from_array,
    open_image,
    open_ome_zarr_container,
)
from ngio.ome_zarr_meta.ngio_specs import AxesSetup, ImageInWellPath, PixelSize

__all__ = [
    "ArrayLike",
    "AxesSetup",
    "Dimensions",
    "Image",
    "ImageInWellPath",
    "Label",
    "OmeZarrContainer",
    "OmeZarrPlate",
    "PixelSize",
    "Roi",
    "RoiPixels",
    "create_empty_ome_zarr",
    "create_empty_plate",
    "create_ome_zarr_from_array",
    "open_image",
    "open_ome_zarr_container",
    "open_ome_zarr_plate",
]

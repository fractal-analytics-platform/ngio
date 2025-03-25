"""Next Generation file format IO."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ngio")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

from ngio.common import ArrayLike, Dimensions
from ngio.hcs import OmeZarrPlate, create_empty_plate, open_omezarr_plate
from ngio.images import (
    Image,
    Label,
    OmeZarrContainer,
    create_empty_omezarr,
    create_omezarr_from_array,
    open_image,
    open_omezarr_container,
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
    "create_empty_omezarr",
    "create_empty_plate",
    "create_omezarr_from_array",
    "open_image",
    "open_omezarr_container",
    "open_omezarr_plate",
]

"""Next Generation file format IO."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ngio")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

from ngio.common import ArrayLike, Dimensions
from ngio.hcs import OmeZarrPlate, OmeZarrWell
from ngio.images import (
    Image,
    Label,
    OmeZarrImage,
    open_omezarr_image,
    open_single_image,
)
from ngio.ome_zarr_meta.ngio_specs import AxesSetup, PixelSize

__all__ = [
    "ArrayLike",
    "AxesSetup",
    "Dimensions",
    "Image",
    "Label",
    "OmeZarrImage",
    "OmeZarrPlate",
    "OmeZarrWell",
    "PixelSize",
    "open_omezarr_image",
    "open_single_image",
]

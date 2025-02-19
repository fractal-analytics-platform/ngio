"""Next Generation file format IO."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ngio")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

from ngio.ome_zarr_meta.ngio_specs import AxesSetup, PixelSize

__all__ = ["AxesSetup", "PixelSize"]

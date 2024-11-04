"""Next Generation file format IO."""

from importlib.metadata import PackageNotFoundError, version

from ngio.core import Image, Label, NgffImage

__all__ = ["Image", "Label", "NgffImage"]


try:
    __version__ = version("ngio")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

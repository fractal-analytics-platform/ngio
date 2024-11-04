"""Next Generation file format IO."""

import os
from importlib.metadata import PackageNotFoundError, version

from ngio.core import Image, Label, NgffImage
from ngio.utils import ngio_logger, set_logger_level

__all__ = ["Image", "Label", "NgffImage", "set_logger_level", "ngio_logger"]

set_logger_level(os.getenv("NGIO_LOGGER_LEVEL", "WARNING"))

try:
    __version__ = version("ngio")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

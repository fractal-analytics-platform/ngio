"""Various utilities for the ngio package."""

import os

from ngio.common._common_types import ArrayLike
from ngio.utils._datasets import download_ome_zarr_dataset, list_ome_zarr_datasets
from ngio.utils._errors import (
    NgioFileExistsError,
    NgioFileNotFoundError,
    NgioTableValidationError,
    NgioValidationError,
    NgioValueError,
)
from ngio.utils._fractal_fsspec_store import fractal_fsspec_store
from ngio.utils._logger import ngio_logger, set_logger_level
from ngio.utils._zarr_utils import (
    AccessModeLiteral,
    StoreOrGroup,
    ZarrGroupHandler,
    open_group_wrapper,
)

set_logger_level(os.getenv("NGIO_LOGGER_LEVEL", "WARNING"))

__all__ = [
    # Zarr
    "AccessModeLiteral",
    "ArrayLike",
    # Errors
    "NgioFileExistsError",
    "NgioFileNotFoundError",
    "NgioTableValidationError",
    "NgioValidationError",
    "NgioValueError",
    "StoreOrGroup",
    "ZarrGroupHandler",
    # Datasets
    "download_ome_zarr_dataset",
    # Fractal
    "fractal_fsspec_store",
    "list_ome_zarr_datasets",
    # Logger
    "ngio_logger",
    "open_group_wrapper",
    "set_logger_level",
]

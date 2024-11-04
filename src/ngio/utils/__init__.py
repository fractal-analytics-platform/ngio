"""Various utilities for the ngio package."""

from ngio.utils._common_types import ArrayLike
from ngio.utils._errors import (
    NgioFileExistsError,
    NgioFileNotFoundError,
    NgioNGFFValidationError,
    NgioTableValidationError,
)
from ngio.utils._logger import ngio_logger, set_logger_level
from ngio.utils._pydantic_utils import BaseWithExtraFields, unique_items_validator

__all__ = [
    "ArrayLike",
    # Pydantic
    "BaseWithExtraFields",
    "unique_items_validator",
    # Logger
    "ngio_logger",
    "set_logger_level",
    # Errors
    "NgioFileExistsError",
    "NgioFileNotFoundError",
    "NgioNGFFValidationError",
    "NgioTableValidationError",
]

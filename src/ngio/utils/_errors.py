# Create a generic error class for the NGFF project


class NgioError(Exception):
    """Base class for all errors in the NGFF project."""

    pass


class NgioFileNotFoundError(NgioError):
    """Error raised when a file is not found."""

    pass


class NgioFileExistsError(NgioError):
    """Error raised when a file already exists."""

    pass


class NgioNGFFValidationError(NgioError):
    """Error raised when a file does not pass validation."""

    pass


class NgioTableValidationError(NgioError):
    """Error raised when a table does not pass validation."""

    pass

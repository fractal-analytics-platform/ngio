# Create a generic error class for the NGFF project


class NgioError(Exception):
    """Base class for all errors in the NGFF project."""

    pass


class NgioFileNotFoundError(NgioError, FileNotFoundError):
    """Error raised when a file is not found."""

    pass


class NgioFileExistsError(NgioError, FileExistsError):
    """Error raised when a file already exists."""

    pass


class NgioValidationError(NgioError, ValueError):
    """Generic error raised when a file does not pass validation."""

    pass


class NgioTableValidationError(NgioError):
    """Error raised when a table does not pass validation."""

    pass


class NgioValueError(NgioError, ValueError):
    """Error raised when a value does not pass a run time test."""

    pass

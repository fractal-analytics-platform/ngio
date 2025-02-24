from typing import Any, Generic, TypeVar

from pydantic import ValidationError

from ngio.common import (
    AccessModeLiteral,
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
)
from ngio.ome_zarr_meta._base_handlers import (
    BaseOmeZarrImageHandler,
    BaseOmeZarrLabelHandler,
)
from ngio.ome_zarr_meta.v04 import OmeZarrV04ImageHandler, OmeZarrV04LabelHandler

_Image_or_Label_Plugin = TypeVar(
    "_Image_or_Label_Plugin", BaseOmeZarrLabelHandler, BaseOmeZarrImageHandler
)


class GenericHandlersManager(Generic[_Image_or_Label_Plugin]):
    """This class is a singleton that manages the available image handler plugins."""

    _instance = None
    _implemented_handlers: dict[str, Any]

    def __new__(cls):
        """Create a new instance of the class if it does not exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._implemented_handlers = {}
        return cls._instance

    def available_handlers(self) -> list[str]:
        """Get the available image handler versions.

        The versions are returned in descending order.
            such that the latest version is the first in the list and the fist to be
            checked.
        """
        return list(reversed(self._implemented_handlers.keys()))

    def get_handler(
        self, store: StoreOrGroup, cache: bool = False, mode: AccessModeLiteral = "a"
    ) -> _Image_or_Label_Plugin:
        """Try to get a handler for the given store based on the metadata version."""
        _errors = {}

        for version, handler in reversed(self._implemented_handlers.items()):
            handler = handler(store=store, cache=cache, mode=mode)
            meta = handler.load(return_error=True)
            if isinstance(meta, ValidationError):
                _errors[version] = meta
                continue
            return handler

        raise NgioValidationError(
            f"Could not load OME-Zarr metadata from any known version. "
            f"Errors: {_errors}"
        )

    def add_handler(
        self, key: str, handler: type[_Image_or_Label_Plugin], overwrite: bool = False
    ):
        """Register a new handler."""
        if key in self._implemented_handlers and not overwrite:
            raise NgioValueError(f"Image handler for version {key} already exists.")
        self._implemented_handlers[key] = handler


class ImageHandlersManager(GenericHandlersManager[BaseOmeZarrImageHandler]):
    def __init__(self):
        super().__init__()
        self.add_handler("0.4", OmeZarrV04ImageHandler)


class LabelHandlersManager(GenericHandlersManager[BaseOmeZarrLabelHandler]):
    def __init__(self):
        super().__init__()
        self.add_handler("0.4", OmeZarrV04LabelHandler)

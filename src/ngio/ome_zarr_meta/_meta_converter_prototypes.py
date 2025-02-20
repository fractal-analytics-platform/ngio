"""Base class for handling OME-NGFF metadata in Zarr groups."""

from typing import Protocol

from pydantic import ValidationError

from ngio.ome_zarr_meta.ngio_specs import NgioImageMeta, NgioLabelMeta

ConverterError = ValidationError | Exception | None


class ImageMetaConverter(Protocol):
    """Protocol for metadata converters."""

    def from_dict(self, meta: dict) -> tuple[bool, NgioImageMeta | ConverterError]:
        """Convert the metadata from a dictionary to the ngio image metadata.

        This function should return a monadic style return:
            * is_success = True -> converted_meta
            * is_success = False -> error


        Ideally if the metadata is valid but the conversion fails for other reasons,
            the error should be raised.

        Args:
            meta (dict): The metadata in dictionary form
                (usually the Zarr group attributes).

        Returns:
            tuple[bool, NgioImageLabelMeta | ConverterError]: Monadic style return.
        """
        ...

    def to_dict(self, meta: NgioImageMeta) -> dict:
        """Convert the ngio image metadata to a dictionary."""
        ...


class LabelMetaConverter(Protocol):
    """Protocol for metadata converters."""

    def from_dict(self, meta: dict) -> tuple[bool, NgioLabelMeta | ConverterError]:
        """Convert the metadata from a dictionary to the ngio label metadata.

        This function should return a monadic style return:
            * is_success = True -> converted_meta
            * is_success = False -> error


        Ideally if the metadata is valid but the conversion fails for other reasons,
            the error should be raised.

        Args:
            meta (dict): The metadata in dictionary form
                (usually the Zarr group attributes).

        Returns:
            tuple[bool, NgioImageLabelMeta | ConverterError]: Monadic style return.
        """
        ...

    def to_dict(self, meta: NgioLabelMeta) -> dict:
        """Convert the ngio label metadata to a dictionary."""
        ...

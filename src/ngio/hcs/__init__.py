"""OME-Zarr HCS objects models."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ngio.images import OmeZarrContainer


class OmeZarrPlate:
    """Placeholder for the OME-Zarr image object."""

    def __init__(self, *args, **kwargs):
        """Initialize the OME-Zarr plate."""
        raise NotImplementedError

    def wells(self) -> list[str]:
        """Return the wells."""
        raise NotImplementedError

    def columns(self) -> list[str]:
        """Return the number of columns."""
        raise NotImplementedError

    def rows(self) -> list[str]:
        """Return the number of rows."""
        raise NotImplementedError

    def get_omezarr_well(self, *args, **kwargs) -> "OmeZarrWell":
        """Return the OME-Zarr well."""
        raise NotImplementedError

    def get_omezarr_image(self, *args, **kwargs) -> "OmeZarrContainer":
        """Return the OME-Zarr image."""
        raise NotImplementedError


class OmeZarrWell:
    """Placeholder for the Image object."""

    def __init__(self, *args, **kwargs):
        """Initialize the OME-Zarr well."""
        raise NotImplementedError

    def acquisitions(self) -> list[str]:
        """Return the acquisition."""
        raise NotImplementedError

    def get_ome_zarr_image(self, *args, **kwargs) -> "OmeZarrContainer":
        """Return the OME-Zarr image."""
        raise NotImplementedError


def open_omezarr_plate(*args, **kwargs):
    """Open an OME-Zarr plate."""
    return OmeZarrPlate(*args, **kwargs)


def open_omezarr_well(*args, **kwargs):
    """Open an OME-Zarr well."""
    return OmeZarrWell(*args, **kwargs)

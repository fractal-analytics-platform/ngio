"""A module for handling the Plate Collection in an OME-Zarr file."""

from ngio.images import OmeZarrContainer
from ngio.ome_zarr_meta import (
    find_plate_meta_handler,
    find_well_meta_handler,
)
from ngio.utils import AccessModeLiteral, StoreOrGroup, ZarrGroupHandler


class OmeZarrWell:
    """A class to handle the Well Collection in an OME-Zarr file."""

    def __init__(self, group_handler: ZarrGroupHandler) -> None:
        """Initialize the LabelGroupHandler.

        Args:
            group_handler: The Zarr group handler that contains the Well.
        """
        self._group_handler = group_handler
        self._meta_handler = find_well_meta_handler(group_handler)

    @property
    def meta_handler(self):
        """Return the metadata handler."""
        return self._meta_handler

    @property
    def meta(self):
        """Return the metadata."""
        return self._meta_handler.meta

    def paths(self, acquisition: int | None = None) -> list[str]:
        """Return the images paths in the well.

        If acquisition is None, return all images paths in the well.
        Else, return the images paths in the well for the given acquisition.

        Args:
            acquisition (int | None): The acquisition id to filter the images.
        """
        return self.meta.paths(acquisition)


class OmeZarrPlate:
    """A class to handle the Plate Collection in an OME-Zarr file."""

    def __init__(self, group_handler: ZarrGroupHandler) -> None:
        """Initialize the LabelGroupHandler.

        Args:
            group_handler: The Zarr group handler that contains the Plate.
        """
        self._group_handler = group_handler
        self._meta_handler = find_plate_meta_handler(group_handler)

    @property
    def meta_handler(self):
        """Return the metadata handler."""
        return self._meta_handler

    @property
    def meta(self):
        """Return the metadata."""
        return self._meta_handler.meta

    @property
    def columns(self) -> list[str]:
        """Return the number of columns in the plate."""
        return self.meta.columns

    @property
    def rows(self) -> list[str]:
        """Return the number of rows in the plate."""
        return self.meta.rows

    @property
    def acquisitions_names(self) -> list[str | None]:
        """Return the acquisitions in the plate."""
        return self.meta.acquisitions_names

    @property
    def acquisitions_ids(self) -> list[int]:
        """Return the acquisitions ids in the plate."""
        return self.meta.acquisitions_ids

    @property
    def wells_paths(self) -> list[str]:
        """Return the wells paths in the plate."""
        return self.meta.wells_paths

    def get_well(self, row: str, column: int | str) -> OmeZarrWell:
        """Get a well from the plate.

        Args:
            row (str): The row of the well.
            column (int | str): The column of the well.

        Returns:
            OmeZarrWell: The well.
        """
        well_path = self.meta.get_well_path(row=row, column=column)
        group_handler = self._group_handler.derive_handler(well_path)
        return OmeZarrWell(group_handler)

    def get_wells(self) -> dict[str, OmeZarrWell]:
        """Get all wells in the plate."""
        wells = {}
        for well_path in self.wells_paths:
            group_handler = self._group_handler.derive_handler(well_path)
            well = OmeZarrWell(group_handler)
            wells[well_path] = well
        return wells

    def get_images(self, acquisition: int | None = None) -> list[OmeZarrContainer]:
        """Get all images in the plate.

        Args:
            acquisition: The acquisition id to filter the images.
        """
        images = []
        for well_path, well in self.get_wells().items():
            for img_path in well.paths(acquisition):
                full_path = f"{well_path}/{img_path}"
                img_group_handler = self._group_handler.derive_handler(full_path)
                images.append(OmeZarrContainer(img_group_handler))
        return images

    def get_well_images(
        self, row: str, column: str | int, acquisition: int | None = None
    ) -> list[OmeZarrContainer]:
        """Get all images in a well.

        Args:
            row: The row of the well.
            column: The column of the well.
            acquisition: The acquisition id to filter the images.
        """
        well_path = self.meta.get_well_path(row=row, column=column)
        group_handler = self._group_handler.derive_handler(well_path)
        well = OmeZarrWell(group_handler)

        images = []
        for path in well.paths(acquisition):
            image_path = f"{well_path}/{path}"
            group_handler = self._group_handler.derive_handler(image_path)
            images.append(OmeZarrContainer(group_handler))

        return images


def open_omezarr_plate(
    store: StoreOrGroup,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
) -> OmeZarrPlate:
    """Open an OME-Zarr plate.

    Args:
        store (StoreOrGroup): The Zarr store or group that stores the plate.
        cache (bool): Whether to use a cache for the zarr group metadata.
        mode (AccessModeLiteral): The
            access mode for the image. Defaults to "r+".
    """
    group_handler = ZarrGroupHandler(store=store, cache=cache, mode=mode)
    return OmeZarrPlate(group_handler)

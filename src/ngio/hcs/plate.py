"""A module for handling the Plate Collection in an OME-Zarr file."""

from ngio.images import OmeZarrContainer
from ngio.ome_zarr_meta import (
    ImageInWellPath,
    NgioPlateMeta,
    NgioWellMeta,
    find_plate_meta_handler,
    find_well_meta_handler,
    get_plate_meta_handler,
    get_well_meta_handler,
)
from ngio.utils import (
    AccessModeLiteral,
    StoreOrGroup,
    ZarrGroupHandler,
)


# Mock lock class that does nothing
class MockLock:
    """A mock lock class that does nothing."""

    def __enter__(self):
        """Enter the lock."""
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the lock."""
        pass


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

    def __repr__(self) -> str:
        """Return a string representation of the plate."""
        return f"Plate([rows x columns] ({len(self.rows)} x {len(self.columns)})"

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

    def _well_path(self, row: str, column: int | str) -> str:
        """Return the well path in the plate."""
        return self.meta.get_well_path(row=row, column=column)

    def _image_path(self, row: str, column: int | str, path: str) -> str:
        """Return the image path in the plate."""
        well = self.get_well(row, column)
        if path not in well.paths():
            raise ValueError(f"Image {path} does not exist in well {row}{column}")
        return f"{self._well_path(row, column)}/{path}"

    def wells_paths(self) -> list[str]:
        """Return the wells paths in the plate."""
        return self.meta.wells_paths

    def images_paths(self, acquisition: int | None = None) -> list[str]:
        """Return the images paths in the plate.

        If acquisition is None, return all images paths in the plate.
        Else, return the images paths in the plate for the given acquisition.

        Args:
            acquisition (int | None): The acquisition id to filter the images.
        """
        images = []
        for well_path, wells in self.get_wells().items():
            for img_path in wells.paths(acquisition):
                images.append(f"{well_path}/{img_path}")
        return images

    def well_images_paths(
        self, row: str, column: int | str, acquisition: int | None = None
    ) -> list[str]:
        """Return the images paths in a well.

        If acquisition is None, return all images paths in the well.
        Else, return the images paths in the well for the given acquisition.

        Args:
            row (str): The row of the well.
            column (int | str): The column of the well.
            acquisition (int | None): The acquisition id to filter the images.
        """
        images = []
        well = self.get_well(row=row, column=column)
        for path in well.paths(acquisition):
            images.append(self._image_path(row=row, column=column, path=path))
        return images

    def get_well(self, row: str, column: int | str) -> OmeZarrWell:
        """Get a well from the plate.

        Args:
            row (str): The row of the well.
            column (int | str): The column of the well.

        Returns:
            OmeZarrWell: The well.
        """
        well_path = self._well_path(row=row, column=column)
        group_handler = self._group_handler.derive_handler(well_path)
        return OmeZarrWell(group_handler)

    def get_wells(self) -> dict[str, OmeZarrWell]:
        """Get all wells in the plate.

        Returns:
            dict[str, OmeZarrWell]: A dictionary of wells, where the key is the well
                path and the value is the well object.
        """
        wells = {}
        for well_path in self.wells_paths():
            group_handler = self._group_handler.derive_handler(well_path)
            well = OmeZarrWell(group_handler)
            wells[well_path] = well
        return wells

    def get_images(self, acquisition: int | None = None) -> dict[str, OmeZarrContainer]:
        """Get all images in the plate.

        Args:
            acquisition: The acquisition id to filter the images.
        """
        images = {}
        for image_path in self.images_paths(acquisition):
            img_group_handler = self._group_handler.derive_handler(image_path)
            images[image_path] = OmeZarrContainer(img_group_handler)
        return images

    def get_image(
        self, row: str, column: int | str, image_path: str
    ) -> OmeZarrContainer:
        """Get an image from the plate.

        Args:
            row (str): The row of the well.
            column (int | str): The column of the well.
            image_path (str): The path of the image.

        Returns:
            OmeZarrContainer: The image.
        """
        image_path = self._image_path(row=row, column=column, path=image_path)
        group_handler = self._group_handler.derive_handler(image_path)
        return OmeZarrContainer(group_handler)

    def get_well_images(
        self, row: str, column: str | int, acquisition: int | None = None
    ) -> dict[str, OmeZarrContainer]:
        """Get all images in a well.

        Args:
            row: The row of the well.
            column: The column of the well.
            acquisition: The acquisition id to filter the images.
        """
        images = {}
        for image_paths in self.well_images_paths(
            row=row, column=column, acquisition=acquisition
        ):
            group_handler = self._group_handler.derive_handler(image_paths)
            images[image_paths] = OmeZarrContainer(group_handler)
        return images

    def _add_image(
        self,
        row: str,
        column: int | str,
        image_path: str,
        acquisition_id: int | None = None,
        acquisition_name: str | None = None,
        atomic: bool = False,
    ) -> StoreOrGroup:
        """Add an image to an ome-zarr plate."""
        if atomic:
            plate_lock = self._group_handler.lock
        else:
            plate_lock = MockLock()

        with plate_lock:
            meta = self.meta
            meta = meta.add_well(row, column, acquisition_id, acquisition_name)
            self.meta_handler.write_meta(meta)
            self.meta_handler._group_handler.clean_cache()

        well_path = self.meta.get_well_path(row=row, column=column)
        group_handler = self._group_handler.derive_handler(well_path)

        if atomic:
            well_lock = group_handler.lock
        else:
            well_lock = MockLock()

        with well_lock:
            attrs = group_handler.load_attrs()
            if len(attrs) == 0:
                # Initialize the well metadata
                # if the group is empty
                well_meta = NgioWellMeta.default_init()
                meta_handler = get_well_meta_handler(group_handler, version="0.4")
            else:
                meta_handler = find_well_meta_handler(group_handler)
                well_meta = meta_handler.meta

            group_handler = self._group_handler.derive_handler(well_path)

            well_meta = well_meta.add_image(path=image_path, acquisition=acquisition_id)
            meta_handler.write_meta(well_meta)
            meta_handler._group_handler.clean_cache()

        return group_handler.get_group(image_path, create_mode=True)

    def atomic_add_image(
        self,
        row: str,
        column: int | str,
        image_path: str,
        acquisition_id: int | None = None,
        acquisition_name: str | None = None,
    ) -> StoreOrGroup:
        """Parallel safe version of add_image."""
        return self._add_image(
            row=row,
            column=column,
            image_path=image_path,
            acquisition_id=acquisition_id,
            acquisition_name=acquisition_name,
            atomic=True,
        )

    def add_image(
        self,
        row: str,
        column: int | str,
        image_path: str,
        acquisition_id: int | None = None,
        acquisition_name: str | None = None,
    ) -> StoreOrGroup:
        """Add an image to an ome-zarr plate."""
        return self._add_image(
            row=row,
            column=column,
            image_path=image_path,
            acquisition_id=acquisition_id,
            acquisition_name=acquisition_name,
            atomic=False,
        )

    def _remove_well(
        self,
        row: str,
        column: int | str,
        atomic: bool = False,
    ):
        """Remove a well from an ome-zarr plate."""
        if atomic:
            plate_lock = self._group_handler.lock
        else:
            plate_lock = MockLock()

        with plate_lock:
            meta = self.meta
            meta = meta.remove_well(row, column)
            self.meta_handler.write_meta(meta)
            self.meta_handler._group_handler.clean_cache()

    def _remove_image(
        self,
        row: str,
        column: int | str,
        image_path: str,
        atomic: bool = False,
    ):
        """Remove an image from an ome-zarr plate."""
        well = self.get_well(row, column)

        if atomic:
            well_lock = well.meta_handler._group_handler.lock
        else:
            well_lock = MockLock()

        with well_lock:
            well_meta = well.meta
            well_meta = well_meta.remove_image(path=image_path)
            well.meta_handler.write_meta(well_meta)
            well.meta_handler._group_handler.clean_cache()
            if len(well_meta.paths()) == 0:
                self._remove_well(row, column, atomic=atomic)

    def atomic_remove_image(
        self,
        row: str,
        column: int | str,
        image_path: str,
    ):
        """Parallel safe version of remove_image."""
        return self._remove_image(
            row=row,
            column=column,
            image_path=image_path,
            atomic=True,
        )

    def remove_image(
        self,
        row: str,
        column: int | str,
        image_path: str,
    ):
        """Remove an image from an ome-zarr plate."""
        return self._remove_image(
            row=row,
            column=column,
            image_path=image_path,
            atomic=False,
        )


def open_ome_zarr_plate(
    store: StoreOrGroup,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
    parallel_safe: bool = True,
) -> OmeZarrPlate:
    """Open an OME-Zarr plate.

    Args:
        store (StoreOrGroup): The Zarr store or group that stores the plate.
        cache (bool): Whether to use a cache for the zarr group metadata.
        mode (AccessModeLiteral): The
            access mode for the image. Defaults to "r+".
        parallel_safe (bool): Whether the group handler is parallel safe.
    """
    group_handler = ZarrGroupHandler(
        store=store, cache=cache, mode=mode, parallel_safe=parallel_safe
    )
    return OmeZarrPlate(group_handler)


def create_empty_plate(
    store: StoreOrGroup,
    name: str,
    images: list[ImageInWellPath] | None = None,
    version: str = "0.4",
    cache: bool = False,
    overwrite: bool = False,
    parallel_safe: bool = True,
) -> OmeZarrPlate:
    """Initialize and create an empty OME-Zarr plate."""
    mode = "w" if overwrite else "w-"
    group_handler = ZarrGroupHandler(
        store=store, cache=True, mode=mode, parallel_safe=False
    )
    meta_handler = get_plate_meta_handler(group_handler, version=version)
    plate_meta = NgioPlateMeta.default_init(
        name=name,
        version=version,
    )
    meta_handler.write_meta(plate_meta)

    if images is not None:
        plate = OmeZarrPlate(group_handler)
        for image in images:
            plate.add_image(
                row=image.row,
                column=image.column,
                image_path=image.path,
                acquisition_id=image.acquisition_id,
                acquisition_name=image.acquisition_name,
            )
    return open_ome_zarr_plate(
        store=store,
        cache=cache,
        mode="r+",
        parallel_safe=parallel_safe,
    )

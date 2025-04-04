"""A module for handling the Plate Collection in an OME-Zarr file."""

from ngio.images import OmeZarrContainer
from ngio.ome_zarr_meta import (
    ImageInWellPath,
    NgffVersion,
    NgioPlateMeta,
    NgioWellMeta,
    find_plate_meta_handler,
    find_well_meta_handler,
    get_plate_meta_handler,
    get_well_meta_handler,
    path_in_well_validation,
)
from ngio.utils import AccessModeLiteral, StoreOrGroup, ZarrGroupHandler


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

    def __repr__(self) -> str:
        """Return a string representation of the well."""
        return f"Well(#images: {len(self.paths())})"

    @property
    def meta_handler(self):
        """Return the metadata handler."""
        return self._meta_handler

    @property
    def meta(self):
        """Return the metadata."""
        return self._meta_handler.meta

    @property
    def acquisition_ids(self) -> list[int]:
        """Return the acquisitions ids in the well."""
        return self.meta.acquisition_ids

    def paths(self, acquisition: int | None = None) -> list[str]:
        """Return the images paths in the well.

        If acquisition is None, return all images paths in the well.
        Else, return the images paths in the well for the given acquisition.

        Args:
            acquisition (int | None): The acquisition id to filter the images.
        """
        return self.meta.paths(acquisition)

    def get_image_store(self, image_path: str) -> StoreOrGroup:
        """Get the image store from the well.

        Args:
            image_path (str): The path of the image.
        """
        return self._group_handler.get_group(image_path, create_mode=True)

    def _add_image(
        self,
        image_path: str,
        acquisition_id: int | None = None,
        strict: bool = True,
        atomic: bool = False,
    ) -> StoreOrGroup:
        """Add an image to an ome-zarr well."""
        image_path = path_in_well_validation(path=image_path)

        if atomic:
            well_lock = self._group_handler.lock
        else:
            well_lock = MockLock()

        with well_lock:
            meta = self.meta.add_image(
                path=image_path, acquisition=acquisition_id, strict=strict
            )
            self.meta_handler.write_meta(meta)
            self.meta_handler._group_handler.clean_cache()

        return self._group_handler.get_group(image_path, create_mode=True)

    def atomic_add_image(
        self,
        image_path: str,
        acquisition_id: int | None = None,
        strict: bool = True,
    ) -> StoreOrGroup:
        """Parallel safe version of add_image."""
        return self._add_image(
            image_path=image_path,
            acquisition_id=acquisition_id,
            atomic=True,
            strict=strict,
        )

    def add_image(
        self,
        image_path: str,
        acquisition_id: int | None = None,
        strict: bool = True,
    ) -> StoreOrGroup:
        """Add an image to an ome-zarr well.

        Args:
            image_path (str): The path of the image.
            acquisition_id (int | None): The acquisition id to filter the images.
            strict (bool): Whether to check if the acquisition id is already exists
                in the well. Defaults to True. If False this might lead to
                acquision in a well that does not exist at the plate level.
        """
        return self._add_image(
            image_path=image_path,
            acquisition_id=acquisition_id,
            atomic=False,
            strict=strict,
        )


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
        return f"Plate([rows x columns] ({len(self.rows)} x {len(self.columns)}))"

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
    def acquisition_ids(self) -> list[int]:
        """Return the acquisitions ids in the plate."""
        return self.meta.acquisition_ids

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

    def get_image_store(
        self, row: str, column: int | str, image_path: str
    ) -> StoreOrGroup:
        """Get the image store from the plate.

        Args:
            row (str): The row of the well.
            column (int | str): The column of the well.
            image_path (str): The path of the image.
        """
        well = self.get_well(row=row, column=column)
        return well.get_image_store(image_path=image_path)

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
        image_path: str | None = None,
        acquisition_id: int | None = None,
        acquisition_name: str | None = None,
        atomic: bool = False,
    ) -> StoreOrGroup | None:
        """Add an image to an ome-zarr plate."""
        if image_path is not None:
            image_path = path_in_well_validation(path=image_path)

        if atomic:
            plate_lock = self._group_handler.lock
        else:
            plate_lock = MockLock()

        with plate_lock:
            meta = self.meta
            meta = meta.add_well(row=row, column=column)
            if acquisition_id is not None:
                meta = meta.add_acquisition(
                    acquisition_id=acquisition_id, acquisition_name=acquisition_name
                )
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
                version = self.meta.plate.version
                version = version if version is not None else "0.4"
                meta_handler = get_well_meta_handler(group_handler, version=version)
            else:
                meta_handler = find_well_meta_handler(group_handler)
                well_meta = meta_handler.meta

            group_handler = self._group_handler.derive_handler(well_path)

            if image_path is not None:
                well_meta = well_meta.add_image(
                    path=image_path, acquisition=acquisition_id, strict=False
                )
            meta_handler.write_meta(well_meta)
            meta_handler._group_handler.clean_cache()

        if image_path is not None:
            return group_handler.get_group(image_path, create_mode=True)
        return None

    def atomic_add_image(
        self,
        row: str,
        column: int | str,
        image_path: str,
        acquisition_id: int | None = None,
        acquisition_name: str | None = None,
    ) -> StoreOrGroup:
        """Parallel safe version of add_image."""
        if image_path is None:
            raise ValueError(
                "Image path cannot be None for atomic add_image. "
                "If your intent is to add a well, use add_well instead."
            )
        group = self._add_image(
            row=row,
            column=column,
            image_path=image_path,
            acquisition_id=acquisition_id,
            acquisition_name=acquisition_name,
            atomic=True,
        )
        if group is None:
            raise ValueError(
                f"Some error occurred while adding image {image_path} "
                f"to well {row}{column}."
            )
        return group

    def add_image(
        self,
        row: str,
        column: int | str,
        image_path: str,
        acquisition_id: int | None = None,
        acquisition_name: str | None = None,
    ) -> StoreOrGroup:
        """Add an image to an ome-zarr plate."""
        if image_path is None:
            raise ValueError(
                "Image path cannot be None for atomic add_image. "
                "If your intent is to add a well, use add_well instead."
            )
        group = self._add_image(
            row=row,
            column=column,
            image_path=image_path,
            acquisition_id=acquisition_id,
            acquisition_name=acquisition_name,
            atomic=False,
        )
        if group is None:
            raise ValueError(
                f"Some error occurred while adding image {image_path} "
                f"to well {row}{column}."
            )
        return group

    def add_well(
        self,
        row: str,
        column: int | str,
    ) -> OmeZarrWell:
        """Add a well to an ome-zarr plate."""
        _ = self._add_image(
            row=row,
            column=column,
            image_path=None,
            acquisition_id=None,
            acquisition_name=None,
            atomic=False,
        )
        return self.get_well(row=row, column=column)

    def add_column(
        self,
        column: int | str,
    ) -> "OmeZarrPlate":
        """Add a column to an ome-zarr plate."""
        meta, _ = self.meta.add_column(column)
        self.meta_handler.write_meta(meta)
        self.meta_handler._group_handler.clean_cache()
        return self

    def add_row(
        self,
        row: str,
    ) -> "OmeZarrPlate":
        """Add a row to an ome-zarr plate."""
        meta, _ = self.meta.add_row(row)
        self.meta_handler.write_meta(meta)
        self.meta_handler._group_handler.clean_cache()
        return self

    def add_acquisition(
        self,
        acquisition_id: int,
        acquisition_name: str,
    ) -> "OmeZarrPlate":
        """Add an acquisition to an ome-zarr plate.

        Be aware that this is not a parallel safe operation.

        Args:
            acquisition_id (int): The acquisition id.
            acquisition_name (str): The acquisition name.
        """
        meta = self.meta.add_acquisition(
            acquisition_id=acquisition_id, acquisition_name=acquisition_name
        )
        self.meta_handler.write_meta(meta)
        self.meta_handler._group_handler.clean_cache()
        return self

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

    def derive_plate(
        self,
        store: StoreOrGroup,
        plate_name: str | None = None,
        version: NgffVersion = "0.4",
        keep_acquisitions: bool = True,
        cache: bool = False,
        overwrite: bool = False,
        parallel_safe: bool = True,
    ) -> "OmeZarrPlate":
        """Derive a new OME-Zarr plate from an existing one.

        Args:
            store (StoreOrGroup): The Zarr store or group that stores the plate.
            plate_name (str | None): The name of the new plate.
            version (NgffVersion): The version of the new plate.
            keep_acquisitions (bool): Whether to keep the acquisitions in the new plate.
            cache (bool): Whether to use a cache for the zarr group metadata.
            overwrite (bool): Whether to overwrite the existing plate.
            parallel_safe (bool): Whether the group handler is parallel safe.
        """
        return derive_ome_zarr_plate(
            ome_zarr_plate=self,
            store=store,
            plate_name=plate_name,
            version=version,
            keep_acquisitions=keep_acquisitions,
            cache=cache,
            overwrite=overwrite,
            parallel_safe=parallel_safe,
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


def _create_empty_plate_from_meta(
    store: StoreOrGroup,
    meta: NgioPlateMeta,
    version: NgffVersion = "0.4",
    overwrite: bool = False,
) -> ZarrGroupHandler:
    """Create an empty OME-Zarr plate from metadata."""
    mode = "w" if overwrite else "w-"
    group_handler = ZarrGroupHandler(
        store=store, cache=True, mode=mode, parallel_safe=False
    )
    meta_handler = get_plate_meta_handler(group_handler, version=version)
    meta_handler.write_meta(meta)
    return group_handler


def create_empty_plate(
    store: StoreOrGroup,
    name: str,
    images: list[ImageInWellPath] | None = None,
    version: NgffVersion = "0.4",
    cache: bool = False,
    overwrite: bool = False,
    parallel_safe: bool = True,
) -> OmeZarrPlate:
    """Initialize and create an empty OME-Zarr plate."""
    plate_meta = NgioPlateMeta.default_init(
        name=name,
        version=version,
    )
    group_handler = _create_empty_plate_from_meta(
        store=store,
        meta=plate_meta,
        version=version,
        overwrite=overwrite,
    )

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


def derive_ome_zarr_plate(
    ome_zarr_plate: OmeZarrPlate,
    store: StoreOrGroup,
    plate_name: str | None = None,
    version: NgffVersion = "0.4",
    keep_acquisitions: bool = True,
    cache: bool = False,
    overwrite: bool = False,
    parallel_safe: bool = True,
) -> OmeZarrPlate:
    """Derive a new OME-Zarr plate from an existing one.

    Args:
        ome_zarr_plate (OmeZarrPlate): The existing OME-Zarr plate.
        store (StoreOrGroup): The Zarr store or group that stores the plate.
        plate_name (str | None): The name of the new plate.
        version (NgffVersion): The version of the new plate.
        keep_acquisitions (bool): Whether to keep the acquisitions in the new plate.
        cache (bool): Whether to use a cache for the zarr group metadata.
        overwrite (bool): Whether to overwrite the existing plate.
        parallel_safe (bool): Whether the group handler is parallel safe.
    """
    if plate_name is None:
        plate_name = ome_zarr_plate.meta.plate.name

    new_meta = ome_zarr_plate.meta.derive(
        name=plate_name,
        version=version,
        keep_acquisitions=keep_acquisitions,
    )
    _ = _create_empty_plate_from_meta(
        store=store,
        meta=new_meta,
        overwrite=overwrite,
        version=version,
    )
    return open_ome_zarr_plate(
        store=store,
        cache=cache,
        mode="r+",
        parallel_safe=parallel_safe,
    )


def open_ome_zarr_well(
    store: StoreOrGroup,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
    parallel_safe: bool = True,
) -> OmeZarrWell:
    """Open an OME-Zarr well.

    Args:
        store (StoreOrGroup): The Zarr store or group that stores the plate.
        cache (bool): Whether to use a cache for the zarr group metadata.
        mode (AccessModeLiteral): The access mode for the image. Defaults to "r+".
        parallel_safe (bool): Whether the group handler is parallel safe.
    """
    group_handler = ZarrGroupHandler(
        store=store, cache=cache, mode=mode, parallel_safe=parallel_safe
    )
    return OmeZarrWell(group_handler)


def create_empty_well(
    store: StoreOrGroup,
    version: NgffVersion = "0.4",
    cache: bool = False,
    overwrite: bool = False,
    parallel_safe: bool = True,
) -> OmeZarrWell:
    """Create an empty OME-Zarr well.

    Args:
        store (StoreOrGroup): The Zarr store or group that stores the well.
        version (NgffVersion): The version of the new well.
        cache (bool): Whether to use a cache for the zarr group metadata.
        overwrite (bool): Whether to overwrite the existing well.
        parallel_safe (bool): Whether the group handler is parallel safe.
    """
    group_handler = ZarrGroupHandler(
        store=store, cache=True, mode="w" if overwrite else "w-", parallel_safe=False
    )
    meta_handler = get_well_meta_handler(group_handler, version=version)
    meta = NgioWellMeta.default_init()
    meta_handler.write_meta(meta)

    return open_ome_zarr_well(
        store=store,
        cache=cache,
        mode="r+",
        parallel_safe=parallel_safe,
    )

"""Abstract class for handling OME-NGFF images."""

from typing import Literal, overload

from ngio.images.image import Image, ImagesContainer
from ngio.images.label import Label, LabelsContainer
from ngio.ome_zarr_meta import NgioImageMeta, PixelSize
from ngio.tables import (
    FeaturesTable,
    MaskingROITable,
    RoiTable,
    Table,
    TablesContainer,
    TypedTable,
)
from ngio.utils import (
    AccessModeLiteral,
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)


def _default_table_container(handler: ZarrGroupHandler) -> TablesContainer | None:
    """Return a default table container."""
    table_handler = handler.derive_handler("tables")
    return TablesContainer(table_handler)


def _default_label_container(handler: ZarrGroupHandler) -> LabelsContainer | None:
    """Return a default label container."""
    label_handler = handler.derive_handler("labels")
    return LabelsContainer(label_handler)


class OmeZarrContainer:
    """This class contains an OME-Zarr image and its associated tables and labels."""

    _images_container: ImagesContainer
    _labels_container: LabelsContainer | None
    _tables_container: TablesContainer | None

    def __init__(
        self,
        store: StoreOrGroup,
        cache: bool = False,
        mode: AccessModeLiteral = "r+",
        table_container: TablesContainer | None = None,
        label_container: LabelsContainer | None = None,
        validate_arrays: bool = True,
    ) -> None:
        """Initialize the OmeZarrContainer."""
        self._group_handler = ZarrGroupHandler(store, cache, mode)
        self._images_container = ImagesContainer(self._group_handler)

        if label_container is None:
            label_container = _default_label_container(self._group_handler)
        self._labels_container = label_container

        if table_container is None:
            table_container = _default_table_container(self._group_handler)
        self._tables_container = table_container

    def __repr__(self) -> str:
        """Return a string representation of the image."""
        return f"OmeZarrContainer({self.image_meta})"

    @property
    def images_container(self) -> ImagesContainer:
        """Return the image container."""
        return self._images_container

    @property
    def labels_container(self) -> LabelsContainer:
        """Return the labels container."""
        if self._labels_container is None:
            raise NgioValidationError("No labels found in the image.")
        return self._labels_container

    @property
    def tables_container(self) -> TablesContainer:
        """Return the tables container."""
        if self._tables_container is None:
            raise NgioValidationError("No tables found in the image.")
        return self._tables_container

    @property
    def image_meta(self) -> NgioImageMeta:
        """Return the image metadata."""
        return self._images_container.meta()

    @property
    def levels(self) -> int:
        """Return the number of levels in the image."""
        return self._images_container.levels

    @property
    def levels_paths(self) -> list[str]:
        """Return the paths of the levels in the image."""
        return self._images_container.levels_paths

    def get_image(
        self,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        highest_resolution: bool = True,
    ) -> Image:
        """Get an image at a specific level."""
        return self._images_container.get(
            path=path,
            pixel_size=pixel_size,
            highest_resolution=highest_resolution,
        )

    def list_tables(self) -> list[str]:
        """List all tables in the image."""
        if self._tables_container is None:
            return []
        return self._tables_container.list()

    @overload
    def get_table(self, table_name: str) -> Table: ...

    @overload
    def get_table(self, table_name: str, check_type: None) -> Table: ...

    @overload
    def get_table(
        self, table_name: str, check_type: Literal["roi_table"]
    ) -> RoiTable: ...

    @overload
    def get_table(
        self, table_name: str, check_type: Literal["masking_roi_table"]
    ) -> MaskingROITable: ...

    @overload
    def get_table(
        self, table_name: str, check_type: Literal["feature_table"]
    ) -> FeaturesTable: ...

    def get_table(self, table_name: str, check_type: TypedTable | None = None) -> Table:
        """Get a table from the image."""
        if self._tables_container is None:
            raise NgioValidationError("No tables found in the image.")

        table = self._tables_container.get(table_name)
        match check_type:
            case "roi_table":
                if not isinstance(table, RoiTable):
                    raise NgioValueError(
                        f"Table '{table_name}' is not a ROI table. "
                        f"Found type: {table.type()}"
                    )
                return table
            case "masking_roi_table":
                if not isinstance(table, MaskingROITable):
                    raise NgioValueError(
                        f"Table '{table_name}' is not a masking ROI table. "
                        f"Found type: {table.type()}"
                    )
                return table
            case "features_table":
                if not isinstance(table, FeaturesTable):
                    raise NgioValueError(
                        f"Table '{table_name}' is not a features table. "
                        f"Found type: {table.type()}"
                    )
                return table
            case None:
                return table
            case _:
                raise NgioValueError(f"Unknown check_type: {check_type}")

    def add_table(
        self,
        name: str,
        table: Table,
        backend: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Add a table to the image."""
        if self._tables_container is None:
            raise NgioValidationError("No tables found in the image.")
        self._tables_container.add(
            name=name, table=table, backend=backend, overwrite=overwrite
        )

    def list_labels(self) -> list[str]:
        """List all labels in the image."""
        if self._labels_container is None:
            return []
        return self._labels_container.list()

    def get_label(self, label_name: str, path: str) -> Label:
        """Get a label from the image."""
        if self._labels_container is None:
            raise NgioValidationError("No labels found in the image.")
        return self._labels_container.get(name=label_name, path=path)

    def derive_label(self, label_name: str, **kwargs) -> Label:
        """Derive a label from an image."""
        if self._labels_container is None:
            raise NgioValidationError("No labels found in the image.")

        ref_image = self.get_image()
        return self._labels_container.derive(
            name=label_name, reference_image=ref_image, **kwargs
        )

    def derive(
        self,
        **kwargs,
    ) -> "OmeZarrContainer":
        """Derive a new image from the current image."""
        raise NotImplementedError


def open_omezarr_container(
    store: StoreOrGroup,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
    validate_arrays: bool = True,
) -> OmeZarrContainer:
    """Open an OME-Zarr image."""
    return OmeZarrContainer(
        store=store,
        cache=cache,
        mode=mode,
        validate_arrays=validate_arrays,
    )


def open_image(
    store: StoreOrGroup,
    path: str | None = None,
    pixel_size: PixelSize | None = None,
    highest_resolution: bool = False,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
) -> Image:
    """Open a single level image from an OME-Zarr image."""
    group_handler = ZarrGroupHandler(store, cache, mode)
    images_container = ImagesContainer(group_handler)
    return images_container.get(
        path=path,
        pixel_size=pixel_size,
        highest_resolution=highest_resolution,
    )

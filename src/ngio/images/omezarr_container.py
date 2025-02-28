"""Abstract class for handling OME-NGFF images."""

from typing import Literal, overload

from ngio.images.abstract_image import Image
from ngio.images.label import Label, LabelGroupHandler
from ngio.ome_zarr_meta import NgioImageMeta, PixelSize, open_omezarr_handler
from ngio.tables import (
    FeaturesTable,
    MaskingROITable,
    RoiTable,
    Table,
    TableContainer,
    TypedTable,
    open_table_group,
)
from ngio.utils import (
    AccessModeLiteral,
    NgioFileNotFoundError,
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
)


class OmeZarrContainer:
    """This class contains an OME-Zarr image and its associated tables and labels."""

    def __init__(
        self,
        store: StoreOrGroup,
        cache: bool = False,
        mode: AccessModeLiteral = "r+",
        table_handler: TableContainer | None = None,
        label_handler: LabelGroupHandler | None = None,
        validate_arrays: bool = True,
    ) -> None:
        """Initialize the OmeZarrContainer."""
        self._omezarr_handler = open_omezarr_handler(
            store=store,
            cache=cache,
            mode=mode,
        )
        self.set_table_handler(table_handler)
        self.set_label_handler(label_handler)
        if validate_arrays:
            self.validate()

    def __repr__(self) -> str:
        """Return a string representation of the image."""
        return f"OmeZarrContainer({self.image_meta})"

    @property
    def image_meta(self) -> NgioImageMeta:
        """Return the image metadata."""
        return self._omezarr_handler.meta

    @property
    def levels(self) -> int:
        """Return the number of levels in the image."""
        return self._omezarr_handler.meta.levels

    @property
    def levels_paths(self) -> list[str]:
        """Return the paths of the levels in the image."""
        return self._omezarr_handler.meta.paths

    def validate(self) -> None:
        """Validate the image."""
        for path in self.levels_paths:
            self.get_image(
                path=path
            )  # this will raise an error if the image is invalid

    def _set_image_handler(self, image_handler: Image | None = None) -> None:
        """Set the image handler."""
        raise NotImplementedError

    def set_table_handler(self, table_handler: TableContainer | None = None) -> None:
        """Set the table handler."""
        if table_handler is not None:
            self._tables_handler = table_handler
            return None

        # try to get the "default" table group
        try:
            _table_group = self._omezarr_handler.group_handler.get_group("tables")
        except NgioFileNotFoundError:
            if self._omezarr_handler.group_handler.mode != "r":
                _table_group = self._omezarr_handler.group_handler.create_group(
                    "tables"
                )
            else:
                _table_group = None

        if _table_group is not None:
            self._tables_handler = open_table_group(_table_group)
        else:
            self._tables_handler = None

    def list_tables(self) -> list[str]:
        """List all tables in the image."""
        if self._tables_handler is None:
            return []
        return self._tables_handler.list()

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
        self, table_name: str, check_type: Literal["features_table"]
    ) -> FeaturesTable: ...

    def get_table(self, table_name: str, check_type: TypedTable | None = None) -> Table:
        """Get a table from the image."""
        if self._tables_handler is None:
            raise NgioValidationError("No tables found in the image.")

        table = self._tables_handler.get(table_name)
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
        if self._tables_handler is None:
            raise NgioValidationError("No tables found in the image.")
        self._tables_handler.add(
            name=name, table=table, backend=backend, overwrite=overwrite
        )

    def set_label_handler(self, label_handler: LabelGroupHandler | None = None) -> None:
        """Set the label handler."""
        if label_handler is not None:
            self._labels_handler = label_handler
            return None

        # try to get the "default" label group
        try:
            _label_group = self._omezarr_handler.group_handler.get_group("labels")
        except NgioFileNotFoundError:
            if self._omezarr_handler.group_handler.mode != "r":
                _label_group = self._omezarr_handler.group_handler.create_group(
                    "labels"
                )
            else:
                _label_group = None

        if _label_group is not None:
            self._labels_handler = LabelGroupHandler(_label_group)
        else:
            self._labels_handler = None

    def list_labels(self) -> list[str]:
        """List all labels in the image."""
        if self._labels_handler is None:
            return []
        return self._labels_handler.list()

    def get_label(self, label_name: str, path: str) -> Label:
        """Get a label from the image."""
        if self._labels_handler is None:
            raise NgioValidationError("No labels found in the image.")
        return self._labels_handler.get(name=label_name, path=path)

    def derive_label(self, label_name: str, **kwargs) -> Label:
        """Derive a label from an image."""
        if self._labels_handler is None:
            raise NgioValidationError("No labels found in the image.")

        ref_image = self.get_image()
        return self._labels_handler.derive(
            name=label_name, reference_image=ref_image, **kwargs
        )

    def get_image(
        self,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        highest_resolution: bool = True,
    ) -> Image:
        """Get an image at a specific level."""
        if path is not None or pixel_size is not None:
            highest_resolution = False
        dataset = self.image_meta.get_dataset(
            path=path, pixel_size=pixel_size, highest_resolution=highest_resolution
        )
        return Image(ome_zarr_handler=self._omezarr_handler, path=dataset.path)

    def derive(
        self,
        **kwargs,
    ) -> "OmeZarrContainer":
        """Derive a new image from the current image."""
        raise NotImplementedError


def open_omezarr_image(
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
    return open_omezarr_image(
        store=store,
        cache=cache,
        mode=mode,
        validate_arrays=False,
    ).get_image(
        path=path,
        pixel_size=pixel_size,
        highest_resolution=highest_resolution,
    )

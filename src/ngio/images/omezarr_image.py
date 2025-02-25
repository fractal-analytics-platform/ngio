"""Abstract class for handling OME-NGFF images."""

# %%
from typing import Literal, overload

from ngio.ome_zarr_meta import NgioImageMeta, open_image_meta
from ngio.tables import (
    FeaturesTable,
    MaskingROITable,
    ROITable,
    Table,
    TypedTable,
    open_table_group,
)
from ngio.utils import (
    AccessModeLiteral,
    NgioFileNotFoundError,
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)


class OmeZarrImage:
    """A class to handle OME-NGFF images."""

    def __init__(
        self,
        store: StoreOrGroup,
        cache: bool = False,
        mode: AccessModeLiteral = "r+",
        validate_arrays: bool = True,
    ) -> None:
        """Initialize the NGFFImage in read mode."""
        self._group_handler = ZarrGroupHandler(store=store, cache=cache, mode=mode)

        try:
            _table_group = self._group_handler.get_group("tables")
        except NgioFileNotFoundError:
            if mode != "r":
                _table_group = self._group_handler.group.create_group("tables")
            else:
                _table_group = None

        if _table_group is not None:
            self._tables_handler = open_table_group(_table_group)
        else:
            self._tables_handler = None

        self._labels_handler = None

        self._image_meta_handler = open_image_meta(
            self._group_handler.group, cache=cache, mode=mode
        )

    @property
    def image_meta(self) -> NgioImageMeta:
        """Return the image metadata."""
        return self._image_meta_handler.load()

    @property
    def levels(self) -> int:
        """Return the number of levels in the image."""
        return self.image_meta.levels

    @property
    def levels_paths(self) -> list[str]:
        """Return the paths of the levels in the image."""
        return self.image_meta.paths

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
    ) -> ROITable: ...

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
                if not isinstance(table, ROITable):
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

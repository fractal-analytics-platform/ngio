"""Module for handling the /tables group in an OME-NGFF file."""

from typing import Literal, Protocol, overload

from ngio.tables._generic_table import GenericTable
from ngio.tables.v1 import FeaturesTableV1, MaskingROITableV1, ROITableV1
from ngio.utils import (
    AccessModeLiteral,
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)

ROITable = ROITableV1
MaskingROITable = MaskingROITableV1
FeaturesTable = FeaturesTableV1


class Table(Protocol):
    """Placeholder class for a table."""

    @staticmethod
    def type() -> str | None:
        """Return the type of the table."""
        ...

    @staticmethod
    def version() -> str | None:
        """Return the version of the table."""
        ...

    @property
    def backend_name(self) -> str | None:
        """The name of the backend."""
        ...

    @classmethod
    def from_store(
        cls,
        store: StoreOrGroup,
        cache: bool = False,
        mode: AccessModeLiteral = "a",
        parallel_safe: bool = False,
    ) -> "Table":
        """Create a new table from a Zarr store."""
        ...

    def set_backend(
        self,
        store: StoreOrGroup,
        backend_name: str | None = None,
        cache: bool = False,
        mode: AccessModeLiteral = "a",
        parallel_safe: bool = False,
    ) -> None:
        """Set the backend store and path for the table."""
        ...

    def consolidate(self) -> None:
        """Consolidate the table on disk."""
        ...


TypedTable = Literal["roi_table", "masking_roi_table", "features_table"]


class TablesManager:
    """A singleton class to manage the available table handler plugins."""

    _instance = None
    _implemented_tables: dict[str, type[Table]]

    def __new__(cls):
        """Create a new instance of the class if it does not exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._implemented_tables = {
                "generic": GenericTable,
            }
        return cls._instance

    @staticmethod
    def _unique_name(table_type: str, version: str) -> str:
        """Return the unique name for a table."""
        return f"{table_type}_v{version}"

    def available_implementations(self) -> list[str]:
        """Get the available table handler versions."""
        return list(self._implemented_tables.keys())

    def get_table(
        self,
        type: str,
        version: str,
        store: StoreOrGroup,
        cache: bool = False,
        mode: AccessModeLiteral = "a",
    ) -> Table:
        """Try to get a handler for the given store based on the metadata version."""
        _errors = {}
        for name, table_cls in self._implemented_tables.items():
            if name != self._unique_name(type, version):
                continue
            try:
                handler = table_cls.from_store(store=store, cache=cache, mode=mode)
                return handler
            except Exception as e:
                _errors[name] = e

        # If no table was found, we can try to load the table from a generic table
        try:
            handler = GenericTable.from_store(store=store, cache=cache, mode=mode)
            return handler
        except Exception as e:
            _errors["generic"] = e

        if len(_errors) == 0:
            raise NgioValidationError(
                f"Could not find a table implementation for {type} v{version}. "
                f"Available tables: {self.available_implementations()}"
            )

        raise NgioValidationError(
            f"Could not load table from any known version. Errors: {_errors}"
        )

    def add_implementation(self, handler: type[Table], overwrite: bool = False):
        """Register a new table handler."""
        table_type = handler.type()
        version = handler.version()
        if table_type is None:
            raise NgioValueError("Table handler must have a type.")

        if version is None:
            raise NgioValueError("Table handler must have a version.")

        table_unique_name = f"{table_type}_v{version}"
        if table_unique_name in self._implemented_tables and not overwrite:
            raise NgioValueError(
                f"Table handler for {table_unique_name} already exists. "
                "Use overwrite=True to replace it."
            )
        self._implemented_tables[table_unique_name] = handler


TablesManager().add_implementation(ROITable)


class TablesHandler:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(
        self, store: StoreOrGroup, cache: bool = False, mode: AccessModeLiteral = "a"
    ) -> None:
        """Initialize the LabelGroupHandler."""
        self._group_handler = ZarrGroupHandler(store, cache, mode)

    def _get_tables_list(self) -> list[str]:
        """Create the /tables group if it doesn't exist."""
        attrs = self._group_handler.load_attrs()
        return attrs.get("tables", [])

    def list(self, filter_types: str | None = None) -> list[str]:
        """List all labels in the group."""
        tables = self._get_tables_list()
        if filter_types is None:
            return tables
        return [
            table for table in tables if self._get_table_type(table) in filter_types
        ]

    def _get_table_type(self, name: str) -> str:
        """Get the type of a table."""
        table_group = self._group_handler.get_group(name)
        return table_group.attrs.get("type", "None")

    def _get_table_version(self, name: str) -> str:
        """Get the version of a table."""
        table_group = self._group_handler.get_group(name)
        return table_group.attrs.get("fractal_table_version", "1")

    def _get(self, name: str) -> Table:
        """Get a label from the group."""
        if name not in self.list():
            raise KeyError(f"Table '{name}' not found in the group.")

        table_group = self._group_handler.get_group(name)
        table_type = self._get_table_type(name)
        version = self._get_table_version(name)

        return TablesManager().get_table(
            type=table_type,
            version=version,
            store=table_group,
            cache=self._group_handler.use_cache,
            mode=self._group_handler.mode,
        )

    @overload
    def get(self, name: str, check_type: Literal["roi_table"]) -> ROITable: ...

    @overload
    def get(
        self, name: str, check_type: Literal["masking_roi_table"]
    ) -> MaskingROITable: ...

    @overload
    def get(
        self, name: str, check_type: Literal["features_table"]
    ) -> FeaturesTable: ...

    @overload
    def get(self, name: str, check_type: None) -> Table: ...

    @overload
    def get(self, name: str) -> Table: ...

    def get(
        self,
        name: str,
        check_type: TypedTable | None = None,
    ) -> Table:
        """Get a table from the group."""
        if name not in self.list():
            raise NgioValueError(f"Table '{name}' not found in the group.")

        match check_type:
            case "roi_table":
                table = self._get(name)
                if not isinstance(table, ROITable):
                    raise NgioValueError(
                        f"Table '{name}' is not an ROI table. "
                        f"Found type: {table.type()}"
                    )
                return table
            case "masking_roi_table":
                table = self._get(name)
                if not isinstance(table, MaskingROITable):
                    raise NgioValueError(
                        f"Table '{name}' is not a masking ROI table. "
                        f"Found type: {table.type()}"
                    )
                return table
            case "features_table":
                table = self._get(name)
                if not isinstance(table, FeaturesTable):
                    raise NgioValueError(
                        f"Table '{name}' is not a features table. "
                        f"Found type: {table.type()}"
                    )
                return table
            case None:
                return self._get(name)

    def add(
        self, name: str, table: Table, backend: str | None, overwrite: bool = False
    ) -> None:
        """Add a table to the group."""
        existing_tables = self._get_tables_list()
        if name in existing_tables and not overwrite:
            raise NgioValueError(f"Table '{name}' already exists in the group.")

        table_group = self._group_handler.create_group(name, overwrite=overwrite)

        if backend is None:
            backend = table.backend_name
        table.set_backend(
            backend_name=backend,
            store=table_group,
            cache=self._group_handler.use_cache,
            mode=self._group_handler.mode,
        )
        table.consolidate()
        if name not in existing_tables:
            existing_tables.append(name)
            self._group_handler.write_attrs({"tables": existing_tables})

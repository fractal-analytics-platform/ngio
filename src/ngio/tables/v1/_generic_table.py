"""Implementation of a generic table class."""

from ngio.tables.abstract_table import AbstractBaseTable
from ngio.tables.backends import (
    BackendMeta,
    SupportedTables,
)
from ngio.utils import ZarrGroupHandler


class GenericTable(AbstractBaseTable):
    """Class to a non-specific table.

    This can be used to load any table that does not have
    a specific definition.
    """

    def __init__(
        self,
        table: SupportedTables | None = None,
        *,
        meta: BackendMeta | None = None,
    ) -> None:
        """Initialize the GenericTable."""
        if meta is None:
            meta = BackendMeta()
        super().__init__(
            meta=meta,
            table=table,
        )

    @staticmethod
    def type() -> str:
        """Return the type of the table."""
        return "generic"

    @staticmethod
    def version() -> str:
        """The generic table does not have a version.

        Since does not follow a specific schema.
        """
        return "1"

    @classmethod
    def from_handler(
        cls, handler: ZarrGroupHandler, backend_name: str | None = None
    ) -> "GenericTable":
        return cls._from_handler(
            handler=handler,
            backend_name=backend_name,
            meta_model=BackendMeta,
        )

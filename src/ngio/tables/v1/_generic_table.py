"""Implementation of a generic table class."""

from ngio.tables.abstract_table import AbstractBaseTable
from ngio.tables.backends import BackendMeta, TableBackendProtocol
from ngio.utils import ZarrGroupHandler


class GenericTable(AbstractBaseTable):
    """Class to a non-specific table.

    This can be used to load any table that does not have
    a specific definition.
    """

    @staticmethod
    def table_type() -> str:
        """Return the type of the table."""
        return "generic_table"

    @staticmethod
    def version() -> str:
        """The generic table does not have a version.

        Since does not follow a specific schema.
        """
        return "1"

    @classmethod
    def from_handler(
        cls,
        handler: ZarrGroupHandler,
        backend: str | TableBackendProtocol | None = None,
    ) -> "GenericTable":
        return cls._from_handler(
            handler=handler,
            backend=backend,
            meta_model=BackendMeta,
        )

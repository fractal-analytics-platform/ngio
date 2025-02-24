"""Module for handling the /tables group in an OME-NGFF file."""

from ngio.utils import AccessModeLiteral, StoreOrGroup, ZarrGroupHandler


class Table:
    """Placeholder class for a table."""

    ...


class TableGroup:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(
        self, group: StoreOrGroup, cache: bool = False, mode: AccessModeLiteral = "a"
    ) -> None:
        """Initialize the LabelGroupHandler."""
        self._group_handler = ZarrGroupHandler(group, cache, mode)

    def list_tables(self) -> list[str]:
        """List all labels in the group."""
        ...

    def get(self, name: str) -> Table:
        """Get a label from the group."""
        ...

    def add(self, name: str, table: Table, overwrite: bool = False) -> None:
        """Add a table to the group."""
        ...

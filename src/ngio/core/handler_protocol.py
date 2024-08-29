"""Handler protocol for all handlers to implement."""

from typing import Protocol, TypeAlias

from zarr.store.common import StoreLike

T = TypeAlias["T"]


class HandlerProtocol[T](Protocol):
    """Basic protocol that all handlers should implement."""

    def __init__(
        self,
        store: StoreLike,
    ):
        """Initialize the handler."""
        ...

    def list(self) -> list[str]:
        """List all items in the store.

        e.g. list all labels or tables managed by the handler.

        Returns:
            list[str]: List of items in the store.
        """
        ...

    def get(self, name: str) -> T:
        """Get an item from the store.

        Args:
            name (str): Name of the item.

        Returns:
            T: The selected item.
        """
        ...

    def create(self, name: str, data: T) -> None:
        """Create an item in the store.

        Args:
            name (str): Name of the item.
            data (T): The item to create.
        """
        ...

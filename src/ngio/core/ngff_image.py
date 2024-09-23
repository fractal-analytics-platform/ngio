"""Abstract class for handling OME-NGFF images."""

from typing import Protocol, TypeVar

from ngio.core.image_handler import Image
from ngio.io import StoreLike, open_group_wrapper
from ngio.ngff_meta import FractalImageLabelMeta, get_ngff_image_meta_handler

T = TypeVar("T")


class HandlerProtocol(Protocol):
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

    def new(self, name: str, **kwargs) -> None:
        """Create an new empty item in the store, based on the reference NgffImage.

        Args:
            name (str): Name of the item.
            **kwargs: Additional keyword arguments.
        """
        ...

    def add(self, name: str, item: T) -> None:
        """Add an item to the store.

        Args:
            name (str): Name of the item.
            item (T): The item to add.
        """
        ...


class NgffImage:
    """A class to handle OME-NGFF images."""

    def __init__(self, store: StoreLike) -> None:
        """Initialize the NGFFImage in read mode."""
        self.store = store
        self.group = open_group_wrapper(store=store, mode="r+")
        self._image_meta = get_ngff_image_meta_handler(
            self.group, meta_mode="image", cache=False
        )

    @property
    def image_meta(self) -> FractalImageLabelMeta:
        """Get the image metadata."""
        return self._image_meta.load_meta()

    @property
    def num_levels(self) -> int:
        """Get the number of levels in the image."""
        return self.image_meta.num_levels

    @property
    def levels_paths(self) -> list[str]:
        """Get the paths of the levels in the image."""
        return self.image_meta.levels_paths

    def get_image(
        self,
        *,
        level_path: str | int | None = None,
        pixel_size: tuple[float, ...] | list[float] | None = None,
        highest_resolution: bool = True,
    ) -> Image:
        """Get an image handler for the given level.

        Args:
            level_path (str | int | None, optional): The path to the level.
            pixel_size (tuple[float, ...] | list[float] | None, optional): The pixel
                size of the level.
            highest_resolution (bool, optional): Whether to get the highest
                resolution level

        Returns:
            ImageHandler: The image handler.
        """
        return Image(
            store=self.group,
            level_path=level_path,
            pixel_size=pixel_size,
            highest_resolution=highest_resolution,
        )

    def derive_new_image(
        self,
        store: StoreLike,
    ) -> "NgffImage":
        """Derive a new image from the current image.

        Args:
            store (StoreLike): The store to create the new image in.

        Returns:
            NgffImage: The new image.
        """
        raise NotImplementedError("Deriving new images is not yet implemented.")

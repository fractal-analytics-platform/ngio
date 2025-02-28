"""A module for handling label images in OME-NGFF files."""

from ngio.images.abstract_image import Image
from ngio.utils import (
    AccessModeLiteral,
    NgioValidationError,
    StoreOrGroup,
    ZarrGroupHandler,
)


class Label(Image):
    """Placeholder class for a label."""

    pass


class LabelsContainer:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(
        self, store: StoreOrGroup, cache: bool = False, mode: AccessModeLiteral = "a"
    ) -> None:
        """Initialize the LabelGroupHandler."""
        self._group_handler = ZarrGroupHandler(store, cache, mode)

        # Validate the group
        # Either contains a labels attribute or is empty
        attrs = self._group_handler.load_attrs()
        if len(attrs) == 0:
            # It's an empty group
            pass
        elif "labels" in attrs and isinstance(attrs["labels"], list):
            # It's a valid group
            pass
        else:
            raise NgioValidationError(
                f"Invalid /labels group. "
                f"Expected a single labels attribute with a list of label names. "
                f"Found: {attrs}"
            )

    def list(self) -> list[str]:
        """Create the /labels group if it doesn't exist."""
        attrs = self._group_handler.load_attrs()
        return attrs.get("labels", [])

    def get(self, name: str, path: str) -> Label:
        """Get a label from the group."""
        raise NotImplementedError

    def derive(
        self,
        name: str,
        reference_image: Image,
        overwrite: bool = False,
        **kwargs,
    ) -> Label:
        """Derive a label from an image."""
        raise NotImplementedError

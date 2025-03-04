"""A module for handling label images in OME-NGFF files."""

from typing import Literal

from ngio.images.abstract_image import AbstractImage, consolidate_image
from ngio.ome_zarr_meta import (
    ImplementedLabelMetaHandlers,
    LabelMetaHandler,
    NgioLabelMeta,
)
from ngio.utils import (
    NgioValidationError,
    ZarrGroupHandler,
)


class Label(AbstractImage[LabelMetaHandler]):
    """Placeholder class for a label."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: LabelMetaHandler | None,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the omezarr file.
            meta_handler: The image metadata handler.

        """
        if meta_handler is None:
            meta_handler = ImplementedLabelMetaHandlers().find_meta_handler(
                group_handler
            )
        super().__init__(
            group_handler=group_handler, path=path, meta_handler=meta_handler
        )

    @property
    def meta(self) -> NgioLabelMeta:
        """Return the metadata."""
        return self._meta_handler.meta

    def consolidate(
        self,
        mode: Literal["dask", "numpy", "coarsen"] = "dask",
    ) -> None:
        """Consolidate the label on disk."""
        consolidate_image(self, mode=mode, order=0)


class LabelsContainer:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(self, group_handler: ZarrGroupHandler) -> None:
        """Initialize the LabelGroupHandler."""
        self._group_handler = group_handler

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
        group_handler = self._group_handler.derive_handler(name)
        return Label(group_handler, path, None)

    def derive(
        self,
        name: str,
        reference_image: AbstractImage,
        overwrite: bool = False,
        **kwargs,
    ) -> Label:
        """Derive a label from an image."""
        raise NotImplementedError

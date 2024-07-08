"""This module provides a class abstraction for a OME-NGFF image."""

import zarr

from ngio.multiscale_handlers import MultiscaleImage, MultiscaleLabel
from ngio.table_handlers import RoiTableHandler


class NgffImage:
    """A class to handle OME-NGFF images."""

    def __init__(self, zarr_url: str, mode: str = "r") -> None:
        """Initialize the NGFFImage in read mode."""
        # setup the main image
        self._zarr_url = zarr_url
        self.group = zarr.open_group(zarr_url, mode=mode)

    @property
    def list_labels(self) -> list[str]:
        """List all the labels in the image."""
        labels_group = self.group.get("labels", None)
        if labels_group is None:
            return []

        list_labels = labels_group.attrs["labels"]
        if not isinstance(list_labels, list):
            raise ValueError("Labels must be a list of strings.")
        return list_labels

    @property
    def list_tables(self) -> list[str]:
        """List all the tables in the image."""
        tables_group = self.group.get("tables", None)
        if tables_group is None:
            return []

        list_tables = tables_group.attrs["tables"]
        if not isinstance(list_tables, list):
            raise ValueError("Tables must be a list of strings.")
        return list_tables

    @property
    def zarr_url(self) -> str:
        """Return the Zarr URL of the image."""
        return self._zarr_url

    def get_multiscale_label(self, label_name: str, level: int = 0) -> MultiscaleLabel:
        """Create a MultiscaleLabel object."""
        return MultiscaleLabel(self.zarr_url, path=f"labels/{label_name}", level=level)

    def get_roi_table(self, table_name: str) -> RoiTableHandler:
        """Create a RoiTableHandler object."""
        return RoiTableHandler(self.zarr_url, table_name)

    def get_multiscale_image(self, level: int = 0) -> MultiscaleImage:
        """Create a MultiscaleImage object."""
        return MultiscaleImage(zarr_url=self.zarr_url, level=level)

    def derive_new_image(
        self,
        zarr_url: str,
        copy_tables: bool | list[str] = False,
        copy_labels: bool | list[str] = False,
    ) -> "NgffImage":
        """Derive a new image from the current image."""
        raise NotImplementedError

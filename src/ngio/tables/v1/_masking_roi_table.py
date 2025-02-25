"""Implementation of the Masking ROI table class.

This class follows the roi_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel

from ngio.common import WorldCooROI
from ngio.utils import AccessModeLiteral, NgioValueError, StoreOrGroup


class MaskingROITableV1Meta(BaseModel):
    """Metadata for the ROI table."""

    fractal_table_version: Literal["1"] = "1"
    type: Literal["masking_roi_table"] = "masking_roi_table"
    backend: str | None = None


class MaskingROITableV1:
    """Class to handle fractal ROI tables.

    To know more about the ROI table format, please refer to the
    specification at:
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(self, rois: Iterable[WorldCooROI] | None = None) -> None:
        """Create a new ROI table."""
        self._meta = MaskingROITableV1Meta()
        self._table_backend = None

        self._rois = {}
        if rois is not None:
            self.add(rois)

    @staticmethod
    def type() -> Literal["masking_roi_table"]:
        """Return the type of the table."""
        return "masking_roi_table"

    @staticmethod
    def version() -> Literal["1"]:
        """Return the version of the fractal table."""
        return "1"

    @property
    def backend_name(self) -> str | None:
        """Return the name of the backend."""
        if self._table_backend is None:
            return None
        return self._table_backend.backend_name

    @classmethod
    def from_store(
        cls,
        store: StoreOrGroup,
        cache: bool = False,
        mode: AccessModeLiteral = "a",
        parallel_safe: bool = False,
    ) -> "MaskingROITableV1":
        """Create a new ROI table from a Zarr store."""
        raise NotImplementedError("Method not implemented.")

    def set_backend(
        self,
        store: StoreOrGroup,
        backend_name: str | None = None,
        cache: bool = False,
        mode: AccessModeLiteral = "a",
        parallel_safe: bool = False,
    ) -> None:
        """Set the backend of the table."""
        raise NotImplementedError("Method not implemented.")

    def rois(self) -> list[WorldCooROI]:
        """List all ROIs in the table."""
        return list(self._rois.values())

    def get(self, roi_name: str) -> WorldCooROI:
        """Get an ROI from the table."""
        if roi_name not in self._rois:
            raise NgioValueError(f"ROI {roi_name} not found in the table.")
        return self._rois[roi_name]

    def add(self, roi: WorldCooROI | Iterable[WorldCooROI]) -> None:
        """Append ROIs to the current table."""
        if isinstance(roi, WorldCooROI):
            roi = [roi]

        for _roi in roi:
            if _roi.name in self._rois:
                raise NgioValueError(f"ROI {_roi.name} already exists in the table.")
            self._rois[_roi.name] = _roi

    def consolidate(self) -> None:
        """Write the current state of the table to the Zarr file."""
        raise NotImplementedError("Method not implemented.")

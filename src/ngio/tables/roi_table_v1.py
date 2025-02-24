"""Implementation of the ROI Table class.

This class follows the roi_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

from collections.abc import Iterable
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from ngio.common import WorldCooROI
from ngio.tables.backends import TableBackendsManager
from ngio.utils import AccessModeLiteral, StoreOrGroup, ZarrGroupHandler

REQUIRED_COLUMNS = [
    "x_micrometer",
    "y_micrometer",
    "z_micrometer",
    "len_x_micrometer",
    "len_y_micrometer",
    "len_z_micrometer",
]


ORIGIN_COLUMNS = [
    "x_micrometer_original",
    "y_micrometer_original",
]

TRANSLATION_COLUMNS = ["translation_x", "translation_y", "translation_z"]

OPTIONAL_COLUMNS = ORIGIN_COLUMNS + TRANSLATION_COLUMNS


class ROITableV1Meta(BaseModel):
    """Metadata for the ROI table."""

    fractal_table_version: Literal["1"] = "1"
    type: Literal["roi_table"] = "roi_table"
    backend: str | None = None


def _dataframe_to_rois(dataframe: pd.DataFrame) -> list[WorldCooROI]:
    """Convert a DataFrame to a WorldCooROI object."""
    raise NotImplementedError


def _rois_to_dataframe(rois: list[WorldCooROI]) -> pd.DataFrame:
    """Convert a list of WorldCooROI objects to a DataFrame."""
    raise NotImplementedError


class ROITableV1:
    """Class to handle fractal ROI tables.

    To know more about the ROI table format, please refer to the
    specification at:
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(self):
        """Create a new ROI table."""
        self._meta = ROITableV1Meta()
        self._rois = []
        self._table_backend = None

    @property
    def index_key(self) -> Literal["FieldIndex"]:
        """Return the index key of the table."""
        return "FieldIndex"

    @property
    def index_type(self) -> Literal["str"]:
        """Return the index type of the table."""
        return "str"

    @classmethod
    def from_store(
        cls,
        store: StoreOrGroup,
        cache: bool = False,
        mode: AccessModeLiteral = "a",
        parallel_safe: bool = False,
    ) -> "ROITableV1":
        """Create a new ROI table from a Zarr store."""
        handler = ZarrGroupHandler(
            store=store, cache=cache, mode=mode, parallel_safe=parallel_safe
        )
        meta = ROITableV1Meta(**handler.load_attrs())
        backend = TableBackendsManager().get_backend(
            backend_name=meta.backend,
            group_handler=handler,
            index_key="FieldIndex",
            index_type="str",
        )

        if not backend.implements_dataframe:
            raise ValueError("The backend does not implement the dataframe protocol.")

        table = cls()
        table._meta = meta
        table._table_backend = backend

        dataframe = backend.load_as_dataframe()
        table._rois = _dataframe_to_rois(dataframe)
        return table

    def _set_backend(self, backend_name: str, store: StoreOrGroup) -> None:
        """Set the backend of the table."""
        handler = ZarrGroupHandler(store=store)
        backend = TableBackendsManager().get_backend(
            backend_name=backend_name,
            group_handler=handler,
            index_key="FieldIndex",
            index_type="str",
        )
        self._table_backend = backend

    def rois(self) -> list[WorldCooROI]:
        """List all ROIs in the table."""
        return self._rois

    def get(self, roi_name: str) -> WorldCooROI:
        """Get an ROI from the table."""
        for roi in self._rois:
            if roi.infos["FieldIndex"] == roi_name:
                return roi
        raise ValueError(f"ROI {roi_name} not found.")

    def add(self, roi: WorldCooROI | Iterable[WorldCooROI]) -> None:
        """Append ROIs to the current table."""
        if isinstance(roi, WorldCooROI):
            roi = [roi]

        for _roi in roi:
            if not isinstance(_roi, WorldCooROI):
                raise ValueError(
                    f"ROI must be an instance of WorldCooROI. Got {type(_roi)} instead."
                )
            self._rois.append(_roi)

    def consolidate(self) -> None:
        """Write the current state of the table to the Zarr file."""
        if self._table_backend is None:
            raise ValueError(
                "No backend set for the table. "
                "Please add the table to a OME-Zarr Image before calling consolidate."
            )

        dataframe = _rois_to_dataframe(self._rois)
        self._table_backend.write_from_dataframe(
            dataframe, metadata=self._meta.model_dump(exclude_none=True)
        )

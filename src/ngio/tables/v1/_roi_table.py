"""Implementation of the ROI Table class.

This class follows the roi_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

from collections.abc import Iterable
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from ngio.common import WorldCooROI
from ngio.tables._validators import validate_columns
from ngio.tables.backends import ImplementedTableBackends
from ngio.utils import AccessModeLiteral, NgioValueError, StoreOrGroup, ZarrGroupHandler

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


def _dataframe_to_rois(dataframe: pd.DataFrame) -> dict[str, WorldCooROI]:
    """Convert a DataFrame to a WorldCooROI object."""
    rois = {}
    for key, row in dataframe.iterrows():
        # check if optional columns are present
        origin = {col: row.get(col, None) for col in ORIGIN_COLUMNS}
        origin = dict(filter(lambda x: x[1] is not None, origin.items()))
        translation = {col: row.get(col, None) for col in TRANSLATION_COLUMNS}
        translation = dict(filter(lambda x: x[1] is not None, translation.items()))

        roi = WorldCooROI(
            name=str(key),
            x=row["x_micrometer"],
            y=row["y_micrometer"],
            z=row["z_micrometer"],
            x_length=row["len_x_micrometer"],
            y_length=row["len_y_micrometer"],
            z_length=row["len_z_micrometer"],
            unit="micrometer",  # type: ignore
            **origin,
            **translation,
        )
        rois[roi.name] = roi
    return rois


def _rois_to_dataframe(rois: dict[str, WorldCooROI], index_key: str) -> pd.DataFrame:
    """Convert a list of WorldCooROI objects to a DataFrame."""
    data = []
    for roi in rois.values():
        row = {
            index_key: roi.name,
            "x_micrometer": roi.x,
            "y_micrometer": roi.y,
            "z_micrometer": roi.z,
            "len_x_micrometer": roi.x_length,
            "len_y_micrometer": roi.y_length,
            "len_z_micrometer": roi.z_length,
        }

        extra = roi.model_extra or {}
        for col in ORIGIN_COLUMNS:
            if col in extra:
                row[col] = extra[col]

        for col in TRANSLATION_COLUMNS:
            if col in extra:
                row[col] = extra[col]
        data.append(row)
    dataframe = pd.DataFrame(data)
    dataframe = dataframe.set_index(index_key)
    return dataframe


class ROITableV1Meta(BaseModel):
    """Metadata for the ROI table."""

    fractal_table_version: Literal["1"] = "1"
    type: Literal["roi_table"] = "roi_table"
    backend: str | None = None


class RoiTableV1:
    """Class to handle fractal ROI tables.

    To know more about the ROI table format, please refer to the
    specification at:
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(self, rois: Iterable[WorldCooROI] | None = None) -> None:
        """Create a new ROI table."""
        self._meta = ROITableV1Meta()
        self._table_backend = None

        self._rois = {}
        if rois is not None:
            self.add(rois)

    @staticmethod
    def type() -> Literal["roi_table"]:
        """Return the type of the table."""
        return "roi_table"

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
    ) -> "RoiTableV1":
        """Create a new ROI table from a Zarr store."""
        handler = ZarrGroupHandler(
            store=store, cache=cache, mode=mode, parallel_safe=parallel_safe
        )
        meta = ROITableV1Meta(**handler.load_attrs())
        backend = ImplementedTableBackends().get_backend(
            backend_name=meta.backend,
            group_handler=handler,
            index_key="FieldIndex",
            index_type="str",
        )

        if not backend.implements_dataframe:
            raise NgioValueError(
                "The backend does not implement the dataframe protocol."
            )

        table = cls()
        table._meta = meta
        table._table_backend = backend

        dataframe = backend.load_as_dataframe()
        dataframe = validate_columns(
            dataframe,
            required_columns=REQUIRED_COLUMNS,
            optional_columns=OPTIONAL_COLUMNS,
        )
        table._rois = _dataframe_to_rois(dataframe)
        return table

    def set_backend(
        self,
        store: StoreOrGroup,
        backend_name: str | None = None,
        cache: bool = False,
        mode: AccessModeLiteral = "a",
        parallel_safe: bool = False,
    ) -> None:
        """Set the backend of the table."""
        handler = ZarrGroupHandler(
            store=store, cache=cache, mode=mode, parallel_safe=parallel_safe
        )
        backend = ImplementedTableBackends().get_backend(
            backend_name=backend_name,
            group_handler=handler,
            index_key="FieldIndex",
            index_type="str",
        )
        self._meta.backend = backend_name
        self._table_backend = backend

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
        if self._table_backend is None:
            raise NgioValueError(
                "No backend set for the table. "
                "Please add the table to a OME-Zarr Image before calling consolidate."
            )

        dataframe = _rois_to_dataframe(self._rois, index_key="FieldIndex")
        dataframe = validate_columns(
            dataframe,
            required_columns=REQUIRED_COLUMNS,
            optional_columns=OPTIONAL_COLUMNS,
        )
        self._table_backend.write_from_dataframe(
            dataframe, metadata=self._meta.model_dump(exclude_none=True)
        )

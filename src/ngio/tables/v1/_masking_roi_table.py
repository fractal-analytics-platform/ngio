"""Implementation of the Masking ROI table class.

This class follows the roi_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel

from ngio.common import WorldCooROI
from ngio.tables._validators import validate_columns
from ngio.tables.backends import ImplementedTableBackends
from ngio.tables.v1._roi_table import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    _dataframe_to_rois,
    _rois_to_dataframe,
)
from ngio.utils import NgioValueError, ZarrGroupHandler


class RegionMeta(BaseModel):
    """Metadata for the region."""

    path: str


class MaskingROITableV1Meta(BaseModel):
    """Metadata for the ROI table."""

    fractal_table_version: Literal["1"] = "1"
    type: Literal["masking_roi_table"] = "masking_roi_table"
    backend: str | None = None
    region: RegionMeta | None = None
    instance_key: str = "label"


class MaskingROITableV1:
    """Class to handle fractal ROI tables.

    To know more about the ROI table format, please refer to the
    specification at:
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(
        self,
        rois: Iterable[WorldCooROI] | None = None,
        reference_label: str | None = None,
    ) -> None:
        """Create a new ROI table."""
        if reference_label is None:
            self._meta = MaskingROITableV1Meta()
        else:
            path = f"../labels/{reference_label}"
            self._meta = MaskingROITableV1Meta(region=RegionMeta(path=path))
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
        return self._table_backend.backend_name()

    @classmethod
    def _from_handler(
        cls, handler: ZarrGroupHandler, backend_name: str | None = None
    ) -> "MaskingROITableV1":
        """Create a new ROI table from a Zarr store."""
        meta = MaskingROITableV1Meta(**handler.load_attrs())

        if backend_name is None:
            backend = ImplementedTableBackends().get_backend(
                backend_name=meta.backend,
                group_handler=handler,
                index_key="label",
                index_type="int",
            )
        else:
            backend = ImplementedTableBackends().get_backend(
                backend_name=backend_name,
                group_handler=handler,
                index_key="label",
                index_type="int",
            )
            meta.backend = backend_name

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

    def _set_backend(
        self,
        handler: ZarrGroupHandler,
        backend_name: str | None = None,
    ) -> None:
        """Set the backend of the table."""
        backend = ImplementedTableBackends().get_backend(
            backend_name=backend_name,
            group_handler=handler,
            index_key="label",
            index_type="int",
        )
        self._meta.backend = backend_name
        self._table_backend = backend

    def rois(self) -> list[WorldCooROI]:
        """List all ROIs in the table."""
        return list(self._rois.values())

    def get(self, label: int) -> WorldCooROI:
        """Get an ROI from the table."""
        _label = str(label)
        if _label not in self._rois:
            raise KeyError(f"ROI {_label} not found in the table.")
        return self._rois[_label]

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

        dataframe = _rois_to_dataframe(self._rois, index_key="label")
        dataframe = validate_columns(
            dataframe,
            required_columns=REQUIRED_COLUMNS,
            optional_columns=OPTIONAL_COLUMNS,
        )
        self._table_backend.write_from_dataframe(
            dataframe, metadata=self._meta.model_dump(exclude_none=True)
        )

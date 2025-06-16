"""Implementation of the ROI Table class.

This class follows the roi_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

# Import _type to avoid name conflict with table.type
from builtins import type as _type
from collections.abc import Iterable
from typing import Generic, Literal, TypeVar

import pandas as pd
from pydantic import BaseModel

from ngio.common import Roi
from ngio.tables._validators import validate_columns
from ngio.tables.backends import BackendMeta, ImplementedTableBackends
from ngio.utils import NgioValueError, ZarrGroupHandler

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


def _dataframe_to_rois(dataframe: pd.DataFrame) -> dict[str, Roi]:
    """Convert a DataFrame to a WorldCooROI object."""
    rois = {}
    for key, row in dataframe.iterrows():
        # check if optional columns are present
        origin = {col: row.get(col, None) for col in ORIGIN_COLUMNS}
        origin = dict(filter(lambda x: x[1] is not None, origin.items()))
        translation = {col: row.get(col, None) for col in TRANSLATION_COLUMNS}
        translation = dict(filter(lambda x: x[1] is not None, translation.items()))

        roi = Roi(
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


def _rois_to_dataframe(rois: dict[str, Roi], index_key: str) -> pd.DataFrame:
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


class RoiTableV1Meta(BackendMeta):
    """Metadata for the ROI table."""

    fractal_table_version: Literal["1"] = "1"
    type: Literal["roi_table"] = "roi_table"


class RegionMeta(BaseModel):
    """Metadata for the region."""

    path: str


class MaskingRoiTableV1Meta(BackendMeta):
    """Metadata for the ROI table."""

    fractal_table_version: Literal["1"] = "1"
    type: Literal["masking_roi_table"] = "masking_roi_table"
    region: RegionMeta | None = None
    instance_key: str = "label"


_roi_meta = TypeVar("_roi_meta", RoiTableV1Meta, MaskingRoiTableV1Meta)


class _GenericRoiTableV1(Generic[_roi_meta]):
    """Class to a non-specific table."""

    _meta: _roi_meta

    def __init__(
        self, meta: _roi_meta | None = None, rois: Iterable[Roi] | None = None
    ) -> None:
        """Create a new ROI table."""
        if meta is None:
            raise NgioValueError("Metadata must be provided.")
        self._meta = meta
        self._table_backend = None

        self._rois = {}
        if rois is not None:
            self.add(rois)

    @staticmethod
    def type() -> str:
        """Return the type of the table."""
        raise NotImplementedError

    @staticmethod
    def version() -> Literal["1"]:
        """Return the version of the fractal table."""
        return "1"

    @staticmethod
    def _index_key() -> str:
        """Return the index key of the table."""
        raise NotImplementedError

    @staticmethod
    def _index_type() -> Literal["int", "str"]:
        """Return the index type of the table."""
        raise NotImplementedError

    @staticmethod
    def _meta_type() -> _type[_roi_meta]:
        """Return the metadata type of the table."""
        raise NotImplementedError

    @property
    def backend_name(self) -> str | None:
        """Return the name of the backend."""
        if self._table_backend is None:
            return None
        return self._table_backend.backend_name()

    @classmethod
    def _from_handler(
        cls, handler: ZarrGroupHandler, backend_name: str | None = None
    ) -> "_GenericRoiTableV1":
        """Create a new ROI table from a Zarr store."""
        meta = cls._meta_type()(**handler.load_attrs())

        if backend_name is None:
            backend = ImplementedTableBackends().get_backend(
                backend_name=meta.backend,
                group_handler=handler,
                index_key=cls._index_key(),
                index_type=cls._index_type(),
            )
        else:
            backend = ImplementedTableBackends().get_backend(
                backend_name=backend_name,
                group_handler=handler,
                index_key=cls._index_key(),
                index_type=cls._index_type(),
            )
            meta.backend = backend_name

        if not backend.implements_pandas:
            raise NgioValueError(
                "The backend does not implement the dataframe protocol."
            )

        # This will be implemented in the child classes
        table = cls()
        table._meta = meta
        table._table_backend = backend

        dataframe = backend.load_as_pandas_df()
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
            index_key=self._index_key(),
            index_type=self._index_type(),
        )
        self._meta.backend = backend_name
        self._table_backend = backend

    def rois(self) -> list[Roi]:
        """List all ROIs in the table."""
        return list(self._rois.values())

    def add(self, roi: Roi | Iterable[Roi], overwrite: bool = False) -> None:
        """Append ROIs to the current table.

        Args:
            roi: A single ROI or a list of ROIs to add to the table.
            overwrite: If True, overwrite existing ROIs with the same name.
        """
        if isinstance(roi, Roi):
            roi = [roi]

        for _roi in roi:
            if not overwrite and _roi.name in self._rois:
                raise NgioValueError(f"ROI {_roi.name} already exists in the table.")
            self._rois[_roi.name] = _roi

    def consolidate(self) -> None:
        """Write the current state of the table to the Zarr file."""
        if self._table_backend is None:
            raise NgioValueError(
                "No backend set for the table. "
                "Please add the table to a OME-Zarr Image before calling consolidate."
            )

        dataframe = _rois_to_dataframe(self._rois, index_key=self._index_key())
        dataframe = validate_columns(
            dataframe,
            required_columns=REQUIRED_COLUMNS,
            optional_columns=OPTIONAL_COLUMNS,
        )
        self._table_backend.write(
            dataframe, metadata=self._meta.model_dump(exclude_none=True), mode="pandas"
        )


class RoiTableV1(_GenericRoiTableV1[RoiTableV1Meta]):
    """Class to handle fractal ROI tables.

    To know more about the ROI table format, please refer to the
    specification at:
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(self, rois: Iterable[Roi] | None = None) -> None:
        """Create a new ROI table."""
        super().__init__(RoiTableV1Meta(), rois)

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        prop = f"num_rois={len(self._rois)}"
        return f"RoiTableV1({prop})"

    @staticmethod
    def type() -> Literal["roi_table"]:
        """Return the type of the table."""
        return "roi_table"

    @staticmethod
    def _index_key() -> str:
        """Return the index key of the table."""
        return "FieldIndex"

    @staticmethod
    def _index_type() -> Literal["int", "str"]:
        """Return the index type of the table."""
        return "str"

    @staticmethod
    def _meta_type() -> _type[RoiTableV1Meta]:
        """Return the metadata type of the table."""
        return RoiTableV1Meta

    def get(self, roi_name: str) -> Roi:
        """Get an ROI from the table."""
        if roi_name not in self._rois:
            raise NgioValueError(f"ROI {roi_name} not found in the table.")
        return self._rois[roi_name]


class MaskingRoiTableV1(_GenericRoiTableV1[MaskingRoiTableV1Meta]):
    """Class to handle fractal ROI tables.

    To know more about the ROI table format, please refer to the
    specification at:
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(
        self,
        rois: Iterable[Roi] | None = None,
        reference_label: str | None = None,
    ) -> None:
        """Create a new ROI table."""
        meta = MaskingRoiTableV1Meta()
        if reference_label is not None:
            path = f"../labels/{reference_label}"
            meta.region = RegionMeta(path=path)
        super().__init__(meta, rois)

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        prop = f"num_rois={len(self._rois)}"
        if self.reference_label is not None:
            prop += f", reference_label={self.reference_label}"
        return f"MaskingRoiTableV1({prop})"

    @staticmethod
    def type() -> Literal["masking_roi_table"]:
        """Return the type of the table."""
        return "masking_roi_table"

    @staticmethod
    def _index_key() -> str:
        """Return the index key of the table."""
        return "label"

    @staticmethod
    def _index_type() -> Literal["int", "str"]:
        """Return the index type of the table."""
        return "int"

    @staticmethod
    def _meta_type() -> _type[MaskingRoiTableV1Meta]:
        """Return the metadata type of the table."""
        return MaskingRoiTableV1Meta

    @property
    def reference_label(self) -> str | None:
        """Return the reference label."""
        path = self._meta.region
        if path is None:
            return None
        path = path.path
        path = path.split("/")[-1]
        return path

    def get(self, label: int) -> Roi:
        """Get an ROI from the table."""
        _label = str(label)
        if _label not in self._rois:
            raise KeyError(f"ROI {_label} not found in the table.")
        return self._rois[_label]

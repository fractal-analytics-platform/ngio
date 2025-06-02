"""Implementation of the ROI Table class.

This class follows the roi_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

# Import _type to avoid name conflict with table.type
from collections.abc import Iterable
from functools import cache
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from ngio.common import Roi
from ngio.tables.abstract_table import (
    AbstractBaseTable,
    TabularData,
)
from ngio.tables.backends import (
    BackendMeta,
    TableBackend,
    convert_to_pandas,
    normalize_pandas_df,
)
from ngio.utils import (
    NgioTableValidationError,
    NgioValueError,
    ZarrGroupHandler,
    ngio_logger,
)

REQUIRED_COLUMNS = [
    "x_micrometer",
    "y_micrometer",
    "z_micrometer",
    "len_x_micrometer",
    "len_y_micrometer",
    "len_z_micrometer",
]

#####################
# Optional columns are not validated at the moment
# only a warning is raised if non optional columns are present
#####################

ORIGIN_COLUMNS = [
    "x_micrometer_original",
    "y_micrometer_original",
]

TRANSLATION_COLUMNS = ["translation_x", "translation_y", "translation_z"]

PLATE_COLUMNS = ["plate_name", "row", "column", "path", "acquisition"]

INDEX_COLUMNS = [
    "FieldIndex",
    "label",
]

OPTIONAL_COLUMNS = ORIGIN_COLUMNS + TRANSLATION_COLUMNS + PLATE_COLUMNS + INDEX_COLUMNS


@cache
def _check_optional_columns(col_name: str) -> None:
    """Check if the column name is in the optional columns."""
    if col_name not in OPTIONAL_COLUMNS:
        ngio_logger.warning(
            f"Column {col_name} is not in the optional columns. "
            f"Standard optional columns are: {OPTIONAL_COLUMNS}."
        )


def _dataframe_to_rois(
    dataframe: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> dict[str, Roi]:
    """Convert a DataFrame to a WorldCooROI object."""
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS

    # Validate the columns of the DataFrame
    _required_columns = set(dataframe.columns).intersection(set(required_columns))
    if len(_required_columns) != len(required_columns):
        raise NgioTableValidationError(
            f"Could not find required columns: {_required_columns} in the table."
        )

    extra_columns = set(dataframe.columns).difference(set(required_columns))

    for col in extra_columns:
        _check_optional_columns(col)

    extras = {}

    rois = {}
    for row in dataframe.itertuples(index=True):
        # check if optional columns are present
        if len(extra_columns) > 0:
            extras = {col: getattr(row, col, None) for col in extra_columns}

        roi = Roi(
            name=str(row.Index),
            x=row.x_micrometer,  # type: ignore
            y=row.y_micrometer,  # type: ignore
            z=row.z_micrometer,  # type: ignore
            x_length=row.len_x_micrometer,  # type: ignore
            y_length=row.len_y_micrometer,  # type: ignore
            z_length=row.len_z_micrometer,  # type: ignore
            unit="micrometer",  # type: ignore
            **extras,
        )
        rois[roi.name] = roi
    return rois


def _table_to_rois(
    table: TabularData,
    index_key: str | None = None,
    index_type: Literal["int", "str"] | None = None,
    required_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Roi]]:
    """Convert a table to a dictionary of ROIs.

    Args:
        table: The table to convert.
        index_key: The column name to use as the index of the DataFrame.
        index_type: The type of the index column in the DataFrame.
        required_columns: The required columns in the DataFrame.

    Returns:
        A dictionary of ROIs.
    """
    dataframe = convert_to_pandas(
        table,
        index_key=index_key,
        index_type=index_type,
    )
    return dataframe, _dataframe_to_rois(dataframe, required_columns=required_columns)


def _rois_to_dataframe(rois: dict[str, Roi], index_key: str | None) -> pd.DataFrame:
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
        for col in extra:
            _check_optional_columns(col)
            row[col] = extra[col]
        data.append(row)
    dataframe = pd.DataFrame(data)
    dataframe = normalize_pandas_df(dataframe, index_key=index_key)
    return dataframe


class GenericRoiTableV1(AbstractBaseTable):
    def __init__(
        self,
        *,
        rois: Iterable[Roi] | None = None,
        meta: BackendMeta,
    ) -> None:
        table = None

        self._rois: dict[str, Roi] | None = None
        if rois is not None:
            self._rois = {}
            self.add(rois)
            table = _rois_to_dataframe(self._rois, index_key=meta.index_key)

        super().__init__(table_data=table, meta=meta)

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        rois = self.rois()
        prop = f"num_rois={len(rois)}"
        class_name = self.__class__.__name__
        return f"{class_name}({prop})"

    @staticmethod
    def table_type() -> str:
        """Return the type of the table."""
        return "generic_roi_table"

    @staticmethod
    def version() -> Literal["1"]:
        """Return the version of the fractal table."""
        return "1"

    @property
    def table_data(self) -> TabularData:
        """Return the table."""
        if self._rois is None:
            return super().table_data

        if len(self.rois()) > 0:
            self._table_data = _rois_to_dataframe(self._rois, index_key=self.index_key)
        return super().table_data

    def set_table_data(
        self, table_data: TabularData | None = None, refresh: bool = False
    ) -> None:
        if table_data is not None:
            if not isinstance(table_data, TabularData):
                raise NgioValueError(
                    "The table must be a pandas DataFrame, polars LazyFrame, "
                    " or AnnData object."
                )

            table_data, rois = _table_to_rois(
                table_data,
                index_key=self.index_key,
                index_type=self.index_type,
                required_columns=REQUIRED_COLUMNS,
            )
            self._table_data = table_data
            self._rois = rois
            return None

        if self._table_data is not None and not refresh:
            return None

        if self._table_backend is None:
            raise NgioValueError(
                "The table does not have a DataFrame in memory nor a backend."
            )

        table_data, rois = _table_to_rois(
            self._table_backend.load(),
            index_key=self.index_key,
            index_type=self.index_type,
            required_columns=REQUIRED_COLUMNS,
        )
        self._table_data = table_data
        self._rois = rois

    def _check_rois(self) -> None:
        """Load the ROIs from the table.

        If the ROIs are already loaded, do nothing.
        If the ROIs are not loaded, load them from the table.
        """
        if self._rois is None:
            self._rois = _dataframe_to_rois(self.dataframe)

    def rois(self) -> list[Roi]:
        """List all ROIs in the table."""
        self._check_rois()
        if self._rois is None:
            return []
        return list(self._rois.values())

    def add(self, roi: Roi | Iterable[Roi], overwrite: bool = False) -> None:
        """Append ROIs to the current table.

        Args:
            roi: A single ROI or a list of ROIs to add to the table.
            overwrite: If True, overwrite existing ROIs with the same name.
        """
        if isinstance(roi, Roi):
            roi = [roi]

        self._check_rois()
        if self._rois is None:
            self._rois = {}

        for _roi in roi:
            if not overwrite and _roi.name in self._rois:
                raise NgioValueError(f"ROI {_roi.name} already exists in the table.")
            self._rois[_roi.name] = _roi

    def get(self, roi_name: str) -> Roi:
        """Get an ROI from the table."""
        self._check_rois()
        if self._rois is None:
            self._rois = {}

        if roi_name not in self._rois:
            raise NgioValueError(f"ROI {roi_name} not found in the table.")
        return self._rois[roi_name]

    @classmethod
    def from_table_data(
        cls, table_data: TabularData, meta: BackendMeta
    ) -> "GenericRoiTableV1":
        """Create a new ROI table from a table data."""
        _, rois = _table_to_rois(
            table=table_data,
            index_key=meta.index_key,
            index_type=meta.index_type,
            required_columns=REQUIRED_COLUMNS,
        )
        return cls(rois=rois.values(), meta=meta)


class RoiTableV1Meta(BackendMeta):
    """Metadata for the ROI table."""

    table_version: Literal["1"] = "1"
    type: Literal["roi_table"] = "roi_table"
    index_key: str | None = "FieldIndex"
    index_type: Literal["str", "int"] | None = "str"


class RoiTableV1(GenericRoiTableV1):
    """Class to handle fractal ROI tables.

    To know more about the ROI table format, please refer to the
    specification at:
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(
        self, rois: Iterable[Roi] | None = None, *, meta: RoiTableV1Meta | None = None
    ) -> None:
        """Create a new ROI table."""
        if meta is None:
            meta = RoiTableV1Meta()

        if meta.index_key is None:
            meta.index_key = "FieldIndex"

        if meta.index_type is None:
            meta.index_type = "str"
        super().__init__(meta=meta, rois=rois)

    @classmethod
    def from_handler(
        cls,
        handler: ZarrGroupHandler,
        backend: TableBackend | None = None,
    ) -> "RoiTableV1":
        table = cls._from_handler(
            handler=handler,
            backend=backend,
            meta_model=RoiTableV1Meta,
        )
        return table

    @staticmethod
    def table_type() -> Literal["roi_table"]:
        """Return the type of the table."""
        return "roi_table"


class RegionMeta(BaseModel):
    """Metadata for the region."""

    path: str


class MaskingRoiTableV1Meta(BackendMeta):
    """Metadata for the ROI table."""

    table_version: Literal["1"] = "1"
    type: Literal["masking_roi_table"] = "masking_roi_table"
    region: RegionMeta | None = None
    instance_key: str = "label"
    index_key: str | None = "label"
    index_type: Literal["int", "str"] | None = "int"


class MaskingRoiTableV1(GenericRoiTableV1):
    """Class to handle fractal ROI tables.

    To know more about the ROI table format, please refer to the
    specification at:
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(
        self,
        rois: Iterable[Roi] | None = None,
        *,
        reference_label: str | None = None,
        meta: MaskingRoiTableV1Meta | None = None,
    ) -> None:
        """Create a new ROI table."""
        if meta is None:
            meta = MaskingRoiTableV1Meta()

        if reference_label is not None:
            meta.region = RegionMeta(path=reference_label)

        if meta.index_key is None:
            meta.index_key = "label"

        if meta.index_type is None:
            meta.index_type = "int"
        meta.instance_key = meta.index_key
        super().__init__(meta=meta, rois=rois)

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        rois = self.rois()
        if self.reference_label is not None:
            prop = f"num_rois={len(rois)}, reference_label={self.reference_label}"
        else:
            prop = f"num_rois={len(rois)}"
        return f"MaskingRoiTableV1({prop})"

    @classmethod
    def from_handler(
        cls,
        handler: ZarrGroupHandler,
        backend: TableBackend | None = None,
    ) -> "MaskingRoiTableV1":
        table = cls._from_handler(
            handler=handler,
            backend=backend,
            meta_model=MaskingRoiTableV1Meta,
        )
        return table

    @staticmethod
    def table_type() -> Literal["masking_roi_table"]:
        """Return the type of the table."""
        return "masking_roi_table"

    @property
    def meta(self) -> MaskingRoiTableV1Meta:
        """Return the metadata of the table."""
        if not isinstance(self._meta, MaskingRoiTableV1Meta):
            raise NgioValueError(
                "The metadata of the table is not of type MaskingRoiTableV1Meta."
            )
        return self._meta

    @property
    def reference_label(self) -> str | None:
        """Return the reference label."""
        path = self.meta.region
        if path is None:
            return None

        path = path.path
        path = path.split("/")[-1]
        return path

    def get(self, label: int | str) -> Roi:  # type: ignore
        """Get an ROI from the table."""
        roi_name = str(label)
        return super().get(roi_name)

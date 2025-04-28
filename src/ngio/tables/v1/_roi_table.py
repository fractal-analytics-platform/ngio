"""Implementation of the ROI Table class.

This class follows the roi_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

# Import _type to avoid name conflict with table.type
from collections.abc import Iterable
from typing import Literal, Self

import pandas as pd
from pydantic import BaseModel

from ngio.common import Roi
from ngio.tables.abstract_table import (
    AbstractBaseTable,
    SupportedTables,
)
from ngio.tables.backends import BackendMeta, convert_to_pandas, normalize_pandas_df
from ngio.utils import NgioTableValidationError, NgioValueError, ZarrGroupHandler

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

PLATE_COLUMNS = ["plate_name", "row", "column", "path", "acquisition"]

OPTIONAL_COLUMNS = ORIGIN_COLUMNS + TRANSLATION_COLUMNS + PLATE_COLUMNS


def validate_columns(
    table_df: pd.DataFrame,
    required_columns: list[str],
    optional_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Validate the columns headers of the table.

    If a required column is missing, a TableValidationError is raised.
    If a list of optional columns is provided, only required and optional columns are
        allowed in the table.

    Args:
        table_df (pd.DataFrame): The DataFrame to validate.
        required_columns (list[str]): A list of required columns.
        optional_columns (list[str] | None): A list of optional columns.
            Default is None.

    Returns:
        pd.DataFrame: The validated DataFrame.
    """
    table_header = table_df.columns
    for column in required_columns:
        if column not in table_header:
            raise NgioTableValidationError(
                f"Could not find required column: {column} in the table"
            )

    if optional_columns is None:
        return table_df

    possible_columns = [*required_columns, *optional_columns]
    for column in table_header:
        if column not in possible_columns:
            raise NgioTableValidationError(
                f"Could not find column: {column} in the list of possible columns. ",
                f"Possible columns are: {possible_columns}",
            )

    return table_df


def _dataframe_to_rois(dataframe: pd.DataFrame) -> dict[str, Roi]:
    """Convert a DataFrame to a WorldCooROI object."""
    rois = {}
    for key, row in dataframe.iterrows():
        # check if optional columns are present
        origin = {col: row.get(col, None) for col in ORIGIN_COLUMNS}
        origin = dict(filter(lambda x: x[1] is not None, origin.items()))
        translation = {col: row.get(col, None) for col in TRANSLATION_COLUMNS}
        translation = dict(filter(lambda x: x[1] is not None, translation.items()))
        plate = {col: row.get(col, None) for col in PLATE_COLUMNS}
        plate = dict(filter(lambda x: x[1] is not None, plate.items()))

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
            **plate,
        )
        rois[roi.name] = roi
    return rois


def _table_to_rois(
    table: SupportedTables,
    index_key: str | None = None,
    index_type: Literal["int", "str"] | None = None,
    required_columns: list[str] | None = None,
    optional_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Roi]]:
    """Convert a table to a dictionary of ROIs.

    Args:
        table: The table to convert.
        index_key: The column name to use as the index of the DataFrame.
        index_type: The type of the index column in the DataFrame.
        required_columns: The required columns in the DataFrame.
        optional_columns: The optional columns in the DataFrame.

    Returns:
        A dictionary of ROIs.
    """
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS
    if optional_columns is None:
        optional_columns = OPTIONAL_COLUMNS

    dataframe = convert_to_pandas(
        table,
        index_key=index_key,
        index_type=index_type,
    )
    return dataframe, _dataframe_to_rois(dataframe)


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
        for col in ORIGIN_COLUMNS:
            if col in extra:
                row[col] = extra[col]

        for col in TRANSLATION_COLUMNS:
            if col in extra:
                row[col] = extra[col]
        data.append(row)
    dataframe = pd.DataFrame(data)
    dataframe = normalize_pandas_df(dataframe, index_key=index_key)
    return dataframe


class _GenericRoiTableV1(AbstractBaseTable):
    def __init__(
        self,
        *,
        meta: BackendMeta,
        rois: Iterable[Roi] | None = None,
        index_key: str | None = None,
        index_type: Literal["int", "str"] | None = None,
    ) -> None:
        table = None
        self._rois = {}
        if rois is not None:
            self.add(rois)
            table = _rois_to_dataframe(self._rois, index_key=meta.index_key)

        super().__init__(
            table=table, meta=meta, index_key=index_key, index_type=index_type
        )

    @staticmethod
    def type() -> str:
        """Return the type of the table."""
        raise NotImplementedError

    @staticmethod
    def version() -> Literal["1"]:
        """Return the version of the fractal table."""
        return "1"

    @property
    def table(self) -> SupportedTables:
        """Return the table."""
        if len(self._rois) > 0:
            self._table = _rois_to_dataframe(self._rois, index_key=self.index_key)

        return super().table

    def set_table(
        self, table: SupportedTables | None = None, refresh: bool = False
    ) -> None:
        if table is not None:
            if not isinstance(table, SupportedTables):
                raise NgioValueError(
                    "The table must be a pandas DataFrame, polars LazyFrame, "
                    " or AnnData object."
                )

            table, rois = _table_to_rois(
                table,
                index_key=self.index_key,
                index_type=self.index_type,
                required_columns=REQUIRED_COLUMNS,
                optional_columns=OPTIONAL_COLUMNS,
            )
            self._table = table
            self._rois = rois
            return None

        if self._table is not None and not refresh:
            return None

        if self._table_backend is None:
            raise NgioValueError(
                "The table does not have a DataFrame in memory nor a backend."
            )

        table, rois = _table_to_rois(
            self._table_backend.load(),
            index_key=self.index_key,
            index_type=self.index_type,
            required_columns=REQUIRED_COLUMNS,
            optional_columns=OPTIONAL_COLUMNS,
        )
        self._table = table
        self._rois = rois

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

    def concatenate(
        self,
        table: Self,
        src_columns: dict[str, str] | None = None,
        dst_columns: dict[str, str] | None = None,
        index_key: str = "index",
    ) -> Self:
        """Concatenate multiple tables into a single table."""
        if src_columns is None:
            src_columns = {}
        if dst_columns is None:
            dst_columns = {}

        src_prefix = "_".join(src_columns.keys())
        dst_prefix = "_".join(dst_columns.keys())

        concat_rois = {}
        for roi in self._rois.values():
            concat_name = f"{src_prefix}{roi.name}"
            concat_rois[concat_name] = Roi(
                name=concat_name,
                x=roi.x,
                y=roi.y,
                z=roi.z,
                x_length=roi.x_length,
                y_length=roi.y_length,
                z_length=roi.z_length,
                unit=roi.unit,
                **src_columns,
            )

        for roi in table._rois.values():
            concat_name = f"{dst_prefix}{roi.name}"
            concat_rois[concat_name] = Roi(
                name=concat_name,
                x=roi.x,
                y=roi.y,
                z=roi.z,
                x_length=roi.x_length,
                y_length=roi.y_length,
                z_length=roi.z_length,
                unit=roi.unit,
                **dst_columns,
            )

        return type(self)(
            meta=self._meta,
            rois=concat_rois.values(),
            index_key=index_key,
            index_type="str",
        )


class RoiTableV1Meta(BackendMeta):
    """Metadata for the ROI table."""

    fractal_table_version: Literal["1"] = "1"
    type: Literal["roi_table"] = "roi_table"
    index_key: str | None = "FieldIndex"
    index_type: Literal["str", "int"] | None = "str"


class RoiTableV1(_GenericRoiTableV1):
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

        meta.index_key = "FieldIndex"
        meta.index_type = "str"
        super().__init__(meta=meta, rois=rois)

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        prop = f"num_rois={len(self._rois)}"
        return f"RoiTableV1({prop})"

    @classmethod
    def from_handler(
        cls, handler: ZarrGroupHandler, backend_name: str | None = None
    ) -> "RoiTableV1":
        table = cls._from_handler(
            handler=handler,
            backend_name=backend_name,
            meta_model=RoiTableV1Meta,
        )
        table._rois = _dataframe_to_rois(table.dataframe)
        return table

    @staticmethod
    def type() -> Literal["roi_table"]:
        """Return the type of the table."""
        return "roi_table"

    def get(self, roi_name: str) -> Roi:
        """Get an ROI from the table."""
        if roi_name not in self._rois:
            raise NgioValueError(f"ROI {roi_name} not found in the table.")
        return self._rois[roi_name]


class RegionMeta(BaseModel):
    """Metadata for the region."""

    path: str


class MaskingRoiTableV1Meta(BackendMeta):
    """Metadata for the ROI table."""

    fractal_table_version: Literal["1"] = "1"
    type: Literal["masking_roi_table"] = "masking_roi_table"
    region: RegionMeta | None = None
    instance_key: str = "label"
    index_key: str | None = "label"
    index_type: Literal["int", "str"] | None = "int"


class MaskingRoiTableV1(_GenericRoiTableV1):
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
        index_key: str | None = None,
        index_type: Literal["int", "str"] | None = None,
    ) -> None:
        """Create a new ROI table."""
        if meta is None:
            meta = MaskingRoiTableV1Meta()

        if reference_label is not None:
            meta.region = RegionMeta(path=reference_label)

        meta.index_key = "label" if index_key is None else index_key
        meta.index_type = "int" if index_type is None else index_type
        meta.instance_key = meta.index_key
        super().__init__(
            meta=meta, rois=rois, index_key=index_key, index_type=index_type
        )

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        if self.reference_label is not None:
            prop = f"num_rois={len(self._rois)}, reference_label={self.reference_label}"
        else:
            prop = f"num_rois={len(self._rois)}"
        return f"MaskingRoiTableV1({prop})"

    @classmethod
    def from_handler(
        cls, handler: ZarrGroupHandler, backend_name: str | None = None
    ) -> "MaskingRoiTableV1":
        table = cls._from_handler(
            handler=handler,
            backend_name=backend_name,
            meta_model=MaskingRoiTableV1Meta,
        )
        table._rois = _dataframe_to_rois(table.dataframe)
        return table

    @staticmethod
    def type() -> Literal["masking_roi_table"]:
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

    def get(self, label: int) -> Roi:
        """Get an ROI from the table."""
        _label = str(label)
        if _label not in self._rois:
            raise KeyError(f"ROI {_label} not found in the table.")
        return self._rois[_label]

"""HCS (High Content Screening) specific metadata classes for NGIO."""

from typing import Literal

from ome_zarr_models.v04.hcs import HCSAttrs
from ome_zarr_models.v04.plate import (
    Acquisition,
    Column,
    Plate,
    Row,
    WellInPlate,
)
from ome_zarr_models.v04.well import WellAttrs
from ome_zarr_models.v04.well_types import WellImage, WellMeta

from ngio.utils import NgioValueError


class NgioWellMeta(WellAttrs):
    """HCS well metadata."""

    @classmethod
    def default_init(
        cls,
        images_paths: list[str],
        acquisition_ids: list[int] | None = None,
        version: Literal["0.4"] | None = None,
    ) -> "NgioWellMeta":
        if acquisition_ids is None:
            _acquisition_ids = [None] * len(images_paths)
        else:
            _acquisition_ids = acquisition_ids

        if len(images_paths) != len(_acquisition_ids):
            raise NgioValueError(
                "Images paths and acquisition ids must have the same length."
            )

        images = [
            WellImage(path=image_path, acquisition=acquisition_id)
            for image_path, acquisition_id in zip(
                images_paths, _acquisition_ids, strict=True
            )
        ]

        return cls(well=WellMeta(images=images, version=version))

    def paths(self, acquisition: int | None = None) -> list[str]:
        """Return the images paths in the well.

        If acquisition is None, return all images paths in the well.
        Else, return the images paths in the well for the given acquisition.

        Args:
            acquisition (int | None): The acquisition id to filter the images.
        """
        if acquisition is None:
            return [images.path for images in self.well.images]
        return [
            images.path
            for images in self.well.images
            if images.acquisition == acquisition
        ]

    def add_image(self, path: str, acquisition: int | None = None) -> "NgioWellMeta":
        """Add an image to the well.

        Args:
            path (str): The path of the image.
            acquisition (int | None): The acquisition id of the image.
        """
        list_of_images = self.well.images
        for image in list_of_images:
            if image.path == path:
                raise NgioValueError(
                    f"Image at path {path} already exists in the well."
                )

        new_image = WellImage(path=path, acquisition=acquisition)
        list_of_images.append(new_image)
        return NgioWellMeta(
            well=WellMeta(images=list_of_images, version=self.well.version)
        )

    def remove_image(self, path: str) -> "NgioWellMeta":
        """Remove an image from the well.

        Args:
            path (str): The path of the image.
        """
        list_of_images = self.well.images
        for image in list_of_images:
            if image.path == path:
                list_of_images.remove(image)
                return NgioWellMeta(
                    well=WellMeta(images=list_of_images, version=self.well.version)
                )
        raise NgioValueError(f"Image at path {path} not found in the well.")


def _stringify_column(column: str | int) -> str:
    """Convert the column to a string.

    Args:
        column (str | int): The column to convert.

    Returns:
        str: The column as a string.
    """
    if isinstance(column, str):
        return column

    # Maybe we should pad the column with zeros
    return str(column)


def _find_row_index(rows: list[str], row: str) -> int | None:
    try:
        return rows.index(row)
    except ValueError:
        return None


def _find_column_index(columns: list[str], column: str | int) -> int | None:
    _num_columns = [int(columns) for columns in columns]
    column = int(column)
    try:
        return _num_columns.index(column)
    except ValueError:
        return None


def _relabel_wells(
    wells: list[WellInPlate], rows: list[Row], columns: list[Column]
) -> list[WellInPlate]:
    new_wells = []
    _rows = [row.name for row in rows]
    _columns = [column.name for column in columns]
    for well in wells:
        row, column = well.path.split("/")
        row_idx = _find_row_index(_rows, row)
        column_idx = _find_column_index(_columns, column)

        if row_idx is None:
            raise NgioValueError(f"Row {row} not found in the plate.")
        if column_idx is None:
            raise NgioValueError(f"Column {column} not found in the plate.")

        new_wells.append(
            WellInPlate(
                path=well.path,
                rowIndex=row_idx,
                columnIndex=column_idx,
            )
        )

    return new_wells


class NgioPlateMeta(HCSAttrs):
    """HCS plate metadata."""

    @classmethod
    def default_init(
        cls,
        rows: list[str] | None = None,
        columns: list[str] | None = None,
        acquisitions_ids: list[int] | None = None,
        acquisitions_names: list[str] | None = None,
        name: str | None = None,
        version: str | None = None,
    ) -> "NgioPlateMeta":
        if rows is None:
            rows = []
        if columns is None:
            columns = []

        if len(rows) != len(columns):
            raise NgioValueError("Rows and columns must have the same length.")

        unique_rows = list(set(rows))
        unique_columns = list(set(columns))

        _wells = []
        for row, column in zip(rows, columns, strict=True):
            # No need to use the _find_row_index and _find_column_index functions
            # because we are sure that the row and column are in the list
            row_idx = unique_rows.index(row)
            column_idx = unique_columns.index(column)
            _wells.append(
                WellInPlate(
                    path=f"{row}/{_stringify_column(column)}",
                    rowIndex=row_idx,
                    columnIndex=column_idx,
                )
            )

        if acquisitions_ids is not None and acquisitions_names is None:
            _acquisitions_ids = acquisitions_ids
            _acquisitions_names = [None] * len(acquisitions_ids)
        elif acquisitions_ids is None and acquisitions_names is not None:
            _acquisitions_names = acquisitions_names
            _acquisitions_ids = list(range(len(acquisitions_names)))
        elif acquisitions_ids is not None and acquisitions_names is not None:
            if len(acquisitions_ids) != len(acquisitions_names):
                raise NgioValueError(
                    "Acquisitions ids and names must have the same length."
                )
            _acquisitions_ids = acquisitions_ids
            _acquisitions_names = acquisitions_names
        else:
            _acquisitions_ids = None
            _acquisitions_names = None

        if _acquisitions_ids is not None and _acquisitions_names is not None:
            _acquisitions = [
                Acquisition(id=acquisition_id, name=acquisition_name)
                for acquisition_id, acquisition_name in zip(
                    _acquisitions_ids, _acquisitions_names, strict=True
                )
            ]
        else:
            _acquisitions = None

        return cls(
            plate=Plate(
                rows=[Row(name=row) for row in unique_rows],
                columns=[
                    Column(name=_stringify_column(column)) for column in unique_columns
                ],
                acquisitions=_acquisitions,
                wells=_wells,
                field_count=None,
                version=version,
                name=name,
            )
        )

    @property
    def columns(self) -> list[str]:
        """Returns the list of columns in the plate."""
        return [columns.name for columns in self.plate.columns]

    @property
    def rows(self) -> list[str]:
        """Returns the list of rows in the plate."""
        return [rows.name for rows in self.plate.rows]

    @property
    def acquisitions_names(self) -> list[str | None]:
        """Return the acquisitions in the plate."""
        if self.plate.acquisitions is None:
            return []
        return [acquisitions.name for acquisitions in self.plate.acquisitions]

    @property
    def acquisitions_ids(self) -> list[int]:
        """Return the acquisitions ids in the plate."""
        if self.plate.acquisitions is None:
            return []
        return [acquisitions.id for acquisitions in self.plate.acquisitions]

    @property
    def wells_paths(self) -> list[str]:
        """Return the wells paths in the plate."""
        return [wells.path for wells in self.plate.wells]

    def get_well_path(self, row: str, column: str | int) -> str:
        """Return the well path for the given row and column.

        Args:
            row (str): The row of the well.
            column (str | int): The column of the well.

        Returns:
            str: The path of the well.
        """
        if row not in self.rows:
            raise NgioValueError(
                f"Row {row} not found in the plate. Available rows are {self.rows}."
            )

        row_idx = self.rows.index(row)

        _num_columns = [int(columns) for columns in self.columns]

        try:
            _column = int(column)
        except ValueError:
            raise NgioValueError(
                f"Column {column} must be an integer or convertible to an integer."
            ) from None

        column_idx = _num_columns.index(_column)

        for well in self.plate.wells:
            if well.columnIndex == column_idx and well.rowIndex == row_idx:
                return well.path

        raise NgioValueError(
            f"Well at row {row} and column {column} not found in the plate."
        )

    def add_well(
        self,
        row: str,
        column: str | int,
        acquisition_id: int | None = None,
        acquisition_name: str | None = None,
        **acquisition_kwargs,
    ) -> "NgioPlateMeta":
        """Add an image to the well.

        Args:
            row (str): The row of the well.
            column (str | int): The column of the well.
            acquisition_id (int | None): The acquisition id of the well.
            acquisition_name (str | None): The acquisition name of the well.
            **acquisition_kwargs: Additional acquisition metadata.
        """
        relabel_wells = False

        row_names = self.rows
        row_idx = _find_row_index(row_names, row)
        if row_idx is None:
            row_names.append(row)
            row_names.sort()
            row_idx = row_names.index(row)
            relabel_wells = True

        rows = [Row(name=row) for row in row_names]

        columns_names = self.columns
        column_idx = _find_column_index(columns_names, column)
        if column_idx is None:
            columns_names.append(_stringify_column(column))
            # sort as numbers
            columns_names.sort(key=lambda x: int(x))
            column_idx = columns_names.index(_stringify_column(column))
            relabel_wells = True

        columns = [Column(name=column) for column in columns_names]

        wells = self.plate.wells
        if relabel_wells:
            wells = _relabel_wells(wells, rows, columns)

        for well_obj in wells:
            if well_obj.rowIndex == row_idx and well_obj.columnIndex == column_idx:
                break
        else:
            wells.append(
                WellInPlate(
                    path=f"{row}/{_stringify_column(column)}",
                    rowIndex=row_idx,
                    columnIndex=column_idx,
                )
            )

        acquisitions = self.plate.acquisitions
        if acquisition_id is not None:
            if acquisitions is None and len(wells) > 0:
                acquisitions = [Acquisition(id=0, name=acquisition_name)]
            elif acquisitions is None:
                acquisitions = []

            for acquisition_obj in acquisitions:
                if acquisition_obj.id == acquisition_id:
                    break
            else:
                acquisitions.append(
                    Acquisition(
                        id=acquisition_id, name=acquisition_name, **acquisition_kwargs
                    )
                )

        new_plate = Plate(
            rows=rows,
            columns=columns,
            acquisitions=acquisitions,
            wells=wells,
            field_count=self.plate.field_count,
            version=self.plate.version,
        )
        return NgioPlateMeta(plate=new_plate)

    def remove_well(self, row: str, column: str | int) -> "NgioPlateMeta":
        """Remove a well from the plate.

        Args:
            row (str): The row of the well.
            column (str | int): The column of the well.
        """
        row_idx = _find_row_index(self.rows, row)
        if row_idx is None:
            raise NgioValueError(f"Row {row} not found in the plate.")

        column_idx = _find_column_index(self.columns, column)
        if column_idx is None:
            raise NgioValueError(f"Column {column} not found in the plate.")

        wells = self.plate.wells
        for well_obj in wells:
            if well_obj.rowIndex == row_idx and well_obj.columnIndex == column_idx:
                wells.remove(well_obj)
                break
        else:
            raise NgioValueError(
                f"Well at row {row} and column {column} not found in the plate."
            )

        new_plate = Plate(
            rows=self.plate.rows,
            columns=self.plate.columns,
            acquisitions=self.plate.acquisitions,
            wells=wells,
            field_count=self.plate.field_count,
            version=self.plate.version,
        )
        return NgioPlateMeta(plate=new_plate)


# %%

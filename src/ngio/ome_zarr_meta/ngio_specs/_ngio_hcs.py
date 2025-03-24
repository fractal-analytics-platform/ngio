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
from pydantic import BaseModel

from ngio.utils import NgioValueError


class ImageInWellPath(BaseModel):
    """Image in a well."""

    row: str
    column: str | int
    path: str
    acquisition_id: int | None = None
    acquisition_name: str | None = None


class NgioWellMeta(WellAttrs):
    """HCS well metadata."""

    @classmethod
    def default_init(
        cls,
        images: list[ImageInWellPath] | None = None,
        version: Literal["0.4"] | None = None,
    ) -> "NgioWellMeta":
        well = cls(well=WellMeta(images=[], version=version))
        if images is None:
            return well

        for image in images:
            well = well.add_image(path=image.path, acquisition=image.acquisition_id)
        return well

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
        images: list[ImageInWellPath] | None = None,
        name: str | None = None,
        version: str | None = None,
    ) -> "NgioPlateMeta":
        plate = cls(
            plate=Plate(
                rows=[],
                columns=[],
                acquisitions=None,
                wells=[],
                field_count=None,
                version=version,
                name=name,
            )
        )

        if images is None:
            return plate

        for image in images:
            plate = plate.add_well(
                row=image.row,
                column=image.column,
                acquisition_id=image.acquisition_id,
                acquisition_name=image.acquisition_name,
            )
        return plate

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

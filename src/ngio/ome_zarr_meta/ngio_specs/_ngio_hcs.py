"""HCS (High Content Screening) specific metadata classes for NGIO."""

from ome_zarr_models.common.well import WellAttrs
from ome_zarr_models.v04.hcs import HCSAttrs


class NgioWellMeta(WellAttrs):
    """HCS well metadata."""

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


class NgioPlateMeta(HCSAttrs):
    """HCS plate metadata."""

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
            raise ValueError(
                f"Row {row} not found in the plate. Available rows are {self.rows}."
            )

        row_idx = self.rows.index(row)

        _num_columns = [int(columns) for columns in self.columns]

        try:
            _column = int(column)
        except ValueError:
            raise ValueError(
                f"Column {column} must be an integer or convertible to an integer."
            ) from None

        column_idx = _num_columns.index(_column)

        for well in self.plate.wells:
            if well.columnIndex == column_idx and well.rowIndex == row_idx:
                return well.path

        raise ValueError(
            f"Well at row {row} and column {column} not found in the plate."
        )

"""Implementing classes to handle tables in the zarr file."""

from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
from pydantic import BaseModel

from ngio.tables.tables import load_table_meta
from ngio.tables.v1 import RoiTable


class ROI(BaseModel):
    """Region of interest (ROI) metadata."""

    field_index: str
    x: float
    y: float
    z: float
    x_length: float
    y_length: float
    z_length: float

    @classmethod
    def from_df_row(cls, df_row: Any) -> "ROI":
        """Create an ROI object from a DataFrame row."""
        return cls(
            field_index=df_row.name,
            x=df_row.x_micrometer,
            y=df_row.y_micrometer,
            z=df_row.z_micrometer,
            x_length=df_row.len_x_micrometer,
            y_length=df_row.len_y_micrometer,
            z_length=df_row.len_z_micrometer,
        )


class RoiTableHandler:
    """Class to handle ROI tables in the zarr file."""

    def __init__(self, zarr_url: str | Path, table_name: str) -> None:
        """Initialize the RoiTableHandler object."""
        self.zarr_url = zarr_url
        self.table_name = table_name
        metadata = load_table_meta(zarr_url=zarr_url, table_name=table_name)

        if isinstance(metadata, RoiTable):
            self._table_metadata = metadata
        else:
            raise ValueError("Unsupported table type")

        self._data = ad.read_zarr(f"{zarr_url}/tables/{table_name}")

    @property
    def data(self) -> ad.AnnData:
        """Return the data of the ROI table as an AnnData object."""
        return self._data

    @property
    def metadata(self) -> RoiTable:
        """Return the metadata of the ROI table."""
        return self._table_metadata

    def to_df(self) -> pd.DataFrame:
        """Return the data of the ROI table as a DataFrame."""
        return self._data.to_df()

    def get_roi(self, field_index: str) -> ROI:
        """Return the ROI object for a given field index."""
        return ROI.from_df_row(self.to_df().loc[field_index])

    def list_roi_index(self) -> list[str]:
        """Return the list of field indices in the ROI table."""
        return list(self.to_df().index)

    def iter_over_roi(self) -> list[ROI]:
        """Iterate over the ROIs in the ROI table."""
        return [self.get_roi(roi) for roi in self.list_roi_index()]

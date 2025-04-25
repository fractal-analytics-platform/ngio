from typing import Literal

import pandas as pd
import polars as pl

from ngio.tables.backends._non_zarr_backends_v1 import NonZarrBaseBackend
from ngio.utils import ZarrGroupHandler


def write_lf_to_csv(path: str, table: pl.DataFrame) -> None:
    """Write a polars DataFrame to a CSV file."""
    table.write_csv(path)


def write_df_to_csv(path: str, table: pd.DataFrame) -> None:
    """Write a pandas DataFrame to a CSV file."""
    table.to_csv(path, index=False)


class CsvTableBackend(NonZarrBaseBackend):
    """A class to load and write small tables in CSV format."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        index_key: str | None = None,
        index_type: None | Literal["int"] | Literal["str"] = None,
    ):
        """Initialize the CsvTableBackend."""
        super().__init__(
            lf_reader=pl.scan_csv,
            df_reader=pd.read_csv,
            lf_writer=write_lf_to_csv,
            df_writer=write_df_to_csv,
            table_name="table.csv",
            group_handler=group_handler,
            index_key=index_key,
            index_type=index_type,
        )

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        return "experimental_csv_v1"

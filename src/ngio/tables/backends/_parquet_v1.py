from typing import Literal

import pandas as pd
import polars as pl

from ngio.tables.backends._non_zarr_backends_v1 import NonZarrBaseBackend
from ngio.utils import ZarrGroupHandler


def write_lf_to_parquet(path: str, table: pl.DataFrame) -> None:
    """Write a polars DataFrame to a Parquet file."""
    table.write_parquet(path)


def write_df_to_parquet(path: str, table: pd.DataFrame) -> None:
    """Write a pandas DataFrame to a Parquet file."""
    table.to_parquet(path, index=False)


class ParquetTableBackend(NonZarrBaseBackend):
    """A class to load and write small tables in Parquet format."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        index_key: str | None = None,
        index_type: None | Literal["int", "str"] = None,
    ):
        """Initialize the ParquetTableBackend."""
        super().__init__(
            lf_reader=pl.scan_parquet,
            df_reader=pd.read_parquet,
            lf_writer=write_lf_to_parquet,
            df_writer=write_df_to_parquet,
            table_name="table.parquet",
            group_handler=group_handler,
            index_key=index_key,
            index_type=index_type,
        )

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        return "experimental_parquet_v1"

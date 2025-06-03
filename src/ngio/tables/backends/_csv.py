import pandas as pd
import polars as pl

from ngio.tables.backends._non_zarr_backends import NonZarrBaseBackend


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
    ):
        """Initialize the CsvTableBackend."""
        super().__init__(
            lf_reader=pl.scan_csv,
            df_reader=pd.read_csv,
            lf_writer=write_lf_to_csv,
            df_writer=write_df_to_csv,
            table_name="table.csv",
        )

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        return "csv"

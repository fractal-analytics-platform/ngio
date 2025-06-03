import pandas as pd
import polars as pl

from ngio.tables.backends._non_zarr_backends import NonZarrBaseBackend


def write_lf_to_parquet(path: str, table: pl.DataFrame) -> None:
    """Write a polars DataFrame to a Parquet file."""
    # make categorical into string (for pandas compatibility)
    schema = table.collect_schema()

    categorical_columns = []
    for name, dtype in zip(schema.names(), schema.dtypes(), strict=True):
        if dtype == pl.Categorical:
            categorical_columns.append(name)

    for col in categorical_columns:
        table = table.with_columns(pl.col(col).cast(pl.Utf8))

    # write to parquet
    table.write_parquet(path)


def write_df_to_parquet(path: str, table: pd.DataFrame) -> None:
    """Write a pandas DataFrame to a Parquet file."""
    table.to_parquet(path, index=False)


class ParquetTableBackend(NonZarrBaseBackend):
    """A class to load and write small tables in Parquet format."""

    def __init__(
        self,
    ):
        """Initialize the ParquetTableBackend."""
        super().__init__(
            lf_reader=pl.scan_parquet,
            df_reader=pd.read_parquet,
            lf_writer=write_lf_to_parquet,
            df_writer=write_df_to_parquet,
            table_name="table.parquet",
        )

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        return "parquet"

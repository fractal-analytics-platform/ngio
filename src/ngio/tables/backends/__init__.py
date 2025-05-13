"""Ngio Tables backend implementations."""

from ngio.tables.backends._abstract_backend import AbstractTableBackend, BackendMeta
from ngio.tables.backends._anndata_v1 import AnnDataBackend
from ngio.tables.backends._csv_v1 import CsvTableBackend
from ngio.tables.backends._json_v1 import JsonTableBackend
from ngio.tables.backends._parquet_v1 import ParquetTableBackend
from ngio.tables.backends._table_backends import (
    ImplementedTableBackends,
    TableBackendProtocol,
)
from ngio.tables.backends._utils import (
    TabularData,
    convert_anndata_to_pandas,
    convert_anndata_to_polars,
    convert_pandas_to_anndata,
    convert_pandas_to_polars,
    convert_polars_to_anndata,
    convert_polars_to_pandas,
    convert_to_anndata,
    convert_to_pandas,
    convert_to_polars,
    normalize_anndata,
    normalize_pandas_df,
    normalize_polars_lf,
    normalize_table,
)

__all__ = [
    "AbstractTableBackend",
    "AnnDataBackend",
    "BackendMeta",
    "CsvTableBackend",
    "ImplementedTableBackends",
    "JsonTableBackend",
    "ParquetTableBackend",
    "TableBackendProtocol",
    "TabularData",
    "convert_anndata_to_pandas",
    "convert_anndata_to_polars",
    "convert_pandas_to_anndata",
    "convert_pandas_to_polars",
    "convert_polars_to_anndata",
    "convert_polars_to_pandas",
    "convert_to_anndata",
    "convert_to_pandas",
    "convert_to_polars",
    "normalize_anndata",
    "normalize_pandas_df",
    "normalize_polars_lf",
    "normalize_table",
]

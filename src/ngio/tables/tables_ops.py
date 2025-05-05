"""Aggregation and filtering operations for tables."""

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
import polars as pl

from ngio.tables import Table, TableType


def _reindex_dataframe(
    dataframe, index_cols: list[str], index_key: str = "ConcatenateIndex"
) -> pd.DataFrame:
    """Reindex a dataframe using an hash of the index columns."""
    # Reindex the dataframe
    old_index = dataframe.index.name
    dataframe = dataframe.reset_index()
    if old_index is not None:
        index_cols.append(old_index)
    dataframe.index = dataframe[index_cols].astype(str).agg("_".join, axis=1)
    dataframe.index.name = index_key
    return dataframe


def _add_const_columns(
    dataframe: pd.DataFrame,
    new_cols: dict[str, str],
    index_key: str | None = None,
) -> pd.DataFrame:
    for col, value in new_cols.items():
        dataframe[col] = value

    if index_key is not None:
        dataframe = _reindex_dataframe(
            dataframe=dataframe,
            index_cols=list(new_cols.keys()),
            index_key=index_key,
        )
    return dataframe


def _add_const_columns_pl(
    dataframe: pl.LazyFrame,
    new_cols: dict[str, str],
    index_key: str = "Index",
) -> pl.LazyFrame:
    dataframe = dataframe.with_columns(
        [pl.lit(value, dtype=pl.String()).alias(col) for col, value in new_cols.items()]
    )

    dataframe = dataframe.with_columns(
        [
            pl.concat_str(
                [pl.col(col) for col in new_cols.keys()],
                separator="_",
            ).alias(index_key)
        ]
    )
    return dataframe


@dataclass
class TableWithExtras:
    """A class to hold a table and its extras."""

    table: Table
    extras: dict[str, str] = field(default_factory=dict)


def _pd_concat(
    tables: Collection[TableWithExtras], index_key: str = "Index"
) -> pd.DataFrame:
    """Concatenate tables from different plates into a single table."""
    if len(tables) == 0:
        raise ValueError("No tables to concatenate.")

    dataframes = []
    for table in tables:
        dataframe = _add_const_columns(
            dataframe=table.table.dataframe, new_cols=table.extras, index_key=index_key
        )
        dataframes.append(dataframe)
    concatenated_table = pd.concat(dataframes, axis=0)
    return concatenated_table


def _pl_concat(
    tables: Collection[TableWithExtras], index_key: str = "Index"
) -> pl.LazyFrame:
    """Concatenate tables from different plates into a single table."""
    if len(tables) == 0:
        raise ValueError("No tables to concatenate.")

    dataframes = []
    for table in tables:
        polars_ls = _add_const_columns_pl(
            dataframe=table.table.lazy_frame,
            new_cols=table.extras,
            index_key=index_key,
        )
        dataframes.append(polars_ls)

    concatenated_table = pl.concat(dataframes, how="vertical")
    return concatenated_table


def conctatenate_tables(
    tables: Collection[TableWithExtras],
    mode: Literal["eager", "lazy"] = "eager",
    index_key: str = "Index",
    table_cls: type[TableType] | None = None,
) -> Table:
    """Concatenate tables from different plates into a single table."""
    if len(tables) == 0:
        raise ValueError("No tables to concatenate.")

    table0 = next(iter(tables)).table

    if mode == "lazy":
        concatenated_table = _pl_concat(tables=tables, index_key=index_key)
    elif mode == "eager":
        concatenated_table = _pd_concat(tables=tables, index_key=index_key)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'eager' or 'lazy'.")

    meta = table0.meta
    meta.index_key = index_key
    meta.index_type = "str"

    if table_cls is not None:
        return table_cls.from_table_data(
            table_data=concatenated_table,
            meta=meta,
        )
    return table0.from_table_data(
        table_data=concatenated_table,
        meta=meta,
    )

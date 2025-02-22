from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import zarr
from anndata import AnnData
from anndata._io.specs import read_elem
from anndata._io.utils import _read_legacy_raw
from anndata._io.zarr import read_dataframe
from anndata.compat import _clean_uns
from anndata.experimental import read_dispatched

from ngio.utils import (
    NgioTableValidationError,
    StoreOrGroup,
    open_group_wrapper,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection


def custom_read_zarr(
    store: StoreOrGroup, elem_to_read: Collection[str] | None = None
) -> AnnData:
    """Read from a hierarchical Zarr array store.

    # Implementation originally from https://github.com/scverse/anndata/blob/main/src/anndata/_io/zarr.py
    # Original implementation would not work with remote storages so we had to copy it
    # here and slightly modified it to work with remote storages.

    Args:
        store (StoreOrGroup): A store or group to read the AnnData from.
        elem_to_read (Collection[str] | None): The elements to read from the store.
    """
    group, _ = open_group_wrapper(store=store, mode="r")

    if elem_to_read is None:
        elem_to_read = [
            "X",
            "obs",
            "var",
            "uns",
            "obsm",
            "varm",
            "obsp",
            "varp",
            "layers",
        ]

    # Read with handling for backwards compat
    def callback(func: Callable, elem_name: str, elem: Any, iospec: Any) -> Any:
        if iospec.encoding_type == "anndata" or elem_name.endswith("/"):
            ad_kwargs = {}
            # Some of these elem fail on https
            # So we only include the ones that are strictly necessary
            # for fractal tables
            # This fails on some https
            # base_elem += list(elem.keys())
            for k in elem_to_read:
                v = elem.get(k)
                if v is not None and not k.startswith("raw."):
                    ad_kwargs[k] = read_dispatched(v, callback)  # type: ignore
            return AnnData(**ad_kwargs)

        elif elem_name.startswith("/raw."):
            return None
        elif elem_name in {"/obs", "/var"}:
            return read_dataframe(elem)
        elif elem_name == "/raw":
            # Backwards compat
            return _read_legacy_raw(group, func(elem), read_dataframe, func)
        return func(elem)

    adata = read_dispatched(group, callback=callback)  # type: ignore

    # Backwards compat (should figure out which version)
    if "raw.X" in group:
        raw = AnnData(**_read_legacy_raw(group, adata.raw, read_dataframe, read_elem))  # type: ignore
        raw.obs_names = adata.obs_names  # type: ignore
        adata.raw = raw  # type: ignore

    # Backwards compat for <0.7
    if isinstance(group["obs"], zarr.Array):
        _clean_uns(adata)

    if not isinstance(adata, AnnData):
        raise ValueError(f"Expected an AnnData object, but got {type(adata)}")
    return adata


def _check_for_mixed_types(series: pd.Series) -> None:
    """Check if the column has mixed types."""
    if series.apply(type).nunique() > 1:  # type: ignore
        raise NgioTableValidationError(
            f"Column {series.name} has mixed types: "
            f"{series.apply(type).unique()}. "  # type: ignore
            "Type of all elements must be the same."
        )


def _check_for_supported_types(series: pd.Series) -> Literal["str", "int", "numeric"]:
    """Check if the column has supported types."""
    if ptypes.is_string_dtype(series):
        return "str"
    if ptypes.is_integer_dtype(series):
        return "int"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    raise NgioTableValidationError(
        f"Column {series.name} has unsupported type: {series.dtype}."
        " Supported types are string and numerics."
    )


def _check_index_key(
    table_df: pd.DataFrame, index_key: str | None, index_type: str = "int"
) -> pd.DataFrame:
    """Check if the index_key correctness.

    - Check if the index_key is present in the data frame.
        (If the index_key is a column in the DataFrame, it is set as the index)
    - Check if the index_key is of the correct type.

    Args:
        table_df (pd.DataFrame): The DataFrame to validate.
        index_key (str): The column name to use as the index of the DataFrame.
        index_type (str): The type of the index column in the DataFrame.
            Either 'str' or 'int'. Default is 'int'.

    Returns:
        pd.DataFrame: The validated DataFrame.
    """
    if index_type not in ["str", "int"]:
        raise ValueError(f"index_type {index_type} not recognized")

    if index_key is None:
        return table_df

    columns = table_df.columns
    if index_key in columns:
        table_df = table_df.set_index(index_key)

    if table_df.index.name != index_key:
        raise NgioTableValidationError(
            f"index_key: {index_key} not found in data frame"
        )

    if index_type == "str":
        if ptypes.is_integer_dtype(table_df.index):
            # Convert the int index to string is generally safe
            table_df.index = table_df.index.astype(str)

        if not ptypes.is_string_dtype(table_df.index):
            raise NgioTableValidationError(
                f"index_key {index_key} must be of string type"
            )

    elif index_type == "int":
        if ptypes.is_string_dtype(table_df.index):
            # Try to convert the string index to int
            try:
                table_df.index = table_df.index.astype(int)
            except ValueError as e:
                if "invalid literal for int() with base 10" in str(e):
                    raise NgioTableValidationError(
                        f"index_key {index_key} must be of "
                        "integer type, but found string. We "
                        "tried implicit conversion failed."
                    ) from None
                else:
                    raise e from e

        if not ptypes.is_integer_dtype(table_df.index):
            raise NgioTableValidationError(
                f"index_key {index_key} must be of integer type"
            )

    else:
        raise NgioTableValidationError(f"index_type {index_type} not recognized")

    return table_df


def dataframe_to_anndata(
    dataframe: pd.DataFrame,
    index_key: str | None = None,
    index_type: str = "int",
) -> ad.AnnData:
    """Convert a table DataFrame to an AnnData object.

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame representing a fractal table.
        index_key (str): The column name to use as the index of the DataFrame.
        index_type (str): The type of the index column in the DataFrame.
            Either 'str' or 'int'. Default is 'int'.
        validators (list[Validator]): A list of functions to further validate the table.
    """
    # Check if the index_key is present in the data frame + optional validations
    dataframe = _check_index_key(dataframe, index_key, index_type)

    # DO NOT SKIP
    # Convert the index to string ALWAYS to avoid casting issues in AnnData
    dataframe.index = dataframe.index.astype(str)

    str_columns, int_columns, num_columns = [], [], []
    for c_name in dataframe.columns:
        column_df = dataframe[c_name]
        _check_for_mixed_types(column_df)  # Mixed types are not allowed in the table
        c_type = _check_for_supported_types(
            column_df
        )  # Only string and numeric types are allowed

        if c_type == "str":
            str_columns.append(c_name)

        elif c_type == "int":
            int_columns.append(c_name)

        elif c_type == "numeric":
            num_columns.append(c_name)

    # Converting all observations to string
    obs_df = dataframe[str_columns + int_columns]
    obs_df.index = dataframe.index

    x_df = dataframe[num_columns]

    if x_df.dtypes.nunique() > 1:
        x_df = x_df.astype("float64")

    if x_df.empty:
        # If there are no numeric columns, create an empty array
        # to avoid AnnData failing to create the object
        x_df = np.zeros((len(obs_df), 0), dtype="float64")

    return ad.AnnData(X=x_df, obs=obs_df)


def anndata_to_dataframe(
    anndata: ad.AnnData,
    index_key: str | None = "label",
    index_type: str = "int",
    validate_index_name: bool = False,
) -> pd.DataFrame:
    """Convert a AnnData object representing a fractal table to a pandas DataFrame.

    Args:
        anndata (ad.AnnData): An AnnData object representing a fractal table.
        index_key (str): The column name to use as the index of the DataFrame.
            Default is 'label'.
        index_type (str): The type of the index column in the DataFrame.
            Either 'str' or 'int'. Default is 'int'.
        validators (list[Validator]): A list of functions to further validate the table.
        validate_index_name (bool): If True, the index name is validated.
    """
    dataframe = anndata.to_df()
    dataframe[anndata.obs_keys()] = anndata.obs

    # Set the index of the DataFrame
    if index_key in dataframe.columns:
        dataframe = dataframe.set_index(index_key)
    elif anndata.obs.index.name is not None:
        if validate_index_name:
            if anndata.obs.index.name != index_key:
                raise NgioTableValidationError(
                    f"Index key {index_key} not found in AnnData object."
                )
        dataframe.index = anndata.obs.index
    elif anndata.obs.index.name is None:
        dataframe.index = anndata.obs.index
        dataframe.index.name = index_key
    else:
        raise NgioTableValidationError(
            f"Index key {index_key} not found in AnnData object."
        )

    dataframe = _check_index_key(dataframe, index_key, index_type)
    return dataframe

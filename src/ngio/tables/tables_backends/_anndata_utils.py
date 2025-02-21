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

from ngio.tables._validators import TableValidator, validate_table
from ngio.utils import (
    NgioTableValidationError,
    StoreOrGroup,
    open_group_wrapper,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable


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


def dataframe_to_anndata(
    dataframe: pd.DataFrame,
    validators: Iterable[TableValidator] | None = None,
) -> ad.AnnData:
    """Convert a table DataFrame to an AnnData object.

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame representing a fractal table.
        validators (Iterable[TableValidator] | None): A collection of functions
            used to validate the table. Default is None.

    """
    dataframe = validate_table(dataframe, validators)

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
    obs_dataframe = dataframe[str_columns + int_columns]
    obs_dataframe.index = dataframe.index

    x_dataframe = dataframe[num_columns]

    if x_dataframe.dtypes.nunique() > 1:
        x_dataframe = x_dataframe.astype("float64")

    if x_dataframe.empty:
        # If there are no numeric columns, create an empty array
        # to avoid AnnData failing to create the object
        x_dataframe = np.zeros((0, 0), dtype="float64")

    return ad.AnnData(X=x_dataframe, obs=obs_dataframe)


def anndata_to_dataframe(
    anndata: ad.AnnData,
    validators: Iterable[TableValidator] | None = None,
) -> pd.DataFrame:
    """Convert a AnnData object representing a fractal table to a pandas DataFrame.

    Args:
        anndata (ad.AnnData): An AnnData object to be converted to a DataFrame.
        validators (Iterable[TableValidator] | None): A collection of functions
            used to validate the table. Default is None.
    """
    dataframe = anndata.to_df()
    dataframe[anndata.obs_keys()] = anndata.obs
    return validate_table(dataframe, validators)

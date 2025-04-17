"""Test suite for the utils module."""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import polars as pl
import pytest
from anndata import AnnData
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame

from ngio.tables.backends._utils import (
    _check_for_mixed_types,
    _check_for_supported_types,
    _validate_cast_index_dtype_df,
    _validate_index_key_df,
    convert_anndata_to_pandas,
    convert_anndata_to_polars,
    convert_pandas_to_anndata,
    convert_pandas_to_polars,
    convert_polars_to_anndata,
    convert_polars_to_pandas,
    normalize_anndata,
    normalize_pandas_df,
    normalize_polars_lf,
)
from ngio.utils import NgioTableValidationError, NgioValidationError, NgioValueError

# -----------------
# Test fixtures
# -----------------


def sample_pandas_df_no_index():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "str_col": ["x", "y", "z"],
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
        }
    )


def sample_pandas_df_int_index():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "str_col": ["x", "y", "z"],
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
        }
    ).set_index("id")


def sample_pandas_df_str_index():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "str_col": ["x", "y", "z"],
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
        }
    ).set_index("id")


def sample_polars_lf():
    """Create a sample polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "id": ["1", "2", "3"],
            "str_col": ["x", "y", "z"],
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
        }
    ).lazy()


def sample_anndata():
    """Create a sample AnnData object for testing."""
    obs = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "str_col": ["x", "y", "z"],
            "int_col": [1, 2, 3],
        }
    ).set_index("id")

    x = np.array([[1.1, 2.2, 3.2]]).T
    return AnnData(X=x, obs=obs)


@pytest.mark.parametrize(
    "in_df, out_df, index_key, index_type, reset_index",
    [
        (sample_pandas_df_no_index(), sample_pandas_df_no_index(), None, None, False),
        (sample_pandas_df_no_index(), sample_pandas_df_str_index(), "id", "str", False),
        (sample_pandas_df_no_index(), sample_pandas_df_int_index(), "id", "int", False),
        (sample_pandas_df_int_index(), sample_pandas_df_int_index(), None, None, False),
        (
            sample_pandas_df_int_index(),
            sample_pandas_df_str_index(),
            "id",
            "str",
            False,
        ),
        (sample_pandas_df_int_index(), sample_pandas_df_no_index(), "id", "str", True),
        (sample_pandas_df_str_index(), sample_pandas_df_no_index(), "id", "str", True),
    ],
)
def test_normalize_pandas_df(in_df, out_df, index_key, index_type, reset_index):
    """Test the normalization of a pandas DataFrame."""
    df = normalize_pandas_df(
        in_df,
        index_key=index_key,
        index_type=index_type,
        reset_index=reset_index,
    )

    pdt.assert_frame_equal(
        df,
        out_df,
        check_dtype=False,
    )


def test_fail_normalize_pandas_df():
    """Test the normalization of a pandas DataFrame."""
    with pytest.raises(NgioTableValidationError):
        normalize_pandas_df(
            sample_pandas_df_no_index(),
            index_key="str_col",
            index_type="int",
            reset_index=False,
        )

    with pytest.raises(NgioTableValidationError):
        normalize_pandas_df(
            sample_pandas_df_no_index(),
            index_key="not_exist",
            index_type="str",
            reset_index=False,
        )

    with pytest.raises(NgioValueError):
        normalize_pandas_df(
            sample_pandas_df_no_index(),
            index_key="id",
            index_type="float",  # type: ignore
            reset_index=False,
        )


def test_normalize_polars_lf():
    """Test the conversion of an AnnData object to a pandas DataFrame."""
    lf = sample_polars_lf()
    normalized_lf = normalize_polars_lf(lf, index_key="id", index_type="str").collect()
    normalized_lf_int = normalize_polars_lf(
        lf, index_key="id", index_type="int"
    ).collect()


def test_normalize_anndata():
    """Test the conversion of an AnnData object to a pandas DataFrame."""
    adata = sample_anndata()
    normalized_adata = normalize_anndata(adata, index_key="str_col")
    normalized_adata = normalize_anndata(adata, index_key="id")

    with pytest.raises(NgioTableValidationError):
        normalize_anndata(adata, index_key="not_exist")

from collections.abc import Callable
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import pandas.api.types as ptypes


class TableValidationError(Exception):
    """Error raised when a table is not formatted correctly."""

    pass


Validator = Callable[[pd.DataFrame], pd.DataFrame]


def _check_for_mixed_types(series: pd.Series) -> None:
    """Check if the column has mixed types."""
    if series.apply(type).nunique() > 1:
        raise TableValidationError(
            f"Column {series.name} has mixed types: "
            f"{series.apply(type).unique()}. "
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
    raise TableValidationError(
        f"Column {series.name} has unsupported type: {series.dtype}."
        " Supported types are string and numerics."
    )


def _check_index_key(
    table_df: pd.DataFrame, index_key: str, index_type: Literal["str", "int"]
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
    columns = table_df.columns
    if index_key in columns:
        table_df = table_df.set_index(index_key)

    if table_df.index.name != index_key:
        raise TableValidationError(f"index_key: {index_key} not found in data frame")

    if index_type == "str":
        if ptypes.is_integer_dtype(table_df.index):
            # Convert the int index to string is generally safe
            table_df.index = table_df.index.astype(str)

        if not ptypes.is_string_dtype(table_df.index):
            raise TableValidationError(f"index_key {index_key} must be of string type")

    elif index_type == "int":
        if ptypes.is_string_dtype(table_df.index):
            # Try to convert the string index to int
            try:
                table_df.index = table_df.index.astype(int)
            except ValueError as e:
                if "invalid literal for int() with base 10" in str(e):
                    raise TableValidationError(
                        f"index_key {index_key} must be of "
                        "integer type, but found string. We "
                        "tried implicit conversion failed."
                    ) from None
                else:
                    raise e from e

        if not ptypes.is_integer_dtype(table_df.index):
            raise TableValidationError(f"index_key {index_key} must be of integer type")

    else:
        raise TableValidationError(f"index_type {index_type} not recognized")

    return table_df


def validate_table(
    table_df: pd.DataFrame,
    index_key: str,
    index_type: Literal["str", "int"],
    validators: list[Validator] | None,
) -> pd.DataFrame:
    """Validate the table DataFrame.

    - Check if the index_key is present in the data frame.
        (If the index_key is a column in the DataFrame, it is set as the index)
    - Check if the index_key is of the correct type.
    - Apply all provided optional validators.

    Args:
        table_df (pd.DataFrame): The DataFrame to validate.
        index_key (str): The column name to use as the index of the DataFrame.
        index_type (str): The type of the index column in the DataFrame.
        validators (list[Validator]): A list of functions to further validate table.

    Returns:
        pd.DataFrame: The validated DataFrame.
    """
    table_df = _check_index_key(table_df, index_key, index_type)

    if validators is None:
        return table_df

    # Apply all provided validators
    for validator in validators:
        table_df = validator(table_df)

    return table_df


def table_df_to_ad(
    table_df: pd.DataFrame,
    index_key: str,
    index_type: Literal["str", "int"] = "int",
    validators: list[Validator] | None = None,
) -> ad.AnnData:
    """Convert a table DataFrame to an AnnData object.

    Args:
        table_df (pd.DataFrame): A pandas DataFrame representing a fractal table.
        index_key (str): The column name to use as the index of the DataFrame.
        index_type (str): The type of the index column in the DataFrame.
            Either 'str' or 'int'. Default is 'int'.
        validators (list[Validator]): A list of functions to further validate the table.
    """
    # Check if the index_key is present in the data frame + optional validations
    table_df = validate_table(
        table_df=table_df,
        index_key=index_key,
        index_type=index_type,
        validators=validators,
    )

    # DO NOT SKIP
    # Convert the index to string ALWAYS to avoid casting issues in AnnData
    table_df.index = table_df.index.astype(str)

    str_columns, int_columns, num_columns = [], [], []
    for c_name in table_df.columns:
        column_df = table_df[c_name]
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
    obs_df = table_df[str_columns + int_columns]
    obs_df.index = table_df.index

    x_df = table_df[num_columns]

    if x_df.dtypes.nunique() > 1:
        x_df = x_df.astype("float64")

    if x_df.empty:
        # If there are no numeric columns, create an empty array
        # to avoid AnnData failing to create the object
        x_df = np.zeros((0, 0), dtype="float64")

    return ad.AnnData(X=x_df, obs=obs_df)


def table_ad_to_df(
    table_ad: ad.AnnData,
    index_key: str = "label",
    index_type: Literal["str", "int"] = "int",
    validators: list[Validator] | None = None,
    validate_index_name: bool = False,
) -> pd.DataFrame:
    """Convert a AnnData object representing a fractal table to a pandas DataFrame.

    Args:
        table_ad (ad.AnnData): An AnnData object representing a fractal table.
        index_key (str): The column name to use as the index of the DataFrame.
            Default is 'label'.
        index_type (str): The type of the index column in the DataFrame.
            Either 'str' or 'int'. Default is 'int'.
        validators (list[Validator]): A list of functions to further validate the table.
        validate_index_name (bool): If True, the index name is validated.
    """
    table_df = table_ad.to_df()
    table_df[table_ad.obs_keys()] = table_ad.obs

    # Set the index of the DataFrame
    if index_key in table_df.columns:
        table_df = table_df.set_index(index_key)
    elif table_ad.obs.index.name is not None:
        if validate_index_name:
            if table_ad.obs.index.name != index_key:
                raise TableValidationError(
                    f"Index key {index_key} not found in AnnData object."
                )
        table_df.index = table_ad.obs.index
    elif table_ad.obs.index.name is None:
        table_df.index = table_ad.obs.index
        table_df.index.name = index_key
    else:
        raise TableValidationError(
            f"Index key {index_key} not found in AnnData object."
        )

    table_df = validate_table(
        table_df=table_df,
        index_key=index_key,
        index_type=index_type,
        validators=validators,
    )
    return table_df


####################################################################################################
#
# Common table validators
#
####################################################################################################


def validate_columns(
    table_df: pd.DataFrame,
    required_columns: list[str],
    optional_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Validate the columns headers of the table.

    If a required column is missing, a TableValidationError is raised.
    If a list of optional columns is provided, only required and optional columns are
        allowed in the table.

    Args:
        table_df (pd.DataFrame): The DataFrame to validate.
        required_columns (list[str]): A list of required columns.
        optional_columns (list[str] | None): A list of optional columns.
            Default is None.

    Returns:
        pd.DataFrame: The validated DataFrame.
    """
    table_header = table_df.columns
    for column in required_columns:
        if column not in table_header:
            raise TableValidationError(f"Column {column} is required in ROI table")

    if optional_columns is None:
        return table_df

    possible_columns = [*required_columns, *optional_columns]
    for column in table_header:
        if column not in possible_columns:
            raise TableValidationError(
                f"Column {column} is not recognized in ROI table"
            )

    return table_df


def validate_unique_index(table_df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the index of the table is unique."""
    if table_df.index.is_unique:
        return table_df

    # Find the duplicates
    duplicates = table_df.index[table_df.index.duplicated()].tolist()
    raise TableValidationError(
        f"Index of the table contains duplicates values. Duplicate: {duplicates}"
    )

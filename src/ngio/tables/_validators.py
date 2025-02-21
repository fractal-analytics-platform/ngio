from collections.abc import Iterable
from typing import Literal, Protocol

import pandas as pd
import pandas.api.types as ptypes

from ngio.utils import NgioTableValidationError


class TableValidator(Protocol):
    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        """Validate the table DataFrame.

        A Validator is just a simple callable that takes a
            DataFrame and returns a DataFrame.

        If the DataFrame is valid, the same DataFrame is returned.
        If the DataFrame is invalid, the Validator can either modify the DataFrame
            to make it valid or raise a NgioTableValidationError.

        Args:
            table (pd.DataFrame): The DataFrame to validate.

        Returns:
            pd.DataFrame: The validated DataFrame.

        """
        ...


def validate_table(
    table_df: pd.DataFrame,
    validators: Iterable[TableValidator] | None = None,
) -> pd.DataFrame:
    """Validate the table DataFrame.

    Args:
        table_df (pd.DataFrame): The DataFrame to validate.
        validators (Collection[Validator] | None): A collection of functions
            used to validate the table. Default is None.

    Returns:
        pd.DataFrame: The validated DataFrame.
    """
    validators = validators or []

    # Apply all provided validators
    for validator in validators:
        table_df = validator(table_df)

    return table_df


####################################################################################################
#
# Common table validators
#
####################################################################################################


def validate_index(
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
            raise NgioTableValidationError(
                f"Could not find required column: {column} in the table"
            )

    if optional_columns is None:
        return table_df

    possible_columns = [*required_columns, *optional_columns]
    for column in table_header:
        if column not in possible_columns:
            raise NgioTableValidationError(
                f"Could not find column: {column} in the list of possible columns. ",
                f"Possible columns are: {possible_columns}",
            )

    return table_df


def validate_unique_index(table_df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the index of the table is unique."""
    if table_df.index.is_unique:
        return table_df

    # Find the duplicates
    duplicates = table_df.index[table_df.index.duplicated()].tolist()
    raise NgioTableValidationError(
        f"Index of the table contains duplicates values. Duplicate: {duplicates}"
    )

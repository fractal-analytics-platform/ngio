from collections.abc import Iterable
from typing import Protocol

import pandas as pd
import pandas.api.types as ptypes

from ngio.utils import NgioTableValidationError, NgioValueError


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
def validate_index_key(
    dataframe: pd.DataFrame, index_key: str | None, overwrite: bool = False
) -> pd.DataFrame:
    """Correctly set the index of the DataFrame.

    This function checks if the index_key is present in the DataFrame.
    If not it tries to set sensible defaults.

    In order:
        - If index_key is None, nothing can be done.
        - If index_key is already the index of the DataFrame, nothing is done.
        - If index_key is in the columns, we set the index to that column.
        - If current index is None, we set the index to the index_key.
        - If current index is not None and overwrite is True,
            we set the index to the index_key.

    """
    if index_key is None:
        # Nothing to do
        return dataframe

    if dataframe.index.name == index_key:
        # Index is already set to index_key correctly
        return dataframe

    if index_key in dataframe.columns:
        dataframe = dataframe.set_index(index_key)
        return dataframe

    if dataframe.index.name is None:
        dataframe.index.name = index_key
        return dataframe

    elif overwrite:
        dataframe.index.name = index_key
        return dataframe
    else:
        raise NgioTableValidationError(
            f"Index key {index_key} not found in DataFrame. "
            f"Current index is {dataframe.index.name}. If you want to overwrite the "
            "index set overwrite=True."
        )


def validate_index_dtype(dataframe: pd.DataFrame, index_type: str) -> pd.DataFrame:
    """Check if the index of the DataFrame has the correct dtype."""
    match index_type:
        case "str":
            if ptypes.is_integer_dtype(dataframe.index):
                # Convert the int index to string is generally safe
                dataframe = dataframe.set_index(dataframe.index.astype(str))

            if not ptypes.is_string_dtype(dataframe.index):
                raise NgioTableValidationError(
                    f"Table index must be of string type, got {dataframe.index.dtype}"
                )

        case "int":
            if ptypes.is_string_dtype(dataframe.index):
                # Try to convert the string index to int
                try:
                    dataframe = dataframe.set_index(dataframe.index.astype(int))
                except ValueError as e:
                    if "invalid literal for int() with base 10" in str(e):
                        raise NgioTableValidationError(
                            "Table index must be of integer type, got str."
                            f" We tried implicit conversion and failed: {e}"
                        ) from None
                    else:
                        raise e from e

            if not ptypes.is_integer_dtype(dataframe.index):
                raise NgioTableValidationError(
                    f"Table index must be of integer type, got {dataframe.index.dtype}"
                )
        case _:
            raise NgioValueError(f"index_type {index_type} not recognized")

    return dataframe


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

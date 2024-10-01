import anndata as ad
import pandas as pd
import pandas.api.types as ptypes


class TableValidationError(Exception):
    """Error raised when a table is not formatted correctly."""

    pass


def _safe_to_df(data_frame: pd.DataFrame, index_key: str) -> pd.DataFrame:
    columns = data_frame.columns

    if index_key not in columns:
        raise TableValidationError(f"index_key {index_key} not found in data frame")

    if not ptypes.is_integer_dtype(data_frame[index_key]):
        raise TableValidationError(f"index_key {index_key} must be of integer type")

    data_frame[index_key] = data_frame[index_key].astype(str)

    str_columns, num_columns = [], []
    for c_name in columns:
        column_df = data_frame[c_name]
        if column_df.apply(type).nunique() > 1:
            raise TableValidationError(
                f"Column {c_name} has mixed types: "
                f"{column_df.apply(type).unique()}. "
                "Type of all elements must be the same."
            )

        if ptypes.is_string_dtype(column_df):
            str_columns.append(c_name)

        elif ptypes.is_numeric_dtype(column_df):
            num_columns.append(c_name)
        else:
            raise TableValidationError(
                f"Column {c_name} has unsupported type: {column_df.dtype}."
                " Supported types are string and numerics."
            )

    obs_df = data_frame[str_columns]
    obs_df.index = obs_df.index.astype(str)
    x_df = data_frame[num_columns]
    x_df = x_df.astype("float32")
    return ad.AnnData(X=x_df, obs=obs_df)


def df_to_andata(
    data_frame: pd.DataFrame,
    index_key: str = "label",
    implicit_conversion: bool = False,
) -> ad.AnnData:
    """Convert a pandas DataFrame representing a fractal table to an AnnData object.

    Args:
        data_frame: A pandas DataFrame representing a fractal table.
        index_key: The column name to use as the index of the DataFrame.
            Default is 'label'.
        implicit_conversion: If True, the function will convert the data frame
            to an AnnData object as it. If False, the function will check the data frame
            for compatibility.
            And correct correctly formatted data frame to AnnData object.
            Default is False.
    """
    if implicit_conversion:
        return ad.AnnData(data_frame)

    return _safe_to_df(data_frame, index_key)


def df_from_andata(andata_table: ad.AnnData, index_key: str = "label") -> pd.DataFrame:
    """Convert a AnnData object representing a fractal table to a pandas DataFrame.

    Args:
        andata_table: An AnnData object representing a fractal table.
        index_key: The column name to use as the index of the DataFrame.
            Default is 'label'.

    """
    data_frame = andata_table.to_df()
    data_frame[andata_table.obs_keys()] = andata_table.obs

    if index_key not in data_frame.columns:
        raise TableValidationError(f"index_key {index_key} not found in data frame.")

    data_frame[index_key] = data_frame[index_key].astype(int)
    return data_frame


def validate_roi_table(
    data_frame: pd.DataFrame,
    required_columns: list[str],
    optional_columns: list[str],
    index_name: str = "FieldIndex",
) -> pd.DataFrame:
    """Validate the ROI table.

    Args:
        data_frame: The ROI table as a DataFrame.
        required_columns: A list of required columns in the ROI table.
        optional_columns: A list of optional columns in the ROI table.
        index_name: The name of the index column in the ROI table.
            Default is 'FieldIndex'.
    """
    if data_frame.index.name != index_name:
        if index_name in data_frame.columns:
            data_frame = data_frame.set_index(index_name)
        else:
            raise TableValidationError(
                f"{index_name} is required in ROI table. It must be the index or a "
                "column"
            )

    table_header = data_frame.columns
    for column in required_columns:
        if column not in table_header:
            raise TableValidationError(f"Column {column} is required in ROI table")

    possible_columns = [*required_columns, *optional_columns]
    for column in table_header:
        if column not in possible_columns:
            raise TableValidationError(
                f"Column {column} is not recognized in ROI table"
            )
    return data_frame

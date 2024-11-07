import pandas as pd
import pytest


class TestValidation:
    def test_validate_unique(self) -> None:
        from ngio.tables._utils import NgioTableValidationError, validate_unique_index

        df = pd.DataFrame.from_records(
            data=[
                {"id": 1, "x": 0.1},
                {"id": 2, "x": 0.3},
                {"id": 3, "x": 0.5},
            ]
        )
        df.set_index("id", inplace=True)
        out_df = validate_unique_index(df)
        assert out_df.equals(df)

        df = pd.DataFrame.from_records(
            data=[
                {"id": 1, "x": 0.1},
                {"id": 1, "x": 0.3},
                {"id": 3, "x": 0.5},
            ]
        )
        df.set_index("id", inplace=True)
        with pytest.raises(NgioTableValidationError):
            validate_unique_index(df)

    def test_validate_column(self) -> None:
        from ngio.tables._utils import NgioTableValidationError, validate_columns

        df = pd.DataFrame.from_records(
            data=[
                {"id": 1, "x": 0.1},
                {"id": 2, "x": 0.3},
                {"id": 3, "x": 0.5},
            ]
        )
        out_df = validate_columns(
            df, required_columns=["id", "x"], optional_columns=["y"]
        )
        assert out_df.equals(df)

        out_df = validate_columns(df, required_columns=["id", "x"])
        assert out_df.equals(df)

        with pytest.raises(NgioTableValidationError):
            validate_columns(df, required_columns=["y"])

        with pytest.raises(NgioTableValidationError):
            validate_columns(df, required_columns=["id"], optional_columns=["y"])

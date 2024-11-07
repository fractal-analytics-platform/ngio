import pandas as pd
import pytest


class TestTableConversion:
    def test_table_conversion1(self) -> None:
        from ngio.tables._utils import (
            NgioTableValidationError,
            table_ad_to_df,
            table_df_to_ad,
        )

        df = pd.DataFrame.from_records(
            data=[
                {"label": 1, "feat1": 0.1},
                {"label": 2, "feat1": 0.3},
                {"label": 3, "feat1": 0.5},
            ]
        )

        with pytest.raises(NgioTableValidationError):
            table_df_to_ad(df, index_key="label2", index_type="str")

        # Index as column
        ad_table = table_df_to_ad(df, index_key="label", index_type="int")

        df_out = table_ad_to_df(ad_table, index_key="label", index_type="int")

        df_out["feat1"].equals(df["feat1"])

        # Set index explicitly
        df.set_index("label", inplace=True)
        ad_table = table_df_to_ad(df, index_key="label", index_type="int")

        df_out = table_ad_to_df(ad_table, index_key="label", index_type="int")

        df_out["feat1"].equals(df["feat1"])

    def test_table_conversion2(self) -> None:
        from ngio.tables._utils import (
            NgioTableValidationError,
            table_ad_to_df,
            table_df_to_ad,
        )

        df = pd.DataFrame.from_records(
            data=[
                {"label": "1a", "feat1": 0.1},
                {"label": "2b", "feat1": 0.3},
                {"label": "3c", "feat1": 0.5},
            ]
        )

        with pytest.raises(NgioTableValidationError):
            table_df_to_ad(df, index_key="label", index_type="int")
        ad_table = table_df_to_ad(df, index_key="label", index_type="str")

        df_out = table_ad_to_df(table_ad=ad_table, index_key="label", index_type="str")

        df_out["feat1"].equals(df["feat1"])

        with pytest.raises(NgioTableValidationError):
            df_out = table_ad_to_df(
                table_ad=ad_table, index_key="label", index_type="int"
            )

    def test_table_conversion3(self) -> None:
        from ngio.tables._utils import (
            NgioTableValidationError,
            table_df_to_ad,
        )

        df = pd.DataFrame.from_records(
            data=[
                {"label": 1.3, "feat1": 0.1},
                {"label": 2.1, "feat1": 0.3},
                {"label": 3.4, "feat1": 0.5},
            ]
        )

        with pytest.raises(NgioTableValidationError):
            table_df_to_ad(df, index_key="label", index_type="int")

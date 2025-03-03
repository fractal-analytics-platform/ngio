from pathlib import Path

import pandas as pd
import pytest

from ngio.tables.tables_container import open_table, write_table
from ngio.tables.v1 import GenericTable


@pytest.mark.parametrize("backend", ["json_v1", "anndata_v1"])
def test_generic_table(tmp_path: Path, backend: str):
    store = tmp_path / "test.zarr"
    test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    table = GenericTable(test_df)

    write_table(store=store, table=table, backend=backend)

    loaded_table = open_table(store=store)
    assert isinstance(loaded_table, GenericTable)
    assert set(loaded_table.dataframe.columns) == {"a", "b"}
    for column in loaded_table.dataframe.columns:
        pd.testing.assert_series_equal(
            loaded_table.dataframe[column], test_df[column], check_index=False
        )

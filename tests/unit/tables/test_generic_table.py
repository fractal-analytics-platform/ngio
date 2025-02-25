from pathlib import Path

import pandas as pd
import pytest

from ngio.tables._generic_table import GenericTable


@pytest.mark.parametrize("backend", ["json", "anndata"])
def test_generic_table(tmp_path: Path, backend: str):
    store = tmp_path / "test.zarr"
    test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    table = GenericTable(test_df)

    table.set_backend(store=store, backend_name=backend)
    table.consolidate()

    loaded_table = GenericTable.from_store(store)

    assert set(loaded_table.dataframe.columns) == {"a", "b"}
    for column in loaded_table.dataframe.columns:
        pd.testing.assert_series_equal(
            loaded_table.dataframe[column], test_df[column], check_index=False
        )

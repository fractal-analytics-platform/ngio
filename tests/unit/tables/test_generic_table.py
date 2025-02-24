from pathlib import Path

import pandas as pd
import pytest

from ngio.tables.generic_table import GenericTable


@pytest.mark.parametrize("backend", ["json", "anndata"])
def test_anndata_backend(tmp_path: Path, backend: str):
    from ngio.tables.backends._anndata_utils import dataframe_to_anndata

    store = str(tmp_path / "test.zarr")
    test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    dataframe_to_anndata(test_df)

    table = GenericTable(test_df)

    table.set_backend("anndata", store)
    table.consolidate()

    new_table = GenericTable.from_store(store)

    assert set(new_table.dataframe.columns) == {"a", "b"}
    for column in new_table.dataframe.columns:
        pd.testing.assert_series_equal(new_table.dataframe[column], test_df[column])

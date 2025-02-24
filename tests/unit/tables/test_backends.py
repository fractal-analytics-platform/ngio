from pathlib import Path

import pandas as pd
import pytest

from ngio.tables.backends import TableBackendsManager
from ngio.tables.backends._anndata import AnnDataBackend
from ngio.tables.backends._json import JsonTableBackend
from ngio.utils import NgioValueError, ZarrGroupHandler


def test_backend_manager(tmp_path: Path):
    manager = TableBackendsManager()

    assert set(manager.available_backends) == {"json", "anndata"}
    manager.add_handler("json2", JsonTableBackend)

    manager2 = TableBackendsManager()
    assert set(manager2.available_backends) == {"json", "anndata", "json2"}
    assert set(manager.available_backends) == {"json", "anndata", "json2"}

    store = tmp_path / "test_backend_manager.zarr"
    handler = ZarrGroupHandler(store=store, cache=True, mode="a")
    backend = manager.get_backend("json", handler)
    assert isinstance(backend, JsonTableBackend)

    backend = manager.get_backend(None, handler)
    assert isinstance(backend, AnnDataBackend)

    with pytest.raises(NgioValueError):
        manager.get_backend("non_existent", handler)

    with pytest.raises(NgioValueError):
        manager.add_handler("json", JsonTableBackend)


def test_json_backend(tmp_path: Path):
    store = tmp_path / "test_json_backend.zarr"
    handler = ZarrGroupHandler(store=store, cache=True, mode="a")
    backend = JsonTableBackend(handler)

    assert backend.backend_name == "json"
    assert not backend.implements_anndata
    assert backend.implements_dataframe

    test_table = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["a", "b", "c"]}
    )
    test_table.index = test_table.index.astype(str)

    backend.write_from_dataframe(test_table, metadata={"test": "test"})
    loaded_table = backend.load_as_dataframe()
    assert loaded_table.equals(test_table)
    assert backend.load_columns() == ["a", "b", "c"]
    assert backend._group_handler.load_attrs() == {"test": "test"}

    assert backend.load_as_dataframe(columns=["a"]).equals(test_table[["a"]])


def test_anndata_backend(tmp_path: Path):
    store = tmp_path / "test_anndata_backend.zarr"
    handler = ZarrGroupHandler(store=store, cache=True, mode="a")
    backend = AnnDataBackend(handler)

    assert backend.backend_name == "anndata"
    assert backend.implements_anndata
    assert backend.implements_dataframe

    test_table = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["a", "b", "c"]}
    )

    backend.write_from_dataframe(test_table, metadata={"test": "test"})
    loaded_table = backend.load_as_dataframe()
    assert set(backend.load_columns()) == {"a", "b", "c"}
    columns = backend.load_columns()

    for column in columns:
        # Since the transformation from anndata to dataframe is not perfect
        # We can only compare the columns
        pd.testing.assert_series_equal(loaded_table[column], test_table[column])

    assert backend._group_handler.load_attrs()["test"] == "test"

    assert backend.load_as_dataframe(columns=["a"]).equals(test_table[["a"]])

    with pytest.raises(NotImplementedError):
        backend.load_as_anndata(columns=["a"])

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

from ngio.tables.backends import TableBackendsManager
from ngio.tables.backends._anndata import AnnDataBackend
from ngio.tables.backends._anndata_utils import (
    anndata_to_dataframe,
    dataframe_to_anndata,
)
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
    backend = AnnDataBackend(handler, index_type="int")

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


@pytest.mark.parametrize(
    "index_label, index_type",
    [(None, "int"), ("label", "str"), ("label", "int")],
)
def test_anndata_to_dataframe(index_label: str | None, index_type: str):
    test_obs = pd.DataFrame({"a": [1, 2, 3], "c": ["a", "b", "c"]})
    test_x = pd.DataFrame({"b": [4.0, 5.0, 6.0]})

    if index_label is None:
        test_obs.index = test_obs.index.astype(str)
    else:
        test_obs.index = pd.Index(["1", "2", "3"], name=index_label)

    anndata = ad.AnnData(obs=test_obs, X=test_x)

    dataframe = anndata_to_dataframe(
        anndata,
        index_key=index_label,
        index_type=index_type,
    )

    for column in test_obs.columns:
        pd.testing.assert_series_equal(
            dataframe[column], test_obs[column], check_index=False
        )

    for column in test_x.columns:
        pd.testing.assert_series_equal(
            dataframe[column], test_x[column], check_index=False
        )

    if index_label is not None:
        assert dataframe.index.name == index_label
        if index_type == "int":
            assert ptypes.is_integer_dtype(dataframe.index)
        elif index_type == "str":
            assert ptypes.is_string_dtype(dataframe.index)


@pytest.mark.parametrize(
    "index_label, index_type",
    [(None, "int"), ("label", "str"), ("label", "int")],
)
def test_dataframe_to_anndata(index_label: str | None, index_type: str):
    test_table = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4.0, 5.0, 6.0],
            "c": ["a", "b", "c"],
            "d": [4.0, 5.0, 6.0],
        }
    )

    if index_label is not None and index_type == "int":
        test_table.index = pd.Index([1, 2, 3], name=index_label)
    elif index_label is not None and index_type == "str":
        test_table.index = pd.Index(["a", "b", "c"], name=index_label)

    anndata = dataframe_to_anndata(test_table, index_key=index_label)

    for column in anndata.obs.columns:
        pd.testing.assert_series_equal(
            anndata.obs[column], test_table[column], check_index=False
        )

    if index_label is not None:
        assert anndata.obs.index.name == index_label

    for i, column in enumerate(anndata.var.index):
        np.testing.assert_allclose(anndata.X[:, i], test_table[column].values)  # type: ignore


@pytest.mark.parametrize(
    "index_label, index_type",
    [(None, "int"), ("label", "str"), ("label", "int")],
)
def test_round_trip(index_label: str | None, index_type: str):
    test_table = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4.0, 5.0, 6.0],
            "c": ["a", "b", "c"],
            "d": [4.0, 5.0, 6.0],
        }
    )

    if index_label is not None and index_type == "int":
        test_table.index = pd.Index([1, 2, 3], name=index_label)
    elif index_label is not None and index_type == "str":
        test_table.index = pd.Index(["a", "b", "c"], name=index_label)

    anndata = dataframe_to_anndata(test_table, index_key=index_label)
    datafame = anndata_to_dataframe(
        anndata, index_key=index_label, index_type=index_type
    )

    for column in datafame.columns:
        pd.testing.assert_series_equal(
            datafame[column], test_table[column], check_index=True
        )

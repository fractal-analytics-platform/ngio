import pytest
import zarr
from conftest import ZARR_PYTHON_V


class TestGroupUtils:
    @property
    def test_attrs(self) -> dict:
        return {"a": 1, "b": 2, "c": 3}

    def test_open_group_wrapper(self, store_fixture):
        from ngio.io import open_group_wrapper

        store, zarr_format = store_fixture
        group = open_group_wrapper(store=store, mode="r+", zarr_format=zarr_format)
        group.attrs.update(self.test_attrs)
        assert dict(group.attrs) == self.test_attrs

    @pytest.mark.skipif(ZARR_PYTHON_V, reason="Zarr V2 does not support remote stores.")
    def test_raise_not_implemented_error(self):
        from ngio.io._zarr_group_utils import open_group_wrapper

        with pytest.raises(NotImplementedError):
            open_group_wrapper(
                store=zarr.store.RemoteStore(url="https://test.com/test.zarr"),
                mode="r",
                zarr_format=3,
            )

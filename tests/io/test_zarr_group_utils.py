class TestGroupUtils:
    @property
    def test_attrs(self) -> dict:
        return {"a": 1, "b": 2, "c": 3}

    def test_update_group_attrs(self, store_fixture):
        from ngio.io.zarr_group_utils import (
            read_group_attrs,
            update_group_attrs,
        )

        store, zarr_format = store_fixture

        update_group_attrs(store=store, attrs=self.test_attrs, zarr_format=zarr_format)
        attrs = read_group_attrs(store=store, zarr_format=zarr_format)
        assert attrs == self.test_attrs, "Attributes were not written correctly."

        update_group_attrs(store=store, attrs={"new": 1}, zarr_format=zarr_format)
        attrs = read_group_attrs(store=store, zarr_format=zarr_format)
        expected = {**self.test_attrs, "new": 1}
        assert attrs == expected, "Attributes were not written correctly."

    def test_overwrite_group_attrs(self, store_fixture):
        from ngio.io.zarr_group_utils import (
            overwrite_group_attrs,
            read_group_attrs,
        )

        store, zarr_format = store_fixture

        overwrite_group_attrs(
            store=store, attrs=self.test_attrs, zarr_format=zarr_format
        )
        attrs = read_group_attrs(store=store, zarr_format=zarr_format)
        assert attrs == self.test_attrs, "Attributes were not written correctly."

    def test_list_group_arrays(self, store_fixture):
        from ngio.io.zarr_group_utils import list_group_arrays

        store, zarr_format = store_fixture

        arrays = list_group_arrays(store=store, zarr_format=zarr_format)
        assert len(arrays) == 3, "Arrays were not listed correctly."

    def test_list_group_groups(self, store_fixture):
        from ngio.io.zarr_group_utils import list_group_groups

        store, zarr_format = store_fixture

        groups = list_group_groups(store=store, zarr_format=zarr_format)
        assert len(groups) == 3, "Groups were not listed correctly."
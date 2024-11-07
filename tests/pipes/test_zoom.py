import numpy as np
import pytest
import zarr


class TestZoom:
    def _test_zoom(
        self, source: zarr.Array, target: zarr.Array, order: int = 1, mode: str = "dask"
    ) -> None:
        from ngio.pipes import on_disk_zoom

        on_disk_zoom(source, target, order=order, mode=mode)

    def test_zoom_3d(self, zarr_zoom_3d_array: tuple[zarr.Array, zarr.Array]) -> None:
        source, target = zarr_zoom_3d_array

        for mode in ["dask", "numpy"]:
            for order in [0, 1, 2]:
                self._test_zoom(source, target, order=order, mode=mode)

    def test_zoom_2d(self, zarr_zoom_2d_array: tuple[zarr.Array, zarr.Array]) -> None:
        source, target = zarr_zoom_2d_array
        self._test_zoom(source, target)

    def test_zoom_4d(self, zarr_zoom_4d_array: tuple[zarr.Array, zarr.Array]) -> None:
        source, target = zarr_zoom_4d_array
        self._test_zoom(source, target)

    def test_zoom_3d_fail(
        self, zarr_zoom_3d_array_shape_mismatch: tuple[zarr.Array, zarr.Array]
    ) -> None:
        source, target = zarr_zoom_3d_array_shape_mismatch
        with pytest.raises(ValueError):
            self._test_zoom(source, target)

        with pytest.raises(ValueError):
            self._test_zoom(source, target[...])

        with pytest.raises(ValueError):
            self._test_zoom(source[...], target)

        with pytest.raises(ValueError):
            _target2 = target.astype("float32")
            self._test_zoom(source, _target2)

        with pytest.raises(AssertionError):
            self._test_zoom(source, target, mode="not_a_mode")

    def _test_coarsen(self, source: zarr.Array, target: zarr.Array) -> None:
        from ngio.pipes._zoom_utils import on_disk_coarsen

        on_disk_coarsen(source, target, aggregation_function=np.mean)

    def test_coarsen_3d(
        self, zarr_zoom_3d_array: tuple[zarr.Array, zarr.Array]
    ) -> None:
        source, target = zarr_zoom_3d_array
        self._test_coarsen(source, target)

    def test_coarsen_2d(
        self, zarr_zoom_2d_array: tuple[zarr.Array, zarr.Array]
    ) -> None:
        source, target = zarr_zoom_2d_array
        self._test_coarsen(source, target)

    def test_coarsen_4d(
        self, zarr_zoom_4d_array: tuple[zarr.Array, zarr.Array]
    ) -> None:
        source, target = zarr_zoom_4d_array
        self._test_coarsen(source, target)

    def test_coarsen_2d_fail(
        self, zarr_zoom_2d_array_not_int: tuple[zarr.Array, zarr.Array]
    ) -> None:
        source, target = zarr_zoom_2d_array_not_int
        with pytest.raises(ValueError):
            self._test_coarsen(source, target)

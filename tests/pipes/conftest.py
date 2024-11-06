# create a zarr 3D array fixture
from pathlib import Path

import numpy as np
import pytest
import zarr


@pytest.fixture
def zarr_zoom_3d_array(tmp_path: Path) -> tuple[zarr.Array, zarr.Array]:
    source = zarr.zeros((3, 64, 64), store=tmp_path / "test_3d_s.zarr")
    source[...] = np.random.rand(3, 64, 64)
    target = zarr.zeros((3, 32, 32), store=tmp_path / "test_3d_t.zarr")
    return source, target


@pytest.fixture
def zarr_zoom_2d_array(tmp_path: Path) -> tuple[zarr.Array, zarr.Array]:
    source = zarr.zeros((64, 64), store=tmp_path / "test_2d_s.zarr")
    source[...] = np.random.rand(64, 64)
    target = zarr.zeros((32, 32), store=str(tmp_path / "test_2d_t.zarr"))
    return source, target


@pytest.fixture
def zarr_zoom_4d_array(tmp_path: Path) -> tuple[zarr.Array, zarr.Array]:
    source = zarr.zeros((3, 3, 64, 64), store=tmp_path / "test_4d_s.zarr")
    source[...] = np.random.rand(3, 3, 64, 64)
    target = zarr.zeros((3, 3, 32, 32), store=tmp_path / "test_4d_t.zarr")
    return source, target


@pytest.fixture
def zarr_zoom_2d_array_not_int(tmp_path: Path) -> tuple[zarr.Array, zarr.Array]:
    source = zarr.zeros((64, 64), store=tmp_path / "test_2d_s.zarr")
    source[...] = np.random.rand(64, 64)
    target = zarr.zeros((30, 30), store=str(tmp_path / "test_2d_t.zarr"))
    return source, target


@pytest.fixture
def zarr_zoom_3d_array_shape_mismatch(tmp_path: Path) -> tuple[zarr.Array, zarr.Array]:
    source = zarr.zeros((3, 3, 64, 64), store=tmp_path / "test_3d_s.zarr")
    source[...] = np.random.rand(3, 3, 64, 64)
    target = zarr.zeros((3, 32, 32), store=tmp_path / "test_3d_t.zarr")
    return source, target

from pathlib import Path

import numpy as np
import pytest

from ngio import create_empty_omezarr


@pytest.mark.parametrize("array_mode", ["numpy", "dask"])
def test_omezarr_container(tmp_path: Path, array_mode: str):
    # Very basic test to check if the container is working
    # to be expanded with more meaningful tests
    store = tmp_path / "omezarr.zarr"
    omezarr = create_empty_omezarr(
        store,
        shape=(10, 20, 30),
        chunks=(1, 20, 30),
        xy_pixelsize=0.5,
        levels=3,
        dtype="uint8",
    )

    assert omezarr.levels == 3
    assert omezarr.levels_paths == ["0", "1", "2"]

    image = omezarr.get_image()
    assert image.shape == (10, 20, 30)
    assert image.dtype == "uint8"
    assert image.chunks == (1, 20, 30)
    assert image.pixel_size.x == 0.5

    array = image.get_array(
        x=slice(None), axes_order=["c", "z", "y", "x"], mode=array_mode
    )

    assert array.shape == (1, 10, 20, 30)

    array = array + 1

    image.set_array(array, x=slice(None), axes_order=["c", "z", "y", "x"])
    image.consolidate(mode=array_mode)

    image = omezarr.get_image(path="2")
    assert np.mean(image.get_array()) == 1

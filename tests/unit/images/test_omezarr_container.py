from pathlib import Path

import numpy as np
import pytest

from ngio import create_empty_omezarr, create_omezarr_from_array, open_omezarr_container
from ngio.utils import fractal_fsspec_store


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

    assert isinstance(omezarr.__repr__(), str)
    assert omezarr.levels == 3
    assert omezarr.levels_paths == ["0", "1", "2"]

    image = omezarr.get_image()

    assert image.shape == (10, 20, 30)
    assert image.dtype == "uint8"
    assert image.chunks == (1, 20, 30)
    assert image.pixel_size.x == 0.5
    assert image.meta.get_highest_resolution_dataset().path == "0"
    assert image.meta.get_lowest_resolution_dataset().path == "2"

    array = image.get_array(
        x=slice(None), axes_order=["c", "z", "y", "x"], mode=array_mode
    )

    assert array.shape == (1, 10, 20, 30)

    array = array + 1

    image.set_array(array, x=slice(None), axes_order=["c", "z", "y", "x"])
    image.consolidate(mode=array_mode)

    # Omemeta
    omezarr.initialize_channel_meta(labels=["channel_x"])
    image = omezarr.get_image()
    assert image.channel_labels == ["channel_x"]
    omezarr.update_percentiles()

    image = omezarr.get_image(path="2")
    assert np.mean(image.get_array()) == 1

    new_omezarr = omezarr.derive_image(tmp_path / "derived.zarr", ref_path="2")

    assert new_omezarr.levels == 3
    new_image = new_omezarr.get_image()
    assert new_image.shape == image.shape


def test_create_omezarr_container(tmp_path: Path):
    # Very basic test to check if the container is working
    # to be expanded with more meaningful tests
    store = tmp_path / "omezarr.zarr"
    create_omezarr_from_array(
        store,
        array=np.zeros((10, 20, 30), dtype="uint8"),
        xy_pixelsize=0.5,
        levels=3,
    )


def test_remote_omezarr_container():
    url = (
        "https://raw.githubusercontent.com/"
        "fractal-analytics-platform/fractal-ome-zarr-examples/"
        "refs/heads/main/v04/"
        "20200812-CardiomyocyteDifferentiation14-Cycle1_B_03_mip.zarr/"
    )

    store = fractal_fsspec_store(url)
    omezarr = open_omezarr_container(store)

    assert omezarr.list_labels() == ["nuclei"]
    # assert omezarr.list_tables() == [
    #    "FOV_ROI_table",
    #    "nuclei_ROI_table",
    #    "well_ROI_table",
    #    "regionprops_DAPI",
    # ]

    _ = omezarr.get_label("nuclei", path="0")
    # _ = omezarr.get_table("well_ROI_table")

from pathlib import Path

import numpy as np
import pytest

from ngio import create_empty_omezarr, open_omezarr_container
from ngio.utils import fractal_fsspec_store


@pytest.mark.parametrize(
    "zarr_name",
    [
        "test_image_yx.zarr",
        "test_image_cyx.zarr",
        "test_image_zyx.zarr",
        "test_image_czyx.zarr",
        "test_image_c1yx.zarr",
        "test_image_tyx.zarr",
        "test_image_tcyx.zarr",
        "test_image_tzyx.zarr",
        "test_image_tczyx.zarr",
    ],
)
def test_open_omezarr_container(images_v04: dict[str, Path], zarr_name: str):
    path = images_v04[zarr_name]
    omezarr = open_omezarr_container(path)

    whole_image_roi = omezarr.build_image_roi_table().get("image")
    image = omezarr.get_image()
    assert image.get_roi(whole_image_roi).shape == image.shape

    label = omezarr.get_label("label")
    roi = image.build_image_roi_table().get("image")
    image.get_roi(roi)
    label.get_roi(roi)


def test_omezarr_tables(cardiomyocyte_tiny_path: Path):
    cardiomyocyte_tiny_path = cardiomyocyte_tiny_path / "B" / "03" / "0"
    omezarr = open_omezarr_container(cardiomyocyte_tiny_path)
    assert omezarr.list_tables() == ["FOV_ROI_table", "well_ROI_table"], (
        omezarr.list_tables()
    )
    assert omezarr.list_roi_tables() == ["FOV_ROI_table", "well_ROI_table"], (
        omezarr.list_roi_tables()
    )

    fov_roi = omezarr.get_table("FOV_ROI_table", check_type=None)
    assert len(fov_roi.rois()) == 2  # type: ignore
    roi_table_1 = omezarr.get_table("well_ROI_table", check_type="generic_roi_table")
    assert len(roi_table_1.rois()) == 1
    roi_table_2 = omezarr.get_table("well_ROI_table", check_type="roi_table")
    assert len(roi_table_2.rois()) == 1

    new_well_roi_table = omezarr.build_image_roi_table()
    omezarr.add_table("new_well_ROI_table", new_well_roi_table)

    assert omezarr.list_tables() == [
        "FOV_ROI_table",
        "well_ROI_table",
        "new_well_ROI_table",
    ]


@pytest.mark.parametrize("array_mode", ["numpy", "dask"])
def test_create_omezarr_container(tmp_path: Path, array_mode: str):
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
    assert omezarr.is_3d
    assert not omezarr.is_time_series
    assert not omezarr.is_multi_channels
    assert not omezarr.is_2d_time_series
    assert not omezarr.is_3d_time_series

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

    new_omezarr = omezarr.derive_image(tmp_path / "derived2.zarr", ref_path="2")

    assert new_omezarr.levels == 3
    new_image = new_omezarr.get_image()
    assert new_image.shape == image.shape

    new_label = new_omezarr.derive_label("new_label")
    assert new_label.shape == image.shape
    assert new_label.meta.axes_mapper.on_disk_axes_names == ["z", "y", "x"]

    assert new_omezarr.list_labels() == ["new_label"]
    assert new_omezarr.list_tables() == []
    assert new_omezarr.list_roi_tables() == []

    # Test masked image instantiation
    masked_image = new_omezarr.get_masked_image(masking_label_name="new_label")
    assert masked_image.shape == image.shape
    masked_label = new_omezarr.get_masked_label(
        label_name="new_label", masking_label_name="new_label"
    )
    assert masked_label.shape == image.shape


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
    _ = omezarr.get_table("well_ROI_table", check_type="roi_table")

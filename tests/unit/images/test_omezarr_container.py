from pathlib import Path

import numpy as np
import pytest

from ngio import create_empty_ome_zarr, open_ome_zarr_container
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
def test_open_ome_zarr_container(images_v04: dict[str, Path], zarr_name: str):
    path = images_v04[zarr_name]
    ome_zarr = open_ome_zarr_container(path)

    whole_image_roi = ome_zarr.build_image_roi_table().get("image")
    image = ome_zarr.get_image()
    assert isinstance(image.__repr__(), str)
    assert image.get_roi(whole_image_roi).shape == image.shape

    label = ome_zarr.get_label("label")
    assert isinstance(label.__repr__(), str)
    roi = image.build_image_roi_table().get("image")
    image.get_roi(roi)
    label.get_roi(roi)


def test_ome_zarr_tables(cardiomyocyte_tiny_path: Path):
    cardiomyocyte_tiny_path = cardiomyocyte_tiny_path / "B" / "03" / "0"
    ome_zarr = open_ome_zarr_container(cardiomyocyte_tiny_path)
    assert ome_zarr.list_tables() == ["FOV_ROI_table", "well_ROI_table"], (
        ome_zarr.list_tables()
    )
    assert ome_zarr.list_roi_tables() == ["FOV_ROI_table", "well_ROI_table"], (
        ome_zarr.list_roi_tables()
    )

    fov_roi = ome_zarr.get_table("FOV_ROI_table")
    assert len(fov_roi.rois()) == 2  # type: ignore
    roi_table_1 = ome_zarr.get_table("well_ROI_table")
    assert len(roi_table_1.rois()) == 1  # type: ignore
    roi_table_2 = ome_zarr.get_table("well_ROI_table")
    assert len(roi_table_2.rois()) == 1  # type: ignore

    new_well_roi_table = ome_zarr.build_image_roi_table()
    ome_zarr.add_table("new_well_ROI_table", new_well_roi_table)

    assert ome_zarr.list_tables() == [
        "FOV_ROI_table",
        "well_ROI_table",
        "new_well_ROI_table",
    ]


@pytest.mark.parametrize("array_mode", ["numpy", "dask"])
def test_create_ome_zarr_container(tmp_path: Path, array_mode: str):
    # Very basic test to check if the container is working
    # to be expanded with more meaningful tests
    store = tmp_path / "ome_zarr.zarr"
    ome_zarr = create_empty_ome_zarr(
        store,
        shape=(10, 20, 30),
        chunks=(1, 20, 30),
        xy_pixelsize=0.5,
        levels=3,
        dtype="uint8",
    )

    assert isinstance(ome_zarr.__repr__(), str)
    assert ome_zarr.levels == 3
    assert ome_zarr.levels_paths == ["0", "1", "2"]
    assert ome_zarr.is_3d
    assert not ome_zarr.is_time_series
    assert not ome_zarr.is_multi_channels
    assert not ome_zarr.is_2d_time_series
    assert not ome_zarr.is_3d_time_series
    assert ome_zarr.space_unit == "micrometer"
    assert ome_zarr.time_unit is None

    ome_zarr.set_axes_units(space_unit="yoctometer", time_unit="yoctosecond")
    assert ome_zarr.space_unit == "yoctometer"
    assert ome_zarr.time_unit is None

    image = ome_zarr.get_image()

    assert image.shape == (10, 20, 30)
    assert image.dtype == "uint8"
    assert image.chunks == (1, 20, 30)
    assert image.pixel_size.x == 0.5
    assert image.meta.get_highest_resolution_dataset().path == "0"
    assert image.meta.get_lowest_resolution_dataset().path == "2"

    array = image.get_as_numpy()
    assert isinstance(array, np.ndarray)
    assert array.shape == (10, 20, 30)

    array_dask = image.get_as_dask()
    assert array_dask.shape == (10, 20, 30)

    array = image.get_array(
        x=slice(None),
        axes_order=["c", "z", "y", "x"],
        mode=array_mode,  # type: ignore
    )

    array = array + 1  # type: ignore

    image.set_array(array, x=slice(None), axes_order=["c", "z", "y", "x"])
    image.consolidate(mode=array_mode)  # type: ignore

    # Omemeta
    ome_zarr.set_channel_meta(labels=["channel_x"])
    image = ome_zarr.get_image()
    assert image.channel_labels == ["channel_x"]
    ome_zarr.set_channel_percentiles()

    image = ome_zarr.get_image(path="2")
    assert np.mean(image.get_array()) == 1  # type: ignore

    new_ome_zarr = ome_zarr.derive_image(tmp_path / "derived2.zarr", ref_path="2")

    assert new_ome_zarr.levels == 3
    new_image = new_ome_zarr.get_image()
    assert new_image.shape == image.shape

    new_label = new_ome_zarr.derive_label("new_label")
    assert new_label.shape == image.shape
    assert new_label.meta.axes_mapper.on_disk_axes_names == ["z", "y", "x"]

    assert new_ome_zarr.list_labels() == ["new_label"]
    assert new_ome_zarr.list_tables() == []
    assert new_ome_zarr.list_roi_tables() == []

    # Test masked image instantiation
    masked_image = new_ome_zarr.get_masked_image(masking_label_name="new_label")
    assert masked_image.shape == image.shape
    masked_label = new_ome_zarr.get_masked_label(
        label_name="new_label", masking_label_name="new_label"
    )
    assert masked_label.shape == image.shape


def test_remote_ome_zarr_container():
    url = (
        "https://raw.githubusercontent.com/"
        "fractal-analytics-platform/fractal-ome-zarr-examples/"
        "refs/heads/main/v04/"
        "20200812-CardiomyocyteDifferentiation14-Cycle1_B_03_mip.zarr/"
    )

    store = fractal_fsspec_store(url)
    ome_zarr = open_ome_zarr_container(store)

    assert ome_zarr.list_labels() == ["nuclei"]
    # assert ome_zarr.list_tables() == [
    #    "FOV_ROI_table",
    #    "nuclei_ROI_table",
    #    "well_ROI_table",
    #    "regionprops_DAPI",
    # ]

    _ = ome_zarr.get_label("nuclei", path="0")
    _ = ome_zarr.get_table("well_ROI_table")


@pytest.mark.parametrize("array_mode", ["numpy", "dask"])
def test_get_and_squeeze(tmp_path: Path, array_mode: str):
    # Very basic test to check if the container is working
    # to be expanded with more meaningful tests
    store = tmp_path / "ome_zarr.zarr"
    ome_zarr = create_empty_ome_zarr(
        store,
        shape=(1, 20, 30),
        xy_pixelsize=0.5,
        levels=1,
        axes_names=["c", "y", "x"],
        dtype="uint8",
    )
    image = ome_zarr.get_image()
    assert image.shape == (1, 20, 30)
    assert image.get_array(axes_order=["c", "y", "x"]).shape == (1, 20, 30)
    image.set_array(
        np.ones((1, 20, 30), dtype="uint8"),
        axes_order=["c", "y", "x"],
    )
    assert image.get_array(axes_order=["y", "x"]).shape == (20, 30)
    image.set_array(
        np.ones((20, 30), dtype="uint8"),
        axes_order=["y", "x"],
    )
    assert image.get_array(axes_order=["y", "x"]).shape == (20, 30)
    image.set_array(
        np.ones((20, 30), dtype="uint8"),
        axes_order=["y", "x"],
    )
    assert image.get_array(axes_order=["x"], y=0, c=0).shape == (30,)
    image.set_array(
        np.ones((30,), dtype="uint8"),
        axes_order=["x"],
        y=0,
        c=0,
    )

    assert image.get_array(c=0, axes_order=["c", "y", "x"]).shape == (1, 20, 30)
    assert image.get_array(c=0, axes_order=["y", "x"]).shape == (20, 30)
    assert image.get_array(c=(0,), axes_order=None).shape == (1, 20, 30)
    assert image.get_array(c=0, axes_order=None).shape == (20, 30)

    # Reordering axes and adding a virtual axis
    assert image.get_array(c=0, axes_order=["c", "x", "y", "virtual"]).shape == (
        1,
        30,
        20,
        1,
    )

    # Test channel_labels
    image.get_as_numpy(channel_label="channel_0")
    image.get_as_dask(channel_label="channel_0")

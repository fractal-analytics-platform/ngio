from pathlib import Path

import pytest

from ngio import OmeZarrWell, create_empty_plate, open_ome_zarr_plate
from ngio.utils import NgioValueError


def test_open_real_ome_zarr_plate(cardiomyocyte_tiny_path: Path):
    cardiomyocyte_tiny_path = cardiomyocyte_tiny_path
    ome_zarr_plate = open_ome_zarr_plate(cardiomyocyte_tiny_path)

    assert isinstance(ome_zarr_plate.__repr__(), str)
    assert ome_zarr_plate.columns == ["03"]
    assert ome_zarr_plate.rows == ["B"]
    assert ome_zarr_plate.acquisitions_ids == [0]
    assert ome_zarr_plate.acquisitions_names == [
        "20200812-CardiomyocyteDifferentiation14-Cycle1"
    ]

    well_path = ome_zarr_plate._well_path("B", "03")
    well_path2 = ome_zarr_plate._well_path("B", 3)
    assert well_path == well_path2
    well = ome_zarr_plate.get_well("B", "03")
    assert well.paths() == ["0"]

    image_path = ome_zarr_plate._image_path("B", "03", "0")
    assert image_path == "B/03/0"

    images_plate = ome_zarr_plate.get_images()
    images_well = ome_zarr_plate.get_well_images("B", "03")
    assert len(images_plate) == 1
    assert len(images_well) == 1


def test_create_and_edit_plate(tmp_path: Path):
    test_plate = create_empty_plate(tmp_path / "test_plate.zarr", name="test_plate")
    assert test_plate.columns == []
    assert test_plate.rows == []
    assert test_plate.acquisitions_ids == []

    test_plate.add_image(row="B", column="03", image_path="0", acquisition_id=0)
    test_plate.add_image(row="B", column="03", image_path="1", acquisition_id=0)

    with pytest.raises(NgioValueError):
        test_plate.add_image(row="B", column="03", image_path="1", acquisition_id=1)

    test_plate.atomic_add_image(row="C", column="02", image_path="1", acquisition_id=1)

    assert test_plate.columns == ["02", "03"]
    assert test_plate.rows == ["B", "C"]
    assert test_plate.acquisitions_ids == [0, 1]

    assert len(test_plate.wells_paths()) == 2

    test_plate.remove_image(row="C", column="02", image_path="1")
    assert len(test_plate.wells_paths()) == 1


def test_create_and_edit_plate_path_normalization(tmp_path: Path):
    test_plate = create_empty_plate(tmp_path / "test_plate.zarr", name="test_plate")
    test_plate.add_image(row="B", column="03", image_path="0_mip", acquisition_id=0)
    test_plate.add_image(
        row="B", column="03", image_path="1_illumination_correction", acquisition_id=0
    )
    assert test_plate.images_paths() == ["B/03/0mip", "B/03/1illuminationcorrection"]


def test_derive_plate_from_ome_zarr(cardiomyocyte_tiny_path: Path, tmp_path: Path):
    ome_zarr_plate = open_ome_zarr_plate(cardiomyocyte_tiny_path)
    test_plate = ome_zarr_plate.derive_plate(tmp_path / "test_plate.zarr")
    assert test_plate.columns == ["03"]
    assert test_plate.rows == ["B"]
    assert test_plate.acquisitions_ids == [0]


def test_add_well(tmp_path: Path):
    test_plate = create_empty_plate(tmp_path / "test_plate.zarr", name="test_plate")
    well = test_plate.add_well(row="B", column="03")
    assert isinstance(well, OmeZarrWell)
    assert test_plate.columns == ["03"]
    assert test_plate.rows == ["B"]
    assert test_plate.acquisitions_ids == []
    assert test_plate.wells_paths() == ["B/03"]

    test_plate.add_column("04")
    test_plate.add_row("C")
    assert test_plate.columns == ["03", "04"]
    assert test_plate.rows == ["B", "C"]
    # No well added in this step
    assert test_plate.wells_paths() == ["B/03"]


def test_add_well_with_acquisition(tmp_path: Path):
    test_plate = create_empty_plate(tmp_path / "test_plate.zarr", name="test_plate")
    test_plate.add_acquisition(acquisition_id=0, acquisition_name="test_acquisition")
    test_plate.add_acquisition(acquisition_id=1, acquisition_name="test_acquisition1")
    assert test_plate.acquisitions_ids == [0, 1]
    assert test_plate.acquisitions_names == ["test_acquisition", "test_acquisition1"]

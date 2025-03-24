from pathlib import Path

import pytest

from ngio import create_empty_plate, open_omezarr_plate
from ngio.utils import NgioValueError


def test_open_real_omezarr_plate(cardiomyocyte_tiny_path: Path):
    cardiomyocyte_tiny_path = cardiomyocyte_tiny_path
    omezarr_plate = open_omezarr_plate(cardiomyocyte_tiny_path)

    assert isinstance(omezarr_plate.__repr__(), str)
    assert omezarr_plate.columns == ["03"]
    assert omezarr_plate.rows == ["B"]
    assert omezarr_plate.acquisitions_ids == [0]
    assert omezarr_plate.acquisitions_names == [
        "20200812-CardiomyocyteDifferentiation14-Cycle1"
    ]

    well_path = omezarr_plate.get_well_path("B", "03")
    well_path2 = omezarr_plate.get_well_path("B", 3)
    assert well_path == well_path2
    well = omezarr_plate.get_well("B", "03")
    assert well.paths() == ["0"]

    image_path = omezarr_plate.get_image_path("B", "03", "0")
    assert image_path == "B/03/0"

    images_plate = omezarr_plate.get_images()
    images_well = omezarr_plate.get_well_images("B", "03")
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

    test_plate.add_image(row="C", column="02", image_path="1", acquisition_id=1)

    assert test_plate.columns == ["02", "03"]
    assert test_plate.rows == ["B", "C"]
    assert test_plate.acquisitions_ids == [0, 1]

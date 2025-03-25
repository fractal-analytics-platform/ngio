from pathlib import Path

import pytest

from ngio import Image, open_image


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
def test_open_image(images_v04: dict[str, Path], zarr_name: str):
    path = images_v04[zarr_name]
    image = open_image(path)
    assert isinstance(image, Image)

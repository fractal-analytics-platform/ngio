import os
import shutil
from pathlib import Path

import pytest

from ngio.utils import download_ome_zarr_dataset

zenodo_download_dir = Path("tests/data/v04/zenodo_images/")
os.makedirs(zenodo_download_dir, exist_ok=True)
cardiomyocyte_tiny_source_path = download_ome_zarr_dataset(
    "CardiomyocyteTiny", download_dir=zenodo_download_dir
)


@pytest.fixture
def cardiomyocyte_tiny_path(tmp_path: Path) -> Path:
    dest_path = tmp_path / cardiomyocyte_tiny_source_path.stem
    shutil.copytree(cardiomyocyte_tiny_source_path, dest_path, dirs_exist_ok=True)
    return dest_path


@pytest.fixture
def images_v04(tmp_path: Path) -> dict[str, Path]:
    source = Path("tests/data/v04/images/")
    dest = tmp_path / "v04" / "images"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, dirs_exist_ok=True)
    return {file.name: file for file in dest.glob("*.zarr")}

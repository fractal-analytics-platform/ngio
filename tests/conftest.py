from pathlib import Path

import pytest

from ngio.common import download_ome_zarr_dataset


@pytest.fixture(scope="function")
def ome_zarr_image_v04_path(tmp_path: Path) -> Path:
    tmp_path = tmp_path / "test_datasets_v04"
    return Path(download_ome_zarr_dataset("CardiomyocyteTiny", download_dir=tmp_path))

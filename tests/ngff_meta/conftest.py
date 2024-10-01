import json
from importlib.metadata import version
from pathlib import Path

import zarr
from packaging.version import Version
from pytest import fixture

zarr_version = version("zarr")
ZARR_PYTHON_V = 2 if Version(zarr_version) < Version("3.0.0a") else 3


@fixture
def ome_zarr_image_v04_path(tmpdir):
    zarr_path = Path(tmpdir) / "test_ome_ngff_v04.zarr"

    if ZARR_PYTHON_V == 3:
        group = zarr.open_group(store=zarr_path, mode="w", zarr_format=2)
    else:
        group = zarr.open_group(store=zarr_path, mode="w", zarr_version=2)

    with open("tests/data/meta_v04/base_ome_zarr_image_meta.json") as f:
        base_ome_zarr_meta = json.load(f)

    base_ome_zarr_meta = base_ome_zarr_meta
    group.attrs.update(base_ome_zarr_meta)
    return zarr_path


@fixture
def ome_zarr_label_v04_path(tmpdir):
    zarr_path = Path(tmpdir) / "test_ome_ngff_image_v04.zarr"

    if ZARR_PYTHON_V == 3:
        group = zarr.open_group(store=zarr_path, mode="w", zarr_format=2)
    else:
        group = zarr.open_group(store=zarr_path, mode="w", zarr_version=2)

    with open("tests/data/meta_v04/base_ome_zarr_label_meta.json") as f:
        base_ome_zarr_meta = json.load(f)

    base_ome_zarr_meta = base_ome_zarr_meta
    group.attrs.update(base_ome_zarr_meta)
    return zarr_path

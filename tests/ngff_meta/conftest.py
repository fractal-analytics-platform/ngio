import json
from pathlib import Path

import zarr
import zarr.store
from pytest import fixture


@fixture
def ome_zarr_image_v04_path(tmpdir):
    zarr_path = Path(tmpdir) / "test_ome_ngff_v04.zarr"

    group = zarr.open_group(store=zarr_path, mode="w", zarr_format=2)

    with open("tests/data/meta_v04/base_ome_zarr_image_meta.json") as f:
        base_ome_zarr_meta = json.load(f)

    base_ome_zarr_meta = base_ome_zarr_meta
    group.attrs.update(base_ome_zarr_meta)
    return zarr_path


@fixture
def ome_zarr_label_v04_path(tmpdir):
    zarr_path = Path(tmpdir) / "test_ome_ngff_image_v04.zarr"

    group = zarr.open_group(store=zarr_path, mode="w", zarr_format=2)

    with open("tests/data/meta_v04/base_ome_zarr_label_meta.json") as f:
        base_ome_zarr_meta = json.load(f)

    base_ome_zarr_meta = base_ome_zarr_meta
    group.attrs.update(base_ome_zarr_meta)
    return zarr_path

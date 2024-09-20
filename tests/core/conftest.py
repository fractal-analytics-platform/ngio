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

    # shape = (3, 10, 256, 256)
    for i, path in enumerate(["0", "1", "2", "3"]):
        shape = (3, 10, 256 // (2**i), 256 // (2**i))
        group.create_array(name=path, fill_value=0, shape=shape)

    return zarr_path

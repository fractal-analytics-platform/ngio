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

    json_path = (
        Path(".") / "tests" / "data" / "meta_v04" / "base_ome_zarr_image_meta.json"
    )
    with open(json_path) as f:
        base_ome_zarr_meta = json.load(f)

    base_ome_zarr_meta = base_ome_zarr_meta
    group.attrs.update(base_ome_zarr_meta)

    # shape = (3, 10, 256, 256)
    for i, path in enumerate(["0", "1", "2", "3", "4"]):
        shape = (3, 10, 256 // (2**i), 256 // (2**i))
        if ZARR_PYTHON_V == 3:
            group.create_array(name=path, fill_value=0, shape=shape)
        else:
            group.zeros(name=path, shape=shape)

    return zarr_path

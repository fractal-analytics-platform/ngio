from importlib.metadata import version
from pathlib import Path

import zarr
from packaging.version import Version
from pytest import fixture

zarr_version = version("zarr")
ZARR_PYTHON_V = 2 if Version(zarr_version) < Version("3.0.0a") else 3


def _create_zarr(tempdir, zarr_format=2):
    zarr_path = Path(tempdir) / f"test_group_v{2}.zarr"

    if ZARR_PYTHON_V == 3:
        group = zarr.open_group(store=zarr_path, mode="w", zarr_format=zarr_format)
    else:
        group = zarr.open_group(store=zarr_path, mode="w", zarr_version=zarr_format)

    for i in range(3):
        if ZARR_PYTHON_V == 3:
            group.create_array(f"array_{i}", shape=(10, 10), dtype="i4")
        else:
            group.empty(f"array_{i}", shape=(10, 10), dtype="i4")

    for i in range(3):
        group.create_group(f"group_{i}")

    return zarr_path


@fixture
def local_zarr_path_v2(tmpdir) -> tuple[Path, int]:
    zarr_path = _create_zarr(tmpdir, zarr_format=2)
    return zarr_path, 2


@fixture
def local_zarr_path_v3(tmpdir) -> tuple[Path, int]:
    zarr_path = _create_zarr(tmpdir, zarr_format=3)
    return zarr_path, 3


@fixture
def local_zarr_str_v2(tmpdir) -> tuple[str, int]:
    zarr_path = _create_zarr(tmpdir, zarr_format=2)
    return str(zarr_path.absolute()), 2


@fixture
def local_zarr_str_v3(tmpdir) -> tuple[Path, int]:
    zarr_path = _create_zarr(tmpdir, zarr_format=3)
    return str(zarr_path.absolute()), 3


@fixture(
    params=[
        "local_zarr_path_v2",
        # "local_zarr_path_v3",
        "local_zarr_str_v2",
        # "local_zarr_str_v3",
        # "local_zarr_store_v2",
    ]
)
def store_fixture(request):
    return request.getfixturevalue(request.param)

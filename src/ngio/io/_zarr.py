from importlib.metadata import version
from pathlib import Path
from typing import Literal

import zarr
from packaging.version import Version

from ngio.utils import NgioFileExistsError, NgioFileNotFoundError

zarr_version = version("zarr")
ZARR_PYTHON_V = 2 if Version(zarr_version) < Version("3.0.0a") else 3

if ZARR_PYTHON_V == 2:
    from zarr.errors import ContainsGroupError, GroupNotFoundError

# Zarr v3 Imports
# import zarr.store
# from zarr.core.common import AccessModeLiteral, ZarrFormat
# from zarr.store.common import StoreLike

AccessModeLiteral = Literal["r", "r+", "w", "w-", "a"]
ZarrFormat = Literal[2, 3]
StoreLike = str | Path  # This type alias more narrrow than necessary
StoreOrGroup = StoreLike | zarr.Group


class ZarrV3Error(Exception):
    pass


def _pass_through_group(
    group: zarr.Group, mode: AccessModeLiteral, zarr_format: ZarrFormat = 2
) -> zarr.Group:
    if ZARR_PYTHON_V == 2:
        if zarr_format == 3:
            raise ZarrV3Error("Zarr v3 is not supported in when using zarr-python v2.")
        else:
            return group

    else:
        if group.metadata.zarr_format != zarr_format:
            raise ValueError(
                f"Zarr format mismatch. Expected {zarr_format}, "
                "got {store.metadata.zarr_format}."
            )
        else:
            return group

    raise ValueError("This should never be reached.")


def _open_group_v2_v3(
    store: StoreOrGroup, mode: AccessModeLiteral, zarr_format: ZarrFormat = 2
) -> zarr.Group:
    """Wrapper around zarr.open_group with some additional checks.

    Args:
        store (StoreOrGroup): The store (can also be a Path/str) or group to open.
        mode (ReadOrEdirLiteral): The mode to open the group in.
        zarr_format (ZarrFormat): The Zarr format to use.

    Returns:
        zarr.Group: The opened Zarr group.
    """
    if ZARR_PYTHON_V == 3:
        return zarr.open_group(store=store, mode=mode, zarr_format=zarr_format)
    else:
        try:
            group = zarr.open_group(store=store, mode=mode)

        except ContainsGroupError as e:
            raise NgioFileExistsError(
                f"A Zarr group already exists at {store}, "
                "consider setting overwrite=True."
            ) from e

        except GroupNotFoundError as e:
            raise NgioFileNotFoundError(f"No Zarr group found at {store}") from e

        return group


def _is_group_readonly(group: zarr.Group) -> bool:
    if ZARR_PYTHON_V == 3:
        return group.store_path.store.mode.readonly

    else:
        return not group.store.is_writeable()

from pathlib import Path

import zarr

from ngio.io._zarr import (
    AccessModeLiteral,
    StoreLike,
    StoreOrGroup,
    ZarrFormat,
    _open_group_v2_v3,
    _pass_through_group,
)
from ngio.utils import ngio_logger

# Zarr v3 Imports
# import zarr.store
# from zarr.core.common import AccessModeLiteral, ZarrFormat
# from zarr.store.common import StoreLike


def _check_store(store: StoreLike) -> StoreLike:
    if isinstance(store, str) or isinstance(store, Path):
        return store

    raise NotImplementedError(
        "RemoteStore is not yet supported. Please use LocalStore."
    )


def open_group_wrapper(
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
    if isinstance(store, zarr.Group):
        _group = _pass_through_group(store, mode=mode, zarr_format=zarr_format)
        ngio_logger.debug(
            f"Passing through group: {_group}, "
            f"located in store: {_group.store.path}"
        )
        return _group

    store = _check_store(store)
    _group = _open_group_v2_v3(store=store, mode=mode, zarr_format=zarr_format)
    ngio_logger.debug(f"Opened located in store: {store}")
    return _group

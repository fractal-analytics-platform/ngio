from pathlib import Path
from typing import Literal

import fsspec
import zarr
from zarr.errors import ContainsGroupError, GroupNotFoundError

from ngio.utils import NgioFileExistsError, NgioFileNotFoundError

AccessModeLiteral = Literal["r", "r+", "w", "w-", "a"]
StoreLike = str | Path  # This type alias more narrrow than necessary
StoreOrGroup = StoreLike | zarr.Group


def _check_store(store: StoreLike) -> StoreLike:
    if isinstance(store, str) or isinstance(store, Path):
        return store

    if isinstance(store, fsspec.mapping.FSMap) or isinstance(
        store, zarr.storage.FSStore
    ):
        return store

    raise NotImplementedError(
        f"Store type {type(store)} is not supported. supported types are: "
        "str, Path, fsspec.mapping.FSMap, zarr.storage.FSStore"
    )


def open_group_wrapper(store: StoreOrGroup, mode: AccessModeLiteral) -> zarr.Group:
    """Wrapper around zarr.open_group with some additional checks.

    Args:
        store (StoreOrGroup): The store (can also be a Path/str) or group to open.
        mode (ReadOrEdirLiteral): The mode to open the group in.

    Returns:
        zarr.Group: The opened Zarr group.
    """
    if isinstance(store, zarr.Group):
        return store

    try:
        store = _check_store(store)
        group = zarr.open_group(store=store, mode=mode)

    except ContainsGroupError as e:
        raise NgioFileExistsError(
            f"A Zarr group already exists at {store}, consider setting overwrite=True."
        ) from e

    except GroupNotFoundError as e:
        raise NgioFileNotFoundError(f"No Zarr group found at {store}") from e

    return group

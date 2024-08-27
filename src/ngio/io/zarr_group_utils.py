"""Collection of helper functions to work with Zarr groups."""

from pathlib import Path

import zarr
from zarr.core.common import AccessModeLiteral, StoreLike, ZarrFormat


def _open_group(
    store: StoreLike, mode: AccessModeLiteral, zarr_version: ZarrFormat = 2
) -> zarr.hierarchy.Group:
    """Wrapper around zarr.open_group with some additional checks."""
    if isinstance(store, str):
        store = Path(store)

    if isinstance(store, Path):
        if not store.exists():
            raise FileNotFoundError(f"Path {store} does not exist. Cannot open group.")

    elif isinstance(store, zarr.store.LocalStore):
        if not store.root.exists():
            raise FileNotFoundError(
                f"Path {store.root} does not exist. Cannot open group."
            )
    elif isinstance(store, zarr.store.RemoteStore):
        raise NotImplementedError(
            "RemoteStore is not yet supported. Please use LocalStore."
        )

    return zarr.open_group(store=store, mode=mode, zarr_version=zarr_version)


def read_group_attrs(store: StoreLike, zarr_version: ZarrFormat = 2) -> dict:
    """Simple helper function to read the attributes of a Zarr group."""
    group = _open_group(store=store, mode="r", zarr_version=zarr_version)
    return group.attrs.asdict()


def update_group_attrs(store: StoreLike, attrs: dict, zarr_version: ZarrFormat) -> None:
    """Simple helper function to update the attributes of a Zarr group."""
    group = _open_group(store=store, mode="a", zarr_version=zarr_version)
    group.attrs.update(attrs)


def overwrite_group_attrs(
    store: StoreLike, attrs: dict, zarr_version: ZarrFormat
) -> None:
    """Simple helper function to overwrite the attributes of a Zarr group."""
    group = _open_group(store=store, mode="a", zarr_version=zarr_version)
    group.attrs.clear()
    group.attrs.update(attrs)


def list_group_arrays(store: StoreLike, zarr_version: ZarrFormat = 2) -> list:
    """Simple helper function to list the arrays in a Zarr group."""
    group = _open_group(store, mode="r", zarr_version=zarr_version)
    return group.list_arrays()


def list_group_groups(store: StoreLike, zarr_version: ZarrFormat = 2) -> list:
    """Simple helper function to list the groups in a Zarr group."""
    group = _open_group(store, mode="r", zarr_version=zarr_version)
    return group.list_groups()

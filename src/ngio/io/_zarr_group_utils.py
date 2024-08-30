from pathlib import Path
from typing import Literal

import zarr
import zarr.store
from zarr.core.common import ZarrFormat
from zarr.store.common import StoreLike

ReadOrEdirLiteral = Literal["r", "r+"]
StoreOrGroup = StoreLike | zarr.Group


def _check_group_editable(group: zarr.Group, mode: ReadOrEdirLiteral) -> None:
    """Check if the group can be opened in the given mode."""
    if mode == "r+" and group.store.mode.readonly:
        raise PermissionError("Store is in read-only mode. Cannot open be edited.")


def open_group(
    store: StoreOrGroup, mode: ReadOrEdirLiteral, zarr_format: ZarrFormat = 2
) -> zarr.Group:
    """Wrapper around zarr.open_group with some additional checks.

    This wrapper is used to limit the modes that can be used to open a group.
    It also allows for "r+" Store to be opened in "r" mode to avoid failing in
    the zarr.open_group.

    Args:
        store (StoreOrGroup): The store (can also be a Path/str) or group to open.
        mode (ReadOrEdirLiteral): The mode to open the group in.
        zarr_format (ZarrFormat): The Zarr format to use.

    Returns:
        zarr.Group: The opened Zarr group.
    """
    if isinstance(store, zarr.Group):
        if mode == "r+":
            _check_group_editable(group=store, mode=mode)

        if store.metadata.zarr_format != zarr_format:
            raise ValueError(
                f"Zarr format mismatch. Expected {zarr_format}, "
                "got {store.metadata.zarr_format}."
            )
        return store

    if mode not in ["r", "r+"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'r' or 'r+'.")

    if zarr_format not in [2, 3]:
        raise ValueError(f"Invalid zarr_format: {zarr_format}. Must be 2 or 3.")

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

        if store.mode.readonly and mode == "r+":
            raise PermissionError("Store is in read-only mode. Cannot open be edited.")

        elif store.mode.update and mode == "r":
            # Reopen the store in read only mode to avoid failing in the zarr.open_group
            store = zarr.store.LocalStore(root=store.root, mode="r")

    elif isinstance(store, zarr.store.RemoteStore):
        raise NotImplementedError(
            "RemoteStore is not yet supported. Please use LocalStore."
        )

    return zarr.open_group(store=store, mode=mode, zarr_format=zarr_format)


def read_group_attrs(store: StoreOrGroup, zarr_format: ZarrFormat = 2) -> dict:
    """Simple helper function to read the attributes of a Zarr group."""
    group = open_group(store=store, mode="r", zarr_format=zarr_format)
    return dict(group.attrs)


def update_group_attrs(
    store: StoreOrGroup, attrs: dict, zarr_format: ZarrFormat
) -> None:
    """Simple helper function to update the attributes of a Zarr group."""
    group = open_group(store=store, mode="r+", zarr_format=zarr_format)
    group.attrs.update(attrs)


def overwrite_group_attrs(
    store: StoreOrGroup, attrs: dict, zarr_format: ZarrFormat
) -> None:
    """Simple helper function to overwrite the attributes of a Zarr group."""
    group = open_group(store=store, mode="r+", zarr_format=zarr_format)
    group.attrs.clear()
    group.attrs.update(attrs)


def list_group_arrays(store: StoreOrGroup, zarr_format: ZarrFormat = 2) -> list:
    """Simple helper function to list the arrays in a Zarr group."""
    group = open_group(store, mode="r", zarr_format=zarr_format)
    return list(group.array_keys())


def list_group_groups(store: StoreOrGroup, zarr_format: ZarrFormat = 2) -> list:
    """Simple helper function to list the groups in a Zarr group."""
    group = open_group(store, mode="r", zarr_format=zarr_format)
    return list(group.group_keys())


def create_new_group(
    store: StoreOrGroup, new_group: str, zarr_format: ZarrFormat = 2
) -> zarr.Group:
    """Simple helper function to create a new Zarr group."""
    group = open_group(store, mode="r+", zarr_format=zarr_format)
    list_existing_groups = list_group_groups(store=group, zarr_format=zarr_format)
    if new_group in list_existing_groups:
        raise ValueError(f"Group {new_group} already exists in the store.")
    return group.create_group(new_group)

from pathlib import Path
from typing import Literal

import zarr
import zarr.store
from zarr.core.common import ZarrFormat
from zarr.store.common import StoreLike

ReadOrEdirLiteral = Literal["r", "r+"]


def _open_group(
    store: StoreLike, mode: ReadOrEdirLiteral, zarr_format: ZarrFormat = 2
) -> zarr.Group:
    """Wrapper around zarr.open_group with some additional checks."""
    assert mode in ["r", "r+"], f"Invalid mode: {mode}. Must be 'r' or 'r+'."
    assert zarr_format in [2, 3], f"Invalid zarr_format: {zarr_format}. Must be 2 or 3."

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
            raise PermissionError(
                "Store is opened in read-only mode. Cannot open be edited."
            )

        elif store.mode.update and mode == "r":
            # Reopen the store in read only mode to avoid failing in the zarr.open_group
            store = zarr.store.LocalStore(str(store.root), mode="r")

    elif isinstance(store, zarr.store.RemoteStore):
        raise NotImplementedError(
            "RemoteStore is not yet supported. Please use LocalStore."
        )

    return zarr.open_group(store=store, mode=mode, zarr_format=zarr_format)


def read_group_attrs(store: StoreLike, zarr_format: ZarrFormat = 2) -> dict:
    """Simple helper function to read the attributes of a Zarr group."""
    group = _open_group(store=store, mode="r", zarr_format=zarr_format)
    return dict(group.attrs)


def update_group_attrs(store: StoreLike, attrs: dict, zarr_format: ZarrFormat) -> None:
    """Simple helper function to update the attributes of a Zarr group."""
    group = _open_group(store=store, mode="r+", zarr_format=zarr_format)
    group.attrs.update(attrs)


def overwrite_group_attrs(
    store: StoreLike, attrs: dict, zarr_format: ZarrFormat
) -> None:
    """Simple helper function to overwrite the attributes of a Zarr group."""
    group = _open_group(store=store, mode="r+", zarr_format=zarr_format)
    group.attrs.clear()
    group.attrs.update(attrs)


def list_group_arrays(store: StoreLike, zarr_format: ZarrFormat = 2) -> list:
    """Simple helper function to list the arrays in a Zarr group."""
    group = _open_group(store, mode="r", zarr_format=zarr_format)
    return list(group.array_keys())


def list_group_groups(store: StoreLike, zarr_format: ZarrFormat = 2) -> list:
    """Simple helper function to list the groups in a Zarr group."""
    group = _open_group(store, mode="r", zarr_format=zarr_format)
    return list(group.group_keys())

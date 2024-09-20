import zarr
import zarr.store
from zarr.core.common import AccessModeLiteral, ZarrFormat
from zarr.store.common import StoreLike

StoreOrGroup = StoreLike | zarr.Group


def _check_store(store: StoreLike) -> StoreLike:
    if isinstance(store, zarr.store.RemoteStore):
        raise NotImplementedError(
            "RemoteStore is not yet supported. Please use LocalStore."
        )
    return store


def _pass_through_group(
    group: zarr.Group, mode: AccessModeLiteral, zarr_format: ZarrFormat = 2
) -> zarr.Group:
    if group.metadata.zarr_format != zarr_format:
        raise ValueError(
            f"Zarr format mismatch. Expected {zarr_format}, "
            "got {store.metadata.zarr_format}."
        )
    return group


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
        return _pass_through_group(store, mode=mode, zarr_format=zarr_format)

    store = _check_store(store)
    return zarr.open_group(store=store, mode=mode, zarr_format=zarr_format)

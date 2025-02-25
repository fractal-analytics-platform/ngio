"""Common utilities for working with Zarr groups in consistent ways."""

from pathlib import Path
from typing import Literal

import fsspec
import zarr
from filelock import BaseFileLock, FileLock
from zarr.errors import ContainsGroupError, GroupNotFoundError
from zarr.storage import DirectoryStore, FSStore, Store

from ngio.utils import NgioFileExistsError, NgioFileNotFoundError, NgioValueError

AccessModeLiteral = Literal["r", "r+", "w", "w-", "a"]
# StoreLike is more restrictive than it could be
# but to make sure we can handle the store correctly
# we need to be more restrictive
NgioSupportedStore = str | Path | fsspec.mapping.FSMap | FSStore | DirectoryStore
GenericStore = Store | NgioSupportedStore
StoreOrGroup = GenericStore | zarr.Group


def _check_store(store) -> NgioSupportedStore:
    """Check the store and return a valid store."""
    if isinstance(store, NgioSupportedStore):
        return store

    raise NotImplementedError(
        f"Store type {type(store)} is not supported. "
        f"Supported types are: {NgioSupportedStore}"
    )


def _check_group(group: zarr.Group, mode: AccessModeLiteral) -> zarr.Group:
    """Check the group and return a valid group."""
    is_read_only = getattr(group, "_read_only", False)
    if is_read_only and mode in ["w", "w-"]:
        raise NgioValueError(
            "The group is read only. Cannot open in write mode ['w', 'w-']"
        )

    if mode == "r" and not is_read_only:
        # let's make sure we don't accidentally write to the group
        group = zarr.open_group(store=group.store, path=group.path, mode="r")

    return group


def open_group_wrapper(
    store: StoreOrGroup, mode: AccessModeLiteral
) -> tuple[zarr.Group, NgioSupportedStore]:
    """Wrapper around zarr.open_group with some additional checks.

    Args:
        store (StoreOrGroup): The store or group to open.
        mode (ReadOrEdirLiteral): The mode to open the group in.

    Returns:
        zarr.Group: The opened Zarr group.
    """
    if isinstance(store, zarr.Group):
        group = _check_group(store, mode)
        if hasattr(group, "store_path"):
            _store = group.store_path
        if isinstance(group.store, DirectoryStore):
            _store = group.store.path
        else:
            _store = group.store

        _store = _check_store(_store)
        return group, _store

    try:
        store = _check_store(store)
        group = zarr.open_group(store=store, mode=mode)

    except ContainsGroupError as e:
        raise NgioFileExistsError(
            f"A Zarr group already exists at {store}, consider setting overwrite=True."
        ) from e

    except GroupNotFoundError as e:
        raise NgioFileNotFoundError(f"No Zarr group found at {store}") from e

    return group, store


class ZarrGroupHandler:
    """A simple wrapper around a Zarr group to handle metadata."""

    def __init__(
        self,
        store: StoreOrGroup,
        cache: bool = False,
        mode: AccessModeLiteral = "a",
        parallel_safe: bool = False,
    ):
        """Initialize the handler.

        Args:
            store (StoreOrGroup): The Zarr store or group containing the image data.
            meta_mode (str): The mode of the metadata handler.
            cache (bool): Whether to cache the metadata.
            mode (str): The mode of the store.
            parallel_safe (bool): If True, the handler will create a lock file to make
                that can be used to make the handler parallel safe.
                Be aware that the lock needs to be used manually.
        """
        _group, _store = open_group_wrapper(store, mode)

        # Make sure the cache is set in the attrs
        # in the same way as the cache in the handler
        _group.attrs.cache = cache

        if parallel_safe:
            if not isinstance(_store, str | Path):
                raise NgioValueError(
                    "The store needs to be a path to use the lock mechanism."
                )
            self._lock_path = f"{_store}.lock"
            self._lock = FileLock(self._lock_path)

        else:
            self._lock_path = None
            self._lock = None

        self._group = _group
        self._mode = mode
        self._store = _store
        self.use_cache = cache
        self._cache = {}

    @property
    def store(self) -> NgioSupportedStore:
        """Return the store of the group."""
        return self._store

    @property
    def group_path(self) -> str:
        """Return the store path."""
        return f"{self._store}/{self._group.path}"

    @property
    def mode(self) -> AccessModeLiteral:
        """Return the mode of the group."""
        return self._mode  # type: ignore

    @property
    def lock(self) -> BaseFileLock | None:
        """Return the lock."""
        return self._lock

    def remove_lock(self) -> None:
        """Return the lock."""
        if self._lock is None or self._lock_path is None:
            return None

        lock_path = Path(self._lock_path)
        if lock_path.exists() and self._lock.lock_counter == 0:
            lock_path.unlink()
            self._lock = None
            self._lock_path = None
            return None

        raise NgioValueError("The lock is still in use. Cannot remove it.")

    @property
    def group(self) -> zarr.Group:
        """Return the group."""
        return self._group

    def add_to_cache(self, key: str, value: object) -> None:
        """Add an object to the cache."""
        if not self.use_cache:
            return None
        self._cache[key] = value

    def get_from_cache(self, key: str) -> object | None:
        """Get an object from the cache."""
        if not self.use_cache:
            return None
        return self._cache.get(key, None)

    def clean_cache(self) -> None:
        """Clear the cached metadata."""
        self._cache = {}

    def load_attrs(self) -> dict:
        """Load the attributes of the group."""
        attrs = self.get_from_cache("attrs")
        if attrs is not None and isinstance(attrs, dict):
            return attrs

        attrs = dict(self.group.attrs)

        self.add_to_cache("attrs", attrs)
        return attrs

    def _write_attrs(self, attrs: dict, overwrite: bool = False) -> None:
        """Write the metadata to the store."""
        is_read_only = getattr(self._group, "_read_only", False)
        if is_read_only:
            raise NgioValueError("The group is read only. Cannot write metadata.")

        # we need to invalidate the current attrs cache
        self.add_to_cache("attrs", None)
        if overwrite:
            self.group.attrs.clear()

        self.group.attrs.update(attrs)

    def write_attrs(self, attrs: dict, overwrite: bool = False) -> None:
        """Write the metadata to the store."""
        # Maybe we should use the lock here
        self._write_attrs(attrs, overwrite)

    def _group_get(self, path: str):
        """Get a group from the group."""
        group_or_array = self.get_from_cache(path)
        if group_or_array is not None:
            return group_or_array

        group_or_array = self.group.get(path, None)
        self.add_to_cache(path, group_or_array)
        return group_or_array

    def get_array(self, path: str) -> zarr.Array:
        """Get an array from the group."""
        array = self._group_get(path)
        if array is None:
            raise NgioFileNotFoundError(f"No array found at {path}")
        if not isinstance(array, zarr.Array):
            raise NgioValueError(
                f"The object at {path} is not an array, but a {type(array)}"
            )
        return array

    def get_group(self, path: str) -> zarr.Group:
        """Get a group from the group."""
        group = self._group_get(path)
        if group is None:
            raise NgioFileNotFoundError(f"No group found at {path}")
        if not isinstance(group, zarr.Group):
            raise NgioValueError(
                f"The object at {path} is not a group, but a {type(group)}"
            )
        return group

    def create_group(self, path: str, overwrite: bool = False) -> zarr.Group:
        """Create a group in the group."""
        try:
            group = self.group.create_group(path, overwrite=overwrite)
        except ContainsGroupError as e:
            raise NgioFileExistsError(
                f"A Zarr group already exists at {path}, "
                "consider setting overwrite=True."
            ) from e
        self.add_to_cache(path, group)
        return group

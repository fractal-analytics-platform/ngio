from __future__ import annotations

from typing import TYPE_CHECKING, Any

import zarr
from anndata import AnnData
from anndata._io.specs import read_elem
from anndata._io.utils import _read_legacy_raw
from anndata._io.zarr import read_dataframe
from anndata.compat import _clean_uns
from anndata.experimental import read_dispatched

from ngio.io import open_group_wrapper

if TYPE_CHECKING:
    from collections.abc import Callable

    from ngio.io import StoreOrGroup


def custom_read_zarr(store: StoreOrGroup) -> AnnData:
    """Read from a hierarchical Zarr array store.

    # Implementation originally from https://github.com/scverse/anndata/blob/main/src/anndata/_io/zarr.py
    # Original implementation would not work with remote storages so we had to copy it
    # here and slightly modified it to work with remote storages.

    Args:
        store (StoreOrGroup): A store or group to read the AnnData from.
    """
    group = open_group_wrapper(store=store, mode="r")

    # Read with handling for backwards compat
    def callback(func: Callable, elem_name: str, elem: Any, iospec: Any) -> Any:
        if iospec.encoding_type == "anndata" or elem_name.endswith("/"):
            ad_kwargs = {}
            base_elem = [
                "X",
                "layers",
                "obs",
                "obsm",
                "obsp",
                "uns",
                "var",
                "varm",
                "varp",
            ]
            # This should make sure that the function behaves the same as the original
            # implementation.
            base_elem += list(elem.keys())
            for k in set(base_elem):
                v = elem.get(k)
                if v is not None and not k.startswith("raw."):
                    ad_kwargs[k] = read_dispatched(v, callback)
            return AnnData(**ad_kwargs)

        elif elem_name.startswith("/raw."):
            return None
        elif elem_name in {"/obs", "/var"}:
            return read_dataframe(elem)
        elif elem_name == "/raw":
            # Backwards compat
            return _read_legacy_raw(group, func(elem), read_dataframe, func)
        return func(elem)

    adata = read_dispatched(group, callback=callback)

    # Backwards compat (should figure out which version)
    if "raw.X" in group:
        raw = AnnData(**_read_legacy_raw(group, adata.raw, read_dataframe, read_elem))
        raw.obs_names = adata.obs_names
        adata.raw = raw

    # Backwards compat for <0.7
    if isinstance(group["obs"], zarr.Array):
        _clean_uns(adata)

    return adata

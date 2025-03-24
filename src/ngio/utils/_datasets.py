"""Download testing OME-Zarr datasets."""

from pathlib import Path

import pooch

from ngio.utils._errors import NgioValueError

_ome_zarr_zoo = {
    "CardiomyocyteTiny": {
        "url": "https://zenodo.org/records/13305156/files/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip",
        "known_hash": "md5:efc21fe8d4ea3abab76226d8c166452c",
        "fname": "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip",
        "processor": pooch.Unzip(extract_dir=""),
    },
    "CardiomyocyteSmallMip": {
        "url": "https://zenodo.org/records/13305316/files/20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip",
        "known_hash": "md5:3ed3ea898e0ed42d397da2e1dbe40750",
        "fname": "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip",
        "processor": pooch.Unzip(extract_dir=""),
    },
}


def list_ome_zarr_datasets() -> list[str]:
    """List available OME-Zarr datasets."""
    return list(_ome_zarr_zoo.keys())


def download_ome_zarr_dataset(
    dataset_name: str,
    download_dir: str | Path = "data",
) -> Path:
    """Download an OME-Zarr dataset.

    To list available datasets, use `list_ome_zarr_datasets`.

    Args:
        dataset_name (str): The dataset name.
        download_dir (str): The download directory. Defaults to "data".
    """
    if dataset_name not in _ome_zarr_zoo:
        raise NgioValueError(f"Dataset {dataset_name} not found in the OME-Zarr zoo.")
    ome_zarr_url = _ome_zarr_zoo[dataset_name]
    pooch.retrieve(
        path=download_dir,
        **ome_zarr_url,
    )
    path = Path(download_dir) / ome_zarr_url["fname"]

    if isinstance(ome_zarr_url["processor"], pooch.Unzip):
        path = path.with_suffix("")
    return path

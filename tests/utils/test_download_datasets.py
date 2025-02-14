from pathlib import Path

from ngio.utils import download_ome_zarr_dataset, list_ome_zarr_datasets


def test_download_ome_zarr_dataset(tmp_path: Path):
    # Download the first dataset in the list
    datasets = list_ome_zarr_datasets()

    download_path = tmp_path / "test_datasets"

    for dataset in datasets:
        path = download_ome_zarr_dataset(dataset, download_dir=download_path)
        assert path.exists()

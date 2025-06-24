"""This script downloads the OME-Zarr datasets for testing purposes."""

from ngio.utils import download_ome_zarr_dataset


def main():
    """Download OME-Zarr datasets."""
    download_ome_zarr_dataset(dataset_name="CardiomyocyteSmallMip", download_dir="data")
    download_ome_zarr_dataset(dataset_name="CardiomyocyteTiny", download_dir="data")

if __name__ == "__main__":
    main()

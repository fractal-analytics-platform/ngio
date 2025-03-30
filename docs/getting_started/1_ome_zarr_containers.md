# 1. OmeZarr Container

Let's see how to open and explore an OME-Zarr image using `ngio`:

```python exec="true" source="material-block" session="get_started"
from pathlib import Path
from ngio import open_ome_zarr_container
from ngio.utils import download_ome_zarr_dataset

# Download a sample dataset
download_dir = Path("./data")
download_dir = Path(".").absolute().parent.parent / "data" # markdown-exec: hide
hcs_path = download_ome_zarr_dataset("CardiomyocyteSmallMip", download_dir=download_dir)
image_path = hcs_path / "B" / "03" / "0"

# Open the OME-Zarr container
ome_zarr_container = open_ome_zarr_container(image_path)
```

The `ome_zarr_container` in is your entry point to working with OME-Zarr images. It provides high-level access to the image metadata, images, labels, and tables.

```pycon exec="true" source="console" session="get_started"
>>> ome_zarr_container
>>> print(ome_zarr_container) # markdown-exec: hide
```

The `ome_zarr_container` will be the starting point for all your image processing tasks.

## Main Concepts

### What is the OmeZarr Container?

The `OmeZarr Container` in ngio is your entry point to working with OME-Zarr images.

It provides:

- **OME-Zarr overview** it can be used to get an overview of the OME-Zarr file, including the number of image metadata, list
of labels, and tables available.
- **Image access** allow to access the images at different resolution levels and pixel sizes
- **Label management** allow to check which labels are available, access them, and create new labels
- **Table management** allow to check which tables are available, access them, and create new tables
- **Derive new OME-Zarr images** allow to create new images based on the original one, with the same or similar metadata

### What is the OmeZarr Container not?

The `OmeZarr Container` object does not allow the user to interact with the image data directly. For that, we need to use the `Image`, `Label`, and `Table` objects.

### OME-Zarr Overview

For example:

=== "Number of Resolution Levels"
    Show the number of resolution levels:
    ```pycon exec="true" source="console" session="get_started"
    >>> ome_zarr_container.levels # Show the number of resolution levels
    >>> print(ome_zarr_container.levels) # markdown-exec: hide
    ```

=== "Available Paths"
    Show the paths to all available resolution levels:
    ```pycon exec="true" source="console" session="get_started"
    >>> ome_zarr_container.levels_paths # Show the paths to all available images
    >>> print(ome_zarr_container.levels_paths) # markdown-exec: hide
    ```

=== "Dimensionality"
    Show if the image is 2D or 3D:
    ```pycon exec="true" source="console" session="get_started"
    >>> ome_zarr_container.is_3d # Get if the image is 3D
    >>> print(ome_zarr_container.is_3d) # markdown-exec: hide
    ```
    or if the image is a time series:
    ```pycon exec="true" source="console" session="get_started"
    >>> ome_zarr_container.is_time_series # Get if the image is a time series
    >>> print(ome_zarr_container.is_time_series) # markdown-exec: hide
    ```

=== "Full Metadata Object"
    ```pycon exec="true" source="console" session="get_started"
    >>> metadata = ome_zarr_container.image_meta
    >>> print(metadata) # markdown-exec: hide
    ```
    The metadata object contains all the information about the image, for example, the channel labels:
    ```pycon exec="true" source="console" session="get_started"
    >>> metadata.channel_labels
    >>> print(metadata.channel_labels) # markdown-exec: hide
    ```

## Advanced Usage

### Creating Derived Images

When processing an image, you might want to create a new image with the same metadata:

```python
# Create a new image based on the original
new_image = ome_zarr_container.derive_image("data/new_ome.zarr", overwrite=True)
```

### Creating Images from Arrays

You can also create OME-NGFF images from scratch:

```python
import numpy as np
from ngio import create_ome_zarr_from_array

# Create a random 3D array
x = np.random.randint(0, 255, (16, 128, 128), dtype=np.uint8)

# Convert to OME-NGFF
new_ome_zarr_image = create_ome_zarr_from_array(
    store="random_ome.zarr", 
    array=x, 
    xy_pixelsize=0.65, 
    z_spacing=1.0
)
```

### Opening Remote OME-Zarr Containers

You can use `ngio` to open remote OME-Zarr containers. 
For publicly available OME-Zarr containers, you can just use the `open_ome_zarr_container` function with a URL.

For example, to open a remote OME-Zarr container hosted on a github repository:

```python
import fsspec
import fsspec.implementations.http

url = (
    "https://raw.githubusercontent.com/"
    "fractal-analytics-platform/fractal-ome-zarr-examples/"
    "refs/heads/main/v04/"
    "20200812-CardiomyocyteDifferentiation14-Cycle1_B_03_mip.zarr/"
)

fs = fsspec.implementations.http.HTTPFileSystem(client_kwargs={})
store = fs.get_mapper(url)
ome_zarr_container = open_ome_zarr_container(store)
```

If you are a Fractal user, you can use the `fractal_fsspec_store` function to open a remote OME-Zarr container. This function will create an authenticated `fsspec` store that can be used to access the OME-Zarr container.

```python
from ngio.utils import fractal_fsspec_store
 
store = fractal_fsspec_store(url="https://fracral_url...", fractal_token="**your_secret_token**")
ome_zarr_container = open_ome_zarr_container(store)
```

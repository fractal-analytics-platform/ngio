# 5. HCS Plates

Ngio provides a simple interface for high-content screening (HCS) plates. An HCS plate is a collection of OME-Zarr images organized in a grid-like structure. Each plates contains columns and rows, and each well in the plate is identified by its row and column indices. Each well can contain multiple images, and each image can belong to a different acquisition.

The HCS plate is represented by the `OmeZarrPlate` class.

Let's open an `OmeZarrPlate` object.

```pycon exec="true" source="console" session="hcs_plate"
>>> from pathlib import Path # markdown-exec: hide
>>> from ngio.utils import download_ome_zarr_dataset
>>> from ngio import open_ome_zarr_plate
>>> download_dir = Path(".").absolute().parent.parent / "data" # markdown-exec: hide
>>> hcs_path = download_ome_zarr_dataset("CardiomyocyteSmallMip", download_dir=download_dir)
>>> ome_zarr_plate = open_ome_zarr_plate(hcs_path)
>>> ome_zarr_plate
>>> print(ome_zarr_plate) # markdown-exec: hide
```

This example plate is very small and contains only a single well.

## Plate overview

The `OmeZarrPlate` object provides a high-level overview of the plate, including rows, columns, and acquisitions. The following methods are available:

=== "Columns"
    Show the columns in the plate:
    ```pycon exec="true" source="console" session="hcs_plate"
    >>> ome_zarr_plate.columns
    >>> print(ome_zarr_plate.columns) # markdown-exec: hide
    ```
=== "Rows"
    Show the rows in the plate:
    ```pycon exec="true" source="console" session="hcs_plate"
    >>> ome_zarr_plate.rows
    >>> print(ome_zarr_plate.rows) # markdown-exec: hide
    ```
=== "Acquisitions"
    Show the acquisitions ids:
    ```pycon exec="true" source="console" session="hcs_plate"
    >>> ome_zarr_plate.acquisition_ids
    >>> print(ome_zarr_plate.acquisition_ids) # markdown-exec: hide
    ```

## Retrieving the path to the images

The `OmeZarrPlate` object provides multiple methods to retrieve the path to the images in the plate.

=== "All Images Paths"
    This will return the paths to all images in the plate:
    ```pycon exec="true" source="console" session="hcs_plate"
    >>> ome_zarr_plate.images_paths()
    >>> print(ome_zarr_plate.images_paths()) # markdown-exec: hide
    ```

=== "All Wells Paths"
    This will return the paths to all wells in the plate:
    ```pycon exec="true" source="console" session="hcs_plate"
    >>> ome_zarr_plate.wells_paths()
    >>> print(ome_zarr_plate.wells_paths()) # markdown-exec: hide
    ```

=== "All Images Paths in a Well"
    This will return the paths to all images in a well:
    ```pycon exec="true" source="console" session="hcs_plate"
    >>> ome_zarr_plate.well_images_paths(row="B", column=3)
    >>> print(ome_zarr_plate.well_images_paths(row="B", column=3)) # markdown-exec: hide
    ```

## Getting the images

The `OmeZarrPlate` object provides a method to get the image objects in a well. The method `get_well_images` takes the row and column indices of the well and returns a list of `OmeZarrContainer` objects.

=== "All Images"
    Get all images in the plate:
    ```pycon exec="true" source="console" session="hcs_plate"
    >>> ome_zarr_plate.get_images()
    >>> ome_zarr_plate
    >>> print(ome_zarr_plate.get_images()) # markdown-exec: hide
    ```
    This dictionary contains the path to the images and the corresponding `OmeZarrContainer` object.

=== "All Images in a Well"
    Get all images in a well:
    ```pycon exec="true" source="console" session="hcs_plate"
    >>> well_images = ome_zarr_plate.get_well_images(row="B", column=3)
    >>> well_images
    >>> print(well_images) # markdown-exec: hide
    ```
    This dictionary contains the path to the images and the corresponding `OmeZarrContainer` object.

=== "Specific Image"
    Get a specific image in a well:
    ```pycon exec="true" source="console" session="hcs_plate"
    >>> ome_zarr_plate.get_image(row="B", column=3, image_path="0")
    >>> print(ome_zarr_plate.get_image(row="B", column=3, image_path="0")) # markdown-exec: hide
    ```
    This will return the `OmeZarrContainer` object for the image in the well.

=== "Filter by Acquisition"
    In these methods, you can also filter the images by acquisition. When available, the `acquisition` parameter can be used to filter the images by acquisition id.
    ```pycon exec="true" source="console" session="hcs_plate"
    >>> well_images = ome_zarr_plate.get_well_images(row="B", column=3, acquisition=0)
    >>> well_images
    >>> print(well_images) # markdown-exec: hide
    ```
    The `acquisition` is not required, and if not provided, an empty dictionary will be returned.

## Creating a plate

Ngio provides a utility function to create a plate.

The first step is to create a list of `ImageInWellPath` objects. Each `ImageInWellPath` object contains the path to the image and the corresponding well.
```python exec="true" source="console" session="hcs_plate"
from ngio import ImageInWellPath
list_of_images = [ImageInWellPath(path="0", row="A", column=0),
                    ImageInWellPath(path="0", row="B", column=1),
                    ImageInWellPath(path="0", row="C", column=1),
                    ImageInWellPath(path="1", row="A", column=0, acquisition_id=1, acquisition_name="acquisition_1"),
]
```

!!! note
    The order in which the images are added is not important. The `rows` and `columns` attributes of the plate will be sorted in alphabetical/numerical order.

Then, you can create the plate using the `create_empty_plate` function.
```pycon exec="true" source="console" session="hcs_plate"
>>> from ngio import create_empty_plate
>>> plate = create_empty_plate(store="new_plate.zarr", name="test_plate", images=list_of_images, overwrite=True)
>>> plate
>>> print(plate) # markdown-exec: hide
```

This has created a new empty plate with the metadata correctly set. But no images have been added yet. 

### Modifying the plate

You can add images or remove images

=== "Add Images"
    To add images to the plate, you can use the `add_image` method. This method takes the row and column indices of the well and the path to the image.
    ```pycon exec="true" source="console" session="hcs_plate"
    >>> print(f"Before adding images: {plate.rows} rows, {plate.columns} columns")
    >>> plate.add_image(row="D", column=0, image_path="0")
    >>> print(f"After adding images: {plate.rows} rows, {plate.columns} columns")
    ```
    This will add a new image to the plate and well metadata.
    !!! note
        The order in which the images are added is not important. The `rows` and `columns` attributes of the plate will be sorted in alphabetical/numerical order.
    !!! warning
        This function is not multiprocessing safe. If you are using multiprocessing, you should use the `atomic_add_image` method instead.

=== "Remove Images"
    To remove images from the plate, you can use the `remove_image` method. This method takes the row and column indices of the well and the path to the image.
    ```pycon exec="true" source="console" session="hcs_plate"
    >>> print(f"Before removing images: {plate.wells_paths()} wells")
    >>> plate.remove_image(row="D", column=0, image_path="0")
    >>> print(f"After removing images: {plate.wells_paths()} wells")
    ```
    This will remove the image metadata from the plate and well metadata.
    !!! warning
        No data will be removed from the store. If an image is saved in the store it will remain there.
        Also the metadata will only be removed from the plate.well metadata. The number of columns and rows will not be updated.
        This function is not multiprocessing safe. If you are using multiprocessing, you should use the `atomic_remove_image` method instead.

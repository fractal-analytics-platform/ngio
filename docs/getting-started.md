# Getting Started

!!! warning
    The library is still in development and is not yet stable. The API is subject to change, bugs and breaking changes are expected.

!!! warning
    The documentation is still under development. It is not yet complete and may contain errors and inaccuracies.

## Installation

Currently, `ngio` is not available on PyPI. You can install it from the source code.

```bash
pip install "ngio[v2]"
```

The `v2` extra installs the latest version of `zarr-python` from the `v2` branch.
`ngio` is currently not completely compatible with the `v3` release of `zarr-python`.

## `ngio` API Overview

`ngio` implements an abstract object base API for handling OME-Zarr files. The three main objects are `NgffImage`, `Image` (`Label`), and `ROITables`.

- `NgffImage` is the main entry point to the library. It is used to open an OME-Zarr Image and manage its metadata. This object can not be used to access the data directly.
  but it can be used to access and create the `Image`, `Label`, and `Tables` objects. Moreover it can be used to derive a new `Ngff` images based on the current one.
- `Image` and `Label` are used to access "ImageLike" objects. They are the main objects to access the data in the OME-Zarr file, manage the metadata, and write data.
- `ROITables` can be used to access specific region of interest in the image. They are tightly integrated with the `Image` and `Label` objects.

## Example Usage

Currently, the library is not yet stable. However, you can see some example usage in our [demo notebooks](./notebooks/ngff-image.ipynb).

- [Basic Usage](./notebooks/ngff-image.ipynb)
- [Image/Label/Tables](./notebooks/ngff-image.ipynb)
- [Processing](./notebooks/ngff-image.ipynb)

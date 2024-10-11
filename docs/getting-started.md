# Getting Started

!!! warning
    The library is still in development and is not yet stable. The API is subject to change, bugs and breaking changes are expected.

!!! warning
    The documentation is still under development. It is not yet complete and may contain errors and inaccuracies.

## Installation

The library can be installed from PyPI using pip:

```bash
pip install "ngio[core]"
```

The `core` extra installs the the `zarr-python` dependency. As of now, `zarr-python` is required to be installed separately, due to the transition to the new `zarr-v3` library.

## `ngio` API Overview

`ngio` implements an abstract object base API for handling OME-Zarr files. The three main objects are `NgffImage`, `Image` (`Label`), and `ROITables`.

- `NgffImage` is the main entry point to the library. It is used to open an OME-Zarr Image and manage its metadata. This object can not be used to access the data directly.
  but it can be used to access and create the `Image`, `Label`, and `Tables` objects. Moreover it can be used to derive a new `Ngff` images based on the current one.
- `Image` and `Label` are used to access "ImageLike" objects. They are the main objects to access the data in the OME-Zarr file, manage the metadata, and write data.
- `ROITables` can be used to access specific region of interest in the image. They are tightly integrated with the `Image` and `Label` objects.

## Example Usage

Currently, the library is not yet stable. However, you can see some example usage in our demo notebooks:

- [Basic Usage](https://fractal-analytics-platform.github.io/ngio/notebooks/basic_usage/)
- [Image/Label/Tables](https://fractal-analytics-platform.github.io/ngio/notebooks/image/)
- [Processing](https://fractal-analytics-platform.github.io/ngio/notebooks/processing/)

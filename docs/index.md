
NGIO is a Python library designed to simplify bio-image analysis workflows, offering an intuitive interface for working with OME-Zarr files.

## What is NGIO?

NGIO is built for the [OME-Zarr](https://ngff.openmicroscopy.org/) file format, a modern, cloud-optimized format for biological imaging data. OME-Zarr stores large, multi-dimensional microscopy images and metadata in an efficient and scalable way.

NGIO's mission is to streamline working with OME-Zarr files by providing a simple, object-based API for opening, exploring, and manipulating OME-Zarr Images and HCS Plates. It also offers comprehensive support for tables and regions of interest (ROIs), making it easy to extract and analyze specific regions in your data.

## Key Features

### 📊 Simple Object-Based API

- Easily open, explore, and manipulate OME-Zarr Images and HCS Plates
- Create and derive new images and labels with minimal boilerplate code

### 🔍 Rich Tables and Regions of Interest (ROI) Support

- Extract and analyze specific regions of interest
- Tight integration with [Fractal's Table Framework](https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/)

### 🔄 Scalable Data Processing (Coming Soon)

- Powerful iterators for processing data at scale
- Efficient memory management for large datasets

## Getting Started

Refer to the [Getting Started](getting_started/0_quickstart.md) guide to integrate NGIO into your workflows. We also provide a collection of [tutorials](tutorials/image_processing.ipynb) to help you get up and running quickly.
For more advanced usage and API documentation, see our [API Reference](api/ngio.md).

## Development Status

!!! warning
    NGIO is under active development and is not yet stable. The API is subject to change, and bugs and breaking changes are expected.

### Available Features

- ✅ OME-Zarr metadata handling and validation
- ✅ Image and label access across pyramid levels
- ✅ ROI and table support
- ✅ Streaming from remote sources
- ✅ Documentation and examples

### Upcoming Features

- Advanced image processing iterators
- Parallel processing capabilities

## Contributors

NGIO is developed at the [BioVisionCenter](https://www.biovisioncenter.uzh.ch/en.html), University of Zurich, by [@lorenzocerrone](https://github.com/lorenzocerrone) and [@jluethi](https://github.com/jluethi).

## License

NGIO is released under the BSD-3-Clause License. See [LICENSE](https://github.com/fractal-analytics-platform/ngio/blob/main/LICENSE) for details.

## Repository

Visit our [GitHub repository](https://github.com/fractal-analytics-platform/ngio) for the latest code, issues, and contributions.

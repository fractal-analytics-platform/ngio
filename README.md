# NGIO - Next Generation file format IO

[![License](https://img.shields.io/pypi/l/ngio.svg?color=green)](https://github.com/lorenzocerrone/ngio/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ngio.svg?color=green)](https://pypi.org/project/ngio)
[![Python Version](https://img.shields.io/pypi/pyversions/ngio.svg?color=green)](https://python.org)
[![CI](https://github.com/lorenzocerrone/ngio/actions/workflows/ci.yml/badge.svg)](https://github.com/lorenzocerrone/ngio/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fractal-analytics-platform/ngio/graph/badge.svg?token=FkmF26FZki)](https://codecov.io/gh/fractal-analytics-platform/ngio)

NGIO is a Python library to streamline OME-Zarr image analysis workflows.

**Main Goals:**

- Abstract object base API for handling OME-Zarr files
- Powefull iterators for processing data using common access patterns
- Tight integration with [Fractal's Table Fractal](https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/)
- Validate OME-Zarr files

To get started, check out the [Getting Started](https://fractal-analytics-platform.github.io/ngio/getting-started/) guide. Or checkout our [Documentation](https://fractal-analytics-platform.github.io/ngio/)

## 🚧 Ngio is Under active Development 🚧

### Roadmap

| Feature | Status | ETA | Description |
|---------|--------|-----|-------------|
| Metadata Handling | ✅ | | Read, Write, Validate OME-Zarr Metadata (0.4 supported, 0.5 ready) |
| OME-Zarr Validation | ✅ | | Validate OME-Zarr files for compliance with the OME-Zarr Specification + Compliance between Metadata and Data |
| Base Image Handling | ✅ | | Load data from OME-Zarr files, retrieve basic metadata, and write data |
| ROI Handling | ✅ | | Common ROI models |
| Label Handling | ✅ | Mid-September | Based on Image Handling |
| Table Validation | ✅ | Mid-September | Validate Table fractal V1 + Compliance between Metadata and Data |
| Table Handling | ✅ | Mid-September | Read, Write ROI, Features, and Masked Tables |
| Basic Iterators | Ongoing | End-September | Read and Write Iterators for common access patterns |
| Base Documentation | ✅ | End-September | API Documentation and Examples |
| Beta Ready Testing | ✅ | End-September | Beta Testing; Library is ready for testing, but the API is not stable |
| Mask Iterators | Ongoing | October | Iterators over Masked Tables |
| Advanced Iterators | Not started | October | Iterators for advanced access patterns |
| Parallel Iterators | Not started | End of the Year | Concurrent Iterators for parallel read and write |
| Full Documentation | Not started | End of the Year | Complete Documentation |
| Release 1.0 (Commitment to API) | Not started | End of the Year | API is stable; breaking changes will be avoided |

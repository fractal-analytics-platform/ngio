# Tables Overview

Ngio's architecture is designed to tightly integrate image and tabular data. For this purpose we developed custom specifications for serializing and deserializing tabular data into OME-Zarr, and semantically typed tables derived from the [fractal table specification](https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/).

## Architecture

The ngio tables architectures is composed of two three main components:

### 1. Table Backends

A backend module is a class that can serialize on disk tabular data into OME-Zarr. We currently support four on-disk file formats:

- **AnnData**: Commonly used in single-cell genomics.
- **Parquet**: A columnar storage file format optimized for large datasets.
- **CSV**: A simple text format for tabular data, easily human readable and writable.
- **JSON**: A lightweight data interchange format that both readable and efficient for small tables.

A more detailed description of the backend module can be found in the [Table Backends documentation](backend.md).

### 2. In Memory Table Objects

These are Python objects that represent the tabular data in memory. They provide a convenient interface for manipulating and analyzing the data without needing to interact directly with the underlying file format. We support the following in-memory table objects:

- **Pandas DataFrame**: The most commonly used data structure for tabular data in Python.
- **Polars LazyFrame**: A fast DataFrame implementation that allows for lazy evaluation and efficient computation on large datasets.
- **AnnData**: A specialized data structure for single-cell genomics data, which goes beyond simple tabular data.

We also provide utilities to convert between these in-memory representations, in a standardized way based on the table type specifications/metadata.

### 3. Table Type Specifications

These specifications define structured tables that standardize common table types used in image analysis. We currently developed five table types:

- **Generic Tables**: A flexible table type that can represent any tabular data.
- **ROI Tables**: A table type specifically designed for representing Regions of Interest (ROIs) in images.
- **Masking ROI Tables**: A specialized table type for representing ROIs that are associated with specific labels in a OME-Zarr label image.
- **Feature Tables**: A table type for representing features extracted from images. This table is also associated with a specific label image.
- **Condition Tables**: A table to represent experimental conditions or metadata associated with images or experiments.

A more detailed description of the table types can be found in the [Table Types documentation](./v1/types.md).

## Tables Groups

Tables in OME-Zarr images are organized into groups of tables. Each of these group is saved in a Zarr group, and can be associated with a specific image or plate. The tables groups are:

- **Image Tables**: These tables are a sub group of the OME-Zarr image group and contain metadata or features related only to that specific image. The `.zarr` hierarchy is based on image [specification in NGFF 0.4](https://ngff.openmicroscopy.org/0.4/index.html#image-layout), and it is a generalization of the OME-Zarr `labels` group.

```bash
image.zarr        # Zarr group for a NGFF image
|
├── 0             # Zarr array for multiscale level 0
├── ...
├── N             # Zarr array for multiscale level N
|
├── labels        # Zarr subgroup with a list of labels associated to this image
|   ├── label_A   # Zarr subgroup for a given label
|   ├── label_B   # Zarr subgroup for a given label
|   └── ...
|
└── tables        # Zarr subgroup with a list of tables associated to this image
    ├── table_1   # Zarr subgroup for a given table
    ├── table_2   # Zarr subgroup for a given table
    └── ...
```

- **Plate Tables**: These tables are a sub group of the OME-Zarr plate group and contain metadata or features related only to that specific plate.
  
```bash
plate.zarr       # Zarr group for a NGFF plate
|
├── A             # Row A of the plate
|   ├── 1         # Column 0 of row A
|   |   ├── 0     # Acquisition 0 of column A1
|   |   ├── 1     # Acquisition 1 of column A1
|   |   └── ...   # Other acquisitions of column A1
...
├── tables        # Zarr subgroup with a list of tables associated to this plate
|   ├── table_1   # Zarr subgroup for a given table
|   ├── table_2   # Zarr subgroup for a given table
|   └── ...
└── ...
```

If a plate table contains per image information, the table should contain a `row`, `column`, and `path_in_well` columns.

## Tables Group Attributes

The Zarr attributes of the tables group must include the key tables, pointing to the list of all tables (this simplifies discovery of tables associated to the current NGFF image), as in

```json
{
    "tables": ["table_1", "table_2"]
}
```

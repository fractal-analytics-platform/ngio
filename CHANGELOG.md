# Changelog

## [Unreleased]

### Features

- add support for time in rois and roi-tables

### API Changes

- change to `Dimension` class. `get_shape` and `get_canonical_shape` have been removed, `get` uses new keyword arguments `default` instead of `strict`.

### Table specs

- add `t_second` and `len_t_second` to ROI tables and masking ROI tables

## [v0.3.3]

### Chores

- improve dataset download process and streamline the CI workflows

## [v0.3.2]

### API Changes

- change table backend default to `anndata_v1` for backward compatibility. This will be chaanged again when ngio `v0.2.x` is no longer supported.

### Bug Fixes

- fix [#13](https://github.com/fractal-analytics-platform/fractal-converters-tools/issues/13) (converters tools)
- fix [#88](https://github.com/fractal-analytics-platform/ngio/issues/88)

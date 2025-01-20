from pathlib import Path

import pytest


class TestNgffImage:
    def test_ngff_image(self, ome_zarr_image_v04_path: Path) -> None:
        from ngio.core.ngff_image import NgffImage

        ngff_image = NgffImage(ome_zarr_image_v04_path)
        image_handler = ngff_image.get_image(path="0")
        ngff_image.__repr__()

        assert ngff_image.num_levels == 5
        assert ngff_image.levels_paths == ["0", "1", "2", "3", "4"]
        assert image_handler.channel_labels == ["DAPI", "nanog", "Lamin B1"]
        assert image_handler.get_channel_idx(label="DAPI") == 0
        assert image_handler.get_channel_idx(wavelength_id="A01_C01") == 0

        new_path = ome_zarr_image_v04_path.parent / "new_ngff_image.zarr"
        new_ngff_image = ngff_image.derive_new_image(
            new_path, "new_image", overwrite=True
        )
        new_image_handler = new_ngff_image.get_image(path="0")

        assert new_ngff_image.levels_paths == ngff_image.levels_paths
        assert new_image_handler.channel_labels == image_handler.channel_labels
        assert new_image_handler.shape == image_handler.shape
        assert new_image_handler.pixel_size.zyx == image_handler.pixel_size.zyx
        assert (
            new_image_handler.on_disk_array.shape == image_handler.on_disk_array.shape
        )
        assert (
            new_image_handler.on_disk_array.chunks == image_handler.on_disk_array.chunks
        )
        new_ngff_image.lazy_init_omero(
            labels=3,
            wavelength_ids=["A01_C01", "A02_C02", "A03_C03"],
            consolidate=True,
        )

        new_ngff_image.update_omero_window(start_percentile=1.1, end_percentile=98.9)

    def test_ngff_image_derive(self, ome_zarr_image_v04_path: Path) -> None:
        from ngio.core.ngff_image import NgffImage

        ngff_image = NgffImage(ome_zarr_image_v04_path)

        for name, table_type in [("test_roi", "roi_table")]:
            table = ngff_image.tables.new(name=name, table_type=table_type)
            table.consolidate()

        ngff_image.labels.derive("test_label")

        new_path = ome_zarr_image_v04_path.parent / "new_ngff_image.zarr"
        new_ngff_image = ngff_image.derive_new_image(
            new_path, "new_image", overwrite=True, copy_labels=True, copy_tables=True
        )

        assert ngff_image.tables.list() == new_ngff_image.tables.list()
        assert ngff_image.labels.list() == new_ngff_image.labels.list()

    @pytest.mark.parametrize(
        "shape, axis, chunks",
        [
            ((1, 4, 1, 1945, 1945), ("t", "c", "z", "y", "x"), (1, 1, 1, 1000, 1000)),
            ((1, 4, 1, 1945, 1945), ("t", "c", "z", "y", "x"), None),
            ((1, 4, 1, 1945, 2000), ("t", "c", "z", "y", "x"), None),
            ((1, 1, 1000, 1000), ("c", "z", "y", "x"), (1, 1, 1000, 1000)),
            ((1, 1, 1000, 1000), ("c", "z", "y", "x"), None),
            ((739, 1033), ("y", "x"), (53, 173)),
        ],
    )
    def test_ngff_image_consolidate(self, tmp_path, shape, axis, chunks) -> None:
        from ngio import NgffImage
        from ngio.core.utils import create_empty_ome_zarr_image

        ome_zarr = tmp_path / "test_consolidate.zarr"
        create_empty_ome_zarr_image(
            ome_zarr,
            on_disk_shape=shape,
            on_disk_axis=axis,
            chunks=chunks,
        )

        image = NgffImage(ome_zarr).get_image()
        image.consolidate()

import json

import pytest


class TestOMEZarrHandlerV04:
    def test_create_image_meta(self, tmp_path):
        from ngio.ngff_meta import create_image_metadata
        from ngio.ngff_meta.v04.zarr_utils import (
            fractal_ngff_image_meta_to_vanilla_v04,
            vanilla_ngff_image_meta_v04_to_fractal,
        )

        meta = create_image_metadata(
            on_disk_axis=("t", "c", "z", "y", "x"),
            pixel_sizes=None,
            xy_scaling_factor=2.0,
            z_scaling_factor=1.0,
            time_spacing=1.0,
            time_units="s",
            levels=5,
            name="test",
            channel_labels=["DAPI", "nanog", "Lamin B1"],
            channel_wavelengths=["A01_C01", "A02_C02", "A03_C03"],
            channel_visualization=None,
            omero_kwargs=None,
            version="0.4",
        )

        meta04 = fractal_ngff_image_meta_to_vanilla_v04(meta)
        meta2 = vanilla_ngff_image_meta_v04_to_fractal(meta04)
        assert meta.channel_labels == meta2.channel_labels
        assert meta.channel_wavelength_ids == meta2.channel_wavelength_ids
        assert meta.axes_names == meta2.axes_names
        assert meta.scale(path="2") == meta2.scale(path="2")

    def test_basic_workflow(self, ome_zarr_image_v04_path):
        from ngio.ngff_meta import get_ngff_image_meta_handler
        from ngio.ngff_meta.v04.zarr_utils import NgffImageMeta04

        handler = get_ngff_image_meta_handler(
            store=ome_zarr_image_v04_path, meta_mode="image"
        )

        meta = handler.load_meta()
        handler.write_meta(meta)

        with open("tests/data/meta_v04/base_ome_zarr_image_meta.json") as f:
            base_ome_zarr_meta = json.load(f)

        saved_meta = NgffImageMeta04(**handler.group.attrs).model_dump(
            exclude_none=True
        )
        assert saved_meta == base_ome_zarr_meta

    def test_basic_workflow_with_cache(self, ome_zarr_image_v04_path):
        from ngio.ngff_meta import get_ngff_image_meta_handler
        from ngio.ngff_meta.v04.zarr_utils import NgffImageMeta04

        handler = get_ngff_image_meta_handler(
            store=ome_zarr_image_v04_path, meta_mode="image", cache=True
        )

        meta = handler.load_meta()
        handler.write_meta(meta)

        with open("tests/data/meta_v04/base_ome_zarr_image_meta.json") as f:
            base_ome_zarr_meta = json.load(f)

        saved_meta = NgffImageMeta04(**handler.group.attrs).model_dump(
            exclude_none=True
        )
        assert saved_meta == base_ome_zarr_meta

    def test_wrong_axis_order(self):
        from pydantic import ValidationError

        from ngio.ngff_meta.v04.specs import NgffImageMeta04

        with open(
            "tests/data/meta_v04/base_ome_zarr_image_meta_wrong_axis_order.json"
        ) as f:
            base_ome_zarr_meta = json.load(f)

        with pytest.raises(ValidationError):
            NgffImageMeta04(**base_ome_zarr_meta)

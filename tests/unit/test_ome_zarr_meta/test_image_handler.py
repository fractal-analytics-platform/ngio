from pathlib import Path

from ngio.ome_zarr_meta import NgioImageMeta, open_image_meta_handler


def test_get_image_handler(ome_zarr_image_v04_path: Path):
    # TODO this is a placeholder test
    # The pooch cache is giving us trouble here
    ome_zarr_image_v04_path = ome_zarr_image_v04_path / "B" / "03" / "0"
    handler = open_image_meta_handler(ome_zarr_image_v04_path, cache=True, mode="a")
    meta = handler.safe_load_meta()
    assert isinstance(meta, NgioImageMeta)
    handler.write_meta(meta)

from pathlib import Path

import pytest

from ngio.tables.v1._roi_table import ROITableV1, WorldCooROI
from ngio.utils import NgioValueError


def test_roi_table_v1(tmp_path: Path):
    rois = {
        "roi1": WorldCooROI(
            name="roi1",
            x=0.0,
            y=0.0,
            z=0.0,
            x_length=1.0,
            y_length=1.0,
            z_length=1.0,
            unit="micrometer",  # type: ignore
        )
    }

    table = ROITableV1(rois=rois.values())

    table.add(
        roi=WorldCooROI(
            name="roi2",
            x=0.0,
            y=0.0,
            z=0.0,
            x_length=1.0,
            y_length=1.0,
            z_length=1.0,
            unit="micrometer",  # type: ignore
        )
    )

    with pytest.raises(NgioValueError):
        table.add(
            roi=WorldCooROI(
                name="roi2",
                x=0.0,
                y=0.0,
                z=0.0,
                x_length=1.0,
                y_length=1.0,
                z_length=1.0,
                unit="micrometer",  # type: ignore
            )
        )

    table.set_backend(store=tmp_path / "roi_table.zarr", backend_name="anndata")
    table.consolidate()

    table2 = ROITableV1.from_store(store=tmp_path / "roi_table.zarr")

    assert len(table2._rois) == 2
    assert table2.get("roi1") == table.get("roi1")
    assert table2.get("roi2") == table.get("roi2")

    with pytest.raises(NgioValueError):
        table2.get("roi3")

    assert table2._meta.backend == "anndata"
    assert table2._meta.fractal_table_version == table2.version()
    assert table2._meta.type == table2.type()

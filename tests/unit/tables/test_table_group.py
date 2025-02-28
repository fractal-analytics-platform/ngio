from pathlib import Path

from ngio.tables.table_handler import TableContainer


def test_table_group(tmp_path: Path):
    table_group = TableContainer(tmp_path / "test.zarr")
    # TODO test the table group
    assert isinstance(table_group, TableContainer)

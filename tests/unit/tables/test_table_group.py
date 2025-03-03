from pathlib import Path

from ngio.tables.tables_container import TablesContainer


def test_table_group(tmp_path: Path):
    table_group = TablesContainer(tmp_path / "test.zarr")
    # TODO test the table group
    assert isinstance(table_group, TablesContainer)

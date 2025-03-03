from pathlib import Path

from ngio.tables.tables_container import TablesContainer, open_tables_container


def test_table_container(tmp_path: Path):
    table_group = open_tables_container(tmp_path / "test.zarr")
    # TODO test the table group
    assert isinstance(table_group, TablesContainer)

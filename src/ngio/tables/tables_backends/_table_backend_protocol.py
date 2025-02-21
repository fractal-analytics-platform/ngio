"""Protocol for table backends handlers."""

from typing import Protocol

from anndata import AnnData
from pandas import DataFrame


class TableBackendProtocol(Protocol):
    def load_as_anndata(self) -> AnnData:
        raise NotImplementedError

    def load_as_dataframe(self) -> DataFrame:
        raise NotImplementedError

    def write_from_dataframe(self, table: DataFrame) -> None:
        raise NotImplementedError

    def write_from_anndata(self, table: AnnData) -> None:
        raise NotImplementedError

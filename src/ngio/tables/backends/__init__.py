"""Ngio Tables backend implementations."""

from ngio.tables.backends._table_backends import (
    TableBackendProtocol,
    TableBackendsManager,
)

__all__ = ["TableBackendProtocol", "TableBackendsManager"]

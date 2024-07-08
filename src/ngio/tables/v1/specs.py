"""Table metadata specifications for the fractal tables."""

from pydantic import BaseModel


class Table(BaseModel):
    """Base class for Fractal tables."""

    fractal_table_version: str
    encoding_type: str
    encoding_version: str


class RoiTable(Table):
    """ROI table metadata."""

    pass


class MaskingRoiTable(Table):
    """Masking ROI table metadata."""

    region: dict[str, str]
    instance_key: str


class FeatureTable(Table):
    """Feature table metadata."""

    region: dict[str, str]
    instance_key: str

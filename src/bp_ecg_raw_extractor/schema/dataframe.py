"""Polars schema definition for the ECG records raw table.

Defines the authoritative column schema used throughout the pipeline.
``pl.String`` is used for all string fields — ``pl.Utf8`` is deprecated
since Polars 0.19 and must not appear anywhere in this codebase.
"""

from typing import Any

import pandera.polars as pa_polar
import polars as pl
import pyarrow as pa
from pandera.typing.polars import DataFrame as PanderaDataFrame

# Canonical schema for ecg_records_raw Iceberg/Parquet table.
# pl.String is the correct type — never pl.Utf8 (deprecated).
SCHEMA: pl.Schema = pl.Schema(
    {
        "file_hash": pl.String,
        "processed_at": pl.Datetime("us", "UTC"),
        "original_path": pl.String,
        "source_zip": pl.String,
        "image_width": pl.Int32,
        "image_height": pl.Int32,
        "compression_ratio": pl.Float64,
        "processing_duration_ms": pl.Int64,
        "pdf_producer": pl.String,
        "pdf_creator": pl.String,
        "ocr_raw_text": pl.String,
        "ocr_confidence_mean": pl.Float64,
        "pdf_raw_text": pl.String,
        "watcher_version": pl.String,
    }
)


def make_empty_dataframe() -> pl.DataFrame:
    """Return an empty Polars DataFrame with the canonical ECG schema.

    Useful as a starting point for batch assembly or for schema validation
    in tests.

    Returns:
        An empty :class:`polars.DataFrame` whose dtypes exactly match
        :data:`SCHEMA`.
    """
    return pl.DataFrame(schema=SCHEMA)


def validate_row(row: dict[str, Any]) -> None:
    """Validate that *row* contains all required schema fields.

    Args:
        row: A dictionary representing a single record to be inserted.

    Raises:
        ValueError: If one or more required fields are absent from *row*.
    """
    missing: list[str] = [k for k in SCHEMA if k not in row]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")


class ECGRecordsSchema(pa_polar.DataFrameModel):
    """Pandera schema for runtime validation of ECG record DataFrames.

    Validates column types and value constraints beyond what Polars dtypes
    enforce.  Use :func:`validate` to apply.
    """

    image_width: int = pa_polar.Field(gt=0)
    image_height: int = pa_polar.Field(gt=0)
    ocr_confidence_mean: float = pa_polar.Field(ge=0.0, le=1.0)
    compression_ratio: float = pa_polar.Field(ge=0.0)
    processing_duration_ms: int = pa_polar.Field(ge=0)

    class Config:  # type: ignore[misc]
        """Pandera model config."""

        strict = False  # allow extra columns


def validate(df: pl.DataFrame) -> pl.DataFrame:
    """Validate a DataFrame against :class:`ECGRecordsSchema`.

    Args:
        df: DataFrame to validate.

    Returns:
        The validated DataFrame (Pandera returns the same object on success).

    Raises:
        pandera.errors.SchemaError: When any constraint is violated.
    """
    validated: PanderaDataFrame[ECGRecordsSchema] = ECGRecordsSchema.validate(df)  # type: ignore[assignment]
    return validated  # type: ignore[return-value]


def dataframe_to_arrow(df: pl.DataFrame) -> pa.Table:
    """Convert a Polars DataFrame to a PyArrow Table for Iceberg writing.

    Args:
        df: A Polars DataFrame conforming to :data:`SCHEMA`.

    Returns:
        A :class:`pyarrow.Table` representation of the same data.
    """
    return df.to_arrow()

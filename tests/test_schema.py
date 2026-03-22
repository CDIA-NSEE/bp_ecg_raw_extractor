"""Tests for bp_ecg_raw_extractor.schema.dataframe."""

from datetime import UTC, datetime
from typing import Any

import polars as pl
import pyarrow as pa
import pytest

from bp_ecg_raw_extractor.schema.dataframe import (
    SCHEMA,
    dataframe_to_arrow,
    make_empty_dataframe,
    validate_row,
)

# Expected field names, in order
_EXPECTED_FIELDS: list[str] = [
    "file_hash",
    "processed_at",
    "original_path",
    "source_zip",
    "image_width",
    "image_height",
    "compression_ratio",
    "processing_duration_ms",
    "pdf_producer",
    "pdf_creator",
    "ocr_raw_text",
    "ocr_confidence_mean",
    "pdf_raw_text",
    "watcher_version",
]

# A complete valid row for use in multiple tests
_VALID_ROW: dict[str, Any] = {
    "file_hash": "abc123",
    "processed_at": datetime.now(UTC),
    "original_path": "/data/exam.zip",
    "source_zip": "exam.zip",
    "image_width": 1200,
    "image_height": 900,
    "compression_ratio": 3.14,
    "processing_duration_ms": 420,
    "pdf_producer": "AdobePDF",
    "pdf_creator": "Scanner v1",
    "ocr_raw_text": "Patient: John",
    "ocr_confidence_mean": 0.95,
    "pdf_raw_text": "ECG report content",
    "watcher_version": "0.1.0",
}


class TestSchema:
    """Unit tests for the SCHEMA constant."""

    def test_schema_has_correct_number_of_fields(self) -> None:
        """SCHEMA must contain exactly 14 fields."""
        assert len(SCHEMA) == 14

    def test_schema_has_all_expected_fields(self) -> None:
        """SCHEMA must contain every expected field name."""
        assert list(SCHEMA.keys()) == _EXPECTED_FIELDS

    def test_string_fields_use_pl_string_not_utf8(self) -> None:
        """All string fields must use pl.String — pl.Utf8 is deprecated."""
        string_fields: list[str] = [
            "file_hash",
            "original_path",
            "source_zip",
            "pdf_producer",
            "pdf_creator",
            "ocr_raw_text",
            "pdf_raw_text",
            "watcher_version",
        ]
        for field in string_fields:
            dtype: pl.DataType = SCHEMA[field]
            assert dtype == pl.String, (
                f"Field '{field}' has dtype {dtype!r}; expected pl.String"
            )

    def test_datetime_field_has_utc_timezone(self) -> None:
        """processed_at must be Datetime with UTC timezone and microsecond precision."""
        dtype: pl.DataType = SCHEMA["processed_at"]
        assert dtype == pl.Datetime("us", "UTC")

    def test_integer_fields_have_correct_types(self) -> None:
        """image_width, image_height → Int32; processing_duration_ms → Int64."""
        assert SCHEMA["image_width"] == pl.Int32
        assert SCHEMA["image_height"] == pl.Int32
        assert SCHEMA["processing_duration_ms"] == pl.Int64

    def test_float_fields_have_correct_types(self) -> None:
        """compression_ratio, ocr_confidence_mean → Float64."""
        assert SCHEMA["compression_ratio"] == pl.Float64
        assert SCHEMA["ocr_confidence_mean"] == pl.Float64


class TestMakeEmptyDataframe:
    """Unit tests for :func:`make_empty_dataframe`."""

    def test_returns_polars_dataframe(self) -> None:
        """make_empty_dataframe must return a pl.DataFrame."""
        df: pl.DataFrame = make_empty_dataframe()
        assert isinstance(df, pl.DataFrame)

    def test_has_zero_rows(self) -> None:
        """The empty DataFrame must have 0 rows."""
        df: pl.DataFrame = make_empty_dataframe()
        assert len(df) == 0

    def test_schema_matches_canonical_schema(self) -> None:
        """The empty DataFrame schema must exactly match SCHEMA."""
        df: pl.DataFrame = make_empty_dataframe()
        assert df.schema == SCHEMA

    def test_has_correct_number_of_columns(self) -> None:
        """The empty DataFrame must have 14 columns."""
        df: pl.DataFrame = make_empty_dataframe()
        assert len(df.columns) == 14


class TestValidateRow:
    """Unit tests for :func:`validate_row`."""

    def test_complete_row_passes_validation(self) -> None:
        """A row containing all required fields must not raise."""
        validate_row(_VALID_ROW)  # Should not raise

    def test_missing_single_field_raises_value_error(self) -> None:
        """Omitting one field must raise ValueError listing that field."""
        incomplete: dict[str, Any] = {
            k: v for k, v in _VALID_ROW.items() if k != "file_hash"
        }
        with pytest.raises(ValueError, match="file_hash"):
            validate_row(incomplete)

    def test_missing_multiple_fields_raises_value_error(self) -> None:
        """Omitting multiple fields must raise ValueError."""
        empty: dict[str, Any] = {}
        with pytest.raises(ValueError):
            validate_row(empty)

    def test_extra_fields_do_not_cause_failure(self) -> None:
        """Extra fields beyond the schema are allowed and must not raise."""
        extra: dict[str, Any] = {**_VALID_ROW, "extra_field": "extra_value"}
        validate_row(extra)  # Should not raise


class TestDataframeToArrow:
    """Unit tests for :func:`dataframe_to_arrow`."""

    def test_empty_dataframe_converts_to_arrow(self) -> None:
        """An empty DataFrame must convert to a PyArrow Table without error."""
        df: pl.DataFrame = make_empty_dataframe()
        arrow_table: pa.Table = dataframe_to_arrow(df)
        assert isinstance(arrow_table, pa.Table)

    def test_arrow_table_has_correct_number_of_columns(self) -> None:
        """PyArrow Table must have the same number of columns as SCHEMA."""
        df: pl.DataFrame = make_empty_dataframe()
        arrow_table: pa.Table = dataframe_to_arrow(df)
        assert len(arrow_table.schema) == 14

    def test_arrow_table_has_zero_rows_for_empty_df(self) -> None:
        """Conversion of empty DataFrame must produce a zero-row Arrow Table."""
        df: pl.DataFrame = make_empty_dataframe()
        arrow_table: pa.Table = dataframe_to_arrow(df)
        assert arrow_table.num_rows == 0

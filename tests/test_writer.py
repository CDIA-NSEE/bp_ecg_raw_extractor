"""Tests for the Iceberg writer module.

All pyiceberg catalog interactions are mocked — no real catalog connection
is made during testing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from bp_ecg_raw_extractor.schema.dataframe import SCHEMA, make_empty_dataframe
from bp_ecg_raw_extractor.writer.iceberg_writer import (
    build_catalog_config,
    write_to_iceberg,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NESSIE_URI = "http://localhost:19120/iceberg"
_TABLE_ID = "bp_ecg.ecg_records_raw"
_WAREHOUSE = "s3://bp-ecg-dev-lake"


def _make_row_df() -> pl.DataFrame:
    """Return a valid single-row DataFrame conforming to SCHEMA."""
    row = {
        "file_hash": "abc123",
        "processed_at": datetime.now(UTC),
        "original_path": "/tmp/x.zip",
        "source_zip": "x.zip",
        "image_width": 100,
        "image_height": 200,
        "compression_ratio": 1.5,
        "processing_duration_ms": 500,
        "pdf_producer": "producer",
        "pdf_creator": "creator",
        "ocr_raw_text": "sample text",
        "ocr_confidence_mean": 0.9,
        "pdf_raw_text": "pdf content",
        "watcher_version": "0.1.0",
    }
    return pl.DataFrame([row], schema=SCHEMA)


# ---------------------------------------------------------------------------
# TestBuildCatalogConfig
# ---------------------------------------------------------------------------


class TestBuildCatalogConfig:
    """Unit tests for :func:`build_catalog_config`."""

    def test_returns_rest_type(self) -> None:
        """Catalog config must specify REST type for Nessie compatibility."""
        config = build_catalog_config(_NESSIE_URI)
        assert config["type"] == "rest"

    def test_uri_is_preserved(self) -> None:
        """The URI passed in must be echoed unchanged in the config dict."""
        config = build_catalog_config(_NESSIE_URI)
        assert config["uri"] == _NESSIE_URI

    def test_returns_dict_of_strings(self) -> None:
        """All values in the config must be plain strings."""
        config = build_catalog_config(_NESSIE_URI)
        for value in config.values():
            assert isinstance(value, str)

    def test_different_uri(self) -> None:
        """A different URI must be reflected correctly."""
        uri = "http://nessie:19120/iceberg"
        config = build_catalog_config(uri)
        assert config["uri"] == uri


# ---------------------------------------------------------------------------
# TestWriteToIceberg
# ---------------------------------------------------------------------------


class TestWriteToIceberg:
    """Unit tests for :func:`write_to_iceberg`."""

    @patch("bp_ecg_raw_extractor.writer.iceberg_writer.load_catalog")
    def test_creates_table_when_not_exists(self, mock_load_catalog: MagicMock) -> None:
        """When table does not exist, create_namespace and create_table are called."""
        mock_catalog = MagicMock()
        mock_catalog.table_exists.return_value = False
        mock_table = MagicMock()
        mock_catalog.create_table.return_value = mock_table
        mock_load_catalog.return_value = mock_catalog

        df = _make_row_df()
        write_to_iceberg(df, _NESSIE_URI, _TABLE_ID, _WAREHOUSE)

        mock_catalog.table_exists.assert_called_once_with(_TABLE_ID)
        mock_catalog.create_namespace.assert_called_once_with("bp_ecg")
        mock_catalog.create_table.assert_called_once()
        mock_table.append.assert_called_once()

    @patch("bp_ecg_raw_extractor.writer.iceberg_writer.load_catalog")
    def test_appends_when_table_exists(self, mock_load_catalog: MagicMock) -> None:
        """When the table exists, only load_table and append are called."""
        mock_catalog = MagicMock()
        mock_catalog.table_exists.return_value = True
        mock_table = MagicMock()
        mock_catalog.load_table.return_value = mock_table
        mock_load_catalog.return_value = mock_catalog

        df = _make_row_df()
        write_to_iceberg(df, _NESSIE_URI, _TABLE_ID, _WAREHOUSE)

        mock_catalog.table_exists.assert_called_once_with(_TABLE_ID)
        mock_catalog.create_table.assert_not_called()
        mock_catalog.load_table.assert_called_once_with(_TABLE_ID)
        mock_table.append.assert_called_once()

    def test_raises_on_invalid_identifier(self) -> None:
        """A table identifier without a dot separator must raise ValueError."""
        df = _make_row_df()
        with pytest.raises(ValueError, match="namespace"):
            write_to_iceberg(df, _NESSIE_URI, "nodotintablename", _WAREHOUSE)

    @patch("bp_ecg_raw_extractor.writer.iceberg_writer.load_catalog")
    def test_empty_dataframe_still_calls_append(
        self, mock_load_catalog: MagicMock
    ) -> None:
        """An empty DataFrame must still trigger table.append."""
        mock_catalog = MagicMock()
        mock_catalog.table_exists.return_value = True
        mock_table = MagicMock()
        mock_catalog.load_table.return_value = mock_table
        mock_load_catalog.return_value = mock_catalog

        df = make_empty_dataframe()
        write_to_iceberg(df, _NESSIE_URI, _TABLE_ID, _WAREHOUSE)

        mock_table.append.assert_called_once()

    @patch("bp_ecg_raw_extractor.writer.iceberg_writer.load_catalog")
    def test_namespace_already_exists_does_not_raise(
        self, mock_load_catalog: MagicMock
    ) -> None:
        """create_namespace raising an exception must be silently ignored."""
        mock_catalog = MagicMock()
        mock_catalog.table_exists.return_value = False
        mock_catalog.create_namespace.side_effect = Exception("already exists")
        mock_table = MagicMock()
        mock_catalog.create_table.return_value = mock_table
        mock_load_catalog.return_value = mock_catalog

        df = _make_row_df()
        # Must not raise
        write_to_iceberg(df, _NESSIE_URI, _TABLE_ID, _WAREHOUSE)

        mock_table.append.assert_called_once()

    @patch("bp_ecg_raw_extractor.writer.iceberg_writer.load_catalog")
    def test_warehouse_location_used_in_table_creation(
        self, mock_load_catalog: MagicMock
    ) -> None:
        """The warehouse_location must be passed to create_table."""
        mock_catalog = MagicMock()
        mock_catalog.table_exists.return_value = False
        mock_table = MagicMock()
        mock_catalog.create_table.return_value = mock_table
        mock_load_catalog.return_value = mock_catalog

        warehouse = "s3://my-custom-lake"
        df = _make_row_df()
        write_to_iceberg(df, _NESSIE_URI, _TABLE_ID, warehouse)

        call_kwargs = mock_catalog.create_table.call_args
        location_arg: str = call_kwargs.kwargs.get("location", "")
        assert location_arg.startswith(warehouse)

    @patch("bp_ecg_raw_extractor.writer.iceberg_writer.load_catalog")
    def test_load_catalog_called_with_correct_config(
        self, mock_load_catalog: MagicMock
    ) -> None:
        """load_catalog must be called with the REST type and the nessie_uri."""
        mock_catalog = MagicMock()
        mock_catalog.table_exists.return_value = True
        mock_table = MagicMock()
        mock_catalog.load_table.return_value = mock_table
        mock_load_catalog.return_value = mock_catalog

        df = _make_row_df()
        write_to_iceberg(df, _NESSIE_URI, _TABLE_ID, _WAREHOUSE)

        mock_load_catalog.assert_called_once_with(
            "nessie",
            type="rest",
            uri=_NESSIE_URI,
            **{"py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO"},
        )

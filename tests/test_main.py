"""Tests for main.py.

Covers :func:`configure_logging` and :func:`process_file`.  The async
polling loop (``run``) and ``main`` entry point are not tested here because
they require a live event loop with network I/O; they are exercised by the
integration tests.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, patch

import polars as pl
import pytest
from PIL import Image

from bp_ecg_raw_extractor.config import Settings
from bp_ecg_raw_extractor.main import configure_logging, process_file
from bp_ecg_raw_extractor.schema.dataframe import SCHEMA

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_MINIO_ENDPOINT = "http://localhost:9000"


@pytest.fixture()
def settings(tmp_path: Any) -> Settings:
    """Minimal Settings object suitable for unit tests."""
    return Settings(
        minio_endpoint=_MINIO_ENDPOINT,
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        environment="dev",
        ocr_workers=1,
        async_concurrency=2,
    )


def _make_compressed_png() -> bytes:
    """Create a tiny zstd-compressed PNG for use as test payload."""
    import zstandard

    img = Image.new("RGB", (100, 50), color=(0, 128, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    cctx = zstandard.ZstdCompressor(level=1)
    return cctx.compress(buf.getvalue())


_SAMPLE_METADATA: dict[str, str] = {
    "content-hash": "deadbeef",
    "source-zip": "exam_001.zip",
    "original-path": "/data/exam_001.zip",
    "image-width": "100",
    "image-height": "50",
    "compression-ratio": "2.5",
    "processing-duration-ms": "120",
    "pdf-producer": "reportlab",
    "pdf-creator": "pytest",
    "watcher-version": "0.1.0",
}


# ---------------------------------------------------------------------------
# configure_logging tests
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    """Tests for :func:`configure_logging`."""

    def test_dev_environment_does_not_raise(self, settings: Settings) -> None:
        """configure_logging must not raise in dev mode."""
        settings.environment = "dev"
        configure_logging(settings)  # must not raise

    def test_prod_environment_does_not_raise(self, settings: Settings) -> None:
        """configure_logging must not raise in production mode."""
        settings.environment = "production"
        configure_logging(settings)

    def test_repeated_calls_do_not_raise(self, settings: Settings) -> None:
        """Calling configure_logging multiple times must not raise."""
        configure_logging(settings)
        configure_logging(settings)


# ---------------------------------------------------------------------------
# process_file tests
# ---------------------------------------------------------------------------


class TestProcessFile:
    """Tests for :func:`process_file`."""

    async def test_returns_dataframe_on_success(self, settings: Settings) -> None:
        """A valid file must produce a single-row DataFrame with the correct schema."""
        compressed = _make_compressed_png()
        process_pool: ProcessPoolExecutor = ProcessPoolExecutor(max_workers=1)
        semaphore = asyncio.Semaphore(1)

        mock_loop = AsyncMock()
        mock_loop.run_in_executor = AsyncMock(return_value=("ocr text", 0.95))

        with (
            patch(
                "bp_ecg_raw_extractor.main.download_object",
                new=AsyncMock(return_value=(compressed, _SAMPLE_METADATA)),
            ),
            patch(
                "bp_ecg_raw_extractor.main.asyncio.get_running_loop",
                return_value=mock_loop,
            ),
        ):
            result = await process_file(
                key="some/key.png.zst",
                metadata=_SAMPLE_METADATA,
                settings=settings,
                process_pool=process_pool,
                semaphore=semaphore,
            )

        process_pool.shutdown(wait=False)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        assert set(result.columns) == set(SCHEMA.keys())

    async def test_dataframe_values_match_metadata(self, settings: Settings) -> None:
        """Row values must be derived from the object metadata."""
        compressed = _make_compressed_png()
        process_pool = ProcessPoolExecutor(max_workers=1)
        semaphore = asyncio.Semaphore(1)

        mock_loop = AsyncMock()
        mock_loop.run_in_executor = AsyncMock(return_value=("hello world", 0.88))

        with (
            patch(
                "bp_ecg_raw_extractor.main.download_object",
                new=AsyncMock(return_value=(compressed, _SAMPLE_METADATA)),
            ),
            patch(
                "bp_ecg_raw_extractor.main.asyncio.get_running_loop",
                return_value=mock_loop,
            ),
        ):
            result = await process_file(
                key="some/key.png.zst",
                metadata=_SAMPLE_METADATA,
                settings=settings,
                process_pool=process_pool,
                semaphore=semaphore,
            )

        process_pool.shutdown(wait=False)
        assert result is not None
        row = result.to_dicts()[0]
        assert row["file_hash"] == "deadbeef"
        assert row["source_zip"] == "exam_001.zip"
        assert row["ocr_raw_text"] == "hello world"
        assert abs(row["ocr_confidence_mean"] - 0.88) < 1e-6

    async def test_returns_none_on_download_error(self, settings: Settings) -> None:
        """When download_object raises, process_file must return None."""
        process_pool = ProcessPoolExecutor(max_workers=1)
        semaphore = asyncio.Semaphore(1)

        with patch(
            "bp_ecg_raw_extractor.main.download_object",
            new=AsyncMock(side_effect=RuntimeError("network error")),
        ):
            result = await process_file(
                key="bad/key.png.zst",
                metadata={},
                settings=settings,
                process_pool=process_pool,
                semaphore=semaphore,
            )

        process_pool.shutdown(wait=False)
        assert result is None

    async def test_returns_none_on_corrupt_compressed_bytes(
        self, settings: Settings
    ) -> None:
        """Corrupt compressed bytes must cause process_file to return None."""
        process_pool = ProcessPoolExecutor(max_workers=1)
        semaphore = asyncio.Semaphore(1)

        with patch(
            "bp_ecg_raw_extractor.main.download_object",
            new=AsyncMock(return_value=(b"not zstd data", _SAMPLE_METADATA)),
        ):
            result = await process_file(
                key="bad/key.png.zst",
                metadata=_SAMPLE_METADATA,
                settings=settings,
                process_pool=process_pool,
                semaphore=semaphore,
            )

        process_pool.shutdown(wait=False)
        assert result is None

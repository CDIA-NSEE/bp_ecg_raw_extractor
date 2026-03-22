"""Tests for storage/minio_client.py.

All S3 interactions are mocked via ``unittest.mock`` — no real MINIO instance
or moto is needed because aioboto3 uses async context managers that are easy
to stub with ``AsyncMock``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from bp_ecg_raw_extractor.storage.minio_client import (
    download_object,
    list_unprocessed_objects,
)

_ENDPOINT = "http://localhost:9000"
_KEY = "minioadmin"
_SECRET = "minioadmin"
_BUCKET = "bp-ecg-dev-images"


def _make_s3_client_cm(mock_s3: Any) -> Any:
    """Return an async context-manager mock wrapping *mock_s3*."""
    cm: AsyncMock = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_s3)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# list_unprocessed_objects
# ---------------------------------------------------------------------------


class TestListUnprocessedObjects:
    """Tests for :func:`list_unprocessed_objects`."""

    async def test_empty_bucket_returns_empty_list(self) -> None:
        """When the bucket has no objects, an empty list is returned."""
        mock_s3 = AsyncMock()
        mock_paginator = MagicMock()

        async def empty_pages(**_kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:  # type: ignore[no-untyped-def]
            yield {"Contents": []}

        mock_paginator.paginate = MagicMock(return_value=empty_pages())
        mock_s3.get_paginator = MagicMock(return_value=mock_paginator)

        mock_session = MagicMock()
        mock_session.client = MagicMock(return_value=_make_s3_client_cm(mock_s3))

        with patch(
            "bp_ecg_raw_extractor.storage.minio_client.aioboto3.Session",
            return_value=mock_session,
        ):
            result = await list_unprocessed_objects(_ENDPOINT, _KEY, _SECRET, _BUCKET)

        assert result == []

    async def test_objects_are_returned(self) -> None:
        """Objects found in the bucket are returned as metadata dicts."""
        mock_s3 = AsyncMock()
        mock_paginator = MagicMock()

        async def one_page(**_kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:  # type: ignore[no-untyped-def]
            yield {
                "Contents": [
                    {
                        "Key": "2024/01/01/abc.png.zst",
                        "Size": 12345,
                        "LastModified": "2024-01-01T00:00:00Z",
                        "ETag": '"abc123"',
                    }
                ]
            }

        mock_paginator.paginate = MagicMock(return_value=one_page())
        mock_s3.get_paginator = MagicMock(return_value=mock_paginator)

        mock_session = MagicMock()
        mock_session.client = MagicMock(return_value=_make_s3_client_cm(mock_s3))

        with patch(
            "bp_ecg_raw_extractor.storage.minio_client.aioboto3.Session",
            return_value=mock_session,
        ):
            result = await list_unprocessed_objects(_ENDPOINT, _KEY, _SECRET, _BUCKET)

        assert len(result) == 1
        assert result[0]["key"] == "2024/01/01/abc.png.zst"
        assert result[0]["size"] == 12345
        assert result[0]["etag"] == "abc123"

    async def test_prefix_is_forwarded(self) -> None:
        """When *prefix* is given, it is passed to paginate."""
        mock_s3 = AsyncMock()
        mock_paginator = MagicMock()
        captured_kwargs: dict[str, Any] = {}

        async def capture_pages(**kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
            captured_kwargs.update(kwargs)
            yield {}

        # Assign the function (not a pre-called instance) so kwargs are captured
        mock_paginator.paginate = capture_pages
        mock_s3.get_paginator = MagicMock(return_value=mock_paginator)

        mock_session = MagicMock()
        mock_session.client = MagicMock(return_value=_make_s3_client_cm(mock_s3))

        with patch(
            "bp_ecg_raw_extractor.storage.minio_client.aioboto3.Session",
            return_value=mock_session,
        ):
            await list_unprocessed_objects(
                _ENDPOINT, _KEY, _SECRET, _BUCKET, prefix="2024/"
            )

        assert captured_kwargs.get("Prefix") == "2024/"

    async def test_no_prefix_omits_prefix_key(self) -> None:
        """When *prefix* is empty, 'Prefix' must not appear in paginate kwargs."""
        mock_s3 = AsyncMock()
        mock_paginator = MagicMock()
        captured_kwargs: dict[str, Any] = {}

        async def capture_pages(**kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
            captured_kwargs.update(kwargs)
            yield {}

        mock_paginator.paginate = capture_pages
        mock_s3.get_paginator = MagicMock(return_value=mock_paginator)

        mock_session = MagicMock()
        mock_session.client = MagicMock(return_value=_make_s3_client_cm(mock_s3))

        with patch(
            "bp_ecg_raw_extractor.storage.minio_client.aioboto3.Session",
            return_value=mock_session,
        ):
            await list_unprocessed_objects(_ENDPOINT, _KEY, _SECRET, _BUCKET)

        assert "Prefix" not in captured_kwargs


# ---------------------------------------------------------------------------
# download_object
# ---------------------------------------------------------------------------


class TestDownloadObject:
    """Tests for :func:`download_object`."""

    async def test_returns_body_and_metadata(self) -> None:
        """Body bytes and metadata dict are both returned."""
        payload = b"zstd compressed data"
        meta = {"content-hash": "abc", "source-zip": "exam.zip"}

        async def fake_chunks(chunk_size: int = 65536) -> AsyncGenerator[bytes, None]:  # type: ignore[no-untyped-def]
            yield payload

        mock_body = MagicMock()
        mock_body.iter_chunks = fake_chunks

        mock_s3 = AsyncMock()
        mock_s3.get_object = AsyncMock(
            return_value={"Body": mock_body, "Metadata": meta}
        )

        mock_session = MagicMock()
        mock_session.client = MagicMock(return_value=_make_s3_client_cm(mock_s3))

        with patch(
            "bp_ecg_raw_extractor.storage.minio_client.aioboto3.Session",
            return_value=mock_session,
        ):
            body, metadata = await download_object(
                _ENDPOINT, _KEY, _SECRET, _BUCKET, "some/key.png.zst"
            )

        assert body == payload
        assert metadata == meta

    async def test_empty_metadata_returns_empty_dict(self) -> None:
        """When the object has no metadata, an empty dict is returned."""

        async def fake_chunks(chunk_size: int = 65536) -> AsyncGenerator[bytes, None]:  # type: ignore[no-untyped-def]
            yield b"data"

        mock_body = MagicMock()
        mock_body.iter_chunks = fake_chunks

        mock_s3 = AsyncMock()
        mock_s3.get_object = AsyncMock(return_value={"Body": mock_body, "Metadata": {}})

        mock_session = MagicMock()
        mock_session.client = MagicMock(return_value=_make_s3_client_cm(mock_s3))

        with patch(
            "bp_ecg_raw_extractor.storage.minio_client.aioboto3.Session",
            return_value=mock_session,
        ):
            _, metadata = await download_object(
                _ENDPOINT, _KEY, _SECRET, _BUCKET, "key"
            )

        assert metadata == {}

    async def test_multiple_chunks_are_joined(self) -> None:
        """Multiple streaming chunks are concatenated into a single bytes value."""

        async def multi_chunks(chunk_size: int = 65536) -> AsyncGenerator[bytes, None]:  # type: ignore[no-untyped-def]
            yield b"hello "
            yield b"world"

        mock_body = MagicMock()
        mock_body.iter_chunks = multi_chunks

        mock_s3 = AsyncMock()
        mock_s3.get_object = AsyncMock(return_value={"Body": mock_body, "Metadata": {}})

        mock_session = MagicMock()
        mock_session.client = MagicMock(return_value=_make_s3_client_cm(mock_s3))

        with patch(
            "bp_ecg_raw_extractor.storage.minio_client.aioboto3.Session",
            return_value=mock_session,
        ):
            body, _ = await download_object(_ENDPOINT, _KEY, _SECRET, _BUCKET, "key")

        assert body == b"hello world"

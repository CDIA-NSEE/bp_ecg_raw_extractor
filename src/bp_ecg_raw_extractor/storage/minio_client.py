"""Async S3/MINIO client utilities using aioboto3.

All operations are asynchronous and use streaming where possible.
``aiobotocore`` is NOT imported directly — it is managed transitively by
``aioboto3`` and must not be added as a direct dependency.
"""

from __future__ import annotations

from typing import Any

import aioboto3
import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


async def list_unprocessed_objects(
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """List objects in a MINIO bucket and return lightweight metadata dicts.

    Args:
        endpoint_url: S3-compatible endpoint (e.g. ``"http://localhost:9000"``).
        access_key: S3 access key id.
        secret_key: S3 secret access key.
        bucket: Name of the bucket to list.
        prefix: Optional key prefix filter.

    Returns:
        A list of dicts, each containing ``"key"``, ``"size"``,
        ``"last_modified"``, and ``"etag"`` fields.  Returns an empty list
        when the bucket contains no matching objects.
    """
    session: aioboto3.Session = aioboto3.Session()
    results: list[dict[str, Any]] = []

    async with session.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ) as s3:
        paginator = s3.get_paginator("list_objects_v2")
        kwargs: dict[str, Any] = {"Bucket": bucket}
        if prefix:
            kwargs["Prefix"] = prefix

        async for page in paginator.paginate(**kwargs):
            for obj in page.get("Contents", []):
                results.append(
                    {
                        "key": obj["Key"],
                        "size": obj.get("Size", 0),
                        "last_modified": obj.get("LastModified"),
                        "etag": obj.get("ETag", "").strip('"'),
                    }
                )

    logger.debug(
        "listed_objects",
        bucket=bucket,
        prefix=prefix,
        count=len(results),
    )
    return results


async def download_object(
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    key: str,
) -> tuple[bytes, dict[str, str]]:
    """Download an object from MINIO and return its bytes and metadata.

    Uses chunked streaming (64 KiB chunks) to avoid loading the entire
    response body into memory at once before assembling the final bytes.

    Args:
        endpoint_url: S3-compatible endpoint URL.
        access_key: S3 access key id.
        secret_key: S3 secret access key.
        bucket: Bucket name.
        key: Object key.

    Returns:
        A ``(body_bytes, metadata)`` tuple where *metadata* is the dict of
        custom object metadata stored by the watcher (string values only).

    Raises:
        Exception: Propagated from ``aioboto3`` / ``botocore`` on network or
            permission errors.
    """
    session: aioboto3.Session = aioboto3.Session()

    async with session.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ) as s3:
        response: dict[str, Any] = await s3.get_object(Bucket=bucket, Key=key)
        metadata: dict[str, str] = response.get("Metadata", {})

        chunks: list[bytes] = []
        async for chunk in response["Body"].iter_chunks(chunk_size=65536):
            chunks.append(chunk)

        body: bytes = b"".join(chunks)

    logger.debug(
        "downloaded_object",
        bucket=bucket,
        key=key,
        size_bytes=len(body),
    )
    return body, metadata

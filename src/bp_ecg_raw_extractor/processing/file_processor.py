"""Per-file processing pipeline for bp_ecg_raw_extractor.

Encapsulates the complete extraction workflow for a single ``.png.zst``
object:

    download → decompress → OCR → download ZIP → parse PDF → write Iceberg

Provides:
- ``_process_once``: single async attempt, no retry.
- ``process_file``: async with semaphore and exponential back-off retries.
- ``process_file_sync``: synchronous wrapper for use in Airflow ``@task``.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import polars as pl
import structlog

from bp_ecg_raw_extractor.config import Settings
from bp_ecg_raw_extractor.decompressor.zstd_reader import decompress_to_image
from bp_ecg_raw_extractor.image_proc.roi import extract_roi_bytes
from bp_ecg_raw_extractor.metrics import (
    extraction_duration_sec,
    files_extracted_total,
    iceberg_writes_total,
    ocr_confidence,
)
from bp_ecg_raw_extractor.ocr.paddle_ocr import get_ocr_text_and_confidence
from bp_ecg_raw_extractor.pdf_parser.pdfplumber_parser import extract_text
from bp_ecg_raw_extractor.schema.dataframe import SCHEMA, validate_row
from bp_ecg_raw_extractor.storage.minio_client import download_object
from bp_ecg_raw_extractor.writer.iceberg_writer import record_exists, write_to_iceberg

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


async def _process_once(
    key: str,
    settings: Settings,
    process_pool: ProcessPoolExecutor,
) -> pl.DataFrame:
    """Single processing attempt for one image file (no retry logic).

    Args:
        key: S3 object key of the ``.png.zst`` file.
        settings: Application settings.
        process_pool: Shared process pool for CPU-bound OCR work.

    Returns:
        A single-row :class:`polars.DataFrame` on success.

    Raises:
        Exception: Any unhandled error from any pipeline stage.
    """
    start_ts: datetime = datetime.now(UTC)

    # 1. Download compressed image and its metadata
    compressed_bytes: bytes
    img_metadata: dict[str, str]
    compressed_bytes, img_metadata = await download_object(
        endpoint_url=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.bucket_images,
        key=key,
    )

    structlog.contextvars.bind_contextvars(
        file_hash=img_metadata.get("content-hash", key[:12]),
        source_zip=img_metadata.get("source-zip", ""),
        environment=settings.environment,
    )

    # Persistent dedup: query Iceberg before any CPU work.
    # run_in_executor keeps the asyncio event loop non-blocking.
    # Returns False when the table doesn't exist yet (first run).
    _file_hash: str = img_metadata.get("content-hash", "")
    if _file_hash:
        _loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        _already_written: bool = await _loop.run_in_executor(
            None,
            record_exists,
            _file_hash,
            settings.nessie_uri,
            settings.iceberg_table_name,
            settings.minio_endpoint,
            settings.minio_access_key,
            settings.minio_secret_key,
        )
        if _already_written:
            logger.info("duplicate_skipped", file_hash=_file_hash, key=key)
            return pl.DataFrame(schema=SCHEMA)

    # 2. Decompress to PIL Image
    pil_image = decompress_to_image(compressed_bytes)
    img_w: int
    img_h: int
    img_w, img_h = pil_image.size

    # 3. Extract OCR ROI and run OCR in process pool (CPU-bound)
    roi_bytes: bytes = extract_roi_bytes(pil_image, settings.ocr_region)
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    ocr_text: str
    ocr_confidence_mean: float
    ocr_text, ocr_confidence_mean = await loop.run_in_executor(
        process_pool,
        get_ocr_text_and_confidence,
        roi_bytes,
        settings.paddle_use_gpu,
        settings.paddle_lang,
    )

    # 4. Download original ZIP from intake bucket and parse PDF
    source_intake_key: str = img_metadata.get("source-intake-key", "")
    pdf_raw_text: str = ""
    if source_intake_key:
        try:
            zip_bytes: bytes
            zip_bytes, _ = await download_object(
                endpoint_url=settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                bucket=settings.bucket_intake,
                key=source_intake_key,
            )
            # Extract PDF from ZIP and parse its text
            with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
                pdf_names: list[str] = [
                    n for n in zf.namelist() if n.lower().endswith(".pdf")
                ]
                if pdf_names:
                    pdf_raw_text = extract_text(zf.read(pdf_names[0]))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pdf_extraction_skipped",
                intake_key=source_intake_key,
                error=str(exc),
            )
    else:
        logger.warning("no_source_intake_key_in_metadata", object_key=key)

    # 5. Assemble DataFrame row
    processing_duration_ms: int = int(
        (datetime.now(UTC) - start_ts).total_seconds() * 1000
    )
    row: dict[str, Any] = {
        "file_hash": img_metadata.get("content-hash", ""),
        "processed_at": datetime.now(UTC),
        "original_path": img_metadata.get("original-path", ""),
        "source_zip": img_metadata.get("source-zip", ""),
        "image_width": int(img_metadata.get("image-width", str(img_w))),
        "image_height": int(img_metadata.get("image-height", str(img_h))),
        "compression_ratio": float(img_metadata.get("compression-ratio", "0.0")),
        "processing_duration_ms": processing_duration_ms,
        "pdf_producer": img_metadata.get("pdf-producer", "unknown"),
        "pdf_creator": img_metadata.get("pdf-creator", "unknown"),
        "ocr_raw_text": ocr_text,
        "ocr_confidence_mean": ocr_confidence_mean,
        "pdf_raw_text": pdf_raw_text,
        "watcher_version": img_metadata.get("watcher-version", "unknown"),
    }

    validate_row(row)
    df: pl.DataFrame = pl.DataFrame([row], schema=SCHEMA)

    # 6. Write row to Iceberg immediately
    write_to_iceberg(
        df=df,
        nessie_uri=settings.nessie_uri,
        table_identifier=settings.iceberg_table_name,
        warehouse_location=f"s3://{settings.bucket_lake}",
        s3_endpoint=settings.minio_endpoint,
        s3_access_key=settings.minio_access_key,
        s3_secret_key=settings.minio_secret_key,
    )

    # 7. Update Prometheus metrics
    elapsed_s: float = (datetime.now(UTC) - start_ts).total_seconds()
    files_extracted_total.inc()
    ocr_confidence.observe(ocr_confidence_mean)
    iceberg_writes_total.inc(len(df))
    extraction_duration_sec.observe(elapsed_s)

    logger.info(
        "file_extracted_ok",
        key=key,
        elapsed_s=round(elapsed_s, 3),
        ocr_confidence=round(ocr_confidence_mean, 4),
    )
    return df


async def process_file(
    key: str,
    settings: Settings,
    process_pool: ProcessPoolExecutor,
    semaphore: asyncio.Semaphore,
) -> pl.DataFrame | None:
    """Process a single ``.png.zst`` file with retries and back-off.

    Runs under *semaphore* to bound the number of concurrent downloads.
    Retries up to 3 times with exponential back-off (2 s → 4 s) before
    giving up and returning ``None``.

    Args:
        key: S3 object key of the ``.png.zst`` file.
        settings: Application settings.
        process_pool: Shared process pool for CPU-bound OCR work.
        semaphore: Semaphore limiting parallel file downloads.

    Returns:
        A single-row :class:`polars.DataFrame` on success, or ``None``
        after all retries are exhausted.
    """
    async with semaphore:
        last_exc: BaseException | None = None
        for attempt in range(1, 4):
            try:
                return await _process_once(key, settings, process_pool)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 3:
                    wait_s: int = 2**attempt
                    logger.warning(
                        "processing_attempt_failed",
                        key=key,
                        attempt=attempt,
                        wait_s=wait_s,
                        error=str(exc),
                    )
                    await asyncio.sleep(wait_s)

        logger.error(
            "processing_failed_all_retries",
            key=key,
            error=str(last_exc),
        )
        structlog.contextvars.clear_contextvars()
        return None


def process_file_sync(key: str, settings: Settings) -> dict[str, Any]:
    """Synchronous entry point for processing a single image file.

    Creates an isolated :class:`~concurrent.futures.ProcessPoolExecutor`
    (spawn context) and a fresh asyncio event loop.  Designed for use in
    Airflow ``@task`` workers which run in their own OS process.

    Args:
        key: S3 object key of the ``.png.zst`` file to process.
        settings: Application settings.

    Returns:
        A dict with ``"status": "ok"``, ``"key"``, and ``"rows"`` on success.

    Raises:
        RuntimeError: When all retry attempts fail.
    """
    mp_ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx) as pool:
        sem: asyncio.Semaphore = asyncio.Semaphore(1)
        df: pl.DataFrame | None = asyncio.run(
            process_file(key, settings, pool, sem)
        )

    if df is None:
        raise RuntimeError(f"Processing failed after all retries: {key}")

    return {"status": "ok", "key": key, "rows": len(df)}

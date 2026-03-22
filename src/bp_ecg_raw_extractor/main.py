"""Async entry point for bp_ecg_raw_extractor.

Polls the MINIO images bucket for ``.png.zst`` files, decompresses them,
extracts an OCR region, runs PaddleOCR in a ``ProcessPoolExecutor`` worker,
assembles a Polars DataFrame row, and appends batches to an Iceberg table.

Usage::

    python -m bp_ecg_raw_extractor.main
"""

from __future__ import annotations

import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import polars as pl
import structlog
from prometheus_client import start_http_server

from bp_ecg_raw_extractor.config import Settings
from bp_ecg_raw_extractor.processing.file_processor import process_file
from bp_ecg_raw_extractor.storage.minio_client import list_unprocessed_objects

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def configure_logging(settings: Settings) -> None:
    """Configure structlog for JSON output in production or console in dev.

    Args:
        settings: Application settings.
    """
    renderer: Any
    if settings.environment == "dev":
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------


async def run(settings: Settings) -> None:
    """Continuously poll MINIO for new files and write results to Iceberg.

    Each :func:`~bp_ecg_raw_extractor.processing.file_processor.process_file`
    call handles validation, OCR, PDF parsing, and Iceberg writing
    atomically for a single object.  Results are gathered with
    ``return_exceptions=True`` so one failure does not abort the whole batch.

    Args:
        settings: Application settings.
    """
    configure_logging(settings)
    start_http_server(port=settings.metrics_port)
    logger.info(
        "extractor_starting",
        environment=settings.environment,
        metrics_port=settings.metrics_port,
    )

    process_pool: ProcessPoolExecutor = ProcessPoolExecutor(
        max_workers=settings.ocr_workers,
        mp_context=multiprocessing.get_context("spawn"),
    )
    semaphore: asyncio.Semaphore = asyncio.Semaphore(settings.async_concurrency)

    try:
        while True:
            objects: list[dict[str, Any]] = await list_unprocessed_objects(
                endpoint_url=settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                bucket=settings.bucket_images,
            )

            if objects:
                logger.info("new_objects_found", count=len(objects))
                tasks: list[asyncio.Task[pl.DataFrame | None]] = [
                    asyncio.create_task(
                        process_file(
                            key=obj["key"],
                            settings=settings,
                            process_pool=process_pool,
                            semaphore=semaphore,
                        )
                    )
                    for obj in objects
                ]
                raw_results: list[
                    pl.DataFrame | None | BaseException
                ] = await asyncio.gather(*tasks, return_exceptions=True)

                successes: list[pl.DataFrame] = [
                    r for r in raw_results if isinstance(r, pl.DataFrame)
                ]
                total_rows: int = sum(len(df) for df in successes)
                logger.info(
                    "batch_complete",
                    total=len(objects),
                    succeeded=len(successes),
                    rows=total_rows,
                )
            else:
                logger.debug("no_new_objects", bucket=settings.bucket_images)

            await asyncio.sleep(60)
    finally:
        process_pool.shutdown(wait=True)
        logger.info("extractor_stopped")


async def run_single(key: str, settings: Settings) -> None:
    """Process a single S3 object key and exit — used by Airflow DockerOperator.

    Runs the same pipeline as :func:`run` but for exactly one file, then
    shuts down the process pool and returns.  Exits with code 1 on failure
    so that Airflow marks the task as failed.

    Args:
        key: S3 object key of the ``.png.zst`` file to process.
        settings: Application settings.
    """
    configure_logging(settings)
    logger.info("single_key_mode_starting", key=key)

    process_pool: ProcessPoolExecutor = ProcessPoolExecutor(
        max_workers=settings.ocr_workers,
        mp_context=multiprocessing.get_context("spawn"),
    )
    semaphore: asyncio.Semaphore = asyncio.Semaphore(1)

    try:
        result = await process_file(
            key=key,
            settings=settings,
            process_pool=process_pool,
            semaphore=semaphore,
        )
        if result is None:
            logger.error("single_key_failed", key=key)
            raise SystemExit(1)
        logger.info("single_key_done", key=key, rows=len(result))
    finally:
        process_pool.shutdown(wait=True)


def main() -> None:
    """CLI entry point — parse settings and start the run loop or process a single key.

    When ``--key`` is supplied (e.g. by Airflow DockerOperator), the process
    handles exactly that one file and exits.  Without ``--key`` the service
    runs in continuous polling mode.
    """
    import argparse

    parser = argparse.ArgumentParser(description="bp_ecg raw extractor")
    parser.add_argument(
        "--key",
        default=None,
        help="S3 object key to process (single-file mode for Airflow DockerOperator)",
    )
    args = parser.parse_args()

    settings: Settings = Settings()  # type: ignore[call-arg]
    if args.key:
        asyncio.run(run_single(args.key, settings))
    else:
        asyncio.run(run(settings))


if __name__ == "__main__":
    main()

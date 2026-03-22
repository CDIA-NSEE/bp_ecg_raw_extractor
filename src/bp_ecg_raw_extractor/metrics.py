"""Prometheus metric singletons for bp_ecg_raw_extractor."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

files_extracted_total: Counter = Counter(
    "extractor_files_extracted_total",
    "Files extracted successfully",
)

ocr_confidence: Histogram = Histogram(
    "extractor_ocr_confidence",
    "OCR confidence score per file",
    buckets=(0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

extraction_duration_sec: Histogram = Histogram(
    "extractor_duration_seconds",
    "End-to-end extraction time",
    buckets=(1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

iceberg_writes_total: Counter = Counter(
    "extractor_iceberg_writes_total",
    "Rows written to Iceberg",
)

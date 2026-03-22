"""bp_ecg_raw_extractor — consumes .png.zst images from MINIO, runs OCR + PDF parsing,
and writes Parquet/Iceberg tables for the CDIA-NSEE bp_ecg pipeline."""

__version__ = "0.1.0"

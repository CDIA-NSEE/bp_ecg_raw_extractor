"""Iceberg table writer using pyiceberg with a Nessie REST catalog.

Writes Polars DataFrames to an Iceberg table via the pyiceberg REST catalog
client.  The table is created on first write with a day-granularity partition
on the ``processed_at`` timestamp column.
"""

from __future__ import annotations

import contextlib

import polars as pl
import pyarrow as pa
import structlog
from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.expressions import EqualTo
from pyiceberg.table import Table
from pyiceberg.transforms import DayTransform

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def build_catalog_config(
    nessie_uri: str,
    s3_endpoint: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
) -> dict[str, str]:
    """Build pyiceberg catalog configuration for a Nessie REST endpoint.

    S3 credentials are included when provided so that PyIceberg can read
    and write Parquet files directly from/to MINIO.

    Args:
        nessie_uri: Full URI of the Nessie REST catalog
            (e.g. ``"http://nessie:19120/iceberg"``).
        s3_endpoint: S3-compatible endpoint URL (e.g. MINIO).  Omit for AWS.
        s3_access_key: S3 access key ID.  Omit when using IAM roles.
        s3_secret_key: S3 secret access key.  Omit when using IAM roles.

    Returns:
        A dict suitable for passing as ``**kwargs`` to
        :func:`pyiceberg.catalog.load_catalog`.
    """
    config: dict[str, str] = {
        "type": "rest",
        "uri": nessie_uri,
        "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
    }
    if s3_endpoint:
        config["s3.endpoint"] = s3_endpoint
        config["s3.path-style-access"] = "true"
        config["s3.region"] = "us-east-1"
    if s3_access_key:
        config["s3.access-key-id"] = s3_access_key
    if s3_secret_key:
        config["s3.secret-access-key"] = s3_secret_key
    return config


def record_exists(
    file_hash: str,
    nessie_uri: str,
    table_identifier: str,
    s3_endpoint: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
) -> bool:
    """Return ``True`` if *file_hash* already exists in the Iceberg table.

    Provides persistent deduplication that survives container restarts —
    unlike the in-memory ``ProcessedKeyStore``.  Returns ``False`` when the
    table does not yet exist (first run), so it is safe to call before the
    table is created.

    Args:
        file_hash: BLAKE3 hex digest to look up in the ``file_hash`` column.
        nessie_uri: Nessie REST catalog URI.
        table_identifier: Fully-qualified table name (``"<ns>.<table>"``).
        s3_endpoint: S3-compatible endpoint URL.
        s3_access_key: S3 access key ID.
        s3_secret_key: S3 secret access key.

    Returns:
        ``True`` if a matching row exists; ``False`` otherwise.
    """
    catalog: Catalog = load_catalog(
        "nessie",
        **build_catalog_config(nessie_uri, s3_endpoint, s3_access_key, s3_secret_key),
    )
    if not catalog.table_exists(table_identifier):
        return False
    table: Table = catalog.load_table(table_identifier)
    scan_result = table.scan(row_filter=EqualTo("file_hash", file_hash)).to_arrow()
    return len(scan_result) > 0


def write_to_iceberg(
    df: pl.DataFrame,
    nessie_uri: str,
    table_identifier: str,
    warehouse_location: str,
    s3_endpoint: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
) -> None:
    """Append a Polars DataFrame to an Iceberg table.

    If the table does not yet exist it is created with:
    - Schema inferred from the PyArrow representation of *df*.
    - A ``days(processed_at)`` partition spec.

    Args:
        df: The Polars DataFrame to append.  Must conform to the canonical
            :data:`~bp_ecg_raw_extractor.schema.dataframe.SCHEMA`.
        nessie_uri: Nessie REST catalog URI.
        table_identifier: Fully-qualified table identifier in
            ``"<namespace>.<table>"`` format (e.g. ``"bp_ecg.ecg_records_raw"``).
        warehouse_location: S3 URI of the data lake bucket
            (e.g. ``"s3://bp-ecg-dev-lake"``).

    Raises:
        ValueError: If *table_identifier* does not contain a ``"."`` separator.
    """
    if "." not in table_identifier:
        raise ValueError(
            f"table_identifier must be in '<namespace>.<table>' format, "
            f"got: {table_identifier!r}"
        )

    namespace: str
    table_name: str
    namespace, table_name = table_identifier.split(".", 1)

    catalog: Catalog = load_catalog(
        "nessie",
        **build_catalog_config(
            nessie_uri,
            s3_endpoint=s3_endpoint,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
        ),
    )
    arrow_table: pa.Table = df.to_arrow()

    table: Table
    if not catalog.table_exists(table_identifier):
        logger.info(
            "creating_iceberg_table",
            table=table_identifier,
            warehouse=warehouse_location,
        )
        # Create namespace — ignore error if it already exists.
        with contextlib.suppress(Exception):
            catalog.create_namespace(namespace)

        table = catalog.create_table(
            table_identifier,
            schema=arrow_table.schema,
            location=f"{warehouse_location}/{table_name}",
        )
        # Add day-granularity partition BEFORE the first write.
        with table.update_spec() as update:
            update.add_field(
                source_column_name="processed_at",
                transform=DayTransform(),
                partition_field_name="processed_at_day",
            )
        logger.info("iceberg_table_created", table=table_identifier)
    else:
        table = catalog.load_table(table_identifier)
        logger.debug("iceberg_table_loaded", table=table_identifier)

    table.append(arrow_table)
    logger.info(
        "iceberg_append_complete",
        table=table_identifier,
        rows=len(df),
    )

"""DAG for bp_ecg raw extractor pipeline.

Polls bp-ecg-{env}-images bucket for new .png.zst objects, then spawns
one bp_ecg_raw_extractor Docker container per file via DockerOperator.
Each container runs ``python -m bp_ecg_raw_extractor.main --key <key>``,
processes the file end-to-end, and exits.  Non-zero exit = Airflow failure.

Uses Airflow 3 TaskFlow API + DockerOperator (apache-airflow-providers-docker).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import boto3
from airflow.models import Connection
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import dag, task

log = logging.getLogger(__name__)

_EXTRACTOR_IMAGE = "ghcr.io/cdia-nsee/bp_ecg_raw_extractor:latest"
_NETWORK = "bp-ecg-lakehouse-net"


def _minio_params() -> tuple[str, str, str]:
    """Return (endpoint_url, access_key, secret_key) from minio_default connection."""
    conn: Connection = Connection.get_connection_from_secrets("minio_default")
    return f"http://{conn.host}:{conn.port}", conn.login or "", conn.password or ""


@dag(
    dag_id="bp_ecg_raw_extractor",
    schedule="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["bp-ecg", "raw-extractor"],
    doc_md="""
    # bp_ecg Raw Extractor DAG

    Polls `bp-ecg-dev-images` for new `.png.zst` objects in the data interval,
    then spawns one `bp_ecg_raw_extractor` Docker container per file.
    Each container runs `python -m bp_ecg_raw_extractor.main --key <key>`,
    processes the file end-to-end, and exits.  Non-zero exit = Airflow task failure.

    **Connection required:** `minio_default` (AWS/S3 type)
    - Host: MinIO endpoint hostname
    - Port: MinIO API port (9000)
    - Login: access key
    - Password: secret key
    """,
)
def bp_ecg_raw_extractor() -> None:
    """Define the bp_ecg raw extractor DAG."""

    @task
    def list_new_objects(
        data_interval_start: datetime | None = None,
        data_interval_end: datetime | None = None,
    ) -> list[str]:
        """List .png.zst keys added to the images bucket during the data interval."""
        endpoint, access_key, secret_key = _minio_params()

        try:
            from airflow.models import Variable
            bucket: str = Variable.get("bp_ecg_images_bucket", default_var="bp-ecg-dev-images")
        except Exception:
            bucket = "bp-ecg-dev-images"

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        new_keys: list[str] = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if not key.endswith(".png.zst"):
                    continue
                lm: datetime = obj["LastModified"]
                if data_interval_start and data_interval_end:
                    ts = lm.replace(tzinfo=None)
                    if (
                        data_interval_start.replace(tzinfo=None)
                        <= ts
                        < data_interval_end.replace(tzinfo=None)
                    ):
                        new_keys.append(key)
                else:
                    new_keys.append(key)

        log.info(
            "Found %d new objects for interval [%s, %s)",
            len(new_keys),
            data_interval_start,
            data_interval_end,
        )
        return new_keys

    @task
    def build_commands(keys: list[str]) -> list[list[str]]:
        """Build one command list per object key — credentials never enter XCom."""
        return [["python", "-m", "bp_ecg_raw_extractor.main", "--key", k] for k in keys]

    # DAG wiring — credentials delivered via Jinja at task-run time, never serialised to XCom.
    # DockerOperator.template_fields includes 'environment', so Jinja is rendered before launch.
    DockerOperator.partial(
        task_id="process_file",
        image=_EXTRACTOR_IMAGE,
        docker_url="unix:///var/run/docker.sock",
        network_mode=_NETWORK,
        auto_remove="force",
        retries=2,
        retry_delay=timedelta(seconds=30),
        mount_tmp_dir=False,
        environment={
            # Jinja resolves from the Airflow connection at execution time — never stored in DB.
            "MINIO_ENDPOINT": "http://{{ conn.minio_default.host }}:{{ conn.minio_default.port }}",
            "MINIO_ACCESS_KEY": "{{ conn.minio_default.login }}",
            "MINIO_SECRET_KEY": "{{ conn.minio_default.password }}",
        },
    ).expand(command=build_commands(list_new_objects()))


# Instantiate the DAG
bp_ecg_raw_extractor()

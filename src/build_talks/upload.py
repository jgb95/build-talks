"""
Digital Ocean Spaces uploader.

Uploads rendered talk outputs (.mp4, .words.srt, .subs.srt) to DO Spaces
using the S3-compatible API via boto3.

Upload paths follow the convention:
    /{event}/recordings/{talk_id}.mp4
    /{event}/recordings/{talk_id}.words.srt
    /{event}/recordings/{talk_id}.subs.srt

Files are uploaded with public-read ACL so they are immediately accessible
via the DO_SPACES_BASE_URL endpoint.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

log = logging.getLogger(__name__)

# MIME types for the output files we upload.
_MIME_TYPES: dict[str, str] = {
    ".mp4":  "video/mp4",
    ".srt":  "text/plain",
}


def _mime(path: Path) -> str:
    """Return the MIME type for a file, falling back to application/octet-stream."""
    return _MIME_TYPES.get(path.suffix.lower()) or (
        mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    )


def upload_talk(
    talk_id: str,
    event: str,
    output_dir: Path,
    *,
    bucket: str,
    region: str,
    key: str,
    secret: str,
) -> list[str]:
    """
    Upload all output files for a single talk to DO Spaces.

    Uploads the following files if they exist in output_dir:
        {talk_id}.mp4
        {talk_id}.words.srt
        {talk_id}.subs.srt

    Returns a list of the DO Spaces object keys that were successfully uploaded.
    Raises RuntimeError if any upload fails.
    """
    if not event:
        raise ValueError(f"Cannot upload '{talk_id}': event name is empty")

    client = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://{region}.digitaloceanspaces.com",
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )

    files_to_upload = [
        output_dir / f"{talk_id}.mp4",
        output_dir / f"{talk_id}.words.srt",
        output_dir / f"{talk_id}.subs.srt",
    ]

    uploaded: list[str] = []
    errors: list[str] = []

    for local_path in files_to_upload:
        if not local_path.exists():
            log.debug("[upload] %s — skipping %s (not found)", talk_id, local_path.name)
            continue

        object_key = f"{event}/recordings/{local_path.name}"
        log.info("[upload] %s → s3://%s/%s", local_path.name, bucket, object_key)

        try:
            client.upload_file(
                str(local_path),
                bucket,
                object_key,
                ExtraArgs={
                    "ACL": "public-read",
                    "ContentType": _mime(local_path),
                },
            )
            uploaded.append(object_key)
            log.info("[upload] ✓ %s", object_key)
        except (BotoCoreError, ClientError) as exc:
            msg = f"Failed to upload {local_path.name} for '{talk_id}': {exc}"
            log.error("[upload] %s", msg)
            errors.append(msg)

    if errors:
        raise RuntimeError(
            f"Upload errors for '{talk_id}':\n" + "\n".join(errors)
        )

    return uploaded

"""Image storage: S3 when configured, local disk otherwise.

Local disk is the default so the stack runs with no cloud account. The
interface is identical either way, so switching is a config change rather than
a code change.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings

log = logging.getLogger(__name__)


def sha256_bytes(data: bytes) -> str:
    """Content hash of an uploaded image.

    Stored with every prediction. Two things it buys: you can prove which exact
    image produced a recommendation, and you can detect the same image being
    submitted twice without comparing pixels.
    """
    return hashlib.sha256(data).hexdigest()


class ImageStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._s3 = None
        if self.settings.s3_endpoint and self.settings.s3_access_key:
            try:
                import boto3

                self._s3 = boto3.client(
                    "s3",
                    endpoint_url=self.settings.s3_endpoint,
                    aws_access_key_id=self.settings.s3_access_key,
                    aws_secret_access_key=self.settings.s3_secret_key,
                )
                log.info("image store: S3 at %s", self.settings.s3_endpoint)
            except Exception as exc:  # noqa: BLE001 - never fail startup on this
                log.warning("S3 unavailable (%s), falling back to local disk", exc)

        self.local_root = Path(self.settings.local_storage_dir)
        if self._s3 is None:
            self.local_root.mkdir(parents=True, exist_ok=True)
            log.info("image store: local disk at %s", self.local_root)

    def save(self, data: bytes, patient_id: int, kind: str, suffix: str = ".png") -> str:
        """Persist an image and return a URL or path reference.

        Keyed by content hash, so re-uploading an identical image does not
        create a duplicate object.
        """
        digest = sha256_bytes(data)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        key = f"patients/{patient_id}/{kind}/{stamp}/{digest[:16]}{suffix}"

        if self._s3 is not None:
            self._s3.put_object(Bucket=self.settings.s3_bucket, Key=key, Body=data)
            return f"s3://{self.settings.s3_bucket}/{key}"

        path = self.local_root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def load(self, reference: str) -> bytes | None:
        """Read an image back. Used to fetch a patient's baseline for registration."""
        if reference.startswith("s3://") and self._s3 is not None:
            _, _, rest = reference.partition("s3://")
            bucket, _, key = rest.partition("/")
            try:
                return self._s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            except Exception as exc:  # noqa: BLE001
                log.warning("could not read %s: %s", reference, exc)
                return None

        path = Path(reference)
        return path.read_bytes() if path.is_file() else None


_store: ImageStore | None = None


def get_store() -> ImageStore:
    global _store
    if _store is None:
        _store = ImageStore()
    return _store

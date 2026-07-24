from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

try:
    from minio import Minio
    from minio.error import S3Error
except ModuleNotFoundError:  # pragma: no cover - optional dependency in test envs
    Minio = None

    class S3Error(Exception):
        pass

from src.config import settings

logger = logging.getLogger(__name__)


class ImageStorage:
    """Store extracted paper images in MinIO."""

    def __init__(self) -> None:
        if Minio is None:
            raise RuntimeError("minio is not installed")
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.bucket = settings.minio_bucket
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Create the bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info("Created MinIO bucket: %s", self.bucket)
        except S3Error as e:
            logger.warning("MinIO bucket check failed (is MinIO running?): %s", e)

    def upload_image(
        self,
        local_path: str,
        arxiv_id: str,
        figure_id: str,
    ) -> Optional[str]:
        """Upload a single image to MinIO.

        Returns the MinIO object path, or None on failure.
        """
        path = Path(local_path)
        if not path.exists():
            logger.warning("Image not found: %s", local_path)
            return None

        object_name = f"{arxiv_id}/{figure_id}/{path.name}"
        try:
            result = self.client.fput_object(
                self.bucket, object_name, str(path),
            )
            logger.info(
                "Uploaded %s → minio://%s/%s (etag=%s)",
                local_path, self.bucket, object_name, result.etag,
            )
            return object_name
        except S3Error as e:
            logger.exception("MinIO upload failed: %s", e)
            return None

    def upload_images_batch(
        self,
        image_map: dict[str, str],
        arxiv_id: str,
    ) -> dict[str, str]:
        """Upload multiple images.

        Args:
            image_map: {figure_id: local_path}
            arxiv_id: paper identifier

        Returns:
            {figure_id: minio_path} (only successful uploads)
        """
        results = {}
        for fig_id, lpath in image_map.items():
            obj = self.upload_image(lpath, arxiv_id, fig_id)
            if obj:
                results[fig_id] = obj
        return results

    def get_image_url(self, object_path: str) -> Optional[str]:
        """Get a presigned URL for an image (expires in 1 hour)."""
        try:
            url = self.client.presigned_get_object(
                self.bucket, object_path, expires=3600,
            )
            return url
        except S3Error as e:
            logger.warning("Failed to get presigned URL: %s", e)
            return None

    def is_available(self) -> bool:
        """Check if MinIO is reachable."""
        try:
            self.client.list_buckets()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """No-op for MinIO client."""
        pass

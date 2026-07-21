"""Lazy storage package exports.

Keep optional dependencies out of import time so unit tests and offline
workflows can use the SQLite-backed paths without MinIO installed.
"""

__all__ = ["PaperDatabase", "ImageStorage"]


def __getattr__(name: str):
    if name == "PaperDatabase":
        from .sqlite import PaperDatabase
        return PaperDatabase
    if name == "ImageStorage":
        from .minio import ImageStorage
        return ImageStorage
    raise AttributeError(name)

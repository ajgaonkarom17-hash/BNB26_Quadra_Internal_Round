"""
File intake helpers: safe filenames, modality detection, size checks.

Keeping this separate means the API router stays thin and the rules are testable.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Optional

from backend.config import UPLOAD_DIR, MAX_UPLOAD_BYTES

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
DOC_EXT = {".txt", ".pdf", ".md", ".csv", ".json", ".log", ".rtf"}

CONTENT_TYPE_PREFIX = {
    "image": "image/",
    "video": "video/",
    "audio": "audio/",
}


def safe_filename(name: str) -> str:
    name = os.path.basename(name or "file")
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    name = name.strip() or "file"
    return name[:120]


def detect_modality(filename: str, content_type: Optional[str] = None) -> Optional[str]:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in DOC_EXT:
        return "document"
    if content_type:
        ct = content_type.lower()
        for modality, prefix in CONTENT_TYPE_PREFIX.items():
            if ct.startswith(prefix):
                return modality
        if ct.startswith("text/"):
            return "document"
    return None


def new_stored_path(investigation_id: str, filename: str) -> str:
    directory = UPLOAD_DIR / investigation_id
    directory.mkdir(parents=True, exist_ok=True)
    unique = uuid.uuid4().hex[:8]
    return str(directory / f"{unique}_{filename}")


def check_size(num_bytes: int) -> Optional[str]:
    if num_bytes <= 0:
        return "File is empty."
    if num_bytes > MAX_UPLOAD_BYTES:
        return (f"File exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB limit.")
    return None

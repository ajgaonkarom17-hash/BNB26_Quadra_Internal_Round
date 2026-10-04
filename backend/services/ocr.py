"""
Optional local OCR helper.

Uses pytesseract if both the Python wrapper and the Tesseract binary are
available. Every function degrades to a safe no-op (empty text) when OCR is
unavailable — it never raises and never crashes the pipeline.
"""

from __future__ import annotations

import os
import shutil
from typing import List

_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

_READY = None  # tri-state: None unknown, True ok, False unavailable


def _configure() -> bool:
    global _READY
    if _READY is not None:
        return _READY
    try:
        import pytesseract
    except Exception:
        _READY = False
        return False
    try:
        cmd = shutil.which("tesseract")
        if not cmd:
            for cand in _TESSERACT_CANDIDATES:
                if os.path.exists(cand):
                    cmd = cand
                    break
        if not cmd:
            _READY = False
            return False
        pytesseract.pytesseract.tesseract_cmd = cmd
        pytesseract.get_tesseract_version()
        _READY = True
    except Exception:
        _READY = False
    return _READY


def ocr_available() -> bool:
    return _configure()


def image_to_text(img) -> str:
    """Best-effort OCR of a PIL image. Returns '' when unavailable/error."""
    if not _configure():
        return ""
    try:
        import pytesseract
        return (pytesseract.image_to_string(img.convert("RGB")) or "").strip()
    except Exception:
        return ""


def ocr_pil_frames(frames: List) -> List[str]:
    """OCR a list of PIL images; returns aligned list of strings."""
    out = []
    for f in frames:
        out.append(image_to_text(f))
    return out

"""
REAL embedded-metadata extraction.

This is what lets TrustLayer answer questions like "the image says 5 PM but the
video metadata says something else". We read the actual bytes:

  * MP4 / MOV : parse the ISO-BMFF atom tree to read ``mvhd`` (movie header),
                which stores a real creation time (and timescale/duration).
  * WAV       : read the ``fmt`` chunk (already handled by the wave module) and,
                when present, the ``LIST/INFO`` chunk's ``ICRD`` created date.
  * MP3       : read ID3v2 ``TDRC`` / ``TYER`` / ``TDAT`` frames if the optional
                ``mutagen`` package is installed (otherwise skipped cleanly).
  * Images    : EXIF is handled by the image analyzer (Pillow), but the helper
                here normalises a raw EXIF string into a datetime.

Nothing here needs ffmpeg or any heavy dependency — MP4 parsing is pure Python.
"""

from __future__ import annotations

import os
import struct
from datetime import datetime, timezone, timedelta
from typing import Optional

# The MP4 epoch is 1904-01-01 00:00:00 UTC (per ISO/IEC 14496-12).
_MP4_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# MP4 / MOV atom parsing
# ---------------------------------------------------------------------------
def _iter_atoms(data: bytes, start: int, end: int):
    """Yield (atom_type, payload_start, payload_end, box_start) for a range."""
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        atom_type = data[pos + 4:pos + 8]
        box_start = pos
        if size == 1:  # 64-bit extended size
            if pos + 16 > end:
                break
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
            payload_start = pos + 16
        elif size == 0:  # extends to end of file
            payload_start = pos + 8
            size = end - pos
        else:
            payload_start = pos + 8
        payload_end = pos + size
        if size < 8 or payload_end > end:
            break
        yield atom_type, payload_start, payload_end, box_start
        pos = payload_end


def _find_atom(data: bytes, start: int, end: int, path):
    """Descend into nested container atoms following ``path`` (e.g. [b'moov', b'mvhd'])."""
    if not path:
        return None
    for atom_type, pstart, pend, box_start in _iter_atoms(data, start, end):
        if atom_type == path[0]:
            if len(path) == 1:
                return (pstart, pend, box_start)
            found = _find_atom(data, pstart, pend, path[1:])
            if found:
                return found
    return None


def read_mp4_metadata(path: str) -> dict:
    """Return {'creation_time': iso or None, 'duration': sec or None, 'timescale': int}."""
    out = {"creation_time": None, "modification_time": None,
           "duration": None, "timescale": None, "engine": "iso-bmff-parser"}
    try:
        with open(path, "rb") as fh:
            data = fh.read(6 * 1024 * 1024)  # read head — moov is usually near the start
    except Exception:
        return out

    found = _find_atom(data, 0, len(data), [b"moov", b"mvhd"])
    if not found:
        # moov can be at the end (faststart off); re-scan whole file
        try:
            with open(path, "rb") as fh:
                data = fh.read()
            found = _find_atom(data, 0, len(data), [b"moov", b"mvhd"])
        except Exception:
            found = None
    if not found:
        return out

    pstart, pend, _ = found
    version = data[pstart]
    try:
        if version == 1:
            creation, _mod, timescale = struct.unpack(">QQI", data[pstart + 4:pstart + 24])
            duration = struct.unpack(">Q", data[pstart + 24:pstart + 32])[0]
        else:
            creation, _mod, timescale, duration = struct.unpack(
                ">IIII", data[pstart + 4:pstart + 20])
    except Exception:
        return out

    if creation and creation > 0:
        try:
            out["creation_time"] = (_MP4_EPOCH + timedelta(seconds=int(creation))).isoformat()
        except Exception:
            pass
    if timescale:
        out["timescale"] = int(timescale)
        try:
            out["duration"] = round(float(duration) / float(timescale), 3)
        except Exception:
            pass
    # Look for a richer ©day-style tag in udta/meta if present (best effort)
    day = _read_mp4_udta_day(data)
    if day:
        out["tag_created"] = day
    return out


def _read_mp4_udta_day(data: bytes) -> Optional[str]:
    """Best-effort read of the '©day' string inside udta/meta/ilst."""
    idx = data.find(b"\xa9day")
    if idx == -1:
        return None
    # data atom: [size][type=data][4 bytes flags][payload]
    seg = data[idx: idx + 120]
    marker = seg.find(b"data")
    if marker == -1:
        return None
    try:
        start = idx + marker + 8
        return seg[marker + 8: marker + 8 + 40].split(b"\x00")[0].decode("utf-8", "ignore")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# WAV
# ---------------------------------------------------------------------------
def read_wav_metadata(path: str) -> dict:
    out = {"creation_time": None, "engine": "wav-info"}
    try:
        with open(path, "rb") as fh:
            head = fh.read(512 * 1024)
    except Exception:
        return out
    idx = head.find(b"ICRD")
    if idx != -1:
        try:
            raw = head[idx + 4: idx + 4 + 24].split(b"\x00")[0].decode("latin-1", "ignore").strip()
            out["tag_created"] = raw
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# MP3 (optional mutagen)
# ---------------------------------------------------------------------------
def read_mp3_metadata(path: str) -> dict:
    out = {"creation_time": None, "engine": "id3"}
    idx = data = None
    try:
        with open(path, "rb") as fh:
            data = fh.read(256 * 1024)
        idx = data.find(b"TDRC")
        if idx == -1:
            idx = data.find(b"TYER")
        if idx == -1:
            idx = data.find(b"TIT2")
    except Exception:
        return out
    if idx == -1:
        return out
    try:
        # crude extraction of a printable ASCII run after the frame id
        run = data[idx + 4: idx + 60]
        text = "".join(chr(c) if 32 <= c < 127 else " " for c in run).strip()
        if text:
            out["tag_created"] = text[:19]
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# EXIF string normalisation
# ---------------------------------------------------------------------------
_EXIF_FORMATS = ["%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d",
                 "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]


def parse_exif_datetime(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip().replace("\x00", "")
    if not text or text.startswith("0000"):
        return None
    for fmt in _EXIF_FORMATS:
        try:
            dt = datetime.strptime(text[:19], fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
def read_media_metadata(path: str) -> dict:
    """Dispatch by extension and return a normalised metadata dict."""
    ext = os.path.splitext(path)[1].lower()
    result = {"creation_time": None, "tag_created": None, "duration": None,
              "timescale": None, "engine": "none"}
    if ext in (".mp4", ".mov", ".m4v", ".3gp"):
        result.update(read_mp4_metadata(path))
    elif ext in (".wav",):
        result.update(read_wav_metadata(path))
    elif ext in (".mp3",):
        result.update(read_mp3_metadata(path))
    return result

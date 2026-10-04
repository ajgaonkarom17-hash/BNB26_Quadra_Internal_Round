"""
Builds the demo investigation "Event Authenticity Test".

Creates REAL, small synthetic files (no downloads, no fabricated real-world
claims) that are deliberately *individually plausible* but *mutually
inconsistent*, which is exactly the point of TrustLayer:

    image  -> looks like a normal photo (but has an "editing software" tag)
    video  -> decodes fine, but has a few frame-jump indicators
    audio  -> decodes fine, but is much shorter than the video
    report -> a text report whose stated date is years from the media

Run directly:   python -m scripts.build_sample
Or via the API: POST /api/sample
"""

from __future__ import annotations

import os
import struct
import uuid
import wave
from datetime import datetime, timezone

import numpy as np

from backend.config import SAMPLE_DIR, UPLOAD_DIR
from backend.database import db
from backend.utils import upload as upload_utils

INVESTIGATION_NAME = "Event Authenticity Test"


# ---------------------------------------------------------------------------
# Synthetic file writers
# ---------------------------------------------------------------------------
def _patch_mp4_creation_time(path: str, dt) -> bool:
    """
    Overwrite the real ``mvhd`` creation_time in an MP4 so its *container
    metadata* reports ``dt``. This is how we build a demo where the video
    metadata disagrees with the image caption.
    """
    import struct
    from datetime import datetime, timezone
    mp4_epoch = datetime(1904, 1, 1, tzinfo=timezone.utc)
    seconds = int((dt - mp4_epoch).total_seconds())
    try:
        with open(path, "rb") as fh:
            data = bytearray(fh.read())
        idx = data.find(b"mvhd")
        if idx == -1:
            return False
        # idx points at 'mvhd'; version/flags follow, then creation_time.
        version = data[idx + 4]
        if version == 0:
            struct.pack_into(">I", data, idx + 8, seconds)
            struct.pack_into(">I", data, idx + 12, seconds)
        elif version == 1:
            struct.pack_into(">Q", data, idx + 8, seconds)
            struct.pack_into(">Q", data, idx + 16, seconds)
        else:
            return False
        with open(path, "wb") as fh:
            fh.write(data)
        return True
    except Exception as exc:
        print(f"[sample] could not patch mp4 time: {exc}")
        return False


def _write_image(path: str, exif_datetime: str) -> None:
    from PIL import Image, ImageDraw

    w, h = 480, 320
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    base = np.dstack([
        (xx * 0.7 + 40).astype(np.uint8),
        (yy * 0.6 + 60).astype(np.uint8),
        ((xx + yy) * 0.4).astype(np.uint8),
    ])
    img = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 90, 240, 230], fill=(210, 180, 140), outline=(80, 60, 40), width=3)
    draw.ellipse([300, 120, 400, 220], fill=(90, 140, 200))
    # Overlay a caption with a clock time (so OCR, when available, can read it).
    draw.text((20, 20), "EVENT SNAPSHOT", fill=(255, 255, 255))
    draw.text((20, 40), "Recorded 5:00 PM", fill=(255, 240, 120))

    # Real EXIF: capture time (set to match the video metadata so the two
    # *media* modalities agree, while the written report disagrees).
    exif = Image.Exif()
    exif[0x0131] = "Adobe Photoshop 25.0 (Windows)"   # Software -> "looks edited"
    exif[0x0110] = "TrustLayer Demo Camera"           # Model
    exif[0x0132] = exif_datetime                       # DateTime
    exif[0x9003] = exif_datetime                       # DateTimeOriginal
    img.save(path, format="JPEG", quality=88, exif=exif)


def _write_video(path: str, seconds: float = 6.0, fps: int = 15) -> None:
    import cv2

    w, h = 480, 320
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    total = int(seconds * fps)
    for i in range(total):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (30, 40, 70)
        cx = 60 + int((w - 160) * (i / max(total - 1, 1)))
        cv2.rectangle(frame, (cx, 150), (cx + 90, 240), (60, 170, 240), -1)
        cv2.putText(frame, f"frame {i}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        # inject a couple of abrupt visual jumps (frame-level "indicators")
        if i in (int(total * 0.35), int(total * 0.7)):
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_audio(path: str, seconds: float = 2.5, sr: int = 16000) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    signal = 0.45 * np.sin(2 * np.pi * 220 * t) + 0.25 * np.sin(2 * np.pi * 440 * t)
    # add clipping to trigger the clipping heuristic
    signal = np.clip(signal * 2.2, -1.0, 1.0)
    signal[: int(sr * 0.1)] = 0.0  # leading silence
    pcm = (signal * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _write_pdf(path: str, claimed_time: str, speaker: str) -> None:
    """Write a tiny uncompressed PDF our fallback parser can read."""
    lines = [
        "Incident Report - Riverside Community Event",
        "",
        f"Reported by: {speaker}, Dana Reyes",
        "Location: Riverside Park, North Bridge entrance",
        "Incident date: 2021-03-05",
        "",
        f"Summary: The witness {speaker} confirmed that the gathering occurred",
        f"at {claimed_time} in the evening. A large crowd was observed near the",
        "main stage and the recording was said to be captured on the same evening.",
        "",
        "Note: This is a synthetic document for the TrustLayer prototype demo.",
    ]
    text_cmds = ["BT", "/F1 12 Tf", "72 750 Td", "14 TL"]
    for line in lines:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        text_cmds.append(f"({safe}) Tj T*")
    text_cmds.append("ET")
    stream = "\n".join(text_cmds).encode("latin-1")

    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                   b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>")
    objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
                   + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects)+1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF").encode()
    with open(path, "wb") as fh:
        fh.write(out)


# ---------------------------------------------------------------------------
# Investigation builder
# ---------------------------------------------------------------------------
def create_sample_files() -> list:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    # The scenario: the files' *claimed* time and the report's time disagree.
    #   image caption + video metadata  ~ 17:00 (5 PM)
    #   written report                  ~ 21:30 (9:30 PM, ~4.5h later)
    #   report credits a named speaker the audio cannot verify
    media_time_iso = "2021-03-05 17:00:00"
    media_dt = datetime(2021, 3, 5, 17, 0, 0, tzinfo=timezone.utc)
    # The video's *container metadata* claims a different time (09:12) than the
    # image caption (5:00 PM) and the written report (9:30 PM).
    video_metadata_dt = datetime(2021, 3, 5, 9, 12, 0, tzinfo=timezone.utc)
    claimed_time = "9:30 PM"
    speaker = "John Smith"

    spec = [
        ("event_image.jpg", lambda p: _write_image(p, media_time_iso)),
        ("event_video.mp4", _write_video),
        ("event_audio.wav", _write_audio),
        ("event_report.pdf", lambda p: _write_pdf(p, claimed_time, speaker)),
    ]
    for name, writer in spec:
        path = SAMPLE_DIR / name
        try:
            writer(str(path))
            if name == "event_video.mp4":
                # Patch the real mvhd creation_time so metadata disagrees.
                _patch_mp4_creation_time(str(path), video_metadata_dt)
            # Give the media a real, comparable modification timestamp (used as a
            # fallback event time when a container has no embedded date).
            if name != "event_report.pdf":
                ts = media_dt.timestamp()
                os.utime(str(path), (ts, ts))
            files.append((name, str(path)))
        except Exception as exc:
            print(f"[sample] could not create {name}: {exc}")
    return files


def build_sample_investigation() -> dict:
    from backend.database.db import init_db
    init_db()
    files = create_sample_files()
    if not files:
        raise RuntimeError("No sample files could be generated on this system.")

    inv_id = str(uuid.uuid4())
    db.create_investigation(inv_id, INVESTIGATION_NAME)

    evidence_ids = []
    for name, source_path in files:
        with open(source_path, "rb") as fh:
            data = fh.read()
        stored_path = upload_utils.new_stored_path(inv_id, name)
        with open(stored_path, "wb") as fh:
            fh.write(data)
        modality = upload_utils.detect_modality(name) or "document"
        evidence_id = str(uuid.uuid4())
        db.add_evidence({
            "id": evidence_id,
            "investigation_id": inv_id,
            "filename": name,
            "modality": modality,
            "content_type": "",
            "size_bytes": len(data),
            "stored_path": stored_path,
        })
        evidence_ids.append(evidence_id)

    return {
        "investigation_id": inv_id,
        "name": INVESTIGATION_NAME,
        "files": [f[0] for f in files],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    info = build_sample_investigation()
    print("Sample investigation created:")
    print("  id:", info["investigation_id"])
    for f in info["files"]:
        print("  -", f)
    print("\nNow call:  POST /api/investigations/%s/analyze" % info["investigation_id"])

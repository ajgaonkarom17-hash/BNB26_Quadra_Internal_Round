"""
VIDEO analyzer.

Pipeline (all real, lightweight):
  * open with OpenCV, read fps / frame count / duration
  * sample ~10-20 evenly spaced frames (configurable)
  * per-frame features: brightness, blur (Laplacian var), edge density,
    frame-difference vs previous sampled frame (temporal jump)
  * temporal consistency: variability of those signals across time; abrupt
    spikes become "suspicious frame / timestamp indicators" (NOT proven
    manipulation localization)
  * optional face detection with OpenCV Haar cascade when present
  * video embedding = projection of aggregated frame features

FFmpeg is NOT required: OpenCV decodes directly. If OpenCV is missing we fall
back to a metadata-only result and clearly mark it.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from backend.services.base import (
    Finding, ModalityResult, finalize, make_result,
    ENGINE_HEURISTIC, ENGINE_MODEL,
)
from backend.utils.helpers import clamp, risk_to_label
from backend.config import (
    VIDEO_MIN_FRAMES, VIDEO_MAX_FRAMES, SUSPICIOUS_THRESHOLD,
    AUTHENTIC_THRESHOLD, EMBEDDING_DIM, MODE,
)

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    CV2_AVAILABLE = False


_FACE_CASCADE = None


def _creation_datetime(iso: str):
    from datetime import datetime
    try:
        return datetime.fromisoformat(iso)
    except Exception:
        return None


def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE
    if not CV2_AVAILABLE:
        return None
    try:
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(path)
        _FACE_CASCADE = cascade if not cascade.empty() else False
    except Exception:
        _FACE_CASCADE = False
    return _FACE_CASCADE or None


def _frame_features(frame) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (160, 120))
    brightness = float(small.mean() / 255.0)
    blur = float(cv2.Laplacian(small, cv2.CV_64F).var())
    edges = cv2.Canny(small, 60, 160)
    edge_density = float(edges.mean() / 255.0)
    return {
        "brightness": brightness,
        "blur": blur,
        "edge_density": edge_density,
        "small_gray": small,
    }


def analyze(path: str, meta: dict) -> ModalityResult:
    result = make_result("video", meta, engine=ENGINE_HEURISTIC)

    if not CV2_AVAILABLE:
        result.error = "OpenCV is not installed; video analysis unavailable."
        result.quality = "unusable"
        result.findings.append(Finding(
            "unavailable", "Video engine unavailable",
            "OpenCV is required to decode video frames.", "high", "missing"))
        return finalize(result, EMBEDDING_DIM, meta.get("filename", "video"))

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        result.error = "Could not open video (unsupported codec or corrupt file)."
        result.quality = "unusable"
        result.findings.append(Finding(
            "corrupt", "Unreadable video",
            "The video could not be decoded and was skipped.", "high", "missing"))
        return finalize(result, EMBEDDING_DIM, meta.get("filename", "video"))

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total / fps if fps > 0 else 0.0

        target = int(clamp(total, VIDEO_MIN_FRAMES, VIDEO_MAX_FRAMES)) if total > 0 else VIDEO_MAX_FRAMES
        target = max(VIDEO_MIN_FRAMES, min(VIDEO_MAX_FRAMES, target))
        indices = np.linspace(0, max(total - 1, 0), num=target, dtype=int) if total > 0 else []

        sampled = []
        prev_small = None
        face_count = 0
        cascade = _get_face_cascade()

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            feats = _frame_features(frame)
            small = feats.pop("small_gray")
            if prev_small is not None:
                diff = float(np.abs(small.astype(np.float32) - prev_small.astype(np.float32)).mean() / 255.0)
            else:
                diff = 0.0
            prev_small = small
            timestamp = float(idx) / fps if fps > 0 else 0.0
            sampled.append({**feats, "frame_index": int(idx),
                            "timestamp": round(timestamp, 3), "frame_diff": diff})
            if cascade is not None:
                try:
                    faces = cascade.detectMultiScale(small, 1.1, 5)
                    face_count += len(faces)
                except Exception:
                    pass
    finally:
        cap.release()

    if not sampled:
        result.error = "No frames could be extracted from the video."
        result.quality = "unusable"
        result.findings.append(Finding(
            "no_frames", "No frames extracted",
            "The video contains no decodable frames.", "high", "missing"))
        return finalize(result, EMBEDDING_DIM, meta.get("filename", "video"))

    # ---- optional OCR over sampled frames -------------------------------
    # Claims must be *stable across frames* to be trusted: a timestamp that
    # appears in >= half of the sampled frames is a strong claim; one that
    # appears once is a weak hint and is not surfaced as evidence.
    from collections import Counter
    from backend.services import ocr as _ocr
    ocr_available = _ocr.ocr_available()
    ocr_texts: list = []
    frame_claims: list = []
    if ocr_available:
        try:
            from PIL import Image as _PILImage
            for s in sampled[::2][:8]:  # every other frame, max 8
                cap2 = cv2.VideoCapture(path)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, int(s["frame_index"]))
                ok2, fr = cap2.read()
                cap2.release()
                if ok2 and fr is not None:
                    pil = _PILImage.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                    ocr_texts.append(_ocr.image_to_text(pil))
        except Exception:
            ocr_texts = []
        # aggregate: count how many sampled frames support each claim value
        per_frame_claims = []
        for txt in ocr_texts:
            if txt.strip():
                from backend.services.claims import claims_from_text
                per_frame_claims.append(claims_from_text(
                    txt, "video", meta.get("evidence_id", ""), 0.7, "ocr"))
            else:
                per_frame_claims.append([])
        supports: Counter = Counter()
        originals: dict = {}
        for cl in per_frame_claims:
            seen_this_frame = set()
            for c in cl:
                key = (c["type"], c["value"])
                if key not in seen_this_frame:
                    supports[key] += 1
                    seen_this_frame.add(key)
                    originals.setdefault(key, c)
        n_ocr = max(len(per_frame_claims), 1)
        from backend.services.claims import make_claim
        for (typ, val), n in supports.items():
            conf = round(n / n_ocr, 3)
            if conf >= 0.5:  # stable claim
                base = originals[(typ, val)]
                frame_claims.append(make_claim(typ, val, base["original"], "video",
                                               meta.get("evidence_id", ""), conf, "ocr-aggregate"))
        # convenience features for cross-modal reasoning
        times = [c["value"] for c in frame_claims if c["type"] == "time_of_day"]
        dts = [c["value"] for c in frame_claims if c["type"] == "timestamp"]
        locs = [c["value"] for c in frame_claims if c["type"] == "location"]
        if times:
            result.features["ocr_times_of_day"] = times
        if dts:
            result.features["ocr_datetimes"] = dts
        if locs:
            result.features["ocr_locations"] = locs

    brightness = np.array([s["brightness"] for s in sampled])
    blur = np.array([s["blur"] for s in sampled])
    edges = np.array([s["edge_density"] for s in sampled])
    diffs = np.array([s["frame_diff"] for s in sampled])

    # --- suspicious timestamps: spikes in frame-difference or blur drop ----
    suspicious = []
    if len(diffs) >= 3:
        d_mean, d_std = float(diffs.mean()), float(diffs.std())
        threshold = d_mean + 2.0 * max(d_std, 1e-6)
        for s in sampled:
            if s["frame_diff"] >= threshold and s["frame_diff"] > 0.05:
                suspicious.append({
                    "timestamp": s["timestamp"],
                    "frame_index": s["frame_index"],
                    "reason": "abrupt frame-to-frame change",
                    "indicator_score": round(clamp(s["frame_diff"] / max(threshold, 1e-6) * 0.5), 3),
                })
    if len(blur) >= 3:
        b_mean, b_std = float(blur.mean()), float(blur.std())
        for s in sampled:
            if b_std > 1e-6 and s["blur"] < b_mean - 2.0 * b_std:
                suspicious.append({
                    "timestamp": s["timestamp"],
                    "frame_index": s["frame_index"],
                    "reason": "localized blur / smoothing spike",
                    "indicator_score": round(clamp(0.4 + (b_mean - s["blur"]) / (b_mean + 1e-6)), 3),
                })
    # de-duplicate by timestamp keep highest score
    best = {}
    for s in suspicious:
        key = round(s["timestamp"], 2)
        if key not in best or s["indicator_score"] > best[key]["indicator_score"]:
            best[key] = s
    suspicious = sorted(best.values(), key=lambda x: x["timestamp"])[:8]

    temporal_variability = float(diffs.std()) if diffs.size else 0.0
    brightness_variability = float(brightness.std()) if brightness.size else 0.0
    edge_variability = float(edges.std()) if edges.size else 0.0

    # indicators (0..1 higher = more suspicious)
    jump_ind = clamp(temporal_variability * 6.0)
    spike_ind = clamp(len(suspicious) / 5.0)
    light_ind = clamp(brightness_variability * 3.0)
    blur_ind = clamp((np.mean(blur) < 20).astype(float) * 0.6)

    indicators = {
        "temporal_jump_indicator": round(jump_ind, 4),
        "suspicious_frame_indicator": round(spike_ind, 4),
        "lighting_variability_indicator": round(light_ind, 4),
        "overall_blur_indicator": round(blur_ind, 4),
    }

    risk = (0.35 * jump_ind + 0.30 * spike_ind + 0.20 * light_ind + 0.15 * blur_ind)

    model_used = False
    if MODE == "model":
        try:
            from backend.models.registry import predict_individual
            model_score = predict_individual("video", {**indicators,
                                                       "duration": duration}, path)
            if model_score is not None:
                risk = float(model_score)
                model_used = True
                result.engine = ENGINE_MODEL
        except Exception:
            model_used = False

    result.score = clamp(risk)
    result.label = risk_to_label(result.score, SUSPICIOUS_THRESHOLD, AUTHENTIC_THRESHOLD)
    result.confidence = clamp(0.4 + 0.4 * min(1.0, len(sampled) / 12.0)
                              - 0.2 * (width < 160 or height < 120))

    result.features = {
        "fps": round(fps, 3),
        "frame_count": total,
        "duration_seconds": round(duration, 3),
        "resolution": f"{width}x{height}",
        "sampled_frames": len(sampled),
        "mean_brightness": round(float(brightness.mean()), 4),
        "mean_blur": round(float(blur.mean()), 3),
        "temporal_variability": round(temporal_variability, 4),
        "brightness_variability": round(brightness_variability, 4),
        "edge_variability": round(edge_variability, 4),
        "faces_detected": int(face_count),
        "face_detector": "opencv-haar" if _get_face_cascade() is not None else "unavailable",
        **indicators,
    }
    # Real embedded metadata (container creation time, duration, tags).
    from backend.services.media_metadata import read_media_metadata
    media_meta = read_media_metadata(path)
    if media_meta.get("creation_time"):
        mdt = _creation_datetime(media_meta["creation_time"])
        if mdt:
            result.features["metadata_time_of_day"] = mdt.strftime("%H:%M")
    result.detail = {
        "suspicious_timestamps": suspicious,
        "frame_timeline": [
            {"timestamp": s["timestamp"], "frame_diff": round(s["frame_diff"], 4),
             "blur": round(s["blur"], 2), "brightness": round(s["brightness"], 4)}
            for s in sampled
        ],
        "model_applied": model_used,
        "media_metadata": media_meta,
        "claims": frame_claims,
        "ocr_available": bool(ocr_available),
    }

    # findings
    if media_meta.get("creation_time"):
        result.findings.append(Finding(
            "metadata_time", "Container metadata time",
            f"Video container creation time is {media_meta['creation_time']}. "
            "This can be compared against other evidence timelines.",
            "info", "supports"))
    else:
        result.findings.append(Finding(
            "no_metadata_time", "No container timestamp",
            "The video container did not expose a usable creation time.",
            "low", "missing"))
    if suspicious:
        result.findings.append(Finding(
            "suspicious_frames", "Suspicious frame indicators",
            f"{len(suspicious)} timestamp(s) show abrupt change or blur spikes. "
            "These are indicators, not confirmed manipulation locations.",
            "medium", "contradicts"))
    else:
        result.findings.append(Finding(
            "temporal_ok", "Temporally stable",
            "No strong frame-level jumps were detected across the sampled frames.",
            "info", "supports"))
    if light_ind > 0.6:
        result.findings.append(Finding(
            "lighting", "Lighting changes",
            "Scene brightness varies sharply between sampled frames.", "low", "contradicts"))
    if face_count > 0:
        result.findings.append(Finding(
            "faces", "Faces detected",
            f"Detected {face_count} face region(s) across sampled frames (Haar cascade).",
            "info", "supports"))
    else:
        result.findings.append(Finding(
            "no_faces", "No faces detected",
            "No face regions were found; identity comparison may be limited.",
            "low", "missing"))

    result.warnings.append(
        "Frame indicators are DEMO / HEURISTIC; they do not localize manipulation exactly.")
    return finalize(result, EMBEDDING_DIM, meta.get("filename", "video"))

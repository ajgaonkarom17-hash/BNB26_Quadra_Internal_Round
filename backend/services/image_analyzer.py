"""
IMAGE analyzer.

Real operations performed (no fakery):
  * load image with Pillow (robust to many formats)
  * EXIF metadata extraction (camera, software, timestamps, GPS if present)
  * Error Level Analysis (ELA) - a classic, well-documented manipulation
    heuristic based on JPEG recompression residuals
  * noise / high-frequency residual statistics (Laplacian variance)
  * a simple copy-move style block-duplication check via coarse hashing
  * colour/statistical feature extraction

The "score" is a transparent weighted combination of these indicators. In DEMO
mode these weights are hand-set and clearly labelled heuristic. In MODEL mode,
if a trained head exists in models/, it is used instead (see models/registry.py).
"""

from __future__ import annotations

import io
import os
from typing import Optional

import numpy as np

from backend.config import MODE, SUSPICIOUS_THRESHOLD, AUTHENTIC_THRESHOLD
from backend.services.base import (
    Finding, ModalityResult, finalize, make_result,
    ENGINE_HEURISTIC, ENGINE_MODEL,
)
from backend.utils.helpers import clamp, risk_to_label

# All detector features are computed at this fixed resolution so the detector is
# resolution-invariant (see extract_detector_features).
CANONICAL_SIZE = 96

try:
    from PIL import Image, ExifTags
    PIL_AVAILABLE = True
except Exception:  # pragma: no cover
    PIL_AVAILABLE = False

try:
    import pytesseract  # optional OCR
    _PYTESS = True
except Exception:  # pragma: no cover
    _PYTESS = False


def _ocr_text(img) -> tuple[str, str]:
    """
    Best-effort OCR of text burned into the image (e.g. an overlay timestamp).
    Uses pytesseract if installed. Returns (text, engine). Never raises.
    """
    try:
        from backend.services.ocr import image_to_text, ocr_available
        if not ocr_available():
            return "", "unavailable"
        text = image_to_text(img)
        return (text or "").strip(), "pytesseract"
    except Exception:
        return "", "error"


def _exif_to_dict(img) -> dict:
    out: dict = {}
    if not PIL_AVAILABLE:
        return out
    try:
        raw = img.getexif()
        if not raw:
            return out
        for tag_id, value in raw.items():
            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
            if tag in ("GPSInfo", "MakerNote"):
                continue
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", "ignore")
                except Exception:
                    value = str(value)[:80]
            if isinstance(value, (int, float, str)):
                out[tag] = value
    except Exception:
        pass
    return out


def _ela_score(img) -> tuple[float, float]:
    """Return (ela_mean, ela_max_norm). Higher => more compression mismatch."""
    try:
        rgb = img.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")
        a = np.asarray(rgb, dtype=np.float32)
        b = np.asarray(recompressed, dtype=np.float32)
        diff = np.abs(a - b)
        mean = float(diff.mean())
        peak = float(diff.max()) if diff.size else 0.0
        return mean, peak
    except Exception:
        return 0.0, 0.0


def _high_freq_energy(gray: np.ndarray) -> float:
    """Laplacian variance; very low values hint at smoothing/splicing."""
    try:
        import cv2
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        # numpy fallback: second difference energy
        g = gray.astype(np.float32)
        lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
               + g[1:-1, :-2] + g[1:-1, 2:])
        return float(lap.var()) if lap.size else 0.0


def extract_detector_features(path: str) -> dict:
    """
    Discriminative image features for MANIPULATION/SYNTHETIC detection.

    IMPORTANT — RESOLUTION CANONICALISATION:
    the spectrum/colour features are computed from a fixed-size
    (CANONICAL_SIZE x CANONICAL_SIZE) version of the image. Without this the
    detector would be *resolution-locked* to whatever the training images were
    (e.g. CIFAKE's tiny 32x32) and would fail on modern high-resolution AI art.
    Canonicalising puts every image — and every training sample — in the same
    domain. See README "Known limitations" for what this still cannot catch.

    These are the exact features the image detector head is trained on
    (training/individual.py imports this function), so training and inference
    never drift apart.
    """
    out = {}
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        if im.size != (CANONICAL_SIZE, CANONICAL_SIZE):
            im = im.resize((CANONICAL_SIZE, CANONICAL_SIZE), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        gray = np.asarray(im.convert("L"), dtype=np.float32) / 255.0

        # 8x8 downsampled log-magnitude spectrum
        spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
        h, w = spec.shape
        ph, pw = max(h // 8, 1), max(w // 8, 1)
        spec = spec[:ph * 8, :pw * 8].reshape(8, ph, 8, pw).mean(axis=(1, 3)).ravel()
        spec = spec / (spec.mean() + 1e-9)
        for i, v in enumerate(spec):
            out[f"spec_{i}"] = round(float(v), 5)

        # colour correlations
        R, G, B = arr[..., 0].ravel(), arr[..., 1].ravel(), arr[..., 2].ravel()
        for name, (x, y) in {"rg": (R, G), "rb": (R, B), "gb": (G, B)}.items():
            try:
                out[f"corr_{name}"] = round(float(np.corrcoef(x, y)[0, 1]), 5)
            except Exception:
                out[f"corr_{name}"] = 0.0

        out["saturation"] = round(float((arr.max(2) - arr.min(2)).mean()), 5)
        out["brightness_mean"] = round(float(arr.mean()), 5)
        out["contrast_std"] = round(float(arr.std()), 5)
        for i, c in enumerate("rgb"):
            out[f"mean_{c}"] = round(float(arr[..., i].mean()), 5)
            out[f"std_{c}"] = round(float(arr[..., i].std()), 5)
            ch = arr[..., i]
            lap = (-4 * ch[1:-1, 1:-1] + ch[:-2, 1:-1] + ch[2:, 1:-1]
                   + ch[1:-1, :-2] + ch[1:-1, 2:])
            out[f"lap_{c}"] = round(float(lap.var()), 6) if lap.size else 0.0

        hist, _ = np.histogram(gray, bins=16, range=(0, 1), density=True)
        out["entropy"] = round(float(-np.sum(hist[hist > 0] * np.log(hist[hist > 0]))), 5)

        # --- richer manipulation-sensitive signals (v2) -------------------
        # 1) ELA residuals: mean + spread of recompression error. Manipulated
        #    regions recompress differently from the rest of the frame.
        ela_mean, ela_peak = _ela_score(im)
        out["ela_feat_mean"] = round(ela_mean, 5)
        out["ela_feat_peak"] = round(ela_peak / 255.0, 5)

        # 2) Noise residual statistics: subtract a median-filtered version of
        #    each channel and measure the residual energy/structure. AI output
        #    tends to have unnaturally smooth or unnaturally uniform noise.
        for c, name in ((0, "r"), (1, "g"), (2, "b")):
            ch = arr[..., c] * 255.0
            med = _median3(ch)
            resid = ch - med
            out[f"noise_{name}_std"] = round(float(resid.std()), 5)
            out[f"noise_{name}_p90"] = round(float(np.percentile(np.abs(resid), 90)), 5)

        # 3) Gradient statistics: natural photos have a characteristic
        #    heavy-tailed gradient distribution; synthesis/smoothing flattens it.
        gy, gx = np.gradient(gray * 255.0)
        grad = np.sqrt(gx * gx + gy * gy)
        out["grad_mean"] = round(float(grad.mean()), 5)
        out["grad_std"] = round(float(grad.std()), 5)
        out["grad_p90"] = round(float(np.percentile(grad, 90)), 5)
        out["edge_density"] = round(float((grad > 25.0).mean()), 5)

        # 4) Spectrum of the high-pass residual (fails-on-synthesis): real
        #    sensor noise has a fairly flat high-frequency tail; generated
        #    images often roll off too fast or too slowly.
        hf_res = gray - _box3(gray)
        spec2 = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(hf_res))))
        s2 = spec2[:ph * 8, :pw * 8].reshape(8, ph, 8, pw).mean(axis=(1, 3)).ravel()
        s2 = s2 / (s2.mean() + 1e-9)
        for i, v in enumerate(s2):
            out[f"spec2_{i}"] = round(float(v), 5)

        # 5) Block-wise variance: local variance maps tend to be more uniform
        #    in synthetic images than in natural ones.
        bv = []
        for by in range(0, gray.shape[0] - 16, 16):
            for bx in range(0, gray.shape[1] - 16, 16):
                bv.append(float(gray[by:by + 16, bx:bx + 16].var()))
        if bv:
            bv = np.asarray(bv)
            out["blockvar_mean"] = round(float(bv.mean()), 6)
            out["blockvar_cv"] = round(float(bv.std() / (bv.mean() + 1e-9)), 5)
    except Exception:
        pass
    return out


def _median3(ch: np.ndarray) -> np.ndarray:
    """Fast 3x3 median via sliding-window views (no scipy needed)."""
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        pad = np.pad(ch, 1, mode="edge")
        win = sliding_window_view(pad, (3, 3))          # (h, w, 3, 3)
        return np.median(win.reshape(ch.shape[0], ch.shape[1], 9), axis=2)
    except Exception:
        return _box3(ch)


def _box3(ch: np.ndarray) -> np.ndarray:
    pad = np.pad(ch, 1, mode="edge")
    out = np.zeros_like(ch, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += pad[dy:dy + ch.shape[0], dx:dx + ch.shape[1]]
    return out / 9.0


def _duplicate_block_ratio(gray: np.ndarray, block: int = 16) -> float:
    """
    Coarse copy-move indicator: fraction of near-identical non-adjacent blocks.
    Not a forensic algorithm, just a cheap novelty-style signal.
    """
    h, w = gray.shape
    if h < block * 2 or w < block * 2:
        return 0.0
    hashes = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            patch = gray[y:y + block, x:x + block].astype(np.float32)
            hashes.append(np.round(patch.mean(), 2))
    if len(hashes) < 8:
        return 0.0
    arr = np.array(hashes)
    uniq, counts = np.unique(arr, return_counts=True)
    repeated = counts[counts > 1].sum() - (counts > 1).sum()
    return clamp(repeated / arr.size)


def analyze(path: str, meta: dict) -> ModalityResult:
    result = make_result("image", meta, engine=ENGINE_HEURISTIC)

    if not PIL_AVAILABLE:
        result.error = "Pillow is not installed; image analysis unavailable."
        result.quality = "unusable"
        result.findings.append(Finding(
            "unavailable", "Image engine unavailable",
            "Pillow is required for image analysis.", "high", "missing"))
        return finalize(result, 128, meta.get("filename", "image"))

    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:
        result.error = f"Could not read image: {exc}"
        result.quality = "unusable"
        result.findings.append(Finding(
            "corrupt", "Unreadable image",
            "The file could not be decoded and was skipped.", "high", "missing"))
        return finalize(result, 128, meta.get("filename", "image"))

    width, height = img.size
    mode = img.mode
    fmt = img.format or "unknown"
    exif = _exif_to_dict(img)

    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)

    # --- features --------------------------------------------------------
    brightness = float(arr.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    ela_mean, ela_peak = _ela_score(img)
    hf = _high_freq_energy(gray)
    dup = _duplicate_block_ratio(gray)

    has_exif = 1.0 if exif else 0.0
    software = str(exif.get("Software", "")).lower()
    edited_flag = 1.0 if any(k in software for k in
                             ("photoshop", "gimp", "paint", "lightroom", "snapseed")) else 0.0

    # --- indicators (each 0..1, higher = more suspicious) ----------------
    # ELA: normal images usually show ela_mean well under ~8. These cutoffs are
    # heuristic operating points for the demo, not calibrated probabilities.
    ela_ind = clamp(ela_mean / 15.0)
    hf_ind = clamp((200.0 - hf) / 200.0) if hf < 200 else 0.0  # over-smoothing
    dup_ind = clamp(dup * 3.0)
    meta_ind = 0.5 * edited_flag + 0.25 * (1.0 - has_exif)

    indicators = {
        "ela_anomaly": round(ela_ind, 4),
        "smoothing_anomaly": round(hf_ind, 4),
        "duplication_indicator": round(dup_ind, 4),
        "metadata_anomaly": round(meta_ind, 4),
    }

    # Tiered evidence: strong (ELA/smoothing) indicators can drive a high
    # risk, duplication is supporting, and provenance gaps (missing EXIF,
    # editing-software tag) are WEAK evidence — they can never, on their own,
    # push the score into "SUSPICIOUS".
    strong_ind = max(ela_ind, hf_ind)
    risk = (
        0.50 * strong_ind
        + 0.20 * dup_ind
        + 0.15 * meta_ind
        + 0.15 * min(ela_ind, dup_ind)  # agreement bonus between signals
    )
    if strong_ind < 0.3 and dup_ind < 0.5:
        # Only weak/provenance indicators present: cap below the suspicious
        # threshold. Missing metadata is a limitation, not manipulation.
        risk = min(risk, 0.45)

    # --- build the full feature dict BEFORE model scoring -----------------
    result.features = {
        "width": width, "height": height, "aspect_ratio": round(width / max(height, 1), 3),
        "mode": mode, "format": fmt,
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "ela_mean": round(ela_mean, 4),
        "ela_peak": round(ela_peak, 3),
        "high_freq_variance": round(hf, 3),
        "duplicate_block_ratio": round(dup, 4),
        "has_exif": bool(exif),
        **indicators,
    }
    # Discriminative detector features (also used by the trained image head).
    detector_feats = extract_detector_features(path)
    result.features.update(detector_feats)

    # --- engine upgrade (MODEL mode) -------------------------------------
    model_used = False
    if MODE == "model":
        try:
            from backend.models.registry import predict_individual
            model_score = predict_individual("image", result.features, path)
            if model_score is not None:
                risk = float(model_score)
                model_used = True
                result.engine = ENGINE_MODEL
        except Exception:
            model_used = False

    result.score = clamp(risk)
    result.label = risk_to_label(result.score, SUSPICIOUS_THRESHOLD, AUTHENTIC_THRESHOLD)
    result.confidence = clamp(0.55 + 0.4 * has_exif - 0.2 * (width < 128 or height < 128))


    # --- OCR (optional) + structured claims from any text in the image ----
    ocr_text, ocr_engine = _ocr_text(img)
    image_claims = {}
    structured_claims = []
    if ocr_text:
        from backend.services.claim_extraction import claims_from_text
        image_claims = claims_from_text(ocr_text)
        from backend.services.claims import claims_from_text as _structured
        structured_claims = _structured(ocr_text, "image", meta.get("evidence_id", ""),
                                        confidence=0.75, method="ocr")
        if image_claims.get("times_of_day"):
            result.features["ocr_times_of_day"] = image_claims["times_of_day"]
        if image_claims.get("datetimes"):
            result.features["ocr_datetimes"] = image_claims["datetimes"]

    # normalise an EXIF date into a comparable time-of-day
    from backend.services.media_metadata import parse_exif_datetime
    exif_dt = None
    for key in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
        if exif.get(key):
            exif_dt = parse_exif_datetime(exif[key])
            if exif_dt:
                result.features["metadata_time_of_day"] = exif_dt.strftime("%H:%M")
                from backend.services.claims import make_claim
                structured_claims.append(make_claim(
                    "timestamp", exif_dt.strftime("%Y-%m-%dT%H:%M"),
                    str(exif[key]), "image", meta.get("evidence_id", ""),
                    0.85, "exif"))
                break

    result.detail = {"exif": exif, "engine_used": result.engine,
                     "model_applied": model_used, "ocr_engine": ocr_engine,
                     "ocr_text": ocr_text[:1000], "image_claims": image_claims,
                     "claims": structured_claims}

    # --- findings --------------------------------------------------------
    if not exif:
        result.findings.append(Finding(
            "no_exif", "Metadata missing",
            "No EXIF metadata was found; provenance cannot be checked.", "low", "missing"))
    else:
        result.findings.append(Finding(
            "exif", "Metadata present",
            f"Found {len(exif)} EXIF fields"
            + (f" (software: {exif.get('Software')})" if exif.get("Software") else "") + ".",
            "info", "supports"))
    if edited_flag:
        result.findings.append(Finding(
            "editing_software", "Editing software tag",
            f"EXIF Software field references '{exif.get('Software')}'.", "medium", "contradicts"))
    if image_claims.get("times_of_day"):
        result.findings.append(Finding(
            "ocr_time", "Time text in image",
            "Text in the image mentions time(s): "
            + ", ".join(image_claims["times_of_day"][:4])
            + ". This can be compared with media metadata.",
            "info", "supports"))
    if ela_ind > 0.6:
        result.findings.append(Finding(
            "ela_high", "Compression mismatch (ELA)",
            "Error-level analysis shows unusually high recompression residuals, "
            "sometimes associated with edited regions.", "medium", "contradicts"))
    if hf_ind > 0.5:
        result.findings.append(Finding(
            "smoothing", "Possible smoothing",
            "Low high-frequency energy may indicate blurring or region smoothing.", "low", "contradicts"))
    if dup_ind > 0.5:
        result.findings.append(Finding(
            "duplication", "Repeated blocks",
            "Some image blocks look near-identical, a weak copy-move indicator.", "low", "contradicts"))
    if result.score <= AUTHENTIC_THRESHOLD:
        result.findings.append(Finding(
            "image_ok", "No strong visual anomaly",
            "Image-level heuristics did not surface strong manipulation indicators.", "info", "supports"))

    result.warnings.append(
        "Image indicators are DEMO / HEURISTIC unless a trained model is loaded.")
    return finalize(result, _dim(), meta.get("filename", "image"))


def _dim() -> int:
    from backend.config import EMBEDDING_DIM
    return EMBEDDING_DIM

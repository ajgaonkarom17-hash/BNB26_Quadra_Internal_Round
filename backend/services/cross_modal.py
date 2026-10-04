"""
CROSS-MODAL REASONING.

This is the heart of TrustLayer: individual files may look fine, but the
*relationships between them* can expose manipulation.

For every pair of available modalities we produce:
  * a Heuristic Consistency Score in 0..1  (1 = fully consistent)
  * a coarse label (CONSISTENT / MIXED / INCONSISTENT)
  * a list of explicit conflicts

The score blends two things:
  1. embedding similarity in the common representation (data-driven)
  2. domain rules (timestamp agreement, audio/video duration alignment,
     face presence, entity/location agreement)

We NEVER call these probabilities. They are heuristic consistency scores.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from typing import Optional

import numpy as np

from backend.utils.helpers import clamp, cosine_similarity

MODALITIES = ["image", "video", "audio", "document"]
PAIR_ORDER = [tuple(p) for p in combinations(MODALITIES, 2)]

# score bands for human labels
CONSISTENT_MIN = 0.70
INCONSISTENT_MAX = 0.40


def _parse_date(value: str):
    if not value:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    value = str(value).strip()
    candidates = [
        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d", "%Y:%m:%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y",
        "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y",
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=dt.tzinfo or timezone.utc)
        except ValueError:
            continue
    # fall back: extract a 4-digit year
    import re
    m = re.search(r"(19|20)\d{2}", value)
    if m:
        return datetime(int(m.group(0)), 1, 1, tzinfo=timezone.utc)
    return None


def _modality_timestamp(entry: dict) -> Optional[datetime]:
    """Best-effort 'implied event time' for one modality."""
    import os
    result = entry["result"]
    modality = result.modality
    detail = result.detail or {}

    if modality == "image":
        exif = detail.get("exif", {}) if isinstance(detail, dict) else {}
        for key in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
            if exif.get(key):
                dt = _parse_date(exif[key])
                if dt:
                    return dt
    if modality in ("video", "audio"):
        media_meta = detail.get("media_metadata", {}) if isinstance(detail, dict) else {}
        if media_meta.get("creation_time"):
            dt = _parse_date(media_meta["creation_time"])
            if dt:
                return dt
    if modality == "document":
        ents = detail.get("entities", {}) if isinstance(detail, dict) else {}
        for d in ents.get("datetimes", []):
            dt = _parse_date(d)
            if dt:
                return dt
        for d in ents.get("dates", []):
            dt = _parse_date(d)
            if dt:
                return dt
    # structured claims (OCR visible text / extraction) as a documented
    # fallback when no embedded metadata timestamp exists
    for c in detail.get("claims", []) or []:
        if c.get("type") == "timestamp" and c.get("value"):
            dt = _parse_date(c["value"])
            if dt:
                return dt
    # No usable *event* time. We deliberately do NOT fall back to the file's
    # filesystem modification time: that is the upload/copy time, not the time
    # the event happened, and using it would create false conflicts.
    return None


def _label(score: float) -> str:
    if score >= CONSISTENT_MIN:
        return "CONSISTENT"
    if score <= INCONSISTENT_MAX:
        return "INCONSISTENT"
    return "MIXED"


def _clock_to_minutes(value: str):
    """'17:00' / '5:00 PM' -> minutes since midnight (or None)."""
    if not value:
        return None
    import re
    text = str(value).strip().upper()
    m = re.match(r"^(\d{1,2}):(\d{2})\s*(AM|PM)?$", text)
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    mer = m.group(3)
    if mer == "PM" and hour < 12:
        hour += 12
    if mer == "AM" and hour == 12:
        hour = 0
    return hour * 60 + minute


def _clock_difference(a: str, b: str):
    """Smallest absolute difference in minutes between two clock times."""
    ma, mb = _clock_to_minutes(a), _clock_to_minutes(b)
    if ma is None or mb is None:
        return None
    diff = abs(ma - mb)
    return min(diff, 1440 - diff)


def _modality_time_of_day(entry) -> Optional[str]:
    """
    Best 'time of day' implied by a modality, prioritising REAL metadata:
      1. embedded media metadata / EXIF (video container, image EXIF)
      2. OCR text in an image
      3. structured claims in a document
    """
    result = entry["result"]
    feats = result.features or {}
    if feats.get("metadata_time_of_day"):
        return feats["metadata_time_of_day"]
    detail = result.detail or {}
    claims = detail.get("image_claims") or {}
    if claims.get("times_of_day"):
        return claims["times_of_day"][0]
    ents = detail.get("entities") or {}
    if ents.get("times_of_day"):
        return ents["times_of_day"][0]
    # structured claims from OCR / extraction (video, audio, document, image)
    for c in detail.get("claims") or []:
        if c.get("type") == "time_of_day" and c.get("value"):
            return c["value"]
        if c.get("type") == "timestamp" and c.get("value"):
            try:
                return c["value"].split("T")[1]
            except Exception:
                continue
    return None


def _modality_claims(entry) -> dict:
    """Structured claims (time/speaker/place) for a modality."""
    result = entry["result"]
    detail = result.detail or {}
    if result.modality == "document":
        return detail.get("entities") or {}
    if result.modality == "image":
        return detail.get("image_claims") or {}
    return {}


def _compare_pair(pair_name: str, a: dict, b: dict) -> dict:
    ra, rb = a["result"], b["result"]
    emb_a, emb_b = a.get("embedding"), b.get("embedding")
    conflicts: list = []
    breakdown: dict = {}

    # 1) common-representation similarity
    if emb_a is not None and emb_b is not None:
        embed_sim = cosine_similarity(emb_a, emb_b)
    else:
        embed_sim = 0.5
    breakdown["embedding_similarity"] = round(embed_sim, 4)

    rule_score = 0.5
    rule_notes = []

    # 2) timestamp agreement (applies to most pairs)
    ta, tb = _modality_timestamp(a), _modality_timestamp(b)
    if ta and tb:
        days = abs((ta - tb).total_seconds()) / 86400.0
        if days <= 2:
            ts_score = 1.0
        elif days <= 30:
            ts_score = clamp(1.0 - (days - 2) / 28.0 * 0.6)
        else:
            ts_score = clamp(max(0.0, 0.4 - (days - 30) / 365.0))
        breakdown["timestamp_days_apart"] = round(days, 2)
        if days > 30:
            conflicts.append({
                "type": "timestamp_conflict",
                "description": (f"{ra.modality} and {rb.modality} imply event times "
                                f"{days:.0f} days apart ({ta.date()} vs {tb.date()})."),
                "severity": "high" if days > 180 else "medium",
            })
        rule_notes.append(ts_score)
    else:
        breakdown["timestamp_days_apart"] = None  # unknown
        conflict_missing = [m for m, t in ((ra.modality, ta), (rb.modality, tb)) if not t]
        if conflict_missing:
            conflicts.append({
                "type": "timestamp_missing",
                "description": (f"No usable timestamp for {', '.join(conflict_missing)}; "
                                "time alignment could not be verified."),
                "severity": "low",
            })

    # 3) time-of-day agreement (image caption/EXIF vs video metadata vs doc)
    tod_a = _modality_time_of_day(a)
    tod_b = _modality_time_of_day(b)
    if tod_a and tod_b:
        diff_min = _clock_difference(tod_a, tod_b)
        breakdown["time_of_day_a"] = tod_a
        breakdown["time_of_day_b"] = tod_b
        breakdown["time_of_day_diff_minutes"] = diff_min
        if diff_min is not None:
            if diff_min <= 30:
                rule_notes.append(1.0)
            elif diff_min <= 120:
                rule_notes.append(clamp(1.0 - (diff_min - 30) / 90.0 * 0.7))
            else:
                rule_notes.append(clamp(max(0.0, 0.3 - (diff_min - 120) / 720.0)))
                hours = diff_min / 60.0
                conflicts.append({
                    "type": "time_of_day_conflict",
                    "description": (
                        f"{ra.modality} implies {tod_a} but {rb.modality} implies "
                        f"{tod_b} (~{hours:.1f}h apart). The claimed time does not "
                        "match the metadata time."),
                    "severity": "high" if diff_min > 240 else "medium",
                })

    # 4) audio/video duration alignment
    if pair_name == "video_audio":
        v_dur = float(ra.features.get("duration_seconds") or 0)
        a_dur = float(rb.features.get("duration_seconds") or 0)
        if v_dur > 0 and a_dur > 0:
            ratio = min(v_dur, a_dur) / max(v_dur, a_dur)
            breakdown["duration_ratio"] = round(ratio, 4)
            rule_notes.append(ratio)
            if ratio < 0.7:
                conflicts.append({
                    "type": "duration_mismatch",
                    "description": (f"Video is {v_dur:.1f}s but audio is {a_dur:.1f}s; "
                                    "they may not belong to the same recording."),
                    "severity": "high" if ratio < 0.5 else "medium",
                })

    # 5) image/video basic agreement (faces + resolution sanity)
    if pair_name == "image_video":
        vid_faces = int(rb.features.get("faces_detected", 0) or 0)
        img_w = int(ra.features.get("width", 0) or 0)
        vid_res = str(rb.features.get("resolution", "0x0"))
        try:
            vid_w = int(vid_res.split("x")[0])
        except Exception:
            vid_w = 0
        breakdown["video_faces_detected"] = vid_faces
        if vid_faces > 0:
            # image has no face detector; note the limitation rather than guess
            rule_notes.append(0.7)
            breakdown["identity_note"] = "video faces found; image-level identity not compared"
        else:
            rule_notes.append(0.6)

    # 6) document/audio claims: speaker + place agreement
    if "document" in pair_name:
        doc_entry = a if ra.modality == "document" else b
        other_entry = b if doc_entry is a else a
        doc_claims = _modality_claims(doc_entry)
        other_claims = _modality_claims(other_entry)
        other_mod = other_entry["result"].modality

        # place consistency (document vs image/video claims)
        doc_places = {p.lower() for p in (doc_claims.get("places") or [])}
        other_places = {p.lower() for p in (other_claims.get("places") or [])}
        if doc_places and other_places:
            if doc_places & other_places:
                rule_notes.append(0.95)
            else:
                rule_notes.append(0.4)
                conflicts.append({
                    "type": "location_conflict",
                    "description": (f"Document location '{sorted(doc_places)[0]}' does "
                                    f"not match {other_mod} location "
                                    f"'{sorted(other_places)[0]}'."),
                    "severity": "medium",
                })

        # speaker claim: document names a speaker, but audio/video can't confirm it
        doc_speakers = doc_claims.get("speakers") or []
        if doc_speakers and other_mod in ("audio", "video"):
            # We cannot biometrically verify a voice; flag as a check we could not
            # complete rather than pretending it matched.
            has_faces = int(other_entry["result"].features.get("faces_detected", 0) or 0)
            if other_mod == "audio":
                rule_notes.append(0.5)
                conflicts.append({
                    "type": "speaker_unverified",
                    "description": (f"Document attributes the recording to "
                                    f"'{doc_speakers[0]}', but speaker identity is not "
                                    "biometrically verified from audio."),
                    "severity": "medium",
                })
            else:
                rule_notes.append(0.6)
                breakdown["speaker_claim"] = doc_speakers[0]
        elif doc_claims.get("locations") and other_mod in ("image", "video"):
            rule_notes.append(0.55)
        else:
            rule_notes.append(0.6)

    if rule_notes:
        rule_score = float(np.mean(rule_notes))
    breakdown["rule_score"] = round(rule_score, 4)

    # weighted blend, then conflict penalty
    consistency = 0.55 * embed_sim + 0.45 * rule_score
    penalty = sum(0.18 if c["severity"] == "high" else 0.10 if c["severity"] == "medium" else 0.03
                  for c in conflicts)
    consistency = clamp(consistency - penalty)
    breakdown["conflict_penalty"] = round(penalty, 4)

    return {
        "pair": pair_name,
        "consistency": round(consistency, 4),
        "label": _label(consistency),
        "conflicts": conflicts,
        "method": "heuristic:embedding+domain-rules",
        "breakdown": breakdown,
    }


def Ra_has_face(result) -> bool:
    detail = result.detail or {}
    return bool(detail.get("faces_detected", 0)) if isinstance(detail, dict) else False


def compare_all(entries: dict) -> list:
    """
    entries: {modality: {"result": ModalityResult, "embedding": np.ndarray,
                         "stored_path": str}}
    Returns list of pair comparison dicts for every pair where BOTH exist.
    """
    available = [m for m in MODALITIES if m in entries and entries[m]["result"] is not None]
    results = []
    for a, b in combinations(available, 2):
        pair_name = f"{a}_{b}"
        results.append(_compare_pair(pair_name, entries[a], entries[b]))
    return results


def aggregate_inconsistency(comparisons: list) -> dict:
    """Summarise pairwise results into one cross-modal risk signal."""
    if not comparisons:
        return {
            "cross_modal_risk": 0.5,
            "mean_consistency": 0.5,
            "min_consistency": 0.5,
            "conflict_count": 0,
            "pairs_compared": 0,
        }
    scores = [c["consistency"] for c in comparisons]
    conflicts = sum(len(c["conflicts"]) for c in comparisons)
    mean_c = float(np.mean(scores))
    # risk = how INCONSISTENT things are; also reward presence of hard conflicts
    risk = clamp((1.0 - mean_c) * 0.8 + min(conflicts, 3) * 0.1)
    return {
        "cross_modal_risk": round(risk, 4),
        "mean_consistency": round(mean_c, 4),
        "min_consistency": round(float(min(scores)), 4),
        "conflict_count": conflicts,
        "pairs_compared": len(comparisons),
    }

"""
Common structured claim representation.

A claim is a small, explicit assertion extracted from one piece of evidence:

    {"type": "timestamp", "value": "2026-10-04T17:00", "original": "...",
     "source": "image", "evidence_id": "...", "confidence": 0.9,
     "method": "ocr"}

Claim types: timestamp, date, time_of_day, location, person, event,
organization, duration, url, email, phone, general.

This module only NORMALISES and STRUCTURES what the analyzers actually
extracted. It never invents values.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Optional


def make_claim(type: str, value: str, original: str, source: str,
               evidence_id: str = "", confidence: float = 0.7,
               method: str = "heuristic") -> dict:
    return {
        "type": type,
        "value": value,
        "original": original,
        "source": source,
        "evidence_id": evidence_id,
        "confidence": round(float(confidence), 4),
        "method": method,
    }


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

def normalize_datetime(value: str) -> Optional[str]:
    """'2026-10-04 17:00' / '2026-10-04T17:00:00' -> '2026-10-04T17:00'."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M")
    except Exception:
        pass
    m = re.search(r"(\d{4}-\d{2}-\d{2})[ T](\d{1,2}):(\d{2})", str(value))
    if m:
        return f"{m.group(1)}T{int(m.group(2)):02d}:{m.group(3)}"
    m = re.search(r"(\d{4}-\d{2}-\d{2})", str(value))
    if m:
        return m.group(1) + "T00:00"
    return None


def normalize_time_of_day(value: str) -> Optional[str]:
    """'5 PM' / '5:00 PM' / '17:00' -> '17:00'."""
    if not value:
        return None
    text = str(value).strip().upper()
    m = re.match(r"^(\d{1,2}):(\d{2})\s*(AM|PM)?$", text)
    if not m:
        m = re.match(r"^(\d{1,2})\s*(AM|PM)$", text)
        if m:
            hour, minute, mer = int(m.group(1)), 0, m.group(2)
        else:
            return None
    else:
        hour, minute, mer = int(m.group(1)), int(m.group(2)), m.group(3)
    if mer == "PM" and hour < 12:
        hour += 12
    if mer == "AM" and hour == 12:
        hour = 0
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return None
    return f"{hour:02d}:{minute:02d}"


def normalize_location(value: str) -> str:
    """Strip punctuation/extra whitespace and lowercase for comparison."""
    if not value:
        return ""
    v = re.sub(r"[^\w\s]", " ", str(value))
    return re.sub(r"\s+", " ", v).strip().lower()


def normalize_name(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


# ---------------------------------------------------------------------------
# extraction of structured claims from already-extracted text/features
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"\bhttps?://[^\s]+|\bwww\.[^\s]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def claims_from_text(text: str, source: str, evidence_id: str = "",
                     confidence: float = 0.7, method: str = "ocr") -> List[dict]:
    """Turn raw (OCR/document) text into structured claims."""
    claims: List[dict] = []
    if not text:
        return claims
    from backend.services.claim_extraction import (
        extract_datetimes, extract_times_of_day, extract_speakers,
        extract_places,
    )
    for dt in extract_datetimes(text):
        norm = normalize_datetime(dt)
        if norm:
            claims.append(make_claim("timestamp", norm, dt, source, evidence_id,
                                     confidence, method))
    for t in extract_times_of_day(text):
        claims.append(make_claim("time_of_day", t, t, source, evidence_id,
                                 confidence, method))
    for p in extract_places(text):
        claims.append(make_claim("location", normalize_location(p), p, source,
                                 evidence_id, confidence, method))
    for s in extract_speakers(text):
        claims.append(make_claim("person", normalize_name(s), s, source,
                                 evidence_id, confidence, method))
    for u in _URL_RE.findall(text):
        claims.append(make_claim("url", u.lower(), u, source, evidence_id,
                                 confidence, method))
    for e in _EMAIL_RE.findall(text):
        claims.append(make_claim("email", e.lower(), e, source, evidence_id,
                                 confidence, method))
    for ph in _PHONE_RE.findall(text):
        # a datetime like "2026-10-04 17:00" is NOT a phone number
        if re.search(r"\d{4}-\d{2}-\d{2}", ph) or re.match(r"^\d{4}", ph):
            continue
        digits = re.sub(r"\D", "", ph)
        if len(digits) >= 7:
            claims.append(make_claim("phone", digits, ph, source, evidence_id,
                                     confidence * 0.8, method))
    # "Location: X" style labels, which the preposition-based regex misses
    for m in re.finditer(r"(?:location|venue|place)\s*[:\-]\s*([^|\n,]+)", text, re.IGNORECASE):
        loc = m.group(1).strip()
        if 2 < len(loc) < 60:
            claims.append(make_claim("location", normalize_location(loc), loc, source,
                                     evidence_id, confidence, method))
    return claims


def claims_from_entities(entities: dict, source: str, evidence_id: str = "",
                         confidence: float = 0.8, method: str = "rule-extraction") -> List[dict]:
    """Structure the document analyzer's entities dict into claims."""
    claims: List[dict] = []
    if not entities:
        return claims
    for dt in entities.get("datetimes", []) or []:
        norm = normalize_datetime(dt)
        if norm:
            claims.append(make_claim("timestamp", norm, str(dt), source, evidence_id, confidence, method))
    for d in entities.get("dates", []) or []:
        norm = normalize_datetime(d)
        if norm:
            claims.append(make_claim("date", norm[:10], str(d), source, evidence_id, confidence, method))
    for t in entities.get("times_of_day", []) or []:
        n = normalize_time_of_day(t)
        if n:
            claims.append(make_claim("time_of_day", n, str(t), source, evidence_id, confidence, method))
    for loc in entities.get("locations", []) or entities.get("places", []) or []:
        claims.append(make_claim("location", normalize_location(loc), str(loc), source, evidence_id, confidence, method))
    for n in entities.get("names", []) or entities.get("speakers", []) or []:
        claims.append(make_claim("person", normalize_name(n), str(n), source, evidence_id, confidence, method))
    for u in entities.get("urls", []) or []:
        claims.append(make_claim("url", str(u).lower(), str(u), source, evidence_id, confidence, method))
    for e in entities.get("emails", []) or []:
        claims.append(make_claim("email", str(e).lower(), str(e), source, evidence_id, confidence, method))
    return claims

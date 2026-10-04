"""
CLAIM + ENTITY EXTRACTION shared across modalities.

A "claim" here is a small structured assertion we can cross-check, mostly:

    time   -> "at 5 PM", "17:00", "2021-03-05 18:30"
    place  -> "at Riverside Park", "near North Bridge"
    speaker-> "John Smith said ...", "Speaker: Jane Doe"

This is deliberately rule-based (dates + a couple of patterns + a small name
heuristic). It is what powers cross-modal checks such as
"image text says 5 PM but video metadata says 09:12".
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

# ISO datetime, common date forms, and clock times
_DATETIME_RE = re.compile(
    r"\b(\d{4})-(\d{2})-(\d{2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?\b")
_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_US_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
_CLOCK_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s?(AM|PM|am|pm)\b")
_CLOCK24_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_MONTH_RE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
    r"(\d{1,2}),?\s+(\d{2,4})\b", re.IGNORECASE)

_SPEAKER_RE = re.compile(
    r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,2})\s+"
    r"(?:said|stated|claimed|reported|confirmed|denied|recorded|narrat\w+)\b")
_SPEAKER_LABEL_RE = re.compile(
    r"\b(?:speaker|voice|interviewee|witness|narrator)\s*[:\-]\s*"
    r"([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,2})", re.IGNORECASE)

_PLACE_RE = re.compile(
    r"\b(?:at|in|near|from|around)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}"
    r"(?:\s+(?:Park|Bridge|Street|Avenue|Road|Station|Airport|Hotel|"
    r"Building|Office|Campus|Square|Centre|Center))?)")


def _parse_clock(hour: int, minute: Optional[int], meridiem: Optional[str]) -> str:
    if meridiem:
        m = meridiem.upper()
        if m == "PM" and hour < 12:
            hour += 12
        if m == "AM" and hour == 12:
            hour = 0
    return f"{hour:02d}:{minute or 0:02d}"


def extract_datetimes(text: str) -> list:
    """Return a list of ISO datetime strings found in text."""
    found = []
    for m in _DATETIME_RE.finditer(text):
        y, mo, d, hh, mm, ss = m.groups()
        try:
            dt = datetime(int(y), int(mo), int(d), int(hh), int(mm),
                          int(ss or 0), tzinfo=timezone.utc)
            found.append(dt.isoformat())
        except ValueError:
            continue
    for m in _MONTH_RE.finditer(text):
        try:
            dt = datetime.strptime(m.group(0).replace(".", ""), "%b %d, %Y")
            found.append(dt.replace(tzinfo=timezone.utc).isoformat())
        except ValueError:
            try:
                dt = datetime.strptime(m.group(0).replace(".", ""), "%B %d, %Y")
                found.append(dt.replace(tzinfo=timezone.utc).isoformat())
            except ValueError:
                continue
    for m in _DATE_RE.finditer(text):
        iso = m.group(0)
        if iso not in " ".join(found):
            found.append(iso + "T00:00:00+00:00")
    for m in _US_DATE_RE.finditer(text):
        a, b, y = m.groups()
        y = int(y)
        if y < 100:
            y += 2000
        try:
            found.append(datetime(y, int(a), int(b), tzinfo=timezone.utc).isoformat())
        except ValueError:
            continue
    return sorted(set(found))


def extract_times_of_day(text: str) -> list:
    """Return HH:MM strings for clock times (12h or 24h), avoiding double matches."""
    times = []
    consumed = []
    for m in _CLOCK_RE.finditer(text):
        times.append(_parse_clock(int(m.group(1)), int(m.group(2) or 0), m.group(3)))
        consumed.append((m.start(), m.end()))
    for m in _CLOCK24_RE.finditer(text):
        # skip if this span overlaps a 12h match (e.g. the "9:30" inside "9:30 PM")
        if any(not (m.end() <= s or m.start() >= e) for s, e in consumed):
            continue
        times.append(_parse_clock(int(m.group(1)), int(m.group(2)), None))
    return sorted(set(times))


def extract_speakers(text: str) -> list:
    speakers = []
    for m in _SPEAKER_RE.finditer(text):
        speakers.append(m.group(1).strip())
    for m in _SPEAKER_LABEL_RE.finditer(text):
        speakers.append(m.group(1).strip())
    # de-dup preserving order
    seen, out = set(), []
    for s in speakers:
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def extract_places(text: str) -> list:
    places = [m.group(1).strip() for m in _PLACE_RE.finditer(text)]
    seen, out = set(), []
    for p in places:
        if p.lower() not in seen and len(p) > 2:
            seen.add(p.lower())
            out.append(p)
    return out


def claims_from_text(text: str) -> dict:
    """Structured assertions we can cross-check against other modalities."""
    if not text:
        return {"datetimes": [], "times_of_day": [], "speakers": [], "places": []}
    return {
        "datetimes": extract_datetimes(text),
        "times_of_day": extract_times_of_day(text),
        "speakers": extract_speakers(text),
        "places": extract_places(text),
    }


def datetime_from_iso(value: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

"""
TEXT / DOCUMENT analyzer.

Supports:
  * .txt / .md / .csv  -> direct text read (encoding-tolerant)
  * .pdf               -> PyMuPDF (fitz) if available, otherwise a minimal
                          built-in PDF text extractor, otherwise an honest error
  * .json / .log       -> treated as text

Extraction (all regex / rule based, no external NLP model required):
  * dates & timestamps
  * names (capitalized sequences)
  * locations (gazetteer keywords)
  * emails, urls, phone-like numbers
  * "claims" = sentences containing assertion verbs / numbers

IMPORTANT: there is NO AI-text detector here. We only measure *surface* signals
(repetition, sentence-length uniformity, templated phrasing) and label them as
heuristics. We never claim a document was written by an AI.
"""

from __future__ import annotations

import os
import re
import zlib
from typing import Optional

import numpy as np

from backend.services.base import (
    Finding, ModalityResult, finalize, make_result,
    ENGINE_HEURISTIC, ENGINE_MODEL,
)
from backend.utils.helpers import clamp, risk_to_label
from backend.config import SUSPICIOUS_THRESHOLD, AUTHENTIC_THRESHOLD, EMBEDDING_DIM

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except Exception:  # pragma: no cover
    FITZ_AVAILABLE = False


DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b",
    r"\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b",
]
LOCATION_KEYWORDS = [
    "street", "avenue", "road", "city", "town", "district", "county", "state",
    "country", "park", "bridge", "building", "office", "campus", "station",
    "airport", "hotel", "restaurant", "north", "south", "east", "west",
]
CLAIM_VERBS = ["confirmed", "reported", "observed", "stated", "claimed", "found",
               "detected", "occurred", "happened", "recorded", "documented"]


def _read_text_file(path: str) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as fh:
                return fh.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def _read_pdf_fitz(path: str) -> tuple[str, int]:
    doc = fitz.open(path)
    try:
        pages = doc.page_count
        chunks = []
        for page in doc:
            chunks.append(page.get_text("text"))
        return "\n".join(chunks), pages
    finally:
        doc.close()


def _read_pdf_minimal(path: str) -> tuple[str, int]:
    """
    Extremely small PDF text extractor: pulls text inside BT..ET blocks from
    uncompressed or FlateDecode streams. Works for many simple generated PDFs
    and serves as a dependency-free fallback. Not a full PDF parser.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    page_count = raw.count(b"/Type /Page") + raw.count(b"/Type/Page")
    texts = []
    # find stream ... endstream blocks
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.S):
        block = match.group(1)
        try:
            if b"FlateDecode" in raw[max(0, match.start() - 200):match.start()]:
                block = zlib.decompress(block)
        except Exception:
            pass
        # extract text in parentheses within BT/ET (best-effort)
        for tmatch in re.finditer(rb"\((?:[^()\\]|\\.)*\)", block):
            piece = tmatch.group(0)[1:-1]
            try:
                texts.append(piece.decode("latin-1", "ignore"))
            except Exception:
                pass
    text = " ".join(texts).strip()
    return text, max(1, page_count)


def _extract_entities(text: str) -> dict:
    dates = []
    for pat in DATE_PATTERNS:
        dates.extend(re.findall(pat, text, flags=re.IGNORECASE))
    dates = sorted(set(d.strip() for d in dates if d.strip()))[:25]

    names = sorted(set(re.findall(r"\b(?:[A-Z][a-z]{2,}\s){1,2}[A-Z][a-z]{2,}\b", text)))[:15]
    emails = sorted(set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)))[:10]
    urls = sorted(set(re.findall(r"https?://[^\s)]+", text)))[:10]
    phones = sorted(set(re.findall(r"\+?\d[\d\s().-]{7,}\d", text)))[:10]

    lower = text.lower()
    locations = sorted(set(k for k in LOCATION_KEYWORDS if k in lower))[:15]

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = []
    for s in sentences:
        sl = s.lower()
        if any(v in sl for v in CLAIM_VERBS) or re.search(r"\b\d{4}\b", s):
            s = s.strip()
            if 20 <= len(s) <= 300:
                claims.append(s)
    claims = claims[:12]

    # Structured, cross-checkable assertions (time / speaker / place).
    from backend.services.claim_extraction import claims_from_text
    structured = claims_from_text(text)

    return {
        "dates": dates, "names": names, "emails": emails, "urls": urls,
        "phones": phones, "locations": locations, "claims": claims,
        "sentence_count": len(sentences),
        "datetimes": structured["datetimes"],
        "times_of_day": structured["times_of_day"],
        "speakers": structured["speakers"],
        "places": structured["places"],
    }


def _surface_style_signals(text: str, entities: dict) -> dict:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    lengths = [len(s.split()) for s in sentences] or [0]
    len_std = float(np.std(lengths))
    mean_len = float(np.mean(lengths)) + 1e-9
    uniformity = clamp(1.0 - (len_std / mean_len))  # 1 => very uniform sentence lengths
    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    vocab_ratio = len(set(words)) / (len(words) + 1e-9)
    repetition = clamp(1.0 - vocab_ratio)
    templated = clamp(len(entities["claims"]) / 6.0)
    return {
        "sentence_length_uniformity": round(uniformity, 4),
        "lexical_repetition": round(repetition, 4),
        "claim_density": round(templated, 4),
    }


def analyze(path: str, meta: dict) -> ModalityResult:
    result = make_result("document", meta, engine=ENGINE_HEURISTIC)
    ext = os.path.splitext(path)[1].lower()

    text, pages, extractor = "", 0, "none"
    try:
        if ext == ".pdf":
            if FITZ_AVAILABLE:
                text, pages = _read_pdf_fitz(path)
                extractor = "pymupdf"
            else:
                text, pages = _read_pdf_minimal(path)
                extractor = "minimal-pdf-fallback"
        else:
            text, pages, extractor = _read_text_file(path), 1, "plain-text"
    except Exception as exc:
        result.error = f"Could not read document: {exc}"
        result.quality = "unusable"
        result.findings.append(Finding(
            "corrupt", "Unreadable document",
            "The document could not be parsed and was skipped.", "high", "missing"))
        return finalize(result, EMBEDDING_DIM, meta.get("filename", "document"))

    text = (text or "").strip()
    if len(text) < 5:
        result.quality = "low"
        result.error = ("No extractable text was found. The PDF may be scanned "
                        "images only (OCR is not implemented in this prototype).")
        result.findings.append(Finding(
            "no_text", "No extractable text",
            "The document yielded little or no text; OCR is not available.", "medium", "missing"))
        result.features = {"pages": pages, "extractor": extractor, "char_count": len(text)}
        return finalize(result, EMBEDDING_DIM, meta.get("filename", "document"))

    entities = _extract_entities(text)
    style = _surface_style_signals(text, entities)
    char_count = len(text)
    word_count = len(re.findall(r"\b\w+\b", text))

    # heuristic risk: documents that are stylistically over-uniform / repetitive
    # and claim-heavy are flagged only weakly. This is surface signal, not
    # AI-text detection.
    uniformity_ind = style["sentence_length_uniformity"]
    repetition_ind = style["lexical_repetition"]
    claim_ind = style["claim_density"]
    sparse_meta_ind = 0.4 if (pages >= 1 and not entities["dates"]) else 0.0

    indicators = {
        "style_uniformity_indicator": round(uniformity_ind, 4),
        "repetition_indicator": round(repetition_ind, 4),
        "claim_density_indicator": round(claim_ind, 4),
        "missing_dates_indicator": round(sparse_meta_ind, 4),
    }
    risk = (0.30 * uniformity_ind + 0.25 * repetition_ind
            + 0.25 * claim_ind + 0.20 * sparse_meta_ind)

    model_used = False
    from backend.config import MODE
    if MODE == "model":
        try:
            from backend.models.registry import predict_individual
            model_score = predict_individual("document", {**indicators}, path)
            if model_score is not None:
                risk = float(model_score)
                model_used = True
                result.engine = ENGINE_MODEL
        except Exception:
            model_used = False

    result.score = clamp(risk)
    # documents are intrinsically hard to judge; cap the label influence
    result.label = risk_to_label(result.score, SUSPICIOUS_THRESHOLD, AUTHENTIC_THRESHOLD)
    result.confidence = clamp(0.35 + 0.4 * min(1.0, word_count / 200.0))

    result.features = {
        "extractor": extractor,
        "pages": pages,
        "char_count": char_count,
        "word_count": word_count,
        "date_count": len(entities["dates"]),
        "name_count": len(entities["names"]),
        "location_count": len(entities["locations"]),
        "claim_count": len(entities["claims"]),
        **style,
        **indicators,
    }
    result.detail = {
        "entities": entities,
        "preview": text[:600],
        "model_applied": model_used,
    }

    if entities["dates"]:
        result.findings.append(Finding(
            "dates", "Dates found",
            f"Extracted {len(entities['dates'])} date/time reference(s) for cross-checking.",
            "info", "supports"))
    else:
        result.findings.append(Finding(
            "no_dates", "No dates found",
            "No dates or timestamps were found to align with media evidence.", "low", "missing"))
    if entities["names"]:
        result.findings.append(Finding(
            "names", "Names found",
            f"Extracted {len(entities['names'])} possible person/organisation name(s).",
            "info", "supports"))
    if entities.get("speakers"):
        result.findings.append(Finding(
            "speakers", "Named speakers",
            "Document attributes statements to: "
            + ", ".join(entities["speakers"][:4]) + ".", "info", "supports"))
    if entities.get("times_of_day"):
        result.findings.append(Finding(
            "times_of_day", "Times of day",
            "Clock times found: " + ", ".join(entities["times_of_day"][:5])
            + ". These can be compared with media metadata.", "info", "supports"))
    if uniformity_ind > 0.6:
        result.findings.append(Finding(
            "uniform_style", "Unusually uniform style",
            "Sentence lengths are very uniform (surface heuristic only, NOT AI detection).",
            "low", "neutral"))
    if result.score <= AUTHENTIC_THRESHOLD:
        result.findings.append(Finding(
            "doc_ok", "No major textual contradiction",
            "Document text did not surface strong internal inconsistencies.", "info", "supports"))

    result.warnings.append(
        "Text signals are surface heuristics. This is NOT an AI-text detector.")
    return finalize(result, EMBEDDING_DIM, meta.get("filename", "document"))
